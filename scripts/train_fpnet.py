"""Train the spectrum -> fingerprint model.

    uv run python scripts/train_fpnet.py --epochs 12

Splits by STRUCTURE, never by spectrum. A molecule measured ten times would
otherwise put near-identical spectra in both train and validation, and the model
would score well by memorising rather than generalising — the same mistake in a
different costume as the degenerate validation holdout in Phase 1.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from casmi.fpnet import FPNet, save

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fpnet"


class SpectraSet(Dataset):
    """Wraps the prepared arrays; converts to tensors in the collate step."""

    def __init__(self, d, idx, fp):
        self.mz, self.inten, self.npk = d["mz"][idx], d["inten"][idx], d["npk"][idx]
        self.prec, self.ce = d["prec"][idx], d["ce"][idx]
        self.add_id, self.ins_id = d["add_id"][idx], d["ins_id"][idx]
        # Store the INDEX into the fingerprint table, not a copy of the row.
        # `fp[y_row[idx]]` materialised one 1,583-float row per spectrum:
        # 896k x 1583 x 4 bytes = 5.7 GB for labels alone, which is what made
        # training hold 11 GB and blocked every validation run alongside it.
        # Many spectra share a structure, so the table is far smaller.
        self.fp, self.y_row = fp, d["y_row"][idx]

    def __len__(self):
        return len(self.npk)

    def __getitem__(self, i):
        # Return only the REAL peaks; the collator decides the padded width.
        n = int(self.npk[i])
        return (self.mz[i][:n], self.inten[i][:n], self.prec[i],
                int(self.add_id[i]), int(self.ins_id[i]), self.ce[i],
                self.fp[self.y_row[i]])


class SortishBatches(torch.utils.data.Sampler):
    """Batch spectra of similar length together, in a shuffled order.

    Dynamic padding alone is not enough. ~20% of spectra sit at the 256-peak
    cap, so at batch size 256 essentially EVERY batch contains at least one long
    spectrum and therefore pads to 256 — exactly what dynamic padding was meant
    to avoid.

    So: shuffle, cut into large pools, sort each pool by length, and slice
    batches from it. Spectra in a batch then have similar lengths and the padded
    width tracks the data. The pool is shuffled each epoch and the batch order
    is shuffled too, so batches are not identical between epochs — which pure
    length-sorting would cause, and which harms training.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, pool_mult: int = 50,
                 seed: int = 0):
        self.lengths = lengths
        self.bs = batch_size
        self.pool = batch_size * pool_mult
        self.rng = np.random.default_rng(seed)
        self.n_batches = len(lengths) // batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(idx), self.pool):
            chunk = idx[start : start + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk) - self.bs + 1, self.bs):
                batches.append(chunk[b : b + self.bs].tolist())
        self.rng.shuffle(batches)
        yield from batches


def collate(batch):
    """Pad each batch to ITS longest spectrum, not to the global maximum.

    The cap is 256 peaks but the median spectrum has 44, so padding everything
    to 256 would run attention — which is quadratic in sequence length — over
    mostly padding. Batch-local padding keeps the cost proportional to the data
    actually present.
    """
    mzs, its, prec, add, ins, ce, y = zip(*batch)
    width = max(len(m) for m in mzs)
    B = len(batch)

    mz = np.zeros((B, width), dtype=np.float32)
    it = np.zeros((B, width), dtype=np.float32)
    pad = np.ones((B, width), dtype=bool)
    for i, (m, v) in enumerate(zip(mzs, its)):
        n = len(m)
        mz[i, :n], it[i, :n] = m, v
        pad[i, :n] = False          # True marks slots attention must ignore

    t = torch.as_tensor
    return (t(mz), t(it), t(pad), t(np.stack(prec)),
            t(np.array(add), dtype=torch.long), t(np.array(ins), dtype=torch.long),
            t(np.stack(ce)), t(np.stack(y), dtype=torch.float32))


@torch.no_grad()
def evaluate(model, loader, dev, loss_fn):
    """Loss plus bit-level accuracy metrics on held-out structures."""
    model.eval()
    tot, n, tp, fp_, fn_ = 0.0, 0, 0, 0, 0
    for mz, it, pad, prec, add, ins, ce, y in loader:
        mz, it, pad = mz.to(dev), it.to(dev), pad.to(dev)
        prec, add, ins, ce, y = (prec.to(dev), add.to(dev), ins.to(dev),
                                 ce.to(dev), y.to(dev))
        logits = model(mz, it, pad, prec, add, ins, ce)
        tot += loss_fn(logits, y).item() * len(y)
        n += len(y)
        pred = logits > 0                       # logit > 0 <=> probability > 0.5
        tp += (pred & (y > 0.5)).sum().item()
        fp_ += (pred & (y < 0.5)).sum().item()
        fn_ += (~pred & (y > 0.5)).sum().item()
    # F1 on the positive bits. Plain accuracy is useless here: only ~3% of bits
    # are set, so predicting all-zero would score 97%.
    prec_ = tp / max(tp + fp_, 1)
    rec = tp / max(tp + fn_, 1)
    f1 = 2 * prec_ * rec / max(prec_ + rec, 1e-9)
    return tot / max(n, 1), prec_, rec, f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=DATA / "fpnet.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(DATA / "train.npz", allow_pickle=True)
    n_bits = int(d["n_bits"])
    # uint8, not float32: 4x smaller, and collate() casts each batch's labels
    # to float32 anyway, which is the only place the float type is needed.
    fp = np.unpackbits(d["fp"], axis=1)[:, :n_bits]
    y_row = d["y_row"]
    print(f"{len(y_row):,} spectra | {fp.shape[0]:,} structures | {n_bits} bits | {dev}")

    # Split by structure. Spectra of the same molecule must never straddle it.
    rng = np.random.default_rng(0)
    structs = np.unique(y_row)
    val_structs = set(rng.choice(structs, int(len(structs) * args.val_frac),
                                 replace=False).tolist())
    is_val = np.array([s in val_structs for s in y_row])
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"train {len(tr_idx):,} spectra / {len(structs) - len(val_structs):,} structures")
    print(f"val   {len(va_idx):,} spectra / {len(val_structs):,} structures")
    assert not (set(y_row[tr_idx]) & set(y_row[va_idx])), "structure leaked across split"

    # Everything the checkpoint records, computed ONCE, here.
    #
    # `d` is a lazy NpzFile: every d["key"] access re-reads and decompresses
    # that array from the zip. The first version of train_keys indexed
    # d["keys"] inside a list comprehension, so it unpickled all 269k keys
    # 255k times -- quadratic, hours per checkpoint save, GPU idle at 1%
    # throughout. Read each array once into a plain variable.
    keys = d["keys"]
    train_keys = np.array([keys[r] for r in structs if r not in val_structs],
                          dtype=object)
    ck_meta = dict(n_bits=n_bits, bit_sel=d["bit_sel"],
                   adduct_list=d["adduct_list"],
                   max_peaks=int(d["mz"].shape[1]), train_keys=train_keys)
    print(f"checkpoint will record {len(train_keys):,} training structures")

    tr_set = SpectraSet(d, tr_idx, fp)
    tr = DataLoader(tr_set,
                    batch_sampler=SortishBatches(tr_set.npk, args.batch),
                    collate_fn=collate, num_workers=4, persistent_workers=True)
    va = DataLoader(SpectraSet(d, va_idx, fp), batch_size=args.batch * 2,
                    collate_fn=collate, num_workers=2, persistent_workers=True)

    model = FPNet(n_bits=n_bits, n_adducts=len(d["adduct_list"]),
                  d=args.dim, layers=args.layers).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par/1e6:.1f}M parameters")

    # Only ~3% of bits are set, so an unweighted BCE is minimised by predicting
    # all zeros. pos_weight raises the cost of missing a set bit to rebalance.
    pos_rate = fp.mean()
    pos_weight = torch.full((n_bits,), float((1 - pos_rate) / pos_rate), device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_loss = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.1
    )
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for mz, it, pad, prec, add, ins, ce, y in tr:
            mz, it, pad = mz.to(dev, non_blocking=True), it.to(dev), pad.to(dev)
            prec, add, ins, ce, y = (prec.to(dev), add.to(dev), ins.to(dev),
                                     ce.to(dev), y.to(dev))
            opt.zero_grad(set_to_none=True)
            if scaler:
                # bfloat16 autocast: ~2x throughput on this GPU, and bf16 has
                # the same exponent range as fp32 so no loss scaling drama.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(mz, it, pad, prec, add, ins, ce), y)
                loss.backward()
            else:
                loss = loss_fn(model(mz, it, pad, prec, add, ins, ce), y)
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(y)
            seen += len(y)

        vl, p, r, f1 = evaluate(model, va, dev, eval_loss)
        print(f"epoch {ep:2d} | train {run/seen:.4f} | val {vl:.4f} "
              f"| bit P {p:.3f} R {r:.3f} F1 {f1:.3f} | {time.time()-t0:.0f}s",
              flush=True)
        if vl < best:
            best = vl
            # max_peaks travels WITH the weights. Without it, inference has to
            # guess the cap the model was trained at, and guessing 128 for a
            # 256-peak model silently truncates every rich spectrum.
            # train_keys (in ck_meta) is the set of structures this model
            # trained on, so a later holdout can exclude them without
            # re-deriving the split from a train.npz that may since have been
            # regenerated with different sampling (docs/09).
            save(model, args.out, val_loss=vl, val_f1=f1, epoch=ep, **ck_meta)

    print(f"best val loss {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
