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
        self.y = fp[d["y_row"][idx]]

    def __len__(self):
        return len(self.npk)

    def __getitem__(self, i):
        n = int(self.npk[i])
        # pad marks slots beyond the real peak count so attention ignores them.
        pad = np.ones(self.mz.shape[1], dtype=bool)
        pad[:n] = False
        return (self.mz[i], self.inten[i], pad, self.prec[i],
                int(self.add_id[i]), int(self.ins_id[i]), self.ce[i], self.y[i])


def collate(batch):
    mz, it, pad, prec, add, ins, ce, y = zip(*batch)
    t = torch.as_tensor
    return (t(np.stack(mz)), t(np.stack(it)), t(np.stack(pad)),
            t(np.stack(prec)), t(np.array(add), dtype=torch.long),
            t(np.array(ins), dtype=torch.long), t(np.stack(ce)),
            t(np.stack(y), dtype=torch.float32))


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
    fp = np.unpackbits(d["fp"], axis=1)[:, :n_bits].astype(np.float32)
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

    tr = DataLoader(SpectraSet(d, tr_idx, fp), batch_size=args.batch, shuffle=True,
                    collate_fn=collate, num_workers=4, drop_last=True,
                    persistent_workers=True)
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
            save(model, args.out, n_bits=n_bits, bit_sel=d["bit_sel"],
                 adduct_list=d["adduct_list"], val_loss=vl, val_f1=f1)

    print(f"best val loss {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
