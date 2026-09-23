"""Fine-tune fpnet to RANK same-mass candidates, not just predict bits.

    uv run python scripts/train_contrastive.py --base data/fpnet/fpnet_v4.pt \
        --out data/fpnet/fpnet_v4c.pt

Input: data/fpnet/contrastive.npz from scripts/prepare_contrastive.py.

THE LOSS
    For a spectrum with logits z and candidate fingerprints c_1..c_L:

        score_j = c_j . z

    That is exactly how inference ranks (score_candidates' log-likelihood,
    minus a term that is identical for every candidate). Softmax over the list
    and take cross-entropy with the true candidate: the model is trained on the
    exact comparison it has to win at test time, against same-mass isomers --
    the hardest negatives that exist.

    The original per-bit BCE stays in the loss as a regulariser. Without it the
    ranking loss alone can drift the logits into a shape that separates this
    training set's lists but no longer means "probability of a substructure".

HONEST EVALUATION
    The in-training metric below is on held-out LISTS of training structures,
    so it is optimistic -- only a sanity check that the loss is doing
    something. The real test is validate_fpnet.py --unseen-only afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_fpnet import SpectraSet, collate, evaluate  # noqa: E402

from casmi.fpnet import load, save  # noqa: E402

DATA = ROOT / "data" / "fpnet"


class ContrastSet(Dataset):
    """A spectrum (via SpectraSet) plus its candidate list."""

    def __init__(self, spectra: SpectraSet, ptr, cand, true_pos, idx):
        self.spectra, self.ptr, self.cand = spectra, ptr, cand
        self.true_pos, self.idx = true_pos, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (self.spectra[j],
                self.cand[self.ptr[j]:self.ptr[j + 1]],
                int(self.true_pos[j]))


def make_collate(fp_packed: np.ndarray, bit_sel: np.ndarray):
    """Collate spectra with train_fpnet's collate; candidates are unpacked here.

    Fingerprints stay PACKED in memory (256 bytes each) and are only unpacked
    for the ~8k candidates in the current batch -- the whole table unpacked
    would be over a gigabyte.
    """
    def fn(batch):
        spectra, lists, tpos = zip(*batch)
        spec = collate(list(spectra))
        L = max(len(l) for l in lists)
        B = len(lists)
        C = np.zeros((B, L, len(bit_sel)), dtype=np.float32)
        M = np.zeros((B, L), dtype=bool)
        for b, l in enumerate(lists):
            bits = np.unpackbits(fp_packed[l], axis=1)[:, bit_sel]
            C[b, :len(l)] = bits
            M[b, :len(l)] = True
        return (*spec, torch.as_tensor(C), torch.as_tensor(M),
                torch.as_tensor(tpos, dtype=torch.long))
    return fn


def rank_logits(z, C, M, log_tau):
    """score_j = c_j . z / tau, padding masked out.

    Computed in float32 even under bf16 autocast: the score is a sum over
    ~1,600 bits, and bf16's 8-bit mantissa would blur the few-unit gaps that
    separate isomers.
    """
    s = torch.einsum("bln,bn->bl", C, z.float()) / log_tau.exp()
    return s.masked_fill(~M, float("-inf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA / "train.npz")
    ap.add_argument("--lists", type=Path, default=DATA / "contrastive.npz")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only this many lists (smoke tests); 0 = all")
    ap.add_argument("--tag", required=True,
                    help="model_tag stored in the checkpoint; the notebook "
                         "asserts it to catch a stale dataset mount")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(args.base, device=dev)
    bit_sel = np.asarray(ck["bit_sel"])
    train_keys = set(np.asarray(ck["train_keys"]).tolist())

    d = np.load(args.data, allow_pickle=True)
    n_bits = int(d["n_bits"])
    assert n_bits == len(bit_sel), "--data does not match the base model's bits"
    fp = np.unpackbits(d["fp"], axis=1)[:, :n_bits]
    keys, y_row = d["keys"], d["y_row"]

    c = np.load(args.lists, allow_pickle=True)
    q_rows, ptr, cand, true_pos = c["q_rows"], c["ptr"], c["cand"], c["true_pos"]
    fp_packed = c["fp_packed"]
    print(f"{len(q_rows):,} candidate lists, {len(fp_packed):,} distinct candidates")

    # Hold out 5% of lists for the (optimistic) in-training sanity metric.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(q_rows))
    if args.limit:
        perm = perm[: args.limit]
    n_va = len(perm) // 20
    va_i, tr_i = perm[:n_va], perm[n_va:]

    spectra = SpectraSet(d, q_rows, fp)
    cfn = make_collate(fp_packed, bit_sel)
    tr = DataLoader(ContrastSet(spectra, ptr, cand, true_pos, tr_i),
                    batch_size=args.batch, shuffle=True, collate_fn=cfn,
                    num_workers=4, persistent_workers=True)
    va = DataLoader(ContrastSet(spectra, ptr, cand, true_pos, va_i),
                    batch_size=args.batch, collate_fn=cfn, num_workers=2)

    # The base model's own val split, to check bit prediction does not degrade.
    va_bits = np.flatnonzero(np.array([keys[r] not in train_keys for r in y_row]))
    if args.limit:
        va_bits = va_bits[: 2000]
    bits_loader = DataLoader(SpectraSet(d, va_bits, fp), batch_size=512,
                             collate_fn=collate, num_workers=2)
    eval_loss = nn.BCEWithLogitsLoss()

    pos_rate = fp.mean()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.full(
        (n_bits,), float((1 - pos_rate) / pos_rate), device=dev))
    # Learnable softmax temperature. Ranking is invariant to it, so it does not
    # need to ship -- it only lets the loss choose a sensible sharpness.
    #
    # Starts at 10, not 1, with its own learning rate. Isomer scores differ by
    # tens of units, so at tau=1 the softmax is near one-hot and confidently
    # wrong: the smoke test's loss was 6.3, worse than guessing uniformly over
    # ~57 candidates (ln 57 = 4.0). Sharing the model's 1e-4 learning rate, tau
    # never moved off 1.00 to fix that itself.
    log_tau = torch.tensor(np.log(10.0), device=dev, requires_grad=True)
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr, "weight_decay": 0.01},
        {"params": [log_tau], "lr": 1e-2, "weight_decay": 0.0},
    ])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, 1e-2], total_steps=args.epochs * len(tr))

    def sanity(tag):
        model.eval()
        hits, rr, n = 0, 0.0, 0
        with torch.no_grad():
            for *spec, C, M, T in va:
                spec = [t.to(dev) for t in spec[:7]]
                z = model(*spec)
                s = rank_logits(z, C.to(dev), M.to(dev), log_tau)
                true_s = s.gather(1, T.to(dev)[:, None])
                rank = 1 + (s > true_s).sum(1)
                hits += (rank == 1).sum().item()
                rr += (1.0 / rank.float()).sum().item()
                n += len(T)
        vl, p, r, f1 = evaluate(model, bits_loader, dev, eval_loss)
        print(f"{tag} | lists top1 {hits/n:.3f} MRR {rr/n:.4f} (optimistic) "
              f"| base-val BCE {vl:.4f} F1 {f1:.3f}", flush=True)
        model.train()
        return vl

    sanity("before  ")
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for *spec, y, C, M, T in tr:
            spec = [t.to(dev, non_blocking=True) for t in spec]
            y, C, M, T = y.to(dev), C.to(dev), M.to(dev), T.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                z = model(*spec)
            loss = (nn.functional.cross_entropy(rank_logits(z, C, M, log_tau), T)
                    + args.bce_weight * bce(z.float(), y))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(T)
            n += len(T)
        print(f"epoch {ep} | loss {tot/n:.4f} | tau {log_tau.exp().item():.2f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        vl = sanity(f"epoch {ep}")

    meta = {k: v for k, v in ck.items() if k not in ("cfg", "state")}
    meta.update(val_loss=vl, model_tag=args.tag, contrastive_from=str(args.base))
    save(model, args.out, **meta)
    print(f"-> {args.out} (tag {args.tag})")


if __name__ == "__main__":
    main()
