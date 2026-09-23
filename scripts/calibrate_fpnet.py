"""Per-bit Platt calibration of the fingerprint model's outputs.

WHY THIS IS NEEDED
    `score_candidates` treats sigmoid(logit) as the probability that a bit is
    set. It is not. Training uses pos_weight to stop the loss being minimised by
    predicting all zeros, which deliberately biases the model toward predicting
    1s — measured precision 0.21 against recall 0.76. Every "probability" it
    emits is therefore systematically too high, and by a different amount for
    every bit.

    Ranking a candidate by the likelihood of its bit pattern under miscalibrated
    probabilities weights the evidence wrongly: a bit the model is confidently
    wrong about counts as much as one it is reliably right about.

WHAT THIS DOES
    Fits a per-bit logistic map  p_i = sigmoid(A_i * z_i + B_i)  on the held-out
    validation split, so the reported probability matches the observed frequency.
    This is the calibration step behind the "modified Platt score" used by
    CSI:FingerID and MSNovelist (Dührkop 2015; Stravs 2022) — those also fold in
    per-bit sensitivity/specificity, which per-bit calibration largely subsumes.

    Deliberately NOT a transcription of the published ModPlatt formula. Its four
    branches are ambiguous as printed, and a scoring function I cannot verify is
    exactly the kind of thing that fails silently. This is the same idea in a
    form whose correctness is checkable — and it is checked, below.

    uv run python scripts/calibrate_fpnet.py --model data/fpnet/fpnet.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the training split and collator verbatim. Re-deriving either here would
# put calibration on different data than training saw.
from train_fpnet import SpectraSet, collate  # noqa: E402

from casmi.fpnet import load, save  # noqa: E402

DATA = ROOT / "data" / "fpnet"


def fit_platt(logits: np.ndarray, labels: np.ndarray, iters: int = 200,
              lr: float = 0.1, device: str = "cpu"):
    """Fit per-bit A, B minimising log-loss of sigmoid(A*z + B).

    All bits are fitted jointly as independent 1-D logistic regressions, which
    on a GPU is one batched optimisation rather than 1,576 separate ones.
    """
    z = torch.as_tensor(logits, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)

    # Start at the identity so a well-calibrated model is left alone.
    A = torch.ones(z.shape[1], device=device, requires_grad=True)
    B = torch.zeros(z.shape[1], device=device, requires_grad=True)
    # strong_wolfe is not optional here. BCEWithLogitsLoss averages over all
    # ~70M elements, so the raw gradient is ~1e-8; with LBFGS's default fixed
    # step the parameters move by nothing and the fit silently returns the
    # identity it started from. A line search rescales each step to the
    # curvature, which makes the fit independent of the loss's scale.
    opt = torch.optim.LBFGS([A, B], max_iter=iters, lr=lr,
                            line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(A * z + B, y)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        final = loss_fn(A * z + B, y).item()
        before = loss_fn(z, y).item()
    return A.detach().cpu().numpy(), B.detach().cpu().numpy(), before, final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DATA / "fpnet.pt")
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(args.model, device=dev)
    d = np.load(DATA / "train.npz", allow_pickle=True)
    n_bits = int(d["n_bits"])
    fp = np.unpackbits(d["fp"], axis=1)[:, :n_bits].astype(np.float32)
    y_row = d["y_row"]

    # EXACTLY the split train_fpnet made — same seed, same fraction, same order.
    # Calibrating on data the model trained on would fit the fit, not the bias.
    rng = np.random.default_rng(0)
    structs = np.unique(y_row)
    val_structs = set(rng.choice(structs, int(len(structs) * args.val_frac),
                                 replace=False).tolist())
    va_idx = np.flatnonzero(np.array([s in val_structs for s in y_row]))
    print(f"calibrating on {len(va_idx):,} held-out spectra / "
          f"{len(val_structs):,} structures, {n_bits} bits")

    loader = DataLoader(SpectraSet(d, va_idx, fp), batch_size=512,
                        collate_fn=collate, num_workers=2)
    zs, ys = [], []
    with torch.no_grad():
        for mz, it, pad, prec, add, ins, ce, y in loader:
            out = model(mz.to(dev), it.to(dev), pad.to(dev), prec.to(dev),
                        add.to(dev), ins.to(dev), ce.to(dev))
            zs.append(out.float().cpu().numpy())
            ys.append(y.numpy())
    Z, Y = np.concatenate(zs), np.concatenate(ys)

    sig = lambda x: np.where(x >= 0, 1 / (1 + np.exp(-np.clip(x, -60, 60))),
                             np.exp(np.clip(x, -60, 60)) / (1 + np.exp(np.clip(x, -60, 60))))
    p_raw = sig(Z)
    print(f"\nBEFORE: mean predicted P(bit=1) {p_raw.mean():.4f} "
          f"vs actual {Y.mean():.4f}  "
          f"-> over-predicts by {p_raw.mean()/Y.mean():.2f}x")

    A, B, before, after = fit_platt(Z, Y, device=dev)
    p_cal = sig(A * Z + B)
    print(f"AFTER : mean predicted P(bit=1) {p_cal.mean():.4f} "
          f"vs actual {Y.mean():.4f}")
    print(f"\nlog-loss {before:.4f} -> {after:.4f}")
    print(f"A: median {np.median(A):.3f} [{A.min():.2f}, {A.max():.2f}]")
    print(f"B: median {np.median(B):.3f} [{B.min():.2f}, {B.max():.2f}]")

    # Calibration must IMPROVE log-loss; it is fitted to do so, and a failure
    # here means the split or the wiring is wrong, not that the idea is.
    assert after < before, "calibration made log-loss worse — check the split"
    assert abs(p_cal.mean() - Y.mean()) < 0.3 * Y.mean(), (
        f"calibrated mean {p_cal.mean():.4f} still far from actual {Y.mean():.4f} "
        "— the optimiser stalled rather than converged")

    save(model, args.model, **{k: v for k, v in ck.items()
                               if k not in ("cfg", "state", "platt_a", "platt_b")},
         platt_a=A, platt_b=B)
    print(f"\n-> saved platt_a / platt_b into {args.model}")


if __name__ == "__main__":
    main()
