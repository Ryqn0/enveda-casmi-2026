"""Train and evaluate the learned ranker against the fixed blend.

    uv run python scripts/train_ranker.py --dump data/ranker_v3.parquet

Input is the per-candidate dump from
    validate_fpnet.py --unseen-only --frag --class1-frac 0.15 --dump ...

Every number printed here is out-of-fold: 5-fold cross-validation grouped by
MOLECULE, so a molecule's candidates are never split between the fold that
trains the ranker and the fold that scores it. The fixed blend is scored on
exactly the same held-out molecules, so the comparison is like-for-like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi.ranker import (FEATURES, RankNet, expected_rr,  # noqa: E402
                          ranker_features)

K = 25
# Training runs on the GPU when there is one (the first run took ~50 min on
# CPU while fpnet training held the GPU). Evaluation stays on CPU: its
# per-molecule loop would otherwise force a GPU sync on every element.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLEND = {"z_fp": 1.0, "z_an": 0.5, "z_fr": 0.25}   # what rank.py ships


def rebuild(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the dumped features with ones from the SERVING function.

    The dump computed features inline, inside the validator; the notebook
    computes them with casmi.ranker.ranker_features. Training on the dumped
    copy would mean training on one computation and serving another -- this
    project's most expensive bug class. So the dump is used only for its RAW
    inputs, and every feature the model sees is rebuilt by the exact function
    that will run at test time. Train and serve cannot disagree by
    construction.

    The comparison below is kept as a diagnostic. It found two real
    discrepancies in the inline version, both from TIES:
      - float32 z-scores of an all-tied channel come out as +/-1 instead of 0
        (the float32 mean is off by one rounding step, so the std is not 0).
        Harmless to the fixed blend -- a constant per molecule cannot reorder
        its candidates -- but a nonlinear ranker trained on it would be skewed.
      - rank features of tied candidates depend on sort stability.
    """
    out = df.copy()
    worst = {f: 0.0 for f in FEATURES}
    for _, g in df.groupby("qid", sort=False):
        f = ranker_features(g.fp_gap.to_numpy(), g.an_raw.to_numpy(),
                            g.fr_raw.to_numpy(), float(g.an_conf.iloc[0]),
                            int(round(np.exp(g.log_npk.iloc[0]))))
        out.loc[g.index, FEATURES] = f
        for j, name in enumerate(FEATURES):
            worst[name] = max(worst[name], np.abs(f[:, j] - g[name].to_numpy()).max())
    print("dumped vs serving-function features, max |difference| (diagnostic):")
    print("  " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    # Features that involve no ties must still agree -- if these drift, the
    # raw inputs themselves are wrong, which rebuilding cannot fix.
    for name in ("an_raw", "fr_raw", "an_conf", "log_ncand", "log_npk"):
        assert worst[name] < 1e-3, f"raw input {name} disagrees: {worst[name]}"
    return out


def to_tensors(df: pd.DataFrame, mean=None, std=None):
    """Pad molecules to a dense [Q, Cmax, F] tensor with a validity mask."""
    groups = [g for _, g in df.groupby("qid", sort=False)]
    cmax = max(len(g) for g in groups)
    Q, F = len(groups), len(FEATURES)
    X = np.zeros((Q, cmax, F), dtype=np.float32)
    M = np.zeros((Q, cmax), dtype=bool)
    Y = np.full(Q, -1, dtype=np.int64)   # index of the true candidate, -1 = absent
    for i, g in enumerate(groups):
        n = len(g)
        X[i, :n] = g[FEATURES].to_numpy(np.float32)
        M[i, :n] = True
        pos = np.flatnonzero(g.label.to_numpy())
        if len(pos):
            Y[i] = pos[0]
    if mean is None:
        # Standardise with statistics from REAL candidates only; padding zeros
        # would drag every mean toward 0.
        mean, std = X[M].mean(0), X[M].std(0) + 1e-6
    X = (X - mean) / std
    X[~M] = 0.0
    return (torch.as_tensor(X), torch.as_tensor(M), torch.as_tensor(Y),
            mean, std)


def train(X, M, Y, hidden: int, steps: int, seed: int = 0) -> RankNet:
    torch.manual_seed(seed)
    net = RankNet(len(FEATURES), hidden).to(DEV)
    X, M, Y = X.to(DEV), M.to(DEV), Y.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    has = Y >= 0   # molecules whose answer is among the candidates
    for _ in range(steps):
        opt.zero_grad()
        s = net(X).masked_fill(~M, float("-inf"))
        # Listwise softmax: maximise P(true candidate) among ITS OWN candidates.
        loss = torch.nn.functional.cross_entropy(s[has], Y[has])
        loss.backward()
        opt.step()
    return net.cpu().eval()   # evaluated and saved on CPU


def mrr(scores: torch.Tensor, M, Y) -> np.ndarray:
    """Reciprocal rank per molecule, 0 if absent or ranked below K."""
    s = scores.masked_fill(~M, float("-inf"))
    out = np.zeros(len(Y))
    for i in range(len(Y)):
        if Y[i] < 0:
            continue
        valid = s[i][M[i]].numpy()        # this molecule's real candidates
        out[i] = expected_rr(valid, int(Y[i]), K)
    return out


def blend_scores(X_raw, M):
    """The fixed blend, on UNstandardised features (it is defined on z-scores)."""
    idx = {f: FEATURES.index(f) for f in BLEND}
    s = sum(w * X_raw[..., idx[f]] for f, w in BLEND.items())
    return s.masked_fill(~M, float("-inf"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ranker.pt")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_parquet(args.dump)
    print(f"{len(df):,} candidates / {df.qid.nunique():,} molecules "
          f"({df.groupby('qid').is_c1.first().mean():.0%} class 1)")
    df = rebuild(df)

    qids = df.qid.unique()
    rng = np.random.default_rng(0)
    fold_of = dict(zip(qids, rng.permutation(len(qids)) % args.folds))
    c1_of = df.groupby("qid").is_c1.first()

    configs = [("linear", 0, 300), ("mlp", 32, 100), ("mlp", 32, 300),
               ("mlp", 32, 1000)]
    oof = {"blend": [], **{f"{n}-{st}": [] for n, _, st in configs}}
    is_c1 = []

    for k in range(args.folds):
        tr = df[df.qid.map(fold_of) != k]
        te = df[df.qid.map(fold_of) == k]
        Xtr, Mtr, Ytr, mean, std = to_tensors(tr)
        Xte, Mte, Yte, _, _ = to_tensors(te, mean, std)
        # raw (unstandardised) copy for the fixed blend
        Xte_raw = Xte * torch.as_tensor(std) + torch.as_tensor(mean)
        oof["blend"].append(mrr(blend_scores(Xte_raw, Mte), Mte, Yte))
        for name, hidden, steps in configs:
            net = train(Xtr, Mtr, Ytr, hidden, steps)
            with torch.no_grad():
                oof[f"{name}-{steps}"].append(mrr(net(Xte), Mte, Yte))
        is_c1.append(np.array([c1_of[q] for q in te.qid.unique()]))

    c1 = np.concatenate(is_c1)
    print(f"\nout-of-fold MRR@{K}, {len(c1):,} molecules "
          f"({c1.sum()} class 1, {(~c1).sum()} class 2):\n")
    print(f"| method | all | class 1 | class 2 |")
    print(f"|---|---:|---:|---:|")
    for name, parts in oof.items():
        r = np.concatenate(parts)
        print(f"| {name} | {r.mean():.4f} | {r[c1].mean():.4f} | "
              f"{r[~c1].mean():.4f} |")

    # Paired comparison against the blend: per-molecule differences, so
    # molecule-to-molecule difficulty cancels and only the METHOD varies.
    base = np.concatenate(oof["blend"])
    print("\npaired difference vs blend (mean +/- 2 standard errors):")
    for name in oof:
        if name == "blend":
            continue
        d = np.concatenate(oof[name]) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {name:10s} {d.mean():+.4f} +/- {2 * se:.4f}")

    # Final model on everything, with the most-used config. Pick it by hand
    # from the table above rather than automatically -- automatic selection
    # on the same folds would quietly overfit the choice.
    X, Mk, Y, mean, std = to_tensors(df)
    net = train(X, Mk, Y, hidden=32, steps=300)
    # The fpnet whose features this ranker learned. The notebook asserts it
    # equals the shipped fpnet's model_tag: a ranker trained on one model's
    # scores and served another's is the same train/serve skew as every
    # other bug in this project, just one layer up.
    fpnet_tag = str(df.fpnet_tag.iloc[0]) if "fpnet_tag" in df else "unknown"
    assert "fpnet_tag" not in df or df.fpnet_tag.nunique() == 1, \
        "dump mixes features from several fpnet models"
    torch.save({"cfg": net.cfg, "state": net.state_dict(), "features": FEATURES,
                "mean": mean, "std": std, "steps": 300,
                "trained_on": str(args.dump), "fpnet_tag": fpnet_tag}, args.out)
    print(f"\n-> {args.out} (mlp-300, trained on all {len(qids):,} molecules, "
          f"fpnet {fpnet_tag})")


if __name__ == "__main__":
    main()
