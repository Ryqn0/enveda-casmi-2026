"""Does the public engine's second ranker improve with leak-free fingerprints?

    uv run python scripts/cv_public_ranker.py

Uses megayak's public simulation rows (data/public/, from
megayak/casmi26-simulated-ranker-rows):

  nofp    51 features, the 12 fingerprint columns zero -- what W088 ships
  merged  the same 51 features with those 12 columns filled from megayak's
          leak-free fingerprint models (sim_rank_rows_fppair.npz)

The two files hold the same candidates in the same order (checked below), so
the ONLY difference between the two configurations is those 12 columns.

Evaluated on the natural-product query set (enveda-np-examples, the one the
public notebooks calibrate on and that reproduced the leaderboard ordering in
docs/12): 5-fold cross-validation grouped by query group, training on every
other set plus the other folds, with the engine's own ranker settings.
Predicted LB uses the public calibration  0.162 * class-1 + 0.220 * class-2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casmi.ranker import expected_rr  # noqa: E402

FP_COLS = [15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30]
# The engine's settings (casmi_engine.RANK), so this measures what would ship.
W_A = (0.35, 0.55)
SEEDS = (0, 1)
GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03,
           min_samples_leaf=80, l2_regularization=1.0)


def load():
    a = np.load(ROOT / "data/public/sim_rank_rows_nofp.npz", allow_pickle=True)
    b = np.load(ROOT / "data/public/sim_rank_rows_fppair.npz", allow_pickle=True)
    for k in ("Y", "S", "G", "Q"):
        assert np.array_equal(a[k], b[k]), f"rows not aligned on {k}"
    X0 = a["X"].astype(np.float32)
    assert np.all(X0[:, FP_COLS] == 0), "expected zero fingerprint columns in nofp"
    X1 = X0.copy()
    X1[:, FP_COLS] = b["X"][:, FP_COLS]
    sets = [str(s) for s in a["sets"]]
    return X0, X1, a["Y"], a["S"], a["G"], a["Q"], sets


def fit_predict(X, Y, S, tr, te):
    """Average of the engine's 4 models (2 class priors x 2 seeds)."""
    p = np.zeros(te.sum())
    for wA in W_A:
        for sd in SEEDS:
            m = HistGradientBoostingClassifier(random_state=sd, **GBM)
            m.fit(X[tr], Y[tr], sample_weight=np.where(S[tr] == 0, wA, 1.0 - wA))
            p += m.predict_proba(X[te])[:, 1]
    return p / (len(W_A) * len(SEEDS))


def main() -> None:
    X0, X1, Y, S, G, Q, sets = load()
    np_q = sets.index("np")
    is_np = Q == np_q
    groups = np.unique(G[is_np])
    rng = np.random.default_rng(0)
    fold_of = dict(zip(groups, rng.permutation(len(groups)) % 5))
    fold = np.array([fold_of.get(g, -1) for g in G])
    print(f"{len(groups)} natural-product query groups, 5 folds")

    rr = {"nofp": {}, "merged": {}}
    for k in range(5):
        te = is_np & (fold == k)
        tr = ~te
        for name, X in (("nofp", X0), ("merged", X1)):
            p = fit_predict(X, Y, S, tr, te)
            for g in np.unique(G[te]):
                sel = G[te] == g
                y = Y[te][sel]
                rr[name][g] = (expected_rr(p[sel], int(np.argmax(y)), 25)
                               if y.any() else 0.0)
        print(f"  fold {k + 1}/5 done", flush=True)

    s_of = {g: int(S[G == g][0]) for g in groups}
    print("\n| config | class 1 | class 2 | predicted LB |")
    print("|---|---:|---:|---:|")
    for name in ("nofp", "merged"):
        c1 = np.mean([v for g, v in rr[name].items() if s_of[g] == 0])
        c2 = np.mean([v for g, v in rr[name].items() if s_of[g] == 1])
        print(f"| {name} | {c1:.4f} | {c2:.4f} | {0.162 * c1 + 0.220 * c2:.4f} |")
    d = np.array([rr["merged"][g] - rr["nofp"][g] for g in groups])
    print(f"\npaired merged - nofp: {d.mean():+.4f} +/- {2 * d.std(ddof=1) / np.sqrt(len(d)):.4f}"
          f" (per query group, n={len(d)})")


if __name__ == "__main__":
    main()
