"""Can a learned STACKER combine the rankers better than our fixed rule? (docs/16)

    uv run python scripts/stack_harness.py

Inputs: the two harness dumps (see blend_harness.py), now carrying the engine's
ranker A and ranker B separately, plus our ranker. Per candidate of the engine's
list, a stacker sees each ranker's score, within-molecule rank and gap to the
best, whether the candidate is in our window, and molecule context.

Every learned option is cross-validated 5-fold by MOLECULE (both simulations
of a molecule in the same fold): a molecule is always scored by a combiner
that never saw it. Compared, paired, with the rule that scored LB 0.355:
    final = 0.9 * engine blend rank + 0.1 * our rank
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casmi.features import rank01  # noqa: E402
from casmi.ranker import expected_rr  # noqa: E402

C1_W, C2_W, K = 0.162, 0.220, 25
FOLDS = 5


def groups():
    """One dict per (molecule, simulation) on the engine's candidate list."""
    eng = pickle.load(open(ROOT / "data/harness/dump_engine_ties.pkl", "rb"))
    own = pickle.load(open(ROOT / "data/own/dump_own.pkl", "rb"))
    mols = sorted(set(eng["c1"]) & set(eng["c2"]) & set(own["c1"]) & set(own["c2"]))
    out = []
    for sim in ("c1", "c2"):
        for m in mols:
            keys, blend, a, b = eng[sim][m]
            od = dict(zip(*own[sim][m]))
            o = np.array([od.get(k, -1.0) for k in keys])
            ra, rb, ro, rbl = (1 - rank01(v) for v in (a, b, o, blend))
            X = np.column_stack([
                a, b, np.maximum(o, 0), (o >= 0).astype(float),   # raw scores, in-our-window
                ra, rb, ro,                                         # ranks (1 = best)
                a - a.max(), b - b.max(), np.where(o >= 0, o - o.max(), -1.0),
                np.full(len(a), np.log(len(a))), np.full(len(a), a.max()),
                np.full(len(a), b.max()), np.full(len(a), max(o.max(), 0)),
            ]).astype(np.float32)
            out.append({"mol": m, "sim": sim, "y": keys == m, "X": X,
                        "ra": ra, "rb": rb, "ro": ro, "rbl": rbl})
    return out, mols


def rr(g, s):
    t = np.flatnonzero(g["y"])
    return expected_rr(s, int(t[0]), K) if len(t) else 0.0


def lb(gs, scores):
    """Per-molecule predicted-LB contribution, molecules in sorted order."""
    per = {}
    for g, s in zip(gs, scores):
        per.setdefault(g["mol"], {})[g["sim"]] = rr(g, s)
    return np.array([C1_W * per[m]["c1"] + C2_W * per[m]["c2"] for m in sorted(per)])


def rule(g, w_pv, w_own):
    return (1 - w_own) * (w_pv * g["ra"] + (1 - w_pv) * g["rb"]) + w_own * g["ro"]


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gs, mols = groups()
    fold = dict(zip(mols, np.random.default_rng(0).permutation(len(mols)) % FOLDS))
    print(f"{len(mols)} molecules x 2 simulations, {FOLDS}-fold by molecule\n")

    submitted = [0.9 * g["rbl"] + 0.1 * g["ro"] for g in gs]
    tuned, logit, gbm = [None] * len(gs), [None] * len(gs), [None] * len(gs)
    grid = list(itertools.product([0.7, 0.8, 0.88, 0.95, 1.0], [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]))
    picks = []
    for k in range(FOLDS):
        tr = [i for i, g in enumerate(gs) if fold[g["mol"]] != k]
        te = [i for i, g in enumerate(gs) if fold[g["mol"]] == k]
        trg = [gs[i] for i in tr]
        # 2. tuned rule: best grid point on the training folds
        best = max(grid, key=lambda w: lb(trg, [rule(g, *w) for g in trg]).mean())
        picks.append(best)
        for i in te:
            tuned[i] = rule(gs[i], *best)
        # 3./4. learned stackers; class-1 / class-2 rows weighted as in the LB formula
        X = np.vstack([gs[i]["X"] for i in tr])
        y = np.concatenate([gs[i]["y"] for i in tr])
        w = np.concatenate([np.full(len(gs[i]["y"]), C1_W if gs[i]["sim"] == "c1" else C2_W) for i in tr])
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
        hg = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                            min_samples_leaf=100, random_state=0).fit(X, y, sample_weight=w)
        for i in te:
            logit[i] = lr.predict_proba(sc.transform(gs[i]["X"]))[:, 1]
            gbm[i] = hg.predict_proba(gs[i]["X"])[:, 1]

    base = lb(gs, submitted)
    print("| combiner | predicted LB | vs submitted rule (+/- 2 SE) |")
    print("|---|---:|---:|")
    for name, s in [("submitted rule (0.9 engine + 0.1 ours)", submitted),
                    ("tuned rule (weights from training folds)", tuned),
                    ("logistic-regression stacker", logit),
                    ("gradient-boosted stacker (depth 3)", gbm)]:
        v = lb(gs, s)
        d = v - base
        print(f"| {name} | {v.mean():.4f} | {d.mean():+.4f} +/- {2 * d.std(ddof=1) / np.sqrt(len(d)):.4f} |")
    print(f"\ntuned rule picked (w_pv, w_own) per fold: {picks}")


if __name__ == "__main__":
    main()
