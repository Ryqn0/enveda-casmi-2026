"""Does our pipeline add anything to the public engine? (Phase 8, docs/16)

    uv run python scripts/engine_harness.py score --nb kaggle/engine_ties/casmi26-engine-ties.ipynb \
        --b-train cv --w-pv 0.88 --dump data/harness/dump_engine_ties.pkl
    uv run python scripts/own_harness.py score --model fpnet_v5 --cv-eval --dump data/own/dump_own.pkl
    uv run python scripts/blend_harness.py

Both harnesses simulate the same 250 enveda-np-examples molecules the same way
(class 1 / class 2, one molecule at a time); each dumped its final per-candidate
scores keyed by structure. Here:

1. Disagreement: per molecule and class, whose top guess is right. If our
   pipeline is never right where the engine is wrong, blending cannot help.
2. Blend: final = (1-w) * engine rank + w * our rank, on the ENGINE's candidates
   (its 10 ppm window contains ours). A candidate outside our window gets our
   worst rank. Ranks are within-molecule and tie-averaged (docs/15), the same
   kind of rank blend the engine uses between its own two rankers.

Truth = candidate key equal to the molecule's key. The engine harness also
de-duplicates tautomers; this script does not, identically for every config,
so paired differences are fair (absolute values differ slightly).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casmi.features import rank01  # noqa: E402
from casmi.ranker import expected_rr  # noqa: E402

C1_W, C2_W, K = 0.162, 0.220, 25
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


def main() -> None:
    eng = pickle.load(open(ROOT / "data/harness/dump_engine_ties.pkl", "rb"))
    own = pickle.load(open(ROOT / "data/own/dump_own.pkl", "rb"))
    mols = sorted(set(eng["c1"]) & set(eng["c2"]) & set(own["c1"]) & set(own["c2"]))
    print(f"{len(mols)} molecules in both harnesses, both classes\n")

    rr = {w: {"c1": [], "c2": []} for w in WEIGHTS}
    tally = {sim: {"both": 0, "engine only": 0, "ours only": 0, "neither": 0} for sim in ("c1", "c2")}
    for sim in ("c1", "c2"):
        for m in mols:
            ke, se = eng[sim][m][:2]
            ko, so = own[sim][m]
            truth = np.flatnonzero(ke == m)
            ours_on_eng = dict(zip(ko, so))
            # Candidates outside our window: below our lowest score.
            so_e = np.array([ours_on_eng.get(k, -1.0) for k in ke])
            ae, ao = 1 - rank01(se), 1 - rank01(so_e)
            for w in WEIGHTS:
                s = (1 - w) * ae + w * ao
                rr[w][sim].append(expected_rr(s, int(truth[0]), K) if len(truth) else 0.0)
            # Top-1 on each side's OWN candidate list.
            e_ok = ke[np.argmax(se)] == m
            o_ok = ko[np.argmax(so)] == m
            tally[sim]["both" if e_ok and o_ok else "engine only" if e_ok
                       else "ours only" if o_ok else "neither"] += 1

    print("Top-1 correct (on each pipeline's own candidates):")
    print("| class | both | engine only | ours only | neither |")
    print("|---|---:|---:|---:|---:|")
    for sim in ("c1", "c2"):
        t = tally[sim]
        print(f"| {sim} | {t['both']} | {t['engine only']} | {t['ours only']} | {t['neither']} |")

    base = C1_W * np.array(rr[0.0]["c1"]) + C2_W * np.array(rr[0.0]["c2"])
    print("\n| w (our share) | class 1 | class 2 | predicted LB | vs engine alone (+/- 2 SE) |")
    print("|---|---:|---:|---:|---:|")
    for w in WEIGHTS:
        c1, c2 = np.array(rr[w]["c1"]), np.array(rr[w]["c2"])
        d = C1_W * c1 + C2_W * c2 - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"| {w:g} | {c1.mean():.4f} | {c2.mean():.4f} | {C1_W * c1.mean() + C2_W * c2.mean():.4f} "
              f"| {d.mean():+.4f} +/- {2 * se:.4f} |")


if __name__ == "__main__":
    main()
