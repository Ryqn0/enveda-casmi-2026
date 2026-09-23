"""Validate the combined ranker on both classes, sweeping the blend weight.

The question this answers is not "does analog help class 2" — that is already
measured — but "does switching the analog channel on COST anything on class 1?"
A blend that lifts class 2 while damaging class 1 may be a net loss, and an
aggregate number would hide it.

Weight 0.0 is identity-only, i.e. the Phase 1 pipeline restricted to the
candidate pool.

    uv run python scripts/validate_combined.py -n 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_class2 import fetch_raw, sample_structures  # noqa: E402

from casmi.library import Library  # noqa: E402
from casmi.pool import ADDUCT_MASS, CandidatePool  # noqa: E402
from casmi.rank import CombinedRanker  # noqa: E402

K = 25


def _rr(cand_keys, total, truth) -> float:
    order = np.argsort(-total)[:K]
    ranked = list(cand_keys[order])
    return 1.0 / (ranked.index(truth) + 1) if truth in ranked else 0.0


def strategies() -> dict:
    """Named blending strategies, each mapping (ident, analog) -> total score.

    Linear blends are the obvious approach, but they force one weight to serve
    two regimes. For a class-2 molecule the truth scores 0 on identity while
    wrong candidates that happen to be in the library score above it, so the
    identity channel is not merely uninformative there — it is harmful.

    Gating instead uses identity only when it is CONFIDENT. A strong identity
    match means the molecule is in the library (class 1); a weak one means it is
    not, and the channel should be ignored rather than averaged in.
    """
    out = {}
    for w in (0.0, 1.0, 4.0, 1e6):
        label = "identity only" if w == 0 else ("analog only" if w > 1e5 else f"linear w={w:g}")
        out[label] = (lambda i, a, w=w: i + w * a)
    for t in (0.3, 0.5, 0.7):
        # Above the threshold identity leads and analog breaks ties; below it,
        # identity is discarded entirely.
        out[f"gated t={t:g}"] = (
            lambda i, a, t=t: (i + 0.5 * a) if (len(i) and i.max() >= t) else a
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default=None)
    args = ap.parse_args()

    lib = Library(ROOT / "data" / "index")
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    ranker = CombinedRanker(lib, pool)
    print(f"library {len(lib.meta):,} | pool {len(pool):,}")

    sources = set(args.source.split(",")) if args.source else None
    holdout = sample_structures(lib, pool, args.n, args.seed, sources)
    needed = {int(r) for h in holdout for r in h["query_rows"]}
    raw = fetch_raw(needed)
    print(f"holdout {len(holdout)} structures, {len(raw)} spectra\n")

    keys = lib.structures.inchikey14.to_numpy()
    strats = strategies()
    results = {(m, name): [] for m in ("class1", "class2") for name in strats}

    t0 = time.time()
    for h in holdout:
        s = h["struct_idx"]
        truth = keys[s]
        rows = [r for r in h["query_rows"] if r in raw]
        if not rows:
            continue
        adduct = raw[rows[0]][1]
        if adduct not in ADDUCT_MASS:
            continue
        spectra = [(raw[r][0], raw[r][2]) for r in rows]

        for mode in ("class1", "class2"):
            # class 1: the molecule has other reference spectra, just not these.
            # class 2: it has none at all.
            exclude = (
                lib.positions_of_rows(h["query_rows"])
                if mode == "class1"
                else lib.spectra_of(s)
            )
            cand, ident, analog, _ = ranker.score_components(
                spectra, adduct, exclude=exclude
            )
            if len(cand) == 0:
                continue
            ck = pool.key[cand]
            for name, fn in strats.items():
                results[(mode, name)].append(_rr(ck, fn(ident, analog), truth))

    dt = time.time() - t0
    n = len(next(iter(results.values())))
    print(f"{n} molecules in {dt:.0f}s ({dt/max(n,1):.1f}s each)\n")

    # Class 1 is ~15.4% of the test set, measured from the first submission
    # (LB 0.130 / class-1 MRR 0.845). Class 2 is estimated at ~43% by assuming
    # the current leader's 0.415 comes from class 1 plus a class-2 MRR near the
    # published 0.612. Class 3 contributes 0 — those structures are in no
    # database, so no candidate pool can hold them.
    P1, P2 = 0.154, 0.43

    print("MRR@25 by strategy (natural-product holdout)")
    print("| strategy | class 1 | class 2 | projected LB |")
    print("|---|---:|---:|---:|")
    best = None
    for name in strats:
        c1 = np.mean(results[("class1", name)])
        c2 = np.mean(results[("class2", name)])
        lb = P1 * c1 + P2 * c2
        print(f"| {name} | {c1:.4f} | {c2:.4f} | {lb:.4f} |")
        if best is None or lb > best[1]:
            best = (name, lb)
    print(f"\nbest by projected LB: {best[0]} ({best[1]:.4f})")
    print(f"(projection assumes P(class1)={P1:.1%}, P(class2)={P2:.0%}; "
          f"P2 is an estimate, so treat the ordering as more reliable than the value)")


if __name__ == "__main__":
    main()
