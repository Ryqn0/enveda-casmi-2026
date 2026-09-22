"""Measure the candidate pool's retrieval ceiling.

Ranking cannot recover a structure the pool never returned, so recall here is a
hard upper bound on everything Phase 2 can achieve. Equally important is how
MANY candidates come back: we have 25 submission slots, so a pool that returns
the answer among 3,000 candidates has only moved the problem.

Sampled from training structures measured on timsTOF (the test instrument),
using their real precursor m/z and adduct.

    uv run python scripts/pool_recall.py -n 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.pool import ADDUCT_MASS, CandidatePool

ROOT = Path(__file__).resolve().parents[1]

PPMS = (5.0, 10.0, 20.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000, help="structures to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    print(f"pool: {len(pool):,} structures")

    tr = pq.read_table(
        ROOT / "data" / "raw" / "train.parquet",
        columns=["inchikey14", "adduct", "precursor_mz", "instrument_type"],
    ).to_pandas()

    # Match the test set: timsTOF only, and only adducts the test set uses.
    tr = tr[tr.instrument_type.eq("timsTOF") & tr.adduct.isin(ADDUCT_MASS)]
    # One measurement per structure — we are testing mass lookup, not spectra.
    tr = tr.drop_duplicates(subset="inchikey14")
    rng = np.random.default_rng(args.seed)
    sample = tr.iloc[rng.choice(len(tr), size=min(args.n, len(tr)), replace=False)]
    print(f"sampled {len(sample):,} structures\n")

    in_pool = np.isin(sample.inchikey14.to_numpy(), pool.key)
    print(f"structure present in pool at all: {in_pool.mean():.1%}")
    print("(training structures were added to the pool, so this should be ~100%;"
          " a shortfall means keys disagree)\n")

    rows = []
    for ppm in PPMS:
        found, sizes = [], []
        for r in sample.itertuples():
            hits = pool.query(float(r.precursor_mz), r.adduct, ppm=ppm)
            sizes.append(len(hits))
            found.append(r.inchikey14 in set(pool.key[hits]) if len(hits) else False)
        found = np.array(found)
        sizes = np.array(sizes)
        rows.append(
            {
                "ppm": ppm,
                "recall": found.mean(),
                "median_candidates": np.median(sizes),
                "p90": np.percentile(sizes, 90),
                "max": sizes.max(),
            }
        )

    df = pd.DataFrame(rows)
    print("| ppm | recall | median cands | p90 | max |")
    print("|---|---:|---:|---:|---:|")
    for r in df.itertuples():
        print(f"| {r.ppm:.0f} | {r.recall:.1%} | {r.median_candidates:.0f} | "
              f"{r.p90:.0f} | {r.max:,} |")

    # What recall is worth depends on where the answer can be placed. Report the
    # share of queries small enough that all candidates fit in 25 slots.
    ppm = 10.0
    sizes = np.array([len(pool.query(float(r.precursor_mz), r.adduct, ppm=ppm))
                      for r in sample.itertuples()])
    print(f"\nat {ppm:.0f} ppm: {(sizes <= 25).mean():.1%} of molecules return "
          f"<=25 candidates (answer guaranteed submittable if retrieved)")
    print(f"           {(sizes > 100).mean():.1%} return >100 (ranking does all the work)")


if __name__ == "__main__":
    main()
