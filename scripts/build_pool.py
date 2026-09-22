"""Build the candidate structure pool from COCONUT + training structures.

    uv run python scripts/build_pool.py

Writes data/pool.parquet, sorted by neutral monoisotopic mass.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from casmi.pool import CandidatePool, build_pool

ROOT = Path(__file__).resolve().parents[1]
COCONUT = ROOT / "data" / "coconut" / "coconut_csv_lite-09-2026.csv"
TRAIN = ROOT / "data" / "raw" / "train.parquet"
OUT = ROOT / "data" / "pool.parquet"


def main() -> None:
    t0 = time.time()
    pool = build_pool(COCONUT, TRAIN, OUT)
    print(f"pool: {len(pool):,} structures in {time.time() - t0:.0f}s")
    print(pool.source.value_counts().to_string())
    print(f"mass range: {pool.exact_mass.min():.1f} - {pool.exact_mass.max():.1f} Da")

    # Cross-check our formula parser against COCONUT's own exact mass column.
    # They are computed independently, so agreement is real evidence the parser
    # is right; a silent error here would quietly exclude correct candidates.
    got = pool[pool.exact_molecular_weight.notna()] if "exact_molecular_weight" in pool else None
    if got is not None and len(got):
        err = (got.exact_mass - got.exact_molecular_weight).abs()
        print(f"\nvs COCONUT's own exact mass ({len(got):,} rows):")
        print(f"  median |diff| {err.median():.2e} Da | p99 {err.quantile(0.99):.2e} "
              f"| max {err.max():.2e}")
        print(f"  within 1 mDa: {(err < 1e-3).mean():.3%}")

    # Sanity: a known molecule must be findable from its measured ion.
    cp = CandidatePool(OUT)
    hits = cp.query(195.087652, "[M+H]+", ppm=5)  # caffeine [M+H]+
    print(f"\ncaffeine [M+H]+ at 5 ppm -> {len(hits)} candidates")
    keys = {cp.key[i] for i in hits}
    print(f"  contains caffeine (RYYVLZVUVIJVGH): {'RYYVLZVUVIJVGH' in keys}")


if __name__ == "__main__":
    main()
