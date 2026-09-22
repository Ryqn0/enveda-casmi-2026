"""Build the spectral library index from train.parquet.

The index is derived data, not source — it lives under data/ and is gitignored.
Rebuild it with:

    uv run python scripts/build_library.py              # full, ~1.6M spectra
    uv run python scripts/build_library.py --limit 5    # smoke test, 5 batches
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from casmi.library import TEST_ADDUCTS, TEST_MZ_RANGE, build

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N batches of 50k rows (smoke test)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "index")
    args = ap.parse_args()

    # Deliberately NOT read off test.parquet: that file is a placeholder made of
    # training rows and is replaced by the real test set at re-run. Fitting the
    # library filter to it would silently exclude most of the real test range.
    mz_range, adducts = TEST_MZ_RANGE, set(TEST_ADDUCTS)
    print(f"library filter: m/z {mz_range[0]:.1f}-{mz_range[1]:.1f}, "
          f"{len(adducts)} adducts -> {args.out}")

    if args.limit:
        # monkeypatch-free smoke test: shrink the source by reading fewer batches
        import casmi.library as lib

        original = lib.iter_library_rows

        def limited(*a, **kw):
            for i, df in enumerate(original(*a, **kw)):
                if i >= args.limit:
                    return
                yield df

        lib.iter_library_rows = limited

    t0 = time.time()
    build(RAW / "train.parquet", args.out, mz_range, adducts)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
