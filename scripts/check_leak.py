"""Check whether test spectra appear verbatim in train.parquet.

Reproducible evidence for the duplication found on 2026-09-22. Every test
spectrum is compared against training rows sharing its precursor mass (to 4 dp)
and peak count, then the full m/z array is compared elementwise.

    uv run python scripts/check_leak.py

Writes data/leak_map.parquet: one row per test spectrum with the matched
training row and the structure that row is labelled with.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Exact float equality, deliberately. np.allclose would keep its default
# rtol=1e-5, which at m/z 400 tolerates ~4 mDa — enough to call two genuinely
# different measurements a duplicate. The claim here is identity, so the test
# must be identity.


def main() -> None:
    test = pd.read_parquet(RAW / "test.parquet")

    # Prefilter on (precursor rounded, peak count) so we only compare full arrays
    # for plausible pairs — a 1213 x 2.5M elementwise comparison is not viable.
    want: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(test.itertuples()):
        want[(round(float(r.precursor_mz), 4), len(r.ms2_mzs))].append(i)

    exact = np.zeros(len(test), dtype=bool)
    match_row = np.full(len(test), -1, dtype=np.int64)
    label: list[tuple | None] = [None] * len(test)

    pf = pq.ParquetFile(RAW / "train.parquet")
    cols = [
        "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "ingest_lib",
    ]
    offset = 0
    for batch in pf.iter_batches(batch_size=100_000, columns=cols):
        df = batch.to_pandas()
        for j, key in enumerate(zip(df.precursor_mz.round(4), df.ms2_mzs.map(len))):
            if key not in want:
                continue
            train_mz = np.asarray(df.ms2_mzs.iloc[j], dtype=np.float64)
            train_it = np.asarray(df.ms2_normalized_intensities.iloc[j], dtype=np.float64)
            for i in want[key]:
                if exact[i]:
                    continue
                q_mz = np.asarray(test.ms2_mzs.iloc[i], dtype=np.float64)
                q_it = np.asarray(
                    test.ms2_normalized_intensities.iloc[i], dtype=np.float64
                )
                if np.array_equal(q_mz, train_mz) and np.array_equal(q_it, train_it):
                    exact[i] = True
                    match_row[i] = offset + j
                    label[i] = (
                        df.normalized_smiles.iloc[j],
                        df.inchikey14.iloc[j],
                        df.ingest_lib.iloc[j],
                    )
        offset += len(df)

    out = test[["molecule_id", "spectrum_id"]].copy()
    out["exact"] = exact
    out["train_row"] = match_row
    out["leak_smiles"] = [x[0] if x else None for x in label]
    out["leak_key"] = [x[1] if x else None for x in label]
    out["leak_lib"] = [x[2] if x else None for x in label]

    per_mol = out.groupby("molecule_id").exact.any()
    disagree = out[out.exact].groupby("molecule_id").leak_key.nunique().gt(1).sum()

    print(f"test spectra with an exact duplicate in train : {exact.sum():>5} / {len(out)} "
          f"({exact.mean():.1%})")
    print(f"test molecules with >=1 duplicate             : {per_mol.sum():>5} / {len(per_mol)} "
          f"({per_mol.mean():.1%})")
    print(f"distinct structures recovered                 : "
          f"{out.loc[out.exact, 'leak_key'].nunique():>5}")
    print(f"molecules whose duplicates disagree           : {disagree:>5}")
    print("source libraries:")
    print(out.loc[out.exact, "leak_lib"].value_counts().to_string())

    out.to_parquet(ROOT / "data" / "leak_map.parquet", index=False)
    print(f"-> data/leak_map.parquet")


if __name__ == "__main__":
    main()
