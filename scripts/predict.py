"""Generate submission.csv from the library-search baseline.

    uv run python scripts/predict.py --index data/index --out submissions/baseline.csv

Candidates are deduplicated by the *metric's* key, not the dataset's. The two
disagree: the metric tautomer-canonicalizes before hashing, so two library
structures can collapse to one answer. Submitting both wastes a slot for no
chance of scoring.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.library import Library
from casmi.metric import K, inchikey14

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

OVERFETCH = 4  # pull K*OVERFETCH candidates so dedup still leaves K


def peaks_of(row) -> np.ndarray:
    return np.stack(
        [
            np.asarray(row.ms2_mzs, dtype=np.float32),
            np.asarray(row.ms2_normalized_intensities, dtype=np.float32),
        ],
        axis=1,
    )


def dedupe(smiles: list[str], k: int = K) -> list[str]:
    """Keep the first occurrence of each metric-equivalent structure."""
    seen: set[str] = set()
    out: list[str] = []
    for smi in smiles:
        key = inchikey14(smi)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(smi)
        if len(out) == k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "baseline.csv")
    args = ap.parse_args()

    lib = Library(args.index)
    test = pd.read_parquet(RAW / "test.parquet")
    print(f"library {len(lib.meta):,} spectra | test {test.molecule_id.nunique()} molecules")

    rows, zero_score = [], 0
    t0 = time.time()
    for mid, grp in test.groupby("molecule_id", sort=True):
        queries = [(float(r.precursor_mz), peaks_of(r)) for r in grp.itertuples()]
        top, scores = lib.rank_molecule(queries, k=K * OVERFETCH)
        if scores[0] == 0:
            zero_score += 1
        cands = dedupe(lib.smiles[top].tolist())
        rows.append({"molecule_id": mid, "smiles": ";".join(cands), "n": len(cands),
                     "top_score": float(scores[0])})

    sub = pd.DataFrame(rows)
    short = (sub.n < K).sum()
    print(f"ranked {len(sub)} molecules in {time.time() - t0:.0f}s")
    print(f"  molecules with no spectral match at all: {zero_score}")
    print(f"  rows with fewer than {K} candidates: {short}")
    print(f"  score distribution: {sub.top_score.describe()[['min','50%','max']].to_dict()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub[["molecule_id", "smiles"]].to_csv(args.out, index=False)
    print(f"-> {args.out}")

    # Guard the submission contract before it ever reaches Kaggle.
    sample = pd.read_csv(RAW / "sample_submission.csv")
    assert set(sub.molecule_id) == set(sample.molecule_id), "molecule_id set mismatch"
    assert len(sub) == len(sample), f"expected {len(sample)} rows, got {len(sub)}"
    assert sub.smiles.str.len().gt(0).all(), "empty prediction string"
    print("submission checks passed")


if __name__ == "__main__":
    main()
