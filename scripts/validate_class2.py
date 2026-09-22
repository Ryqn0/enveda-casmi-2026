"""Measure class-2 performance: the molecule is in the database, not the library.

Simulates competition class 2 exactly:
  - the structure IS in the candidate pool (it is a known compound), and
  - EVERY spectrum of it is hidden from the spectral library (no public spectra).

Identity search scores 0 here by construction, which is what Phase 1 measured.
This script asks whether analog propagation can do better, and compares against
the honest baseline of ordering mass-matched candidates at random.

    uv run python scripts/validate_class2.py -n 150
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rdkit import DataStructs

from casmi.analog import morgan_fp, rank_by_analogy
from casmi.library import Library
from casmi.pool import ADDUCT_MASS, CandidatePool

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

MAX_QUERY_SPECTRA = 4
TOP_ANALOGS = 50
PPM = 5.0
K = 25


def sample_structures(lib: Library, pool: CandidatePool, n: int, seed: int):
    """Pick timsTOF structures that are present in the candidate pool."""
    m = lib.meta
    tims = m[m.instrument_type.eq("timsTOF")]
    per_struct = tims.groupby("struct_idx").row_id.apply(list)

    in_pool = set(pool.key)
    keys = lib.structures.inchikey14.to_numpy()
    eligible = per_struct[[keys[s] in in_pool for s in per_struct.index]]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible.index.to_numpy(), size=min(n, len(eligible)),
                        replace=False)
    return [
        {"struct_idx": int(s), "query_rows": list(rng.permutation(eligible.loc[s])[:MAX_QUERY_SPECTRA])}
        for s in chosen
    ]


def fetch_raw(row_ids: set[int]) -> dict:
    """Pull raw peaks plus precursor and adduct for the given train rows."""
    out = {}
    pf = pq.ParquetFile(RAW / "train.parquet")
    cols = ["precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]
    offset = 0
    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        n = batch.num_rows
        hit = [r for r in range(offset, offset + n) if r in row_ids]
        if hit:
            df = batch.to_pandas()
            for r in hit:
                row = df.iloc[r - offset]
                out[r] = (
                    float(row.precursor_mz),
                    row.adduct,
                    np.stack(
                        [
                            np.asarray(row.ms2_mzs, dtype=np.float32),
                            np.asarray(row.ms2_normalized_intensities, dtype=np.float32),
                        ],
                        axis=1,
                    ),
                )
        offset += n
        if len(out) == len(row_ids):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppm", type=float, default=PPM)
    args = ap.parse_args()

    lib = Library(ROOT / "data" / "index")
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    print(f"library {len(lib.meta):,} spectra | pool {len(pool):,} structures")

    holdout = sample_structures(lib, pool, args.n, args.seed)
    print(f"holdout: {len(holdout)} structures")

    needed = {int(r) for h in holdout for r in h["query_rows"]}
    raw = fetch_raw(needed)
    print(f"fetched {len(raw)}/{len(needed)} raw spectra\n")

    rng = np.random.default_rng(args.seed)
    rr_analog, rr_random, n_cands, recall, analog_tan = [], [], [], [], []
    t0 = time.time()

    for h in holdout:
        s = h["struct_idx"]
        truth = lib.structures.inchikey14.to_numpy()[s]
        queries = [raw[r] for r in h["query_rows"] if r in raw]
        if not queries:
            continue

        # Class 2: the structure has NO public spectra at all.
        exclude = lib.spectra_of(s)

        # Candidates of the right neutral mass, from the first spectrum's
        # precursor and adduct (all spectra of a molecule agree on these).
        mz, adduct, _ = queries[0]
        if adduct not in ADDUCT_MASS:
            continue
        cand = pool.query(mz, adduct, ppm=args.ppm)
        if len(cand) == 0:
            continue
        cand_keys = pool.key[cand]
        hit = truth in set(cand_keys)
        recall.append(hit)
        n_cands.append(len(cand))

        # Fuse analog evidence across the molecule's spectra, keeping each
        # structure's best score.
        best = np.zeros(lib.n_structures, dtype=np.float32)
        for q_mz, _, peaks in queries:
            idx, sc = lib.analog_structures(q_mz, peaks, top_n=TOP_ANALOGS,
                                            exclude=exclude)
            np.maximum.at(best, idx, sc)
        a_idx = np.argpartition(-best, TOP_ANALOGS - 1)[:TOP_ANALOGS]
        a_idx = a_idx[np.argsort(-best[a_idx])]

        # How chemically close is the best analog we found to the true answer?
        # This decides whether the score is real or an artefact of the holdout:
        # a training library full of near-congeners would hand us analogs that a
        # genuinely novel natural product would never have.
        truth_fp = morgan_fp(lib.smiles[s])
        best_tan = 0.0
        if truth_fp is not None:
            for a in a_idx[:10]:
                afp = morgan_fp(lib.smiles[a])
                if afp is not None:
                    best_tan = max(best_tan, DataStructs.TanimotoSimilarity(truth_fp, afp))
        analog_tan.append(best_tan)

        scores = rank_by_analogy(pool.smiles[cand], lib.smiles[a_idx], best[a_idx])
        order = np.argsort(-scores)[:K]
        ranked = cand_keys[order]
        rr_analog.append(1.0 / (list(ranked).index(truth) + 1) if truth in ranked else 0.0)

        # Baseline: the same candidates in random order. This is what retrieval
        # alone is worth, and the number analog ranking must beat.
        shuffled = cand_keys[rng.permutation(len(cand))][:K]
        rr_random.append(1.0 / (list(shuffled).index(truth) + 1) if truth in shuffled else 0.0)

    dt = time.time() - t0
    rr_analog, rr_random = np.array(rr_analog), np.array(rr_random)
    n_cands = np.array(n_cands)

    print(f"{len(rr_analog)} molecules in {dt:.0f}s ({dt/max(len(rr_analog),1):.1f}s each)")
    print(f"candidates at {args.ppm:.0f} ppm: median {np.median(n_cands):.0f} "
          f"| p90 {np.percentile(n_cands,90):.0f}")
    print(f"truth retrieved into pool: {np.mean(recall):.1%}\n")
    print(f"  identity search (Phase 1) MRR@25 = 0.0000   (zero by construction)")
    print(f"  random order of candidates MRR@25 = {rr_random.mean():.4f}")
    print(f"  ANALOG PROPAGATION        MRR@25 = {rr_analog.mean():.4f}"
          f"  | top-1 {(rr_analog==1).mean():.1%}"
          f" | in-25 {(rr_analog>0).mean():.1%}")

    # The headline number is meaningless without this breakdown. If the score
    # only holds up where a near-twin analog exists, it will not survive a test
    # set of genuinely novel natural products.
    tan = np.array(analog_tan)
    print("\nMRR by chemical similarity of the best analog to the truth:")
    print("| analog Tanimoto | n | analog MRR | random MRR |")
    print("|---|---:|---:|---:|")
    for lo, hi, label in [
        (0.0, 0.4, "< 0.4 (distant)"),
        (0.4, 0.6, "0.4-0.6"),
        (0.6, 0.8, "0.6-0.8"),
        (0.8, 1.01, ">= 0.8 (near-twin)"),
    ]:
        m = (tan >= lo) & (tan < hi)
        if m.sum():
            print(f"| {label} | {m.sum()} | {rr_analog[m].mean():.3f} "
                  f"| {rr_random[m].mean():.3f} |")

    pd.DataFrame({"rr_analog": rr_analog, "rr_random": rr_random,
                  "n_cands": n_cands, "analog_tanimoto": tan}).to_csv(
        ROOT / "data" / "val_class2.csv", index=False)


if __name__ == "__main__":
    main()
