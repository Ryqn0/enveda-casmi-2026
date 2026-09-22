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


def sample_structures(lib: Library, pool: CandidatePool, n: int, seed: int,
                      sources: set[str] | None = None):
    """Pick structures present in the candidate pool, optionally by source library.

    Default is timsTOF, matching the test instrument. But in this dataset only
    `enveda-180` and `enveda-np-examples` are timsTOF, and enveda-180 is a
    congeneric screening library — many close structural cousins measured
    together. Analog propagation will look better there than on natural
    products, so `sources` allows sampling from the public natural-product
    libraries instead, at the cost of an instrument mismatch.
    """
    m = lib.meta
    if sources:
        ingest = pq.read_table(
            RAW / "train.parquet", columns=["ingest_lib"]
        ).to_pandas().ingest_lib.to_numpy()
        row_lib = ingest[m.row_id.to_numpy()]
        sel = m[np.isin(row_lib, list(sources))]
        print(f"restricted to {sorted(sources)}: {len(sel):,} spectra")
    else:
        sel = m[m.instrument_type.eq("timsTOF")]
    per_struct = sel.groupby("struct_idx").row_id.apply(list)

    in_pool = set(pool.key)
    keys = lib.structures.inchikey14.to_numpy()
    eligible = per_struct[[keys[s] in in_pool for s in per_struct.index]]

    # At least 2 spectra, and never take them all as queries. Class 2 hides
    # every spectrum anyway, but the SAME sampler feeds the class-1 case, where
    # consuming all of them leaves nothing in the library to match against and
    # silently turns class 1 into class 2. That bug already cost one round of
    # wrong numbers; the guard belongs here, not in the caller.
    eligible = eligible[eligible.map(len) >= 2]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible.index.to_numpy(), size=min(n, len(eligible)),
                        replace=False)
    out = []
    for s in chosen:
        ids = eligible.loc[s]
        take = list(rng.permutation(ids)[: min(MAX_QUERY_SPECTRA, len(ids) - 1)])
        assert 0 < len(take) < len(ids), "queries must leave a spectrum behind"
        out.append({"struct_idx": int(s), "query_rows": take})
    return out


def fetch_raw(row_ids: set[int]) -> dict:
    """Pull raw peaks plus precursor, adduct and instrument for given train rows.

    instrument_type is returned because FPNet takes it as an input: telling the
    model a QTOF spectrum came from a timsTOF makes it score badly for a reason
    unrelated to its quality.
    """
    out = {}
    pf = pq.ParquetFile(RAW / "train.parquet")
    cols = ["precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities",
            "instrument_type", "collision_energy_ev"]
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
                    row.instrument_type,
                    # Mean over a stepped acquisition, matching how the
                    # training set collapsed collision_energy_ev to one number.
                    float(np.mean(row.collision_energy_ev))
                    if row.collision_energy_ev is not None
                    and len(row.collision_energy_ev)
                    else 0.0,
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
    ap.add_argument("--top-analogs", type=int, default=TOP_ANALOGS)
    # Systematic calibration bias to remove. timsTOF (the test instrument)
    # reads +0.98 ppm high; see CandidatePool.neutral_mass.
    ap.add_argument("--offset-ppm", type=float, default=0.0)
    ap.add_argument("--source", default=None,
                    help="comma-separated ingest_lib values, e.g. gnps,massbank,mona,riken")
    args = ap.parse_args()

    lib = Library(ROOT / "data" / "index")
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    print(f"library {len(lib.meta):,} spectra | pool {len(pool):,} structures")

    sources = set(args.source.split(",")) if args.source else None
    holdout = sample_structures(lib, pool, args.n, args.seed, sources)
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
        mz, adduct, _, _, _ = queries[0]
        if adduct not in ADDUCT_MASS:
            continue
        cand = pool.query(mz, adduct, ppm=args.ppm, offset_ppm=args.offset_ppm)
        if len(cand) == 0:
            continue
        cand_keys = pool.key[cand]
        hit = truth in set(cand_keys)
        recall.append(hit)
        n_cands.append(len(cand))

        # Fuse analog evidence across the molecule's spectra, keeping each
        # structure's best score.
        best = np.zeros(lib.n_structures, dtype=np.float32)
        for q_mz, _, peaks, _, _ in queries:
            idx, sc = lib.analog_structures(q_mz, peaks, top_n=args.top_analogs,
                                            exclude=exclude)
            np.maximum.at(best, idx, sc)
        a_idx = np.argpartition(-best, args.top_analogs - 1)[:args.top_analogs]
        a_idx = a_idx[np.argsort(-best[a_idx])]

        # How chemically close is the best analog we found to the true answer?
        # This decides whether the score is real or an artefact of the holdout:
        # a training library full of near-congeners would hand us analogs that a
        # genuinely novel natural product would never have.
        # Over ALL analogs used for ranking, not just the top few: a near-twin
        # sitting at rank 40 still contributes to the score, so measuring only
        # the top 10 would file that molecule under "distant" and understate how
        # much the result depends on close relatives.
        truth_fp = morgan_fp(lib.smiles[s])
        best_tan = 0.0
        if truth_fp is not None:
            for a in a_idx:
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

    # CALIBRATION, measured against the leaderboard on 2026-09-22.
    #
    # The aggregate MRR over this holdout reported 0.621. The actual class-2
    # performance implied by LB 0.240 is at most 0.392 and probably nearer 0.30.
    # The aggregate is optimistic because every holdout molecule is a PUBLISHED
    # compound, so the training library contains its relatives. Test molecules
    # are unpublished -- Enveda measured them precisely because nobody else had
    # -- and analog propagation needs a spectral cousin to exist.
    #
    # The low-Tanimoto bucket does NOT have that problem: those molecules have
    # no close relative in the library either, which is the situation a novel
    # compound is actually in. It read 0.396 while the truth was 0.24-0.39.
    #
    # So the hard bucket is the number to optimise. The aggregate is reported
    # below it for context only.
    hard = tan < 0.4
    if hard.sum():
        print(f"\nCALIBRATED ESTIMATE (best analog Tanimoto < 0.4, n={hard.sum()}):")
        print(f"  class-2 MRR@25 = {rr_analog[hard].mean():.4f}")
        print("  This tracks the leaderboard. The aggregate below does not --")
        print("  it ran 0.621 when the real value was at most 0.392.")

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
