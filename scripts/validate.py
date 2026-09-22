"""Local validation for the library-search baseline.

The leaderboard allows only a handful of submissions per day, so we need a
local score. We simulate the test set by holding out training structures that
were measured on timsTOF (the test instrument) inside the test mass window,
then querying with their *raw* spectra while hiding parts of the library.

Two masking modes, which bracket the real test set:

  known   Hide only the query measurements themselves. The structure's other
          spectra stay visible. This is competition "class 1": the molecule is
          in the library, but this particular measurement is new.

  novel   Hide every spectrum of the structure. This is "class 3": the molecule
          is genuinely absent from the library. Identity search cannot score
          above zero here by construction — running it makes the ceiling of a
          pure library approach concrete rather than theoretical.

    uv run python scripts/validate.py --index data/index -n 250
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.library import Library
from casmi.metric import reciprocal_rank

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

MAX_QUERY_SPECTRA = 4  # test molecules have a median of 3


MIN_SPECTRA = 3  # need >=1 left in the library after taking queries


def pick_holdout(lib: Library, n: int, seed: int) -> pd.DataFrame:
    """Sample structures that look like test molecules: timsTOF, >=3 spectra.

    The query spectra are hidden from the library, so a structure must keep at
    least one spectrum behind or the `known` mode silently degenerates into the
    `novel` one — which is exactly the bug this guard exists to prevent.
    """
    m = lib.meta
    tims = m[m.instrument_type == "timsTOF"]
    per_struct = tims.groupby("struct_idx").row_id.apply(list)
    eligible = per_struct[per_struct.map(len) >= MIN_SPECTRA]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible.index.to_numpy(), size=min(n, len(eligible)),
                        replace=False)
    rows = []
    for s in chosen:
        ids = eligible.loc[s]
        take = list(rng.permutation(ids)[: min(MAX_QUERY_SPECTRA, len(ids) - 1)])
        rows.append({"struct_idx": int(s), "query_rows": take})
    out = pd.DataFrame(rows)
    left = np.array(
        [
            len(lib.spectra_of(r.struct_idx)) - len(lib.positions_of_rows(r.query_rows))
            for r in out.itertuples()
        ]
    )
    assert (left > 0).all(), "holdout leaves a structure unrepresented in the library"
    out["remaining"] = left
    return out


def fetch_raw(row_ids: set[int]) -> dict[int, tuple[float, np.ndarray]]:
    """One streaming pass over train.parquet to pull raw peaks for given rows.

    Raw, not the cleaned library copy: test spectra arrive uncleaned (median 230
    peaks vs 42 in train), so validating on cleaned spectra would flatter us.
    """
    want = row_ids
    out: dict[int, tuple[float, np.ndarray]] = {}
    pf = pq.ParquetFile(RAW / "train.parquet")
    offset = 0
    cols = ["precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        n = batch.num_rows
        hit = [r for r in range(offset, offset + n) if r in want]
        if hit:
            df = batch.to_pandas()
            for r in hit:
                row = df.iloc[r - offset]
                peaks = np.stack(
                    [
                        np.asarray(row.ms2_mzs, dtype=np.float32),
                        np.asarray(row.ms2_normalized_intensities, dtype=np.float32),
                    ],
                    axis=1,
                )
                out[r] = (float(row.precursor_mz), peaks)
        offset += n
        if len(out) == len(want):
            break
    return out


def run_mode(lib: Library, holdout: pd.DataFrame, raw: dict, mode: str) -> pd.DataFrame:
    results = []
    for rec in holdout.itertuples():
        queries = [raw[r] for r in rec.query_rows if r in raw]
        if not queries:
            continue
        if mode == "known":
            exclude = lib.positions_of_rows(rec.query_rows)
        elif mode == "novel":
            exclude = lib.spectra_of(rec.struct_idx)
        else:
            raise ValueError(mode)

        top, scores = lib.rank_molecule(queries, k=25, exclude=exclude)
        truth = lib.smiles[rec.struct_idx]
        rr = reciprocal_rank(lib.smiles[top].tolist(), truth)
        results.append(
            {
                "struct_idx": rec.struct_idx,
                "n_queries": len(queries),
                "remaining": rec.remaining,
                "rr": rr,
                "top_score": float(scores[0]),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "data" / "index")
    ap.add_argument("-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["known", "novel"])
    args = ap.parse_args()

    lib = Library(args.index)
    print(f"library: {len(lib.meta):,} spectra / {lib.n_structures:,} structures")

    holdout = pick_holdout(lib, args.n, args.seed)
    print(f"holdout: {len(holdout)} structures")

    needed = {int(r) for rows in holdout.query_rows for r in rows}
    t0 = time.time()
    raw = fetch_raw(needed)
    print(f"fetched {len(raw)}/{len(needed)} raw spectra in {time.time() - t0:.0f}s")

    for mode in args.modes:
        t0 = time.time()
        res = run_mode(lib, holdout, raw, mode)
        dt = time.time() - t0
        hit1 = (res.rr == 1.0).mean()
        hit25 = (res.rr > 0).mean()
        print(
            f"\n[{mode}] MRR@25 = {res.rr.mean():.4f} | "
            f"top-1 {hit1:.1%} | found-in-25 {hit25:.1%} | "
            f"{len(res)} molecules | {dt:.0f}s ({dt / max(len(res),1):.2f}s each)"
        )
        # Score depends steeply on how well the structure is represented in the
        # library, so always show it — an aggregate alone hides the story.
        by_depth = res.groupby(res.remaining.clip(upper=4)).rr.agg(["mean", "count"])
        print("  by library depth remaining (4 = 4+):")
        for depth, r in by_depth.iterrows():
            print(f"    {depth}: MRR {r['mean']:.3f}  (n={int(r['count'])})")
        res.to_csv(ROOT / "data" / f"val_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
