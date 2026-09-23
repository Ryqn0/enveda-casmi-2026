"""Build the training set for the spectrum -> fingerprint model.

WHAT THE MODEL LEARNS
    A spectrum cannot be inverted to a structure directly. But it CAN be mapped
    to a description of the structure: a fingerprint saying which substructures
    are present. That is a well-posed supervised problem — fixed-size output,
    millions of labelled examples — and once you can predict a fingerprint from
    a spectrum you can rank any candidate by how well its true fingerprint
    matches the predicted one.

    This is the CSI:FingerID idea, and unlike analog propagation it needs NO
    relative in the library. That is exactly our weak spot: class-2 molecules
    whose nearest spectral cousin is chemically distant.

    uv run python scripts/prepare_fpnet_data.py --n-spectra 600000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import DataStructs, RDLogger

from casmi.analog import morgan_fp
from casmi.fpnet import select_peaks
from casmi.library import TEST_ADDUCTS, TEST_MZ_RANGE

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "fpnet"

# 256, not 128. Measured: 69.8% of TEST spectra exceed 128 peaks and 45.8%
# exceed 256, while 31.7% of training spectra were already hitting the 128 cap.
# The training median is only 44, so the cost is bounded -- most spectra are
# nowhere near the cap.
MAX_PEAKS = 256          # per spectrum, most intense kept
FP_SIZE = 2048           # Morgan bit vector length before selection

# Bits set in fewer than this fraction of structures carry almost no signal and
# cannot be learned; bits set in nearly all carry none either. Keeping the
# middle band shrinks the target and stops the loss being dominated by bits
# whose answer is always the same.
MIN_FREQ, MAX_FREQ = 0.005, 0.95

# Adducts and instruments become learned embeddings, so they need integer ids.
# A fixed order is required: the model's embedding table is indexed by it, and
# a reordering between training and inference would silently scramble them.
ADDUCT_LIST = sorted(TEST_ADDUCTS)
INSTRUMENT_LIST = ["timsTOF", "Orbitrap", "QTOF", "other"]


def bits_of(smiles: str) -> np.ndarray | None:
    """Morgan fingerprint as a uint8 array of 0/1, or None if unparseable."""
    fp = morgan_fp(smiles)
    if fp is None:
        return None
    arr = np.zeros(FP_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def instrument_id(s) -> int:
    """Map free-text instrument to a small id. Train has 82 distinct strings."""
    if not isinstance(s, str):
        return 3
    t = s.lower()
    if "timstof" in t:
        return 0
    if "orbitrap" in t or "qft" in t or "itft" in t:
        return 1
    if "tof" in t:
        return 2
    return 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spectra", type=int, default=600_000)
    # Cap spectra per STRUCTURE instead of sampling uniformly over spectra.
    #
    # Uniform sampling is what taught the model to memorise. Spectra per
    # structure are wildly skewed -- median 4, max 1,438 -- so a handful of
    # heavily-measured molecules dominate the training set while the rare tail
    # is crowded out. At 900k uniform we covered 221,792 of 268,969 available
    # structures; at cap=4 we cover ALL of them for 943k spectra, i.e. the same
    # compute buys 21% more distinct molecules.
    #
    # This matters because fpnet loses to analog search on molecules it has not
    # seen (docs/09): its weakness is generalisation, and structure diversity is
    # the lever that targets generalisation directly.
    ap.add_argument("--cap-per-structure", type=int, default=0,
                    help="0 = old uniform sampling")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    lo, hi = TEST_MZ_RANGE
    cols = ["inchikey14", "normalized_smiles", "adduct", "instrument_type",
            "precursor_mz", "collision_energy_ev", "ms2_mzs",
            "ms2_normalized_intensities"]

    # ---- pass 1: which rows qualify, and which structures appear -----------
    t0 = time.time()
    scal = pq.read_table(
        RAW / "train.parquet",
        columns=["inchikey14", "adduct", "precursor_mz", "normalized_smiles"],
    ).to_pandas()
    ok = (
        scal.precursor_mz.between(lo, hi)
        & scal.adduct.isin(TEST_ADDUCTS)
        & scal.normalized_smiles.notna()
    )
    rows = np.flatnonzero(ok.to_numpy())
    print(f"{len(rows):,} eligible spectra ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    if args.cap_per_structure:
        # Shuffle first so the kept spectra per structure are a random subset
        # (not the first N by file order, which correlates with source library
        # and therefore with instrument).
        d = pd.DataFrame({"row": rows,
                          "key": scal.inchikey14.to_numpy()[rows]})
        d = d.sample(frac=1.0, random_state=args.seed)
        rows = np.sort(d.groupby("key").head(args.cap_per_structure)
                       .row.to_numpy())
        print(f"capped at {args.cap_per_structure}/structure -> {len(rows):,} "
              f"spectra over {d.key.nunique():,} structures")
    elif len(rows) > args.n_spectra:
        rows = np.sort(rng.choice(rows, size=args.n_spectra, replace=False))
    keep = set(rows.tolist())
    print(f"sampling {len(rows):,}")

    # ---- fingerprint every structure we will need --------------------------
    smi_of_key = (
        scal.iloc[rows][["inchikey14", "normalized_smiles"]]
        .drop_duplicates("inchikey14")
        .set_index("inchikey14")
        .normalized_smiles
    )
    print(f"fingerprinting {len(smi_of_key):,} structures...")
    t0 = time.time()
    keys, mats = [], []
    for k, smi in smi_of_key.items():
        b = bits_of(smi)
        if b is not None:
            keys.append(k)
            mats.append(b)
    fp_all = np.stack(mats)
    print(f"  {fp_all.shape} in {time.time()-t0:.0f}s")

    # Select the informative bit band and freeze that choice.
    freq = fp_all.mean(axis=0)
    sel = np.flatnonzero((freq >= MIN_FREQ) & (freq <= MAX_FREQ))
    print(f"kept {len(sel):,}/{FP_SIZE} bits with frequency in "
          f"[{MIN_FREQ}, {MAX_FREQ}]")
    fp_sel = fp_all[:, sel]
    key_to_row = {k: i for i, k in enumerate(keys)}

    # ---- pass 2: stream the peaks -----------------------------------------
    n = len(rows)
    mz = np.zeros((n, MAX_PEAKS), dtype=np.float32)
    inten = np.zeros((n, MAX_PEAKS), dtype=np.float32)
    npk = np.zeros(n, dtype=np.int16)
    prec = np.zeros(n, dtype=np.float32)
    add_id = np.zeros(n, dtype=np.int8)
    ins_id = np.zeros(n, dtype=np.int8)
    ce = np.zeros(n, dtype=np.float32)
    y_row = np.full(n, -1, dtype=np.int32)

    add_index = {a: i for i, a in enumerate(ADDUCT_LIST)}
    pf = pq.ParquetFile(RAW / "train.parquet")
    offset, w = 0, 0
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        m = batch.num_rows
        hits = [r for r in range(offset, offset + m) if r in keep]
        if hits:
            df = batch.to_pandas()
            for r in hits:
                row = df.iloc[r - offset]
                fr = key_to_row.get(row.inchikey14)
                if fr is None:
                    continue
                if len(row.ms2_mzs) == 0:
                    continue
                # The SAME function inference uses. Previously this logic was
                # duplicated in predict_logits and the two drifted apart.
                a, b = select_peaks(row.ms2_mzs, row.ms2_normalized_intensities,
                                    MAX_PEAKS)
                k = a.size
                mz[w, :k], inten[w, :k], npk[w] = a, b, k
                prec[w] = row.precursor_mz
                add_id[w] = add_index.get(row.adduct, 0)
                ins_id[w] = instrument_id(row.instrument_type)
                cev = row.collision_energy_ev
                ce[w] = float(np.mean(cev)) if cev is not None and len(cev) else 0.0
                y_row[w] = fr
                w += 1
        offset += m
        if w and w % 100_000 < 50_000 and hits:
            print(f"  {w:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)

    print(f"wrote {w:,} spectra in {time.time()-t0:.0f}s")
    sl = slice(0, w)
    np.savez_compressed(
        OUT / "train.npz",
        mz=mz[sl], inten=inten[sl], npk=npk[sl], prec=prec[sl],
        add_id=add_id[sl], ins_id=ins_id[sl], ce=ce[sl], y_row=y_row[sl],
        fp=np.packbits(fp_sel, axis=1), n_bits=len(sel), bit_sel=sel,
        keys=np.array(keys, dtype=object),
        adduct_list=np.array(ADDUCT_LIST, dtype=object),
    )
    print(f"-> {OUT / 'train.npz'}")


if __name__ == "__main__":
    main()
