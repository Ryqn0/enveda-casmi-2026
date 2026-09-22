"""Spectral library: build a Flash Entropy index over the training spectra,
then rank candidate structures for a query molecule.

The competition gives 1-9 spectra per molecule (different collision energies).
We score each spectrum independently against the library, reduce to one score
per *structure* (a structure may have hundreds of library spectra), then fuse
across the molecule's spectra into a single ranked list.

Index layout on disk (`data/index/`):
    flash/        the ms_entropy index (its own binary format)
    meta.parquet  one row per indexed spectrum, in index order
    structures.parquet  one row per unique InChIKey14
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import FlashEntropySearch

# Peaks kept per library spectrum before indexing. The raw data has spectra with
# up to 73k peaks, which is noise; capping bounds index memory at ~1KB/spectrum.
MAX_PEAKS = 128
NOISE_THRESHOLD = 0.01  # drop peaks below 1% of the base peak

# The real test set, per the competition data description — NOT derived from the
# shipped test.parquet, which is a placeholder built from training rows and is
# both narrower (245-460 m/z, 7 adducts) and replaced at re-run.
TEST_ADDUCTS = frozenset(
    {
        "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
        "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
    }
)
# Monoisotopic masses run 157-1159 Da; adduct offsets span roughly -19 to +46,
# so precursor m/z can fall anywhere in this window. Padded either side.
TEST_MZ_RANGE = (130.0, 1220.0)

LIB_COLUMNS = [
    "inchikey14",
    "normalized_smiles",
    "adduct",
    "instrument_type",
    "precursor_mz",
    "ms2_mzs",
    "ms2_normalized_intensities",
]


def _top_peaks(mzs, intensities, n: int = MAX_PEAKS) -> np.ndarray:
    """(N,2) float32 array of the n most intense peaks, sorted by m/z.

    ms_entropy wants [[mz, intensity], ...] ascending in m/z.
    """
    mz = np.asarray(mzs, dtype=np.float32)
    it = np.asarray(intensities, dtype=np.float32)
    if mz.size > n:
        keep = np.argpartition(it, -n)[-n:]  # n largest intensities, unordered
        mz, it = mz[keep], it[keep]
    order = np.argsort(mz)
    return np.stack([mz[order], it[order]], axis=1)


def iter_library_rows(
    train_path: Path,
    mz_range: tuple[float, float],
    adducts: set[str],
    batch_size: int = 50_000,
):
    """Stream train.parquet, yielding filtered batches.

    Streaming matters: the file is 3GB and the peak columns dominate it. Reading
    it whole would not fit alongside the index we are building.
    """
    pf = pq.ParquetFile(train_path)
    lo, hi = mz_range
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=LIB_COLUMNS):
        df = batch.to_pandas()
        # row_id is the row's position in train.parquet, kept so validation can
        # pull the *raw* spectrum back out and exclude exactly that measurement.
        df["row_id"] = np.arange(offset, offset + len(df), dtype=np.int64)
        offset += len(df)
        df = df[
            df.precursor_mz.between(lo, hi)
            & df.adduct.isin(adducts)
            & df.normalized_smiles.notna()
        ]
        if len(df):
            yield df


def build(
    train_path: Path,
    out_dir: Path,
    mz_range: tuple[float, float],
    adducts: set[str],
) -> None:
    """Build and persist the Flash Entropy index plus aligned metadata."""
    spectra, meta = [], []
    for df in iter_library_rows(train_path, mz_range, adducts):
        for i, row in enumerate(df.itertuples(index=False)):
            spectra.append(
                {
                    "id": len(spectra),
                    "precursor_mz": float(row.precursor_mz),
                    "peaks": _top_peaks(row.ms2_mzs, row.ms2_normalized_intensities),
                }
            )
        meta.append(
            df[["inchikey14", "normalized_smiles", "adduct", "instrument_type", "row_id"]]
        )
    meta = pd.concat(meta, ignore_index=True)
    print(f"  collected {len(spectra):,} spectra")

    fes = FlashEntropySearch()
    ordered = fes.build_index(
        spectra,
        max_peak_num=MAX_PEAKS,
        noise_threshold=NOISE_THRESHOLD,
        clean_spectra=True,
    )
    # build_index sorts by precursor m/z — metadata MUST follow that permutation
    # or every score maps to the wrong molecule.
    perm = np.fromiter((s["id"] for s in ordered), dtype=np.int64, count=len(ordered))
    meta = meta.iloc[perm].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flash").mkdir(exist_ok=True)
    fes.write(out_dir / "flash")

    # structure table: one row per unique InChIKey14, with a representative SMILES
    structures = (
        meta.groupby("inchikey14", sort=True)
        .agg(smiles=("normalized_smiles", "first"))
        .reset_index()
    )
    # map each indexed spectrum to its structure row
    meta["struct_idx"] = pd.Index(structures.inchikey14).get_indexer(meta.inchikey14)
    assert (meta.struct_idx >= 0).all(), "every spectrum must map to a structure"

    meta.to_parquet(out_dir / "meta.parquet", index=False)
    structures.to_parquet(out_dir / "structures.parquet", index=False)
    print(f"  indexed {len(meta):,} spectra / {len(structures):,} structures")


class Library:
    """Loaded index + the lookup tables needed to rank structures."""

    def __init__(self, out_dir: Path):
        self.fes = FlashEntropySearch(path_data=out_dir / "flash", low_memory=False)
        self.fes.read(out_dir / "flash")
        self.meta = pd.read_parquet(out_dir / "meta.parquet")
        self.structures = pd.read_parquet(out_dir / "structures.parquet")
        self.struct_idx = self.meta.struct_idx.to_numpy()
        self.n_structures = len(self.structures)
        self.smiles = self.structures.smiles.to_numpy()

        # Group library spectra by structure so validation can hide a structure
        # (or a single measurement of it) without rebuilding the index.
        self._order = np.argsort(self.struct_idx, kind="stable")
        self._starts = np.searchsorted(
            self.struct_idx[self._order], np.arange(self.n_structures + 1)
        )

        self._row_index = pd.Index(self.meta.row_id)

    def spectra_of(self, struct: int) -> np.ndarray:
        """Library positions holding spectra of this structure."""
        return self._order[self._starts[struct] : self._starts[struct + 1]]

    def positions_of_rows(self, row_ids) -> np.ndarray:
        """Library positions for given train.parquet row ids (missing dropped)."""
        pos = self._row_index.get_indexer(pd.Index(row_ids))
        return pos[pos >= 0]

    def score_spectrum(
        self,
        precursor_mz: float,
        peaks: np.ndarray,
        exclude: np.ndarray | None = None,
    ) -> np.ndarray:
        """Best entropy similarity per structure for one query spectrum.

        Returns a vector of length n_structures. A structure scores as well as
        its single best-matching library spectrum — averaging over its spectra
        would punish structures that happen to have many noisy entries.

        `exclude` is library *spectrum* positions to ignore, zeroed before the
        per-structure reduction. Validation uses it to drop the query's own
        library entry, which would otherwise match itself at similarity 1.0.
        """
        cleaned = self.fes.clean_spectrum_for_search(
            precursor_mz=precursor_mz, peaks=peaks, noise_threshold=NOISE_THRESHOLD
        )
        if len(cleaned) == 0:
            return np.zeros(self.n_structures, dtype=np.float32)
        per_spectrum = self.fes.identity_search(
            precursor_mz=precursor_mz,
            peaks=cleaned,
            ms1_tolerance_in_da=0.01,
            ms2_tolerance_in_da=0.02,
        )
        if exclude is not None and len(exclude):
            per_spectrum = per_spectrum.copy()
            per_spectrum[exclude] = 0.0
        out = np.zeros(self.n_structures, dtype=np.float32)
        np.maximum.at(out, self.struct_idx, per_spectrum)
        return out

    def rank_molecule(
        self,
        spectra: list[tuple[float, np.ndarray]],
        k: int = 25,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse a molecule's spectra into one ranked candidate list.

        `spectra` is [(precursor_mz, peaks), ...] for the same molecule.
        `exclude` is library spectrum positions to hide from every query.

        Fusion is the mean over query spectra: a structure that matches at every
        collision energy beats one that matches at a single energy by luck.
        Returns (structure indices, scores), both length k, best first.
        """
        total = np.zeros(self.n_structures, dtype=np.float32)
        for mz, peaks in spectra:
            total += self.score_spectrum(mz, peaks, exclude=exclude)
        total /= max(len(spectra), 1)
        k = min(k, self.n_structures)
        top = np.argpartition(-total, k - 1)[:k]
        top = top[np.argsort(-total[top])]  # argpartition is unordered; sort it
        return top, total[top]
