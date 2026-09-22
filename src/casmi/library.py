"""Spectral library: index the training spectra, then rank structures for a query.

This file is embedded verbatim into the Kaggle submission notebook by
`scripts/package_kaggle.py`, so it is commented for reading, not just for
maintenance. It is the whole of the Phase 1 pipeline.

THE IDEA IN THREE SENTENCES
    A mass spectrum is a set of (fragment mass, intensity) pairs. Two spectra of
    the same molecule look alike, so we can identify an unknown by finding the
    training spectrum it most resembles and returning that molecule's structure.
    "Resembles" is measured by entropy similarity, and the search is made fast by
    an index that only compares spectra whose precursor masses already match.

WHY THIS CANNOT SOLVE THE WHOLE PROBLEM
    It can only ever return a structure that is already in the training set. The
    competition deliberately includes molecules that have no public spectra
    (class 2) or are unknown to chemistry entirely (class 3). Those score zero
    here, by construction, no matter how good the similarity function is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ms_entropy import FlashEntropySearch

# --------------------------------------------------------------------------
# Tuning constants
# --------------------------------------------------------------------------

# Keep at most this many peaks per library spectrum. The raw data contains
# spectra with up to 73,318 peaks, nearly all of it electronic noise. Capping
# bounds index memory at roughly 1 KB per spectrum (128 peaks x 2 values x 4
# bytes) and removes noise that would only dilute the similarity score.
MAX_PEAKS = 128

# Discard peaks shorter than 1% of the tallest peak in the same spectrum.
# Detector noise lives down there; real fragments generally do not.
NOISE_THRESHOLD = 0.01

# The REAL test set, taken from the competition's data description page.
# Deliberately NOT measured from the shipped test.parquet: that file is a
# placeholder built out of training rows and is swapped for the hidden test set
# when the notebook is re-run for scoring. Fitting these constants to the
# placeholder (245-460 m/z, 7 adducts) would silently discard most of the real
# test range.
TEST_ADDUCTS = frozenset(
    {
        # An "adduct" is what the molecule picked up to become an electrically
        # charged ion, which is the only form a mass spectrometer can weigh.
        # [M+H]+ means "the molecule plus one proton"; its measured mass is
        # therefore the molecule's mass plus ~1.007.
        "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
        "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
    }
)

# Test molecules weigh 157-1159 Da. Adduct offsets span about -19 to +46, so the
# measured precursor m/z can land anywhere in this padded window.
TEST_MZ_RANGE = (130.0, 1220.0)

# Columns read from train.parquet. Listing them explicitly matters: the two peak
# columns are ~2.7 GB of the 3 GB file, and asking for columns we do not need
# would multiply the memory cost for nothing.
LIB_COLUMNS = [
    "inchikey14",          # 14-char hash of the 2D structure - our identity key
    "normalized_smiles",   # the structure itself, as a text string
    "adduct",              # e.g. "[M+H]+"
    "instrument_type",     # e.g. "timsTOF"
    "precursor_mz",        # measured mass of the whole ionised molecule
    "ms2_mzs",             # array of fragment masses
    "ms2_normalized_intensities",  # array of fragment heights, aligned to above
]


def _top_peaks(mzs, intensities, n: int = MAX_PEAKS) -> np.ndarray:
    """Return the n most intense peaks as an (N, 2) array sorted by mass.

    ms_entropy expects peaks shaped [[mz, intensity], [mz, intensity], ...] in
    ascending mass order, so this both truncates and reorders.
    """
    # float32 halves memory versus float64 and is far finer than the instrument's
    # actual mass accuracy, so no real precision is lost.
    mz = np.asarray(mzs, dtype=np.float32)
    it = np.asarray(intensities, dtype=np.float32)

    if mz.size > n:
        # argpartition finds the n largest intensities WITHOUT sorting the rest.
        # That is O(size) instead of O(size log size) - it matters because this
        # runs two million times. The -n index means "the last n after
        # partitioning", i.e. the n biggest.
        keep = np.argpartition(it, -n)[-n:]
        mz, it = mz[keep], it[keep]

    # argpartition returned them in arbitrary order; restore ascending mass.
    order = np.argsort(mz)

    # Glue the two 1-D arrays into one (N, 2) array: column 0 mass, column 1
    # intensity.
    return np.stack([mz[order], it[order]], axis=1)


def iter_library_rows(
    train_path: Path,
    mz_range: tuple[float, float],
    adducts: set[str],
    batch_size: int = 50_000,
):
    """Stream train.parquet in batches, yielding only the rows worth indexing.

    Streaming rather than reading the file whole is not an optimisation, it is a
    requirement: the file is 3 GB on disk and expands well beyond that in memory,
    and we still have to build a multi-gigabyte index alongside it.
    """
    # ParquetFile opens the file's metadata without reading any row data.
    pf = pq.ParquetFile(train_path)
    lo, hi = mz_range

    # Running count of rows seen so far, used to give every row a stable id.
    offset = 0

    # iter_batches yields chunks instead of the whole table, so peak memory stays
    # flat regardless of file size.
    for batch in pf.iter_batches(batch_size=batch_size, columns=LIB_COLUMNS):
        df = batch.to_pandas()

        # row_id is this row's absolute position in train.parquet. Validation
        # uses it to pull a spectrum's raw peaks back out and to hide that exact
        # measurement from the library. Computed before filtering, so the numbers
        # stay aligned to the real file.
        df["row_id"] = np.arange(offset, offset + len(df), dtype=np.int64)
        offset += len(df)

        # Three filters, combined with & (elementwise AND across the whole
        # column at once - far faster than looping over rows):
        df = df[
            df.precursor_mz.between(lo, hi)     # inside the test mass window
            & df.adduct.isin(adducts)           # an adduct the test set uses
            & df.normalized_smiles.notna()      # has a structure label at all
        ]

        # Skip empty batches so the caller never handles a zero-row frame.
        if len(df):
            yield df


def build(
    train_path: Path,
    out_dir: Path,
    mz_range: tuple[float, float],
    adducts: set[str],
) -> None:
    """Build the searchable index and save it, with metadata, to out_dir."""
    # spectra: what ms_entropy indexes. meta: everything we need to interpret a
    # result (which molecule was it?). They must stay in lockstep.
    spectra, meta = [], []

    for df in iter_library_rows(train_path, mz_range, adducts):
        # itertuples is used rather than iterrows because it is roughly 10x
        # faster - it yields lightweight namedtuples instead of building a pandas
        # Series per row.
        for row in df.itertuples(index=False):
            spectra.append(
                {
                    # "id" is our own field. ms_entropy carries unknown keys
                    # through untouched, which is how we recover the original
                    # ordering below. Its value is simply this spectrum's
                    # position in the list we are building.
                    "id": len(spectra),
                    "precursor_mz": float(row.precursor_mz),
                    "peaks": _top_peaks(row.ms2_mzs, row.ms2_normalized_intensities),
                }
            )
        # Append the whole batch's metadata at once. Row order here matches the
        # order the loop above appended spectra, which is what keeps them aligned.
        meta.append(
            df[["inchikey14", "normalized_smiles", "adduct", "instrument_type", "row_id"]]
        )

    # Stitch the per-batch frames into one table. ignore_index renumbers rows
    # 0..N-1 so positions line up with the spectra list.
    meta = pd.concat(meta, ignore_index=True)
    print(f"  collected {len(spectra):,} spectra")

    fes = FlashEntropySearch()

    # build_index does the heavy lifting: cleans every spectrum, then builds the
    # lookup structures that make searching millions of spectra fast.
    ordered = fes.build_index(
        spectra,
        max_peak_num=MAX_PEAKS,
        noise_threshold=NOISE_THRESHOLD,
        clean_spectra=True,
    )

    # ---------------------------------------------------------------------
    # THE SUBTLE PART. build_index SORTS the spectra by precursor mass and
    # returns them in the new order. It may also drop spectra that end up empty
    # after cleaning. If we left `meta` in its original order, every similarity
    # score would be attributed to the WRONG molecule - and nothing would crash.
    # So we read back each surviving spectrum's "id" (its original position) and
    # reorder meta to match exactly.
    # ---------------------------------------------------------------------
    perm = np.fromiter((s["id"] for s in ordered), dtype=np.int64, count=len(ordered))
    meta = meta.iloc[perm].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "flash").mkdir(exist_ok=True)
    fes.write(out_dir / "flash")  # ms_entropy's own binary format

    # One row per unique structure. Many spectra share a structure (a
    # well-studied molecule can have hundreds), and predictions are per
    # structure, so we need this collapsed view.
    structures = (
        meta.groupby("inchikey14", sort=True)     # sort=True fixes a stable order
        .agg(smiles=("normalized_smiles", "first"))  # any of its SMILES will do
        .reset_index()
    )

    # For each indexed spectrum, which row of `structures` does it belong to?
    # get_indexer looks up every value at once and returns integer positions,
    # which is what lets the scoring step stay pure numpy.
    meta["struct_idx"] = pd.Index(structures.inchikey14).get_indexer(meta.inchikey14)

    # A -1 would mean a spectrum whose structure is missing from the table, which
    # can only happen if the two were built from different data. Fail loudly.
    assert (meta.struct_idx >= 0).all(), "every spectrum must map to a structure"

    meta.to_parquet(out_dir / "meta.parquet", index=False)
    structures.to_parquet(out_dir / "structures.parquet", index=False)
    print(f"  indexed {len(meta):,} spectra / {len(structures):,} structures")


class Library:
    """A loaded index plus the lookup tables needed to rank structures."""

    def __init__(self, out_dir: Path):
        # Construct then read: ms_entropy wants the path at construction time and
        # loads the arrays on .read().
        self.fes = FlashEntropySearch(path_data=out_dir / "flash", low_memory=False)
        self.fes.read(out_dir / "flash")

        self.meta = pd.read_parquet(out_dir / "meta.parquet")
        self.structures = pd.read_parquet(out_dir / "structures.parquet")

        # Pull the hot columns out of pandas into plain numpy arrays once.
        # Indexing a numpy array is far cheaper than indexing a DataFrame, and
        # these are touched on every single query.
        self.struct_idx = self.meta.struct_idx.to_numpy()
        self.n_structures = len(self.structures)
        self.smiles = self.structures.smiles.to_numpy()

        # Group spectra by structure so validation can hide one structure (or one
        # measurement of it) without rebuilding the whole index.
        # Sorting by struct_idx puts every structure's spectra in one contiguous
        # run; searchsorted then records where each run starts. "stable" keeps
        # equal elements in their original relative order, so results are
        # reproducible between runs.
        self._order = np.argsort(self.struct_idx, kind="stable")
        self._starts = np.searchsorted(
            self.struct_idx[self._order], np.arange(self.n_structures + 1)
        )

        # Lookup from train.parquet row_id back to a position in this index.
        self._row_index = pd.Index(self.meta.row_id)

    def spectra_of(self, struct: int) -> np.ndarray:
        """Index positions of every spectrum belonging to one structure."""
        # Slice out this structure's contiguous run set up in __init__.
        return self._order[self._starts[struct] : self._starts[struct + 1]]

    def positions_of_rows(self, row_ids) -> np.ndarray:
        """Index positions for given train.parquet row ids; unknown ids dropped."""
        pos = self._row_index.get_indexer(pd.Index(row_ids))
        return pos[pos >= 0]  # get_indexer returns -1 for "not found"

    def score_spectrum(
        self,
        precursor_mz: float,
        peaks: np.ndarray,
        exclude: np.ndarray | None = None,
    ) -> np.ndarray:
        """Score every structure against ONE query spectrum.

        Returns an array of length n_structures. `exclude` lists index positions
        to ignore; validation uses it to hide the query's own library entry,
        which would otherwise match itself at a perfect 1.0.
        """
        # Apply the same cleaning to the query that the library received, so like
        # is compared with like. This also removes peaks at or near the precursor
        # mass: that peak is the intact molecule that failed to fragment, it is
        # present in every candidate's spectrum, and so it tells us nothing.
        cleaned = self.fes.clean_spectrum_for_search(
            precursor_mz=precursor_mz, peaks=peaks, noise_threshold=NOISE_THRESHOLD
        )

        # A spectrum can be entirely noise and clean away to nothing. Scoring it
        # would crash, so return all-zeros and let the caller carry on.
        if len(cleaned) == 0:
            return np.zeros(self.n_structures, dtype=np.float32)

        # THE ACTUAL SEARCH. "identity" means only library spectra whose
        # precursor mass matches the query's are considered - the same molecule
        # must weigh the same. Returns one similarity per indexed spectrum.
        #   ms1_tolerance: how far the whole molecule's mass may differ (0.01 Da)
        #   ms2_tolerance: how far a fragment's mass may differ (0.02 Da)
        per_spectrum = self.fes.identity_search(
            precursor_mz=precursor_mz,
            peaks=cleaned,
            ms1_tolerance_in_da=0.01,
            ms2_tolerance_in_da=0.02,
        )

        if exclude is not None and len(exclude):
            # Copy first: the array may be a view into ms_entropy's own memory,
            # and writing to it would corrupt the index for later queries.
            per_spectrum = per_spectrum.copy()
            per_spectrum[exclude] = 0.0

        # Collapse per-spectrum scores into per-structure scores.
        out = np.zeros(self.n_structures, dtype=np.float32)

        # np.maximum.at scatters with a maximum: for each spectrum, write its
        # score into its structure's slot, keeping whichever is larger. The ".at"
        # form is essential - a plain out[idx] = vals would keep only the LAST
        # write per structure rather than the best.
        # A structure is judged by its single best-matching spectrum, because
        # averaging would penalise well-studied molecules for also having noisy
        # entries in the library.
        np.maximum.at(out, self.struct_idx, per_spectrum)
        return out

    def analog_structures(
        self,
        precursor_mz: float,
        peaks: np.ndarray,
        top_n: int = 50,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find library structures that are chemically RELATED to the query.

        `identity_search` demands the precursor masses match, so it can only
        find the same molecule. `hybrid_search` allows a constant mass shift
        between query and library spectrum, which is exactly what a chemical
        modification looks like: add a methyl group and every fragment
        containing it moves by the same 14 Da while the rest stay put.

        That makes this the tool for competition class 2 — the molecule itself
        has no reference spectrum, but a cousin of it does.

        Returns (structure indices, scores) for the top_n best analogs.
        """
        cleaned = self.fes.clean_spectrum_for_search(
            precursor_mz=precursor_mz, peaks=peaks, noise_threshold=NOISE_THRESHOLD
        )
        if len(cleaned) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        # No ms1_tolerance argument here: allowing the precursor to differ is
        # the entire point of a hybrid search.
        sims = self.fes.hybrid_search(
            precursor_mz=precursor_mz, peaks=cleaned, ms2_tolerance_in_da=0.02
        )
        if exclude is not None and len(exclude):
            sims = sims.copy()
            sims[exclude] = 0.0

        per_struct = np.zeros(self.n_structures, dtype=np.float32)
        np.maximum.at(per_struct, self.struct_idx, sims)

        n = min(top_n, self.n_structures)
        top = np.argpartition(-per_struct, n - 1)[:n]
        top = top[np.argsort(-per_struct[top])]
        return top, per_struct[top]

    def rank_molecule(
        self,
        spectra: list[tuple[float, np.ndarray]],
        k: int = 25,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse a molecule's spectra into one ranked candidate list.

        `spectra` is [(precursor_mz, peaks), ...] - all measurements of the SAME
        molecule, typically at different collision energies. The competition
        scores per molecule, not per spectrum, so they must be combined.

        Returns (structure indices, scores), best first, both of length k.
        """
        total = np.zeros(self.n_structures, dtype=np.float32)

        # Sum each spectrum's per-structure scores...
        for mz, peaks in spectra:
            total += self.score_spectrum(mz, peaks, exclude=exclude)

        # ...then divide to get the mean. Using the mean rather than the maximum
        # rewards structures that match at EVERY collision energy over ones that
        # matched strongly at a single energy by chance. max(...,1) guards
        # against a molecule that somehow arrived with no spectra.
        total /= max(len(spectra), 1)

        # Never ask for more candidates than exist.
        k = min(k, self.n_structures)

        # Find the k best without sorting all ~269,000 structures. Negating turns
        # "largest" into "smallest", which is what argpartition finds.
        top = np.argpartition(-total, k - 1)[:k]

        # argpartition guarantees the right k, but in arbitrary order. Rank is the
        # entire metric here, so sort those k properly.
        top = top[np.argsort(-total[top])]

        return top, total[top]
