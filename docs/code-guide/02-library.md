# Part 2 — `library.py`: searching 2.1 million reference spectra

**Written:** Phase 1. **Result:** LB 0.130.
**Used by:** `build_library.py` (builds the index with `build`), and every
validation script plus `rank.py` (they load it with `Library`).

The source file is already commented line by line, because it's embedded in
the Kaggle notebook — read it side by side with this part. Here the focus is
the **concepts** each line relies on, and one query followed end to end.

---

## The idea

Two spectra of the same molecule look alike. So: index every training
spectrum; for a query, find the most similar library spectra; return their
molecules. It can only ever return molecules that *have* a training spectrum —
class 1. Classes 2 and 3 score zero here by construction.

**Two stages, two moments in time:**

```
ONE TIME (build_library.py, ~20 min)          EVERY QUERY (validation / notebook)
train.parquet (3 GB)                          query spectrum
   │ iter_library_rows: stream + filter          │ Library.score_spectrum / analog_structures
   │ _top_peaks: keep 128 biggest peaks          │   clean like the library
   ▼                                             │   Flash Entropy search
FlashEntropySearch.build_index                   │   per-spectrum → per-structure (maximum.at)
   │ (sorts by precursor mass!)                  ▼
   ▼                                          scores for all 268,932 structures
data/index/  flash/ + meta.parquet + structures.parquet
```

---

## Constants (lines 30–76)

| constant | value | why |
|---|---|---|
| `MAX_PEAKS` | 128 | some raw spectra have 73,318 peaks, nearly all noise; caps memory at ~1 KB per spectrum |
| `NOISE_THRESHOLD` | 0.01 | drop peaks under 1% of the tallest — detector noise |
| `TEST_ADDUCTS` | 10 adducts | from the data **description page**, not the placeholder file |
| `TEST_MZ_RANGE` | (130, 1220) | test molecules weigh 157–1159 Da, padded for adduct offsets |
| `LIB_COLUMNS` | 7 columns | read only what's needed: the peak columns are 2.7 GB of the 3 GB file |

**`frozenset`** is an immutable set: fast "is X in it?" checks, and nothing can
accidentally modify it later.

> **Lesson baked in (line 44):** these constants were taken from the competition's
> documentation on purpose. Fitting them to the downloadable `test.parquet`
> (a placeholder) would have thrown away most of the real test range.

---

## `_top_peaks(mzs, intensities, n)` (lines 79–103)

Keeps the `n` most intense peaks, returned sorted by mass as an `(N, 2)` array.

**Concepts:**
- `np.asarray(x, dtype=np.float32)`: converts to a numpy array without copying
  if it already is one. `float32` = 4 bytes per number instead of 8 — half the
  memory, still far more precise than the instrument.
- `np.argpartition(it, -n)[-n:]`: returns the **positions** of the `n` largest
  values without fully sorting (O(size) instead of O(size·log size)). It runs
  2 million times, so this matters.
- `np.argsort(mz)`: positions that would sort by mass. `mz[order]` uses those
  positions to reorder — **fancy indexing**.
- `np.stack([a, b], axis=1)`: glues two 1-D arrays as columns → shape `(N, 2)`.

> **Note from the golden test (docs/14 §9.2):** `argpartition` picks
> *arbitrarily* among peaks tied at the cutoff. `fpnet.select_peaks` was made
> deterministic for that reason. This function still uses `argpartition`
> because the index was built with it; rebuilding with a deterministic rule
> would be the consistent choice next time the index is rebuilt.

---

## `iter_library_rows(train_path, mz_range, adducts, batch_size)` (lines 106–147)

Streams `train.parquet` in chunks of 50,000 rows and **yields** only useful rows.

**Concepts:**
- **Generators and `yield`:** a function containing `yield` doesn't run all at
  once. Each `for df in iter_library_rows(...)` asks it for the next batch; it
  runs until the next `yield`, hands the batch over, and pauses. Memory stays
  flat no matter how big the file is.
- **`pq.ParquetFile(path)` + `iter_batches(columns=...)`:** Parquet is a
  columnar format, so asking for 7 columns really reads only those 7.
- **`row_id` computed before filtering** (line 134): every row keeps its
  absolute position in the original file. Validation later uses it to hide one
  exact measurement from the library.
- **Boolean masks** (lines 139–143):
  ```python
  df = df[df.precursor_mz.between(lo, hi) & df.adduct.isin(adducts) & df.normalized_smiles.notna()]
  ```
  Each condition produces a column of `True/False`; `&` combines them
  element-wise; `df[mask]` keeps the `True` rows. It's vectorised: the whole
  column in one C-level operation, instead of a Python loop.

---

## `build(train_path, out_dir, mz_range, adducts)` (lines 150–234)

Builds the index and saves three things to `data/index/`.

1. **Collect** (lines 159–185): two lists grown in lockstep — `spectra` (what
   the search engine needs: `id`, `precursor_mz`, `peaks`) and `meta` (what we
   need to interpret a hit: which molecule, which row). `itertuples()` is ~10×
   faster than `iterrows()` because it yields light tuples, not pandas Series.
2. **Index** (lines 188–197): `FlashEntropySearch().build_index(...)` cleans
   every spectrum and builds the lookup structures.
3. **The subtle part** (lines 199–208):
   ```python
   perm = np.fromiter((s["id"] for s in ordered), dtype=np.int64, count=len(ordered))
   meta = meta.iloc[perm].reset_index(drop=True)
   ```
   `build_index` **re-sorts** the spectra by precursor mass (and may drop
   empty ones). Our own `"id"` field survives the re-sort, so reading it back
   gives the permutation, and `meta` is reordered to match. Without these two
   lines every score would be attributed to the **wrong molecule** — and
   nothing would crash. `np.fromiter` builds an array straight from a
   generator, without an intermediate list.
4. **One row per structure** (lines 217–221): `groupby("inchikey14").agg(...)`
   collapses many spectra to one row per molecule (predictions are per molecule).
5. **Map spectrum → structure** (line 226):
   `pd.Index(keys).get_indexer(values)` looks up every value at once and returns
   integer positions (−1 if missing). The `assert` right after refuses −1s.
6. **Save**: `fes.write(...)` (the engine's binary format) and two Parquet files.

---

## `class Library` (lines 237–416)

**Concepts first:** a **class** bundles data (*attributes*, `self.x`) with
functions that use it (*methods*). `__init__` runs when you write
`Library(path)`; `self` is the object being built.

### `__init__(self, out_dir)` (lines 240–268)

- Loads the index and the two Parquet tables.
- Pulls hot columns out of pandas into plain numpy arrays once (lines 252–254):
  numpy indexing is much cheaper, and these are touched on every query.
- **Runs of spectra per structure** (lines 262–265):
  ```python
  self._order = np.argsort(self.struct_idx, kind="stable")
  self._starts = np.searchsorted(self.struct_idx[self._order], np.arange(self.n_structures + 1))
  ```
  Sorting spectra by structure puts each molecule's spectra next to each
  other; `searchsorted` finds where each molecule's run begins. Then "all
  spectra of molecule 42" is a single slice: `_order[_starts[42]:_starts[43]]`.
  `kind="stable"` keeps ties in their original order, so results are reproducible.

### `spectra_of(struct)` and `positions_of_rows(row_ids)` (lines 270–278)

- `spectra_of` returns that slice — used by validation to **hide a whole
  molecule** (the class-2 simulation).
- `positions_of_rows` maps file row ids → index positions — used to hide **only
  the query spectra** (the class-1 simulation). `pos[pos >= 0]` drops "not found".

### `score_spectrum(precursor_mz, peaks, exclude)` (lines 280–334) — one query

Following one query through:

1. **Clean the query like the library was cleaned** (line 296) — same noise
   threshold, and it removes the unfragmented precursor peak (present in every
   candidate, so it carries no information).
2. **Empty after cleaning → all zeros** (line 302), rather than crashing.
3. **`identity_search`** (line 310): only library spectra whose precursor mass
   matches within 0.01 Da; fragment peaks match within 0.02 Da. Returns one
   similarity per indexed spectrum (2.1M numbers).
4. **Hide excluded spectra** (lines 317–321):
   ```python
   per_spectrum = per_spectrum.copy()
   per_spectrum[exclude] = 0.0
   ```
   **Views vs copies:** a numpy array can be a *view* into someone else's
   memory. Writing zeros into a view of the engine's internal array would
   corrupt the index for every later query. `.copy()` first makes it ours.
5. **Per spectrum → per structure** (line 333):
   ```python
   np.maximum.at(out, self.struct_idx, per_spectrum)
   ```
   For each spectrum `i`, set `out[struct_idx[i]] = max(out[...], per_spectrum[i])`.
   Why `.at`: plain `out[idx] = vals` with repeated indices keeps only the
   **last** write. `.at` applies every write — so each molecule gets its
   **best** matching spectrum.

### `analog_structures(precursor_mz, peaks, top_n, exclude)` (lines 336–377)

Same shape as `score_spectrum`, with one crucial difference: **`hybrid_search`**
instead of `identity_search`. It allows the precursor mass to differ and
matches fragments both at their own mass and **shifted by the precursor
difference**. A chemical modification (e.g. +14 Da for a methyl group) shifts
exactly the fragments that contain it — so this finds **structural cousins**,
the basis of Phase 2's class-2 strategy. Returns the top 50 cousins.

**Top-k without sorting everything** (lines 374–376), a pattern used all over
the project:

```python
top = np.argpartition(-per_struct, n - 1)[:n]      # the n best, unordered (fast)
top = top[np.argsort(-per_struct[top])]            # sort just those n
```

Negating turns "largest" into "smallest", which is what `argpartition` finds.

### `rank_molecule(spectra, k, exclude)` (lines 379–416)

The Phase 1 prediction: a molecule has several spectra (different collision
energies); score each, **average** them, return the top 25. The average
rewards structures that match at every energy over a lucky single match.
`total /= max(len(spectra), 1)` guards against division by zero.

---

## Try it

```bash
uv run python scripts/build_library.py     # one time: builds data/index/ (~20 min)
uv run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from casmi.library import Library
lib = Library(Path('data/index'))
print(len(lib.meta), 'spectra /', lib.n_structures, 'structures')
print('spectra of structure 0:', lib.spectra_of(0))
"
```
