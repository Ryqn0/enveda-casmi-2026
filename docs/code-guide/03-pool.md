# Part 3 — `pool.py`: candidate structures by mass

**Written:** Phase 2. **Result:** together with `analog.py`, LB 0.130 → 0.240.
**Used by:** `build_pool.py` (`build_pool`); `rank.py`, `app.py`, and every
validation script (`CandidatePool`, `ADDUCT_MASS`).

---

## The idea

Phase 1 could only return molecules that have a training spectrum. Class 2
molecules have no spectrum, so they weren't badly ranked — they were **absent**.
The fix is retrieval: list every known structure whose mass matches the
measurement, then rank that list. This file does the listing.

**The one physics fact you need:** the instrument weighs the **ion**, not the
molecule. `[M+H]+` means "the molecule plus a proton", so

```
measured m/z = neutral mass + adduct mass        (all ions here carry charge 1)
neutral mass = measured m/z − adduct mass
```

The neutral mass is a property of the structure alone, so it can be compared
with a database of structures.

---

## `MONOISOTOPIC` (lines 30–41)

```python
MONOISOTOPIC = {"C": 12.000000, "H": 1.0078250319, "N": 14.0030740052, ...}
```

A **dict** mapping element symbol → mass of its **most abundant isotope**.
The periodic-table average would be wrong here: chlorine averages 35.45 (a mix
of ³⁵Cl and ³⁷Cl), but the spectrometer sees each isotope separately, so the
monoisotopic ³⁵Cl = 34.9689 is right. Carbon is exactly 12 by definition.

## `ELECTRON` and `ADDUCT_MASS` (lines 43–64)

```python
ELECTRON = 0.00054858
_H = MONOISOTOPIC["H"]
_H2O = 2 * _H + MONOISOTOPIC["O"]
ADDUCT_MASS = {
    "[M+H]+":   _H - ELECTRON,
    "[M+Na]+":  MONOISOTOPIC["Na"] - ELECTRON,
    "[M-H]-":   -_H + ELECTRON,
    "[M-H2O+H]+": _H - ELECTRON - _H2O,
    ...
}
```

- What each adduct **adds** to the neutral mass.
- **The electron correction:** a proton is a hydrogen *atom* minus its
  electron: 1.007825 − 0.000549 = 1.007276. Ignoring 0.00055 Da sounds tiny,
  but it's a 1.5 ppm error at m/z 350 — as large as the search window itself.
- Positive ions **lost** an electron (subtract it); negative ions **gained** one (add it).
- `[M-H2O+H]+` subtracts a whole water: the molecule lost water in the source.
- `_H` and `_H2O` are helper constants (the underscore: internal to this module).

## `_FORMULA_TOKEN` (line 69) — a regular expression

```python
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
```

- A **regular expression** is a text pattern. `re.compile` prepares it once.
- `r"..."` is a **raw string**: backslashes stay literal (so `\d` reaches the
  regex engine intact).
- `([A-Z][a-z]?)` — group 1: an uppercase letter, optionally one lowercase
  (`C`, `Cl`, `Br`). `(\d*)` — group 2: zero or more digits (the count).
- `findall("C8H10N4O2")` → `[("C","8"), ("H","10"), ("N","4"), ("O","2")]`.

## `formula_mass(formula)` (lines 72–97)

```python
    if not isinstance(formula, str) or not formula:
        return None
    if any(c in formula for c in "+-."):
        return None
    total = 0.0
    consumed = 0
    for sym, count in _FORMULA_TOKEN.findall(formula):
        if sym not in MONOISOTOPIC:
            return None
        total += MONOISOTOPIC[sym] * (int(count) if count else 1)
        consumed += len(sym) + len(count)
    if consumed != len(formula):
        return None
    return total
```

Line by line:
1. Not a string, or empty → `None`.
2. `+`/`-` mean a **charged** formula; `.` means a **hydrate or salt**
   (`C6H12O6.H2O`). Neither is a neutral molecule's mass → reject.
3. For each `(element, count)` token: unknown element → reject;
   `int(count) if count else 1` — a **conditional expression**: no digits means
   one atom (`O` in `CO` = 1 oxygen).
4. `consumed` counts the characters the regex actually used. `findall`
   **silently skips** characters it can't match, so comparing the count with the
   formula's length catches stray junk (`C6H12x`) instead of ignoring it.

> **Why refuse rather than guess:** a slightly wrong mass puts the right answer
> *outside* the search window forever, silently. Returning `None` drops that
> row, loudly countable.

## `build_pool(coconut_csv, train_parquet, out)` (lines 100–152)

Builds `data/pool.parquet`, one row per structure, sorted by mass:

1. **COCONUT** (read with only 4 columns): `canonical_smiles`,
   `standard_inchi_key` (first 14 characters → `inchikey14`),
   `molecular_formula`, and COCONUT's own mass (kept for a sanity check).
2. **Training structures**: `drop_duplicates(subset="inchikey14")` — one row per molecule.
3. `pd.concat(...)`: stack both tables.
4. Drop rows missing a key or SMILES.
5. `drop_duplicates(subset="inchikey14", keep="last")` — a structure in both
   sources is kept **once**, labelled `train` (it's last). *This later confused
   a COCONUT-coverage count (docs/11): check membership by key, not by label.*
6. `pool.molecular_formula.map(formula_mass)` — `.map` applies a function to
   every value of a column; rows it can't weigh are dropped.
7. **`sort_values("exact_mass")`** — the whole point: a sorted mass column
   allows binary search.
8. Save to Parquet.

The build script compares our masses with COCONUT's own: median difference
2.4×10⁻⁶ Da, 99.98% within 1 mDa.

## `class CandidatePool` (lines 155–212)

### `__init__(self, path)`

```python
        self.df = pd.read_parquet(path)
        self.mass = self.df.exact_mass.to_numpy()
        self.smiles = self.df.smiles.to_numpy()
        self.key = self.df.inchikey14.to_numpy()
        assert np.all(np.diff(self.mass) >= 0), "pool must be sorted by mass"
```

- Three numpy arrays in the **same order**: position `i` is one structure's
  mass, SMILES and key. Everything else refers to candidates by **position**.
- `np.diff` = differences between neighbours; all ≥ 0 means sorted. Binary
  search on an unsorted array returns garbage **without an error**, so this
  assertion guards it.

### `neutral_mass(precursor_mz, adduct, offset_ppm)`

```python
        add = ADDUCT_MASS.get(adduct)
        if add is None:
            return None
        m = precursor_mz - add
        return m / (1.0 + offset_ppm * 1e-6)
```

- `.get(key)` returns `None` for an unknown adduct instead of raising.
- **Calibration correction:** timsTOF reads **+0.98 ppm high** on average. If
  the true mass is `m_true`, the instrument reports `m_true × (1 + 0.98e-6)`, so
  dividing removes the bias and **centres** the search window on the truth.
  (The per-library numbers are in the docstring: enveda-180 +0.98, gnps +1.55,
  massbank +1.87, riken +3.29. Natural products on timsTOF turned out to be
  ~+1.54 — docs/13.)

### `query(precursor_mz, adduct, ppm, offset_ppm)` — the search

```python
        m = self.neutral_mass(precursor_mz, adduct, offset_ppm)
        if m is None or m <= 0:
            return np.empty(0, dtype=np.int64)
        tol = m * ppm * 1e-6
        lo, hi = np.searchsorted(self.mass, [m - tol, m + tol])
        return np.arange(lo, hi, dtype=np.int64)
```

- **ppm** = parts per million: the window scales with mass, because
  instrument error does.
- **`np.searchsorted(sorted, values)`** = binary search: for each value, the
  position where it would be inserted to keep the array sorted. With 722,737
  masses that's ~20 comparisons instead of 722,737.
- The candidates are every position between `lo` and `hi`: a **half-open**
  interval `[m − tol, m + tol)`. (The browser demo reproduces exactly this
  boundary; the golden test checks it.)
- Returns **positions**, which index `self.smiles` and `self.key` directly.

### Worked example: caffeine

| step | value |
|---|---|
| measured ion `[M+H]+` | 195.087652 |
| minus `ADDUCT_MASS["[M+H]+"]` (1.007276) | 194.080376 |
| divide by (1 + 0.98×10⁻⁶) | **194.080185** Da |
| tolerance at 5 ppm: 194.08 × 5×10⁻⁶ | ±0.00097 Da |
| window | [194.079215, 194.081156) |
| result | **17 candidates** — caffeine and 16 isomers of the same formula |

Isomers have **exactly** the same formula, hence the same mass. No mass window
can separate them; that's why ranking is the real problem.

### `_self_check()`

- `formula_mass("C8H10N4O2") ≈ 194.080376` (caffeine).
- Unknown element, charged formula, hydrate, empty → all `None`.
- `194.080376 + ADDUCT_MASS["[M+H]+"] ≈ 195.087652` — the adduct table matches
  a known measurement.

---

## Try it

```bash
uv run python src/casmi/pool.py
uv run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from casmi.pool import CandidatePool, formula_mass
p = CandidatePool(Path('data/pool.parquet'))
print(formula_mass('C8H10N4O2'))
c = p.query(195.087652, '[M+H]+', ppm=5, offset_ppm=0.98)
print(len(c), 'candidates; first:', p.smiles[c[0]])
"
```
