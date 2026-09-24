# Part 8a — scripts: building data and validating

Scripts are the **entry points**: each one is something you run from the
command line. They stay thin: they parse arguments, call functions from
`src/casmi/`, and print or save results. The logic lives in `src` so that
scripts, the notebook and the app all share one implementation.

This part covers the scripts that build data and measure performance, in the
order they were written.

---

## Patterns every script shares

```python
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ...

if __name__ == "__main__":
    main()
```

- **`Path(__file__).resolve().parents[1]`**: `__file__` is this script's
  path; `.resolve()` makes it absolute; `.parents[0]` is `scripts/`,
  `.parents[1]` is the project root. So paths work from any working directory.
- **`/` on `Path` objects** joins path parts: `ROOT / "data" / "raw"`.
- **`argparse`** turns command-line flags into `args.n`, `args.seed`… `-n`
  and `--seed` become attributes; `type=int` converts the text; `default` is
  used when the flag is absent. `--offset-ppm` becomes `args.offset_ppm`
  (hyphens → underscores — the source of a Phase 2 bug where a search for
  `offset_ppm` missed the `--offset-ppm` line and it was added twice).
- **Seeds** (`np.random.default_rng(seed)`): every random choice is
  reproducible, so two runs with the same seed sample the same molecules.
- **Scripts importing from other scripts**: `sys.path.insert(0, str(Path(__file__).resolve().parent))`
  lets `validate_fpnet.py` do `from validate_class2 import sample_structures`.
  The `# noqa: E402` comment tells the linter "yes, I know this import isn't at
  the top — it has to come after the path change".

---

## `eda.py` — first look at the data (Phase 0)

Writes `docs/01-data-notes.md`. Read it for *what* the data looks like; read
the script for *how* to profile data cheaply.

- **Only scalar columns** (`SCALARS`): the peak arrays are 2.7 GB of the 3 GB
  file and aren't needed to count things.
- `say()` both prints and collects lines, so one script produces the console
  output and the Markdown file.
- `counts(series, name)`: `value_counts()` → a Markdown table of the most
  common values with their share.
- **The library-search ceiling** (lines 136–187): for each test molecule, how
  many training structures have a matching precursor mass (±15 ppm, same
  adduct)? Uses the same trick as the pool: sort masses once, then
  `np.searchsorted(mz, [lo, hi])` per molecule. Result: the median molecule has
  dozens of mass-matched candidates for 25 slots — "ranking does the real work",
  a conclusion that held all project.
- The warning block at the top was added after the placeholder discovery: the
  "test set" section describes the **placeholder** file.

## `build_library.py` — build the spectral index (Phase 1)

```python
    mz_range, adducts = TEST_MZ_RANGE, set(TEST_ADDUCTS)
    ...
    build(RAW / "train.parquet", args.out, mz_range, adducts)
```

Calls `library.build` (Part 2) with the test window from the documentation.

**`--limit N` smoke test** (lines 36–48): replaces `library.iter_library_rows`
with a wrapper that stops after N batches:

```python
        original = lib.iter_library_rows
        def limited(*a, **kw):
            for i, df in enumerate(original(*a, **kw)):
                if i >= args.limit:
                    return
                yield df
        lib.iter_library_rows = limited
```

- `*a, **kw` accept **any** positional and keyword arguments and pass them on
  unchanged — a generic wrapper.
- Reassigning a module's function at run time is **monkeypatching**. It works
  because `build` looks up `iter_library_rows` by name when it runs. Useful for
  a quick test; in real code, prefer passing a parameter.

## `build_pool.py` — build the candidate pool (Phase 2)

Calls `pool.build_pool` (Part 3), then **checks itself**:
- our computed masses vs COCONUT's own (`exact_molecular_weight`): median
  difference 2.4×10⁻⁶ Da, 99.98% within 1 mDa;
- caffeine `[M+H]+` at 5 ppm returns candidates and contains caffeine's key.

A build script that verifies its own output is cheap insurance.

## `check_leak.py` — the (retracted) leak check (Phase 1)

Finds test spectra that exist verbatim in the training file.

- **Prefilter, then compare** (lines 36–38): comparing 1,213 test spectra
  with 2.5M training spectra element by element is too slow, so first bucket by
  `(precursor rounded to 4 decimals, number of peaks)` in a `defaultdict(list)`
  (a dict that creates an empty list for a new key automatically). Only pairs in
  the same bucket are compared in full.
- **Exact comparison** (line 64): `np.array_equal`, deliberately. The first
  version used `np.allclose`, whose default relative tolerance (1e-5)
  accepts ~4 mDa differences at m/z 400 — enough to call different spectra
  "identical". The claim was identity, so the test must be identity.
- The finding (every test spectrum is a training row) was real; the
  conclusion ("a leak") was wrong: the file is a documented placeholder.
  `data/leak_map.parquet` later proved useful anyway — it showed the
  placeholder is 100% enveda-180 (docs/12).

## `pool_recall.py` — the retrieval ceiling (Phase 2)

For sampled timsTOF training molecules: is the true structure among the
candidates at 5, 10 and 20 ppm, and how many candidates come back?

```python
    in_pool = np.isin(sample.inchikey14.to_numpy(), pool.key)
```

- `np.isin(a, b)`: for each element of `a`, is it in `b`?
- **The flaw found later (docs/11):** it samples *training* molecules, and the
  pool contains every training structure by construction — so recall could
  only ever be ~100%. It correctly measures that mass arithmetic works; it can't
  measure whether a real test answer is in the pool. The script even prints
  that caveat (line 54) — the numbers were then over-read anyway.

## `validate.py` — the first holdout (Phase 1)

Simulates the test set with training molecules measured on timsTOF.

**`pick_holdout(lib, n, seed)`**: choose molecules with ≥3 spectra, take up
to 4 as queries, and **leave at least one behind**:

```python
        take = list(rng.permutation(ids)[: min(MAX_QUERY_SPECTRA, len(ids) - 1)])
    ...
    assert (left > 0).all(), "holdout leaves a structure unrepresented in the library"
```

This guard exists because of the **degenerate holdout bug**: without it, some
"known" molecules had *all* their spectra taken as queries, so "known" silently
measured "novel" (0.2653 reported vs 0.8446 true).

**`fetch_raw(row_ids)`**: one streaming pass over the file, collecting the
**raw** peaks of just the needed rows. Raw, not cleaned: test spectra arrive
uncleaned (median 230 peaks vs 42 after cleaning), so validating on cleaned
spectra would flatter the method. `if len(out) == len(want): break` stops
reading early once everything is found.

**`run_mode(lib, holdout, raw, mode)`**: two simulations:
- `known` → hide only the query spectra (`positions_of_rows`) → class 1;
- `novel` → hide every spectrum of the molecule (`spectra_of`) → class 3 for
  library search (scores 0 by construction).

Results are also broken down **by how many spectra remain** in the library —
"an aggregate alone hides the story".

## `validate_class2.py` — class-2 simulation and the shared sampler (Phase 2+)

Two roles: a Phase 2 experiment, and the home of two functions reused by later
validators.

### `sample_structures(lib, pool, n, seed, sources, unseen_only, seen_keys, coconut_only)`

The **one** holdout sampler used by `validate_combined.py` and
`validate_fpnet.py`. It grew one filter per lesson:

| argument | filter | lesson |
|---|---|---|
| (default) | timsTOF molecules | match the test instrument (Phase 2) |
| `sources` | only these source libraries | Phase 2's natural-product holdout |
| `seen_keys` | exclude molecules a model trained on | the validation leak (docs/09) |
| `unseen_only` | fallback that re-derives the split | same, for old checkpoints |
| `coconut_only` | only natural products (in COCONUT) | right instrument, wrong chemistry (docs/12) |

- `set - set` (difference) and `set & set` (intersection) do the filtering.
- The "≥2 spectra, leave one behind" guard lives **inside** the sampler
  (lines 113–127), because the degenerate-holdout bug came back once through a
  second sampler that didn't have it. **A guard belongs in the shared function.**

### `fetch_raw(row_ids)`

Like `validate.py`'s, plus adduct, instrument and **collision energy** (the
model needs them). A stepped acquisition's energies are averaged, matching how
training collapsed them to one number.

### `main()` — the Phase 2 experiment

For each molecule (class 2: all spectra hidden), rank candidates by analogy and
compare with **random order of the same candidates** — the baseline that shows
what ranking adds beyond retrieval. It also measures how chemically close the
best cousin is to the truth (lines 237–252) and reports MRR **by that
closeness**. That breakdown mattered: the aggregate was optimistic because
published compounds have published relatives. (The long "CALIBRATION" comment
records the reasoning at the time; the n-too-small caveat for the "hard bucket"
came later, docs/07.)

## `validate_combined.py` — identity vs analog, both classes (Phase 2)

Question: does switching on the analog channel **cost** anything on class 1?

- `ranker.score_components(...)` (Part 6) computes identity and analog once
  per molecule and simulation.
- **`strategies()`** returns a dict of *functions*, each turning
  `(ident, analog)` into scores: linear blends at several weights, and **gated**
  blends ("use identity only when it's confident").
  ```python
  out[label] = (lambda i, a, w=w: i + w * a)
  ```
  The **`w=w` default argument** is essential: a lambda looks up variables when
  it's *called*, not when it's created, so without it every lambda would use the
  last value of `w` from the loop. The default freezes the current value.
- **Projected LB** = `0.154 × class1 + 0.43 × class2` — an estimate, flagged in
  the output as "trust the ordering more than the value".
- Result at the time: analog-only won (hybrid search subsumes identity), which
  is why `rank.py` stopped using the identity channel.

## `predict.py` — the Phase 1 submission, offline

Ranks every test molecule with `Library.rank_molecule` and writes
`submissions/baseline.csv`.

- **`dedupe(smiles)`**: two library structures can collapse to the **same**
  metric key (tautomers), and submitting both wastes a slot. So candidates are
  de-duplicated by `metric.inchikey14` — the metric's key, not the dataset's.
  `OVERFETCH = 4` fetches 100 candidates so 25 remain after de-duplication.
- **Submission contract checks** before anything reaches Kaggle (lines 86–91):
  same molecule set as `sample_submission.csv`, same row count, no empty rows.
