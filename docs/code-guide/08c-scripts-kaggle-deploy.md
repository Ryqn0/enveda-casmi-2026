# Part 8c — scripts: shipping to Kaggle, the public engine, deployment

How the code in `src/` becomes one Kaggle notebook, how that notebook is tested
before it costs a submission, the public-engine experiments, and the first
deployment attempt.

---

## `package_kaggle.py` — from repo to Kaggle (Phase 1, hardened every phase)

This competition runs notebooks **with internet off**: no `pip install` from
PyPI, no `git clone`. Everything must arrive as an attached Kaggle **dataset**
or be written into the notebook itself. The script does both:

```bash
uv run python scripts/package_kaggle.py           # build only, print next steps
uv run python scripts/package_kaggle.py --push    # build AND push the notebook
```

### Constants (lines 29–36)

`DATASET = "ryqn00/casmi26-deps"` (the files), `KERNEL =
"ryqn00/casmi-library-baseline"` (the notebook), and the paths: `STAGE`
(folder that becomes the dataset), `SOURCE` (`kaggle/submit_notebook.py`, what
we edit) and `NOTEBOOK` (the generated `.ipynb`, never edited by hand).

### `build_dataset()` (lines 39–98) — the files

1. Recreate `data/kaggle_deps/` from scratch (`shutil.rmtree`, then `mkdir`),
   so no stale file from a previous build can ship.
2. Copy the **wheels** (`.whl`, pre-built Python packages: RDKit and
   ms_entropy) **flat** into the folder. A sub-folder would be zipped on upload
   and break `pip --find-links`.
3. Copy `pool.parquet` (34 MB). Rebuilding it on Kaggle would mean uploading the
   577 MB COCONUT dump.
4. Copy the model and the ranker under **fixed names** (`fpnet.pt`,
   `ranker.pt`), so the notebook never needs to know version numbers.
5. **Three-way tag check** (lines 75–87), before anything is copied:
   ```python
   fp_tag = torch.load(fp_ck_path, ...).get("model_tag")
   rk_tag = torch.load(rk_ck_path, ...).get("fpnet_tag")
   nb_tag = re.search(r'EXPECTED_FPNET_TAG = "([^"]+)"', (...submit_notebook.py).read_text())
   assert fp_tag and rk_tag == fp_tag, ...
   assert nb_tag and nb_tag.group(1) == fp_tag, ...
   ```
   - the ranker must have been trained on **this** model's scores;
   - the notebook must expect **this** model.
   - `re.search` finds the pattern in the notebook's text. In the pattern,
     `(...)` is a *capture group* and `[^"]+` means "one or more characters
     that aren't a quote". `.group(1)` returns the captured text, e.g. `v4c`.

   Each mismatch is a different mistake, and here it fails in seconds instead of
   25 minutes into a Kaggle run.
6. Write `dataset-metadata.json`, the file the Kaggle CLI reads to know which
   dataset this folder is.

### `to_notebook()` (lines 101–217) — the notebook

`submit_notebook.py` is a normal Python file with **cell markers**:
`# %%` starts a code cell, `# %% [markdown]` a text cell (the convention VS Code
and Jupytext also use). Keeping the source as `.py` means it diffs, lints and
reads like code; a raw `.ipynb` is JSON.

1. `text.split('"""', 2)[-1]`: split at the first two `"""` and keep what
   follows, which drops the module docstring (it describes the file, not the
   notebook).
2. `body.split("\n# %%")`: one chunk per cell. The first cell has no newline
   before its marker, so it's stripped separately (lines 113–114).
3. **`# %%EMBED src/casmi/pool.py`** (lines 119–135): the cell becomes the
   **whole file**, read from `src/`. Kaggle runs exactly the code that the
   validators ran, not a copy that might have drifted.
   - Embedded modules share **one namespace** (Jupyter cells all see the same
     variables), and there's no `casmi` package on Kaggle. So every
     `from casmi.x import y` line is deleted, and **embed order** (pool →
     analog → library → fpnet → fragment → ranker → rank) makes sure `y` is
     already defined when it's needed.
   - `ln.lstrip().startswith(...)`: `lstrip()` first, because imports **inside
     functions** are indented. v7 missed exactly one (`    from casmi.analog
     import morgan_fp`), which only fired when the function was called, and
     crashed 8 minutes into the run.
4. Markdown chunks: strip the `# ` comment prefix from every line.
5. Build the `.ipynb` **JSON by hand** (lines 148–177): a list of cells, each
   with `cell_type`, `source` (a list of lines, `splitlines(True)` keeps the
   `\n`), and for code cells empty `outputs`. `nbformat: 4` is the notebook
   file-format version. No library is needed; the format is simple.
6. Write `kernel-metadata.json`: what `kaggle kernels push` reads. It sets
   `"enable_internet": False` (the competition requires it), `"is_private":
   True`, and attaches the deps dataset plus the competition data.
7. **Self-check** (lines 206–217): any code line still containing both `casmi.`
   and `import` stops the build. It's the v7 bug, caught locally.

### `push()` (lines 220–233)

```python
subprocess.run(["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle")], check=True)
```

- `subprocess.run([...])` runs another program. Passing a **list** (not one
  string) avoids shell-quoting problems.
- `kaggle kernels push -p DIR`: upload the notebook described by
  `DIR/kernel-metadata.json` as a new version, which Kaggle then runs.
- `check=True`: raise an error if the command fails.

**Why `--push` rebuilds first:** v3 was pushed from a notebook generated 45
seconds *before* the fix it was meant to contain, so it shipped stale code.
Doing both in one command removes that gap.

## `kaggle/submit_notebook.py` — the notebook itself

About 400 lines, mostly Markdown and `EMBED` markers. The cells, in order:

| step | what runs |
|---|---|
| intro (Markdown) | the method, the three classes, the leaderboard history |
| 1 | offline install of the wheels |
| 2–7 | `EMBED` pool, analog, library, fpnet, fragment, ranker, rank |
| 8 | find inputs, build the spectral index |
| 9 | load model and pool, safety asserts, rank every molecule |
| 10 | check the submission contract, write `submission.csv` |

### Step 1 — offline install (lines 80–115)

```python
wheel_dirs = sorted({p.parent for p in INPUT.rglob("*.whl")})
subprocess.run([sys.executable, "-m", "pip", "install",
                "--no-index", f"--find-links={WHEELS}", "--no-deps",
                "ms_entropy", "rdkit"], check=True)
```

- `INPUT.rglob("*.whl")`: search **all** sub-folders for wheels. Mount paths are
  *discovered*, not hard-coded: Kaggle mounts datasets deeper than most
  tutorials say (`/kaggle/input/datasets/<user>/<name>`).
- `{... for p in ...}` is a **set comprehension** (no duplicate folders).
- `sys.executable`: the Python running this notebook, so pip installs into
  **this** environment.
- `--no-index`: never contact PyPI. `--find-links=DIR`: look for packages in
  DIR. `--no-deps`: don't try to resolve dependencies (their needs, numpy and
  Pillow, are already installed; with `--no-index` pip would find nowhere to
  look).
- RDKit is pinned to **2026.3.3**, the version the organisers score with: the
  tautomer canonicalisation behind the answer key changes between releases.

### Step 8 — `find_one(name)` (lines 242–261)

```python
hits = sorted(INPUT.rglob(name))
if not hits:
    have = sorted(p.name for p in INPUT.rglob("*") if p.is_file())[:40]
    raise FileNotFoundError(f"{name!r} not found ... Present: {have}")
return hits[0]
```

Find a file by name anywhere under `/kaggle/input`. The first version used
`next(INPUT.rglob(name))`, which raises a bare `StopIteration` with no
message. v6 died that way: Kaggle had mounted an **older version** of the
dataset, without `fpnet.pt`. Listing what *is* present turns the failure into a
diagnosis. `{name!r}` formats with `repr()`, so the name appears in quotes.

Then `build(...)` (Part 2) writes the index to `/tmp`, not `/kaggle/working`:
it's over 2 GB, and everything in `working/` is collected as notebook output.

### Step 9 — safety asserts, then the loop (lines 288–383)

- `DEV = "cuda" if torch.cuda.is_available() else "cpu"`: GPU if there is one.
- **Stale-mount asserts** (lines 306–317): `max_peaks == 256` and
  `model_tag == "v4c"`. If Kaggle mounted an old dataset version, the run fails
  in seconds instead of scoring 18 minutes of the wrong model. (Val-loss
  thresholds were tried first; they broke when the validation set changed. An
  explicit tag doesn't.)
- `assert ranker.ranker is None` and `platt is None`: both measured worse, so
  the notebook proves they're off.
- `peaks_of(row)`: the test file stores m/z and intensities as two parallel
  lists; `np.stack([...], axis=1)` makes the `(N, 2)` array the library code
  expects.
- The loop itself is shown in Part 6.

### Step 10 — the submission contract (lines 392–402)

Five asserts: no duplicate `molecule_id`, no nulls, 1 to 25 guesses per molecule
(`str.count(";") + 1`), and exactly the test's molecule set. A rejected file
costs a submission slot. `to_csv(index=False)` stops pandas writing its row
numbers as an extra column.

## `dryrun_notebook.py` — run the generated notebook locally (Phase 5)

```bash
uv run python scripts/dryrun_notebook.py -n 20
```

Runs the **generated `.ipynb`**, not the source modules, so it tests exactly
what ships, embedding included. It was written after v7's crash: `ast.parse`
and syntax checks can't find an import that only fails when a function is
**called**. Running the code can.

- **`fake_input(tmp)`** (lines 31–42): builds a fake `/kaggle/input` tree of
  **symlinks** (small files that point to the real ones) in the same layout as
  Kaggle's, so `find_one` and the wheel search run unmodified.
- **Substitutions** (lines 56–72): only the things that can't work locally are
  swapped: the pip call becomes `(lambda *a, **k: None)(` (a function that
  does nothing), the index build becomes `pass` (reuse `data/index`), the output
  goes to the temp folder, and the test set is cut to N molecules.
  **Each must match exactly once** (the assert); if the notebook changes shape,
  the dry run fails instead of quietly running something else.
- **One `exec` per cell** (lines 78–80):
  ```python
  ns = {"__name__": "__main__"}
  for i, cell in enumerate(cells):
      exec(compile(cell, f"{NB.name}[cell {i}]", "exec"), ns)
  ```
  `exec` runs a string as Python code; `ns` is the shared dict of variables,
  like Jupyter's single namespace. `compile(..., filename, ...)` sets the name
  shown in tracebacks, so an error says which cell. Joining the cells into one
  string would fail: each embedded module starts with `from __future__ import
  annotations`, which is only legal at the top of a file.
- `tempfile.TemporaryDirectory()` in a `with` block: a folder deleted
  automatically at the end.

---

## The public engine (Phase 6)

`kaggle/public_engine/casmi26-public-engine.ipynb` is an **unmodified copy** of
Jiachen Li's public notebook "CASMI26 Next Direct W088" (built on work by
prvsiyan, haideptry and megayak), pushed as our kernel with credit. It scored
**0.341**. It's not our code, so this guide doesn't go through it line by
line; docs/13 explains how it works and why it beats ours. What's useful to
know here is its structure:

| cell | content |
|---|---|
| 6 | offline install of its wheels |
| 8 | `%%writefile pv.py`: prvsiyan's similarity kernels, ranking features, MetFrag-lite |
| 10 | `%%writefile pv_fp.py`: prvsiyan's fingerprint model |
| 12 | `%%writefile casmi_engine.py`: library, pool, channels, both rankers, blend |
| 14 | the run: `E.main(...)` with `w_pv=0.88` |

**`%%writefile name.py`** is a Jupyter *cell magic*: instead of running the
cell, it saves the cell's text to a file, which later cells `import`. That's
their equivalent of our `EMBED`.

## `cv_public_ranker.py` — test an idea on the engine offline (Phase 6)

Question: the engine's ranker B is trained with its 12 fingerprint columns set
to **zero**. Does filling them from megayak's **leak-free** fingerprint models
help?

- **`load()`**: two row files from megayak's public dataset. `nofp` has the
  zeros; `fppair` has the filled columns. They must describe **the same
  candidates in the same order** (asserted on labels `Y`, class `S`, query group
  `G` and query set `Q`), so the **only** difference between the two
  configurations is those 12 columns.
  - `X1 = X0.copy(); X1[:, FP_COLS] = b["X"][:, FP_COLS]`: fancy indexing with a
    list of column numbers replaces exactly those columns.
- **`fit_predict(...)`**: reproduces the engine's own ranker, 4
  `HistGradientBoostingClassifier` models (2 class-weightings × 2 seeds),
  averaged. A **gradient-boosted tree ensemble** builds hundreds of small
  decision trees, each correcting the previous ones' mistakes; "Hist" = it
  bins feature values into histograms first, which makes it fast.
  `sample_weight` gives class-1 rows weight `wA` and class-2 rows `1 − wA`.
  `predict_proba(...)[:, 1]` = the probability of "this is the true candidate".
- **`main()`**: 5-fold CV over the natural-product query groups
  (`enveda-np-examples`, the set that reproduced the leaderboard ordering). Rows
  from all other query sets are always in training. Scores use our fair
  `expected_rr` (imported from `src/casmi/ranker.py`). The predicted LB is
  `0.162·c1 + 0.220·c2` (the public calibration).
- `fold_of.get(g, -1)`: rows outside the natural-product set get fold −1, so
  they're never in a test fold.

Result: ranker B class 2 went from 0.655 to 0.734 offline.

## `make_engine_fpb.py` — build engine variants by patching (Phase 6)

```bash
uv run python scripts/make_engine_fpb.py 0.88 kaggle/engine_fpb_w088 casmi26-engine-fpb-w088
```

Three **positional** arguments via `sys.argv` (the command line as a list;
`argv[0]` is the script's name): blend weight, output folder, kernel slug.

It loads the public notebook's JSON and **edits its code as text**:

```python
def sub(s, old, new):
    assert s.count(old) == 1, f"expected exactly one match for:\n{old}"
    return s.replace(old, new)
```

Every edit must match **exactly once**, the same guard as the dry run. If the
upstream notebook changes, the script stops instead of producing a half-patched
engine. The edits:

1. Add `FP_COLS` and two globals for a train/serve comparison.
2. `compute_channels` gets a second model set, `models2`, and computes
   `zlog2`, a second set of logits from the leak-free models. Same function,
   different weights.
3. `fit_rankers` fills ranker B's training columns from the `fppair` file,
   **after** asserting the files are aligned and differ in exactly `FP_COLS`.
4. `score_molecules` serves ranker B with `zlog2` and records those features.
5. **Train/serve log**: the 12 features' mean/std as trained vs as served. The
   populations differ (simulation vs test), so they won't match exactly, but a
   wrong definition shows up as a different **scale**.
6. The run cell finds `ours_fpnet_*.pt` (asserting exactly 2) and passes the
   new arguments. The first Markdown cell is rewritten to credit the authors
   and describe the one change.

Result: **0.332** (w_pv 0.88) and **0.330** (0.50), both **below** the
unmodified 0.341. The offline gain didn't transfer. The likely reason
(docs/13) is **ensemble diversity**: B was valuable because it differed from
ranker A. With fingerprints it got stronger on its own but more similar to A,
so the blend of the two gained less. Each LB gap is only about 1 standard error,
but both point the same way.

## `deploy_space.py` — the first deployment attempt (Phase 6)

Assembles a **Gradio** Space (a Python web app hosted by Hugging Face) from the
repo and uploads it:

- A temp folder receives `app/app.py`, the three `casmi` modules it needs (plus
  an empty `__init__.py`, which makes `casmi/` an importable package), the model,
  the pool, `requirements.txt` and a `README.md`. On Hugging Face, the README's
  YAML header (between `---` lines) is the Space's configuration: SDK,
  version, entry file.
- `requirements.txt` pins **CPU-only** torch (`--extra-index-url
  .../whl/cpu`): a free Space has no GPU, and the CUDA build is ~2 GB larger.
- `read_token()` reads `HF_TOKEN` from `.env` (gitignored) and never prints it.
- `HfApi`: `create_repo(..., private=not args.public, exist_ok=True)` (create
  the Space, or reuse it), `add_space_variable` (environment variables the app
  reads for its file paths), `upload_folder` (one commit of the whole folder).
- `--dry-run DIR`: assemble only, to test the exact layout locally.
- **Private by default**: the pool contains structures from the competition's
  training file, which shouldn't be redistributed while it runs.

**Outcome:** Hugging Face answered **402 Payment Required**: Gradio and Docker
Spaces now need a paid (PRO) account. The script works and is kept, but the
deployed demo is the **static** in-browser version (Part 9).

## `pubchem_coverage.py` — check before a big download (Phase 5)

Plan under test: add PubChem (~100M+ structures) to the pool. Before a
multi-GB download, a 2-minute check: are test-like molecules even in PubChem?

- `in_pubchem(smiles)`: a POST request to PubChem's REST API ("fastidentity,
  same_connectivity", the same notion of "same molecule" as the metric).
  `urllib.parse.urlencode` builds the form body; `urlopen(URL, data=body)` sends
  it. Returns `True`/`False`, or **`None` when the request itself failed**, so
  network errors aren't counted as "not in PubChem". PubChem answers **404**
  for "no match", which is a real `False`, not an error.
- `time.sleep(0.25)`: PubChem allows 5 requests per second.

**Outcome:** the API was down at the time, so the check never ran, and the
10 GB download went ahead anyway (docs/11). The question was then answered
another way (docs/12): test-like natural products are **99.6% in COCONUT**, so
the pool was already the right database and PubChem was dropped. Lesson: when a
cheap check fails, find another cheap check before doing the expensive thing.
