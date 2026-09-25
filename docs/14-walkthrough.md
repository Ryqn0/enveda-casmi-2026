# The full walkthrough — CASMI 2026, from zero to 0.341

This is the teaching recap of the whole project: what we built, phase by phase;
what every piece of code does; the commands we ran and what each flag means;
what went wrong and why; and the lessons worth carrying into your next project.

The other docs (00–13) are the detailed lab notebook, written as things
happened. This one is the story told once, start to finish, with hindsight.

**Final state:** best public leaderboard score **0.341** (rank ~390 of ~1,227);
later **0.349** with one fix of ours and **0.355** with our own pipeline blended in
(see the postscripts to section 8, docs/15 and docs/16);
private GitHub repo, an in-browser demo on a free Hugging Face Space.

---

## Contents

1. [The problem in one page](#1-the-problem-in-one-page)
2. [Phase 0 — setup and the metric](#2-phase-0--setup-and-the-metric)
3. [Phase 1 — spectral library search (0.130)](#3-phase-1--spectral-library-search-0130)
4. [Phase 2 — candidate pool + analog propagation (0.240)](#4-phase-2--candidate-pool--analog-propagation-0240)
5. [Phase 3 — the fingerprint model (0.196 → 0.244)](#5-phase-3--the-fingerprint-model-0196--0244)
6. [Phase 4 — features and calibration (0.228)](#6-phase-4--features-and-calibration-0228)
7. [Phase 5 — honest validation, and the wrong chemistry (0.204, 0.234)](#7-phase-5--honest-validation-and-the-wrong-chemistry-0204-0234)
8. [Phase 6 — the public engine (0.341)](#8-phase-6--the-public-engine-0341)
9. [Wrap-up — GitHub and the in-browser demo](#9-wrap-up--github-and-the-in-browser-demo)
10. [Code map — every file, one line each](#10-code-map--every-file-one-line-each)
11. [Command cheat sheet](#11-command-cheat-sheet)
12. [Lessons for your next project](#12-lessons-for-your-next-project)
13. [What's next](#13-whats-next)

---

## 1. The problem in one page

**The task.** A mass spectrometer smashes a molecule into fragments and weighs
them. The result — a *tandem mass spectrum* (MS/MS) — is a list of peaks:
fragment mass (m/z) and how much of it was seen (intensity). The competition
gives you those spectra and asks: **what molecule was it?** You answer with 25
guesses, best first, each written as SMILES (a text encoding of a structure,
like `Cn1c(=O)c2c(ncn2C)n(C)c1=O` for caffeine).

**The score: MRR@25.** For each molecule, if the right answer is your k-th
guess you get `1/k` (1st = 1.0, 2nd = 0.5, 10th = 0.1); not in your 25 = 0.
Average over all molecules. "Right" means same **InChIKey14** — the first 14
characters of the InChIKey, which encode which atoms connect to which, ignoring
3D details (stereochemistry). Two different SMILES of the same molecule
therefore count as the same answer.

**The three classes** (the host's framing, and the key to everything):

| class | meaning | what can reach it |
|---|---|---|
| 1 | a reference spectrum of this molecule is public | match it against a spectral library |
| 2 | the structure is in a database (COCONUT/PubChem), but never measured | pick it from a database by other evidence |
| 3 | the structure is in no database at all | only generating new structures could |

We later derived that class 1 is ~15% of the test set and, from the public
engines' calibration, class 3 is ~62% — which nobody scores on. So the ceiling
for a retrieval pipeline is around 0.38, and the leader at ~0.415 is near it.

**The rules of the game.**
- **Code competition:** you don't upload predictions; you submit a *notebook*.
  Kaggle re-runs it on hidden data, **offline** (no internet), within 9 hours.
- The `test.parquet` you can download is a **placeholder made of training
  rows**; the real test is swapped in only at scoring time.
- RDKit is pinned to **2026.3.3**, because the canonicalisation that decides
  whether two SMILES "match" can change between versions.

---

## 2. Phase 0 — setup and the metric

**Goal:** a working environment and a local scorer that agrees with Kaggle's.

**Environment.** `uv` manages Python and packages: `pyproject.toml` lists the
dependencies, `uv.lock` pins exact versions, `.venv/` holds them. Secrets
(Kaggle and Hugging Face tokens) live in `.env`, which is git-ignored and was
verified never committed.

**The metric — `src/casmi/metric.py` + `official_metric.py`.** The host published
their scoring notebook. We **vendored it verbatim** (copied, untouched) as
`official_metric.py`, and `metric.py` only adds what we need on top: a cached
`inchikey14()` and a per-molecule `reciprocal_rank()`. Its self-check asserts
our mean equals the host's score to 1e-12.

> **Why vendor instead of rewriting:** a hand-written scorer that differs by one
> canonicalisation rule would silently mis-grade every experiment. Copying the
> host's code makes "our score = their score" true by construction.

**Kaggle authentication.** The CLI 2.x uses the new token (`KAGGLE_API_TOKEN` or
`~/.kaggle/access_token`), not the old `kaggle.json` key. The CLI lives in the
project environment, so it's always `uv run kaggle …` (plain `kaggle` gave
"command not found").

**EDA — `scripts/eda.py`.** Profiled both splits: 2.15M training spectra,
~269k distinct structures, instruments, adducts, mass ranges. Test is 100%
timsTOF, ~400 molecules / ~1,213 spectra.

---

## 3. Phase 1 — spectral library search (0.130)

**Idea:** class 1 molecules have public spectra. Index all 2.1M training
spectra; for each test spectrum, find the most similar library spectra and
return their structures.

**`src/casmi/library.py`** (commented line by line, because it's embedded in
the notebook):
- Uses **Flash Entropy Search** (`ms_entropy`): *spectral entropy similarity*,
  a fast, robust way to compare two peak lists.
- `build_index` **sorts spectra by precursor mass** internally (and may drop
  some). The metadata must be re-ordered with the same permutation, or every
  score maps to the wrong molecule. That's the `perm = …; meta.iloc[perm]` line.
- `np.maximum.at(out, struct_idx, per_spectrum)` collapses many spectra per
  structure into one score per structure — `.at` matters: plain fancy-index
  assignment silently drops repeated indices.
- Two searches: `identity_search` (precursor must match) and `hybrid_search`
  (tolerates a constant mass shift — used in Phase 2).

**Result: LB 0.130.** Locally, class-1 MRR 0.845 and class-3 0.000. And
`0.130 / 0.845 ≈ 15.4%` — **the score itself told us the hidden share of
class 1.**

**The false alarm (docs/02).** We found all 1,213 test spectra bitwise
identical to training rows and reported a *data leak*. It wasn't: the data page
says the downloadable test set **is** a placeholder built from training rows.
Retracted. Two lessons in one:
- **Read the documentation before concluding from files.** Your question
  "have you checked the scoring?" led to reading the docs, which led to the truth.
- **`np.allclose` is not "exact".** Its default `rtol=1e-5` tolerated ~4 mDa at
  m/z 400. `scripts/check_leak.py` now uses `np.array_equal`.

---

## 4. Phase 2 — candidate pool + analog propagation (0.240)

**Idea:** class 2 molecules have no spectrum, so a library can't return them.
Instead: (1) list every known structure with the right mass (*retrieval*), then
(2) rank that list by other evidence (*ranking*).

**`src/casmi/pool.py` — the candidate pool.**
- `formula_mass()` computes monoisotopic masses from formulas, and **refuses**
  unknown elements, charged or hydrated formulas rather than guessing.
- `ADDUCT_MASS`: an *adduct* is what the molecule picked up to become an ion
  (`[M+H]+` = a proton, `[M+Na]+` = sodium). Neutral mass = measured m/z −
  adduct mass (with the electron-mass correction).
- `CandidatePool.query(m/z, adduct, ppm, offset_ppm)` returns every structure
  within ±ppm of the neutral mass. **ppm** (parts per million) because
  instrument error scales with mass.
- `offset_ppm = 0.98`: timsTOF reads **+0.98 ppm high** on average (measured
  from `precursor_error_ppm`). Correcting it centres the window.
- The pool = **COCONUT** (open natural-product database) ∪ training structures:
  722,737 structures, verified against COCONUT's own masses (median diff 2.4e-6 Da).

**`src/casmi/analog.py` — analog propagation.** `hybrid_search` finds library
spectra that differ from the query by a constant mass shift — the signature of
a *chemical modification* (add a methyl group and every fragment containing it
shifts by 14 Da). Those are structural cousins. Each candidate is then scored by
**Tanimoto similarity** (overlap of Morgan fingerprints — bit vectors of
substructures) to the cousins, weighted by `similarity**4` (the power 4 was
tuned, and independently confirmed by a public notebook's sweep).
Watch the line with `scores[keep] = np.maximum(scores[keep], …)`: writing
`np.maximum(…, out=scores[keep])` looks equivalent but writes into a *copy* —
the self-check caught every score staying 0.

**Decisions measured, not assumed:**
- Identity search was **dropped** from ranking: hybrid search subsumes it (a
  zero-shift hybrid match *is* an identity match). *Right at the time — but see
  Phase 6: keeping an explicit library channel mattered later.*
- `_pad()` fills short lists up to 25 by mass proximity — empty slots are free
  lottery tickets.

**The Kaggle machinery — `scripts/package_kaggle.py`, `kaggle/submit_notebook.py`.**
- The notebook source is a `.py` file with `# %%` cells; `# %%EMBED src/casmi/x.py`
  lines are replaced by the module's full source, so the notebook is
  **self-contained** (no internet to install our package) yet there's **one
  source of truth**.
- Dependencies (RDKit, ms_entropy) ship as **wheels in a Kaggle dataset**,
  installed with `pip --no-index --find-links`.
- `find_one()` discovers input paths instead of hard-coding them (Kaggle's mount
  paths are deeper than documented), and says what *is* mounted when it fails.
- `--push` regenerates **and** pushes in one step, after a run shipped a notebook
  generated 45 seconds before its own fix.

**Bug worth remembering — the degenerate holdout.** 170 of 250 validation
structures had no library spectra left after we removed the query spectra, so
"novel" molecules were being scored as "known" (0.2653 reported vs 0.8446 true).
Fixed by reserving ≥1 spectrum, with an `assert`, **inside** the sampler — and
it recurred when a second sampler was written without it. A guard belongs in
the shared function, not in each caller.

**Result: LB 0.240.**

---

## 5. Phase 3 — the fingerprint model (0.196 → 0.244)

**Idea:** a neural network that reads a spectrum and predicts which
substructures the molecule contains (fingerprint bits). Then rank candidates by
how well their true fingerprint fits the prediction. It needs no spectral
relative — the hope was to help where analogy fails.

**`src/casmi/fpnet.py` — FPNet.**
- **Set transformer** over peaks: attention between all peaks, no positional
  order (a spectrum is a *set*). Self-check asserts permutation invariance and
  that padding never leaks into results.
- Each peak is embedded with **sinusoids** of its m/z *and* its **neutral loss**
  (precursor − fragment: the same lost group gives the same loss in any molecule).
- A global token carries precursor mass, collision energy, adduct, instrument;
  mean + max pooling; output = 1,583 fingerprint logits.
- Training loss: binary cross-entropy with **`pos_weight`** (only ~3% of bits
  are set, so without it "predict all zeros" wins).
- `score_candidates`: log-likelihood of the candidate's bits. We later proved
  that for ranking this reduces **exactly** to `cand_fp @ logits` — a dot product.

**Training — `scripts/prepare_fpnet_data.py`, `scripts/train_fpnet.py`.**
Split **by structure** (spectra of one molecule never straddle train/val — an
`assert` guards it). OneCycle learning rate with best-checkpoint saving.
*Sortish* batching (group spectra of similar length) cut the attention cost to
0.14× because ~20% of spectra sit at the peak cap.

**Submission 3 scored 0.196 — below Phase 2.** Four causes, all silent:

| bug | training | inference |
|---|---|---|
| collision energy | real values (median 32 eV) | hard-coded 0 |
| peak selection | top-N by **intensity** | first N by position — i.e. the **lowest-mass** peaks |
| peak cap | 256 | hard-coded 128 in the validator |
| blend weight | — | 0.25, chosen on the **wrong instrument** |

This is **train/serve skew**: the same quantity computed differently in
training and at inference. Nothing crashed; everything produced plausible
numbers. Fixing the first two alone moved offline MRR 0.60 → 0.77 with no
retraining. **The structural fix:** one `select_peaks()` used everywhere, and
`max_peaks` stored *inside the checkpoint* (a model's preprocessing is part of
the model).

**Kernel v7 died 8 minutes in** on `from casmi.analog import morgan_fp`
indented inside a function — the embed stripper only removed top-level imports,
and the import only fired when called. `ast.parse` can't catch that. Fix: strip
indented imports, assert none survive, **and execute the notebook's cells
locally** (`scripts/dryrun_notebook.py`, finished in the wrap-up).

**Result: LB 0.244** (v2, weight 0.5) — only +0.004 over analog alone.

---

## 6. Phase 4 — features and calibration (0.228)

**Engineered features (v3).** Log-intensity and `m/z ÷ precursor` added per
peak. Val loss 0.2217 → 0.2187.

**Modified Platt score (from CSI:FingerID / MSNovelist) — failed.** `pos_weight`
makes the model over-predict bits 4.8×, so we calibrated each bit
(`sigmoid(A·z + B)`). Probabilities got 3× better (log-loss 0.2187 → 0.0732);
**ranking got worse**. The algebra explains it: ranking is `cand_fp @ (A·z + B)`,
and `B ≈ −3.2` becomes a penalty of 3.2 per bit the candidate sets — a prior
about molecule size, not evidence from the spectrum. **Better probabilities are
not better rankings.**

**Weight 0.25 again → LB 0.228.** The holdout said 0.25 beat 0.5. The
leaderboard disagreed — for the second time. That pattern is what Phase 5 chased.

---

## 7. Phase 5 — honest validation, and the wrong chemistry (0.204, 0.234)

This phase is where most of the learning happened.

### 7.1 The validation leak (docs/09)

**80.3% of the holdout molecules were in fpnet's training set.** The analog
channel excludes the query molecule's spectra at search time, so it was graded
honestly; fpnet *memorised* those molecules, so it was graded on memory. The
comparison was rigged in fpnet's favour, so the holdout kept recommending more
fpnet weight.

Fix: every checkpoint stores **`train_keys`** (the structures it trained on);
`--unseen-only` samples only molecules the model never saw;
`--exclude-seen-by` makes two models sit **the same exam**.

> **Excluding the query from the *index* is not the same as excluding it from
> the *model*.**

### 7.2 New strategies, measured (docs/10 — on what turned out to be the wrong chemistry)

- **In-silico fragmentation** (`src/casmi/fragment.py`, MetFrag-style): break a
  candidate's bonds, compute fragment masses, score the fraction of observed
  intensity they explain. Needs no library at all.
- **Structure-capped training data (v4)**: at most 4 spectra per structure →
  all 268,969 structures for the same compute.
- **Contrastive fine-tune (v4c)** (`scripts/train_contrastive.py`, Spec2Mol
  idea): train on the real test-time task — pick the true structure among ~60
  same-mass isomers.
- **Learned ranker** (`src/casmi/ranker.py`): a small neural net combining the
  channels, trained on dumped per-candidate features.

**Bugs of my own this phase** (each a small lesson):
- `d["keys"][r]` inside a loop: `d` is a lazy `NpzFile`, so each access
  re-read the whole array — **quadratic**; the GPU sat at 1%. Read once.
- Labels copied per spectrum: 5.7 GB. Storing an **index** instead: 11 GB → 4 GB.
- `pgrep -f` matching **its own** command line — three times. Wait on a PID.
- WSL2's wall clock **jumps**; time durations with `time.monotonic()`.
- A float32 **z-score of an all-tied column** comes out ±1 instead of 0. The
  fix was structural: the ranker trains only on features rebuilt by the
  *serving* function (`ranker_features`), so train and serve cannot disagree.

**Submission: v4c + fragmentation + learned ranker → 0.204.** The best offline
number of the project (0.92) and the *worst* recent leaderboard score.

### 7.3 The PubChem detour (docs/11 — withdrawn)

I found the "100% recall" check was **circular** (it tested training molecules,
which are in the pool by construction), and that only 0.2% of timsTOF molecules
are in COCONUT — and concluded the pool needed PubChem. We downloaded 10 GB.
Then:

### 7.4 Right instrument, wrong chemistry (docs/12)

| source library | structures | in COCONUT | instrument |
|---|---:|---:|---|
| **enveda-180** | 182,941 | **0.0%** | timsTOF |
| **enveda-np-examples** | 250 | **99.6%** | timsTOF |

Both timsTOF, completely different chemistry: enveda-180 is a synthetic
screening library; the test-like set is **natural products**. My "timsTOF
holdout" was 99.9% enveda-180. I had matched the instrument and missed the
chemistry — and every decision since Phase 3 had been tuned on the wrong
molecules. PubChem was dropped: for natural products, COCONUT is right.

**The test that decides whether a holdout can be trusted:** does it rank the
submissions you *already scored* the way the leaderboard did?

| config | natural-product holdout | leaderboard |
|---|---:|---:|
| v2 + 0.5·analog | 0.535 | 0.244 |
| analog only | 0.524 | 0.240 |
| v4c + learned ranker | 0.510 | 0.204 |
| v3 + 0.25·analog | 0.495 | 0.228 |

5 of 6 pairs agree (the timsTOF holdout had them backwards). It also revealed
**why** the ranker hurt: trained on enveda-180, it collapsed class-1 accuracy on
natural products (0.63 → 0.39).

**Tie handling.** Our scorers counted rank as "1 + candidates scored *strictly*
higher", which puts the truth first among ties — flattering tie-heavy methods.
Fixed with `expected_rr` (average over random tie order).

**Submission: v4c + 0.5·analog → 0.234**, a single-variable change vs 0.244.
Within noise (see below), so v2's 0.244 stood.

### 7.5 The leaderboard is noisy too

With ~400 test molecules, two similar configs differ by ±0.01–0.015 from chance
alone. **Differences below ~0.02 are not evidence.** 0.204 vs 0.244 (~3 SE) was
real; 0.234 vs 0.244 (~1 SE) was not.

---

## 8. Phase 6 — the public engine (0.341)

**What we found.** The top public notebooks (by prvsiyan, haideptry, megayak,
Jiachen Li) had independently reached our conclusions: the same pool, the same
fingerprint-model leak, and calibration on `enveda-np-examples`. **Their
individual channels score the same as ours** (analog 0.52, fingerprint 0.49,
fragmentation 0.24). The gap was the **combiner**:

1. **Gradient-boosted rankers** over 31–51 raw features, not a fixed z-score
   blend (z-scores throw away how strong the evidence is in absolute terms).
2. **A dedicated library channel** (best, mean, adduct-shifted, count): class 1
   at 0.87–0.93 vs our 0.78.
3. Rankers trained on **2,250 simulated natural-product molecules** from 7
   libraries, each as class 1 and class 2.
4. Validated on natural products from day one.

**Reproduced: LB 0.341** (`kaggle/public_engine/`, an unmodified, credited copy
of W088; code verified byte-identical before running; reviewed for network
access first).

**Experiment: leak-free fingerprints for ranker B** (`scripts/make_engine_fpb.py`,
`scripts/cv_public_ranker.py`). Offline, ranker B alone improved clearly
(class 2 0.655 → 0.734; the harness reproduced the author's own 0.655 first).
Leaderboard: **0.332 and 0.330** — slightly worse. Most likely reason:
**ensemble diversity**. B's value came from being *different* from ranker A;
with fingerprints it became stronger alone but more similar to A.

> **Evaluate what you ship, not a component of it.**

**Postscript (Phase 7, docs/15): 0.349.** Taking that lesson literally, an
offline harness now runs the whole engine on the 250 natural-product
molecules. Reproducing the authors' published numbers exposed a bug: rank
features broke ties by list position, and the true answer's position differed
between training rows and serving. Averaging ties on both sides: harness
+0.0044 ± 0.0045, leaderboard **0.341 → 0.349**.

**Postscript (Phase 8, docs/16): 0.355.** Our own pipeline was rebuilt with
the engine's lessons: a learned ranker over raw evidence, and a fingerprint
model (v5) that never saw the evaluation molecules. Alone it still scores 0.244:
the harness overrates it. But it is right on different molecules than the
engine (42 of 492 molecule-simulations where the engine is wrong), so a 10%
share of its ranking in the engine's final order took the leaderboard
**0.349 → 0.355**.

One more false alarm handled correctly: the run's feature diagnostics looked
shifted — because a notebook run's `test.parquet` is the **placeholder** (every
molecule has a perfect library match). Diagnostics on the placeholder describe
the placeholder.

---

## 9. Wrap-up — GitHub and the in-browser demo

### 9.1 GitHub (private)

- Checked **the whole git history** for token-like strings (a secret committed
  once stays in history forever, even if deleted).
- **Private**, because Kaggle's rules say code shared publicly *during* a
  competition must be shared on Kaggle. Make it public after the deadline.
- `main` fast-forwarded to the documented 0.341 state.

### 9.2 The demo: why it runs in the browser

Hugging Face started charging for Gradio **and** Docker Spaces; only *static*
Spaces are free. A static server only hands out files — so the computing moved
into the visitor's browser:

| piece | Python | browser |
|---|---|---|
| model | PyTorch | exported to **ONNX** (`scripts/export_onnx.py`), run by onnxruntime-web |
| chemistry | RDKit | **RDKit.js** (RDKit compiled to WebAssembly) |
| pool | 35 MB parquet | 2,593 shards by mass (`scripts/build_web_pool.py`); a query fetches one |
| logic | `fpnet.py`, `pool.py` | `web/core.js`, one module shared by page and tests |

"Static" describes the **server**, not the page: when you change the adduct, your
browser computes a different neutral mass and requests a different shard file.
The dynamic part is all client-side.

**Golden tests** (`scripts/make_golden.py` → `web/test/golden.mjs`): the Python
pipeline writes every intermediate for 31 real spectra; the JavaScript must
reproduce them. Result: fingerprints 718/718 identical, logits within 1.6e-5,
scores within 1.9e-12, identical rankings. And the test **found a bug in the
Python**: `np.argpartition` broke intensity ties at the 256-peak cutoff
arbitrarily (and possibly differently across numpy versions). Now a
deterministic rule shared by both languages.

**Two deployment bugs** (normal for browser work):
1. `import("vendor/x.mjs")` without `./` is a *package name*, not a path →
   build absolute URLs from `location.href`.
2. On a **private** Space, the dynamic `import()` arrived without the Space's
   authentication (401) while plain `fetch()` worked → fetch the files and hand
   them over from memory (a `blob:` URL and raw bytes).

Also fixed on the way: the Gradio app was still truncating spectra to 128
peaks — the Phase 3 skew, never fixed in the app.

---

## 10. Code map — every file, one line each

**`src/casmi/` — our pipeline**

| file | what it does |
|---|---|
| `official_metric.py` | the host's scorer, vendored verbatim |
| `metric.py` | cached InChIKey14 + per-molecule reciprocal rank on top of it |
| `library.py` | Flash Entropy index over 2.1M spectra; identity and hybrid search |
| `pool.py` | candidate pool: formula masses, adduct offsets, mass-window query |
| `analog.py` | Morgan fingerprints + ranking candidates by similarity to spectral cousins |
| `fpnet.py` | the spectrum→fingerprint transformer, peak selection, candidate scoring |
| `fragment.py` | in-silico fragmentation channel (off: no gain on natural products) |
| `ranker.py` | learned ranker, shared feature function, fair tie scoring (`expected_rr`) |
| `rank.py` | combines the channels for one molecule; used by the notebook |

**`scripts/` — entry points**

| file | what it does |
|---|---|
| `eda.py` | profile the data |
| `build_library.py` / `build_pool.py` | build the spectral index / candidate pool |
| `check_leak.py` | the (retracted) leak check, now with exact comparison |
| `pool_recall.py` | retrieval ceiling (note: circular for training molecules) |
| `validate.py`, `validate_class2.py`, `validate_combined.py` | early holdouts; `validate_class2.py` now holds the shared sampler |
| `prepare_fpnet_data.py` / `train_fpnet.py` | fpnet training data and training |
| `calibrate_fpnet.py` | Platt calibration experiment |
| `validate_fpnet.py` | the main holdout: `--unseen-only`, `--coconut-only`, `--frag`, `--dump` |
| `compare_dumps.py` | paired comparison of full configs on identical molecules |
| `prepare_contrastive.py` / `train_contrastive.py` | contrastive fine-tune (v4c) |
| `train_ranker.py` | learned ranker, grouped cross-validation |
| `run_np_calibration.sh` | natural-product holdout for three models |
| `pubchem_coverage.py` | the PubChem check (API was down; kept for the record) |
| `package_kaggle.py` / `dryrun_notebook.py` | build + push our notebook / run it locally |
| `cv_public_ranker.py` / `make_engine_fpb.py` | public-engine experiment offline CV / variant builder |
| `predict.py` | command-line prediction |
| `deploy_space.py` | Gradio Space deploy (needs HF PRO now) |
| `export_onnx.py`, `build_web_pool.py`, `make_golden.py`, `build_web.py` | the in-browser demo pipeline |

**Elsewhere:** `kaggle/submit_notebook.py` (our notebook source),
`kaggle/public_engine/` (W088, credited), `kaggle/engine_fpb_*` (our variants),
`app/app.py` (Gradio demo), `web/` (in-browser demo + golden test).

---

## 11. Command cheat sheet

Every command we used repeatedly, with every flag.

**uv (Python environment)**

| command | meaning |
|---|---|
| `uv sync --extra dev` | install locked dependencies; `--extra dev` adds the optional "dev" group |
| `uv run python x.py` | run inside the project environment |
| `uv run --extra app …` | include the optional "app" dependencies (Gradio) |
| `uv run --with numba …` | add a package **for this command only**, without touching `pyproject.toml` |
| `uv add scikit-learn` | add a dependency and record it in `pyproject.toml` + `uv.lock` |

**Kaggle CLI**

| command | meaning |
|---|---|
| `kaggle datasets version -p DIR -m MSG` | upload a new version of a dataset from `DIR`; `-m` = version note |
| `kaggle datasets status OWNER/NAME` | is the new version `ready`? (wait before pushing a notebook) |
| `kaggle datasets files OWNER/NAME` | list a dataset's files and sizes |
| `kaggle datasets download OWNER/NAME -f FILE -p DIR` | `-f` one file, `-p` destination |
| `kaggle kernels push -p DIR` | upload the notebook in `DIR` and start a run |
| `kaggle kernels status USER/SLUG` | `RUNNING` / `COMPLETE` / `ERROR` |
| `kaggle kernels output USER/SLUG -p DIR` | download the run's log and output files |
| `kaggle kernels pull REF -p DIR -m` | download a notebook's source; `-m` also its metadata |
| `kaggle kernels list --competition C --sort-by scoreDescending --page-size N` | top public notebooks |
| `kaggle competitions submit -c C -k USER/SLUG -v N -f submission.csv -m MSG` | submit notebook version `N`; `-f` is **required** for code competitions |
| `kaggle competitions submissions -c C` | your submissions and scores |

**git / GitHub**

| command | meaning |
|---|---|
| `git switch -c NAME` | create a branch and move onto it |
| `git merge --ff-only BRANCH` | move `main` forward to `BRANCH`, refusing if histories diverged |
| `git log -p --all \| grep …` | search **every version of every file** (the secret scan) |
| `git check-ignore -v FILE` | which `.gitignore` rule ignores this file |
| `gh repo create NAME --private --source . --remote origin --push` | create the GitHub repo from this folder, register it as `origin`, upload |

**Shell tools that bit us**

| command | meaning / lesson |
|---|---|
| `nohup CMD > log 2>&1 &` | run in background, survive the shell closing, send output and errors to `log` |
| `tail --pid=PID -f /dev/null` | wait until process `PID` exits — **wait on PIDs, not `pgrep -f` patterns** |
| `pkill -f "[p]attern"` | the `[p]` trick stops the pattern matching its own command line |
| `python -u` | unbuffered output, so logs appear while a long job runs |

**Node / web**

| command | meaning |
|---|---|
| `npm install -D PKG` | install as a dev-only dependency (tests), not shipped to visitors |
| `node test/golden.mjs` | run the JS-vs-Python golden test |
| `uv run python -m http.server -d web 8000` | serve `web/` locally at http://localhost:8000 |

---

## 12. Lessons for your next project

These are ordered by how much they cost us.

1. **Choose the validation set first, and prove it tracks the leaderboard.**
   A holdout is trustworthy only if it ranks your already-scored submissions
   the way the leaderboard did. We spent three phases optimising against a
   holdout that ranked them backwards. Check this before optimising anything.

2. **Two ways a holdout lies — check both.** *Leakage* (it overlaps the model's
   training data) and *distribution* (it's not the population being scored).
   We fixed the first and it hid the second.

3. **Read the top public work early.** The public notebooks documented the
   chemistry mismatch and the fingerprint leak we spent days rediscovering.
   Reusing credited public work is normal on Kaggle — and it's how we went from
   0.244 to 0.341.

4. **Train/serve skew is the default failure, not an edge case.** Every
   serious bug here was the same quantity computed two ways. Defences that
   worked: one shared function per concept; preprocessing config stored in the
   model file; golden tests comparing the two sides; features rebuilt by the
   serving code.

5. **Know your noise floor.** On ~400 test molecules, ±0.01–0.015 is chance.
   Change one thing per submission, and don't celebrate or panic below ~0.02.

6. **Evaluate what you ship.** A component that improves alone can make an
   ensemble worse (diversity matters). Measure the whole pipeline.

7. **Keep the strongest simple signal.** Don't let a blend dilute an exact
   match that is right ~90% of the time.

8. **Guards belong in shared code.** An `assert` in one caller protects one
   caller; the degenerate-holdout bug came back through a second sampler.

9. **Be suspicious of your best offline number.** Our best (0.92) coincided
   with our worst recent leaderboard score (0.204).

10. **Negative results are results.** Platt calibration, cross-ring
    fragmentation, PubChem, the leak-free fingerprints — each is written up,
    with why it failed. That record is what stopped us repeating them.

11. **Environment hygiene saves hours:** wait on PIDs; unbuffered logs; don't
    trust WSL wall-clock durations; read what `uv add/remove` will uninstall;
    check a paid-tier change before designing around a free tier.

12. **Decide where the computation runs before building a demo.** It sets the
    cost, the privacy, and how much code you'll write.

---

## 13. What's next

The goal you set: **0.35**. Realistic path:

1. Build an offline harness that runs the **whole public engine** (both rankers
   and the blend) on `enveda-np-examples`, reproducing its published offline
   numbers first.
2. Test one change at a time against it, submitting only changes that move it
   by more than the noise.
3. Candidate ideas from the engine authors' own to-do list: candidate pool
   expansion for class 2 (glycosylation/hydroxylation derivatives), combining a
   molecule's different adduct spectra before ranking, polarity-restricted analogs.

After the competition ends: make the GitHub repo and the Space public.
