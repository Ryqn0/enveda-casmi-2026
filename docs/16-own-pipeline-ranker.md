# Phase 8 — our own pipeline, rebuilt with the public engine's lessons

Goal: take **our** 0.244 pipeline (candidate pool, analog search, fingerprint
model, combined by a fixed formula) and apply what docs/13 and docs/15 taught:
a learned combiner over raw evidence, a library channel, honest features, and
an evaluation that reproduces the test.

## 1. Honest fingerprints: hold structures out *before* training

v4c had trained on **240 of the 250** `enveda-np-examples` structures (through
their spectra in other libraries), so its class-2 scores there were inflated.

`scripts/own_holdout.py` picks the structures no model may see:
- **eval:** the 250 `enveda-np-examples` structures (every number is measured
  here; predicted LB = `0.162·class1 + 0.220·class2`, the public calibration);
- **train:** 2,000 natural products (in COCONUT, ≥2 spectra), 400 from each
  of gnps, riken, mona, msdial, massbank, whose simulations become the
  ranker's training rows.

`prepare_fpnet_data.py --exclude-keys` drops all 2,250 from the fingerprint
model's data; **v5** is v4's recipe on the rest (12 epochs, val loss 0.2178).
Honest fingerprint-only score on eval: **0.343** (v5), 0.387 (v5c, contrastive).

## 2. Features: raw evidence, one implementation

`src/casmi/features.py` is used by the harness **and** the notebook:

| channel | per candidate |
|---|---|
| `id_max`, `id_mean` | identity search over the molecule's spectra (we computed this since Phase 2 and never used it) |
| `hyb_self` | the candidate's **own** library spectra matched with a precursor shift: the same molecule measured as another adduct (new `Library.hybrid_scores`) |
| `n_lib` | how many library spectra the candidate has |
| `analog` | Tanimoto × similarity propagation from the top cousins (as before) |
| `fp` | fingerprint log-likelihood |
| `frag` | explained-intensity fragment score |
| `ppm` | mass error |

Each channel becomes raw value, within-molecule rank (**ties averaged**,
docs/15), gap to the best, and z-score, plus molecule context (candidate
count, best cousin similarity, best identity…): 36 features in total.
Molecules measured as **several adducts** (160 of the 250) are handled
throughout: the median neutral mass sets the window, and fingerprint logits
are averaged per adduct group.

## 3. The harness (`scripts/own_harness.py`)

Same protocol as the engine harness: one molecule at a time, class 1 = hide its
query library's spectra, class 2 = hide every spectrum of its structure.
Stages: `evidence` (searches, ~1 h, once) → `fp` (per model) → `score`
(train + evaluate, minutes) → `export` (rows for the notebook). Candidate
windows contain the truth for 244/250 eval molecules.

## 4. What went wrong first, and why

**Class 1 collapsed** (0.72 vs 0.90 for plain analog). Two causes:

1. Training class-1 simulations hid only the 4 query spectra, leaving
   same-lab near-duplicates behind. Fixed: hide the whole query library, as
   for eval. It helped less than expected.
2. **The real one:** in every class-2 simulation the true answer is hidden,
   so *every candidate with library spectra is wrong*. The ranker learned
   "library spectra → suspicious" and, in 78 of 244 class-1 molecules,
   preferred a candidate with **no spectra** over a truth with identity 0.9.

What fixed most of it: training on the **timsTOF** rows too. The eval
molecules were folded in 5-fold by molecule (each still scored by a model that
never saw it): class 1 0.73 → **0.86**. Monotonic constraints alone did nothing.

## 5. Results (250 eval molecules, honest v5)

| config | class 1 | class 2 | predicted LB |
|---|---:|---:|---:|
| **learned ranker (+ timsTOF rows, CV)** | 0.858 | **0.652** | **0.282** |
| analog only (≈ submission 2, LB 0.240) | **0.902** | 0.588 | 0.276 |
| fixed blend z(fp) + 0.5·z(analog) (≈ 0.244 config) | 0.819 | 0.555 | 0.255 |

Experiments after that, all within noise (paired, predicted LB):

| change | effect |
|---|---|
| class-1 row weight 0.2 … 0.85 | flat, 0.281–0.283: it slides score between the classes along the LB formula |
| v5c fingerprints | +0.0019 ± 0.0046 |
| ¼ / ½ of the 2,000 training molecules | +0.004 / +0.002 ± 0.004 |
| timsTOF molecules **only** | −0.0015 ± 0.0051 |

The last two say that **similarity to the test matters more than quantity**:
~200 timsTOF molecules teach the ranker as much as 2,000 from other
instruments. The harness now can't resolve changes below ~±0.005.

## 6. Submission

Notebook v13 (`kaggle/submit_notebook.py`): the ranker is **trained inside the
notebook** from shipped rows (`ranker_rows.npz`, 276,872 rows, ~1 min), so no
pickled scikit-learn model has to survive a version change. Tag checks tie the
rows, the fingerprint model and the notebook together. Empty mass windows are
now padded from the wider window too. 400 molecules ranked in 13 min (CPU).

Caveat going in: for our pipeline the harness has run *high* (analog-only
predicted 0.276, real 0.240), while for the public engine it ran low (0.306 →
0.341). The leaderboard decides.

Submission 56548009: **LB 0.244**, the same as the old fixed blend (0.244).
The harness had predicted about +0.02 on the leaderboard scale; the observed
+0.000 is 1–2 standard errors of leaderboard noise below that.

## 7. Why didn't the harness gain show up?

**Hypothesis tested: memorised fingerprints.** Class-1 test molecules have
public spectra, so every shipped fingerprint model trained on them; the
harness used v5, which never saw the eval molecules in either class. Making the
harness test-like (class-1 rows from v4c, which had seen 1,931 of those
simulations' molecules; class 2 from v5):

| config | class 1 | class 2 | predicted LB |
|---|---:|---:|---:|
| learned ranker | 0.861 | 0.662 | 0.285 |
| analog only | 0.902 | 0.588 | 0.276 |
| fixed blend | 0.864 | 0.555 | 0.262 |

The effect is real but small (fingerprints alone, class 1: 0.34 → 0.44), and
the harness still ranks the fixed blend below analog-only, the reverse of the
leaderboard (0.244 vs 0.240). So it isn't the main explanation.

**What is known:** on these 250 molecules the harness has run **high** for our
pipeline twice (analog-only 0.276 → LB 0.240; ranker ~+0.02 → +0.000) and
**low** for the public engine (0.306 → 0.341). For our pipeline, the eval set
is not a faithful stand-in for the test, for a reason not yet identified. The
ranker may still be a small real gain hidden by leaderboard noise; one
submission can't tell.

**Systematic differences checked** (the downloadable `test.parquet` is a
placeholder, so the two pipelines can't be compared on the real test):
- *Candidate window.* Ours is ±4 ppm with a +0.98 ppm correction; the engine's
  is ±10 ppm (30 ppm fallback). `enveda-np-examples` reads higher than
  enveda-180 (raw median +1.54 vs +0.98 ppm; +2.0 for [M+H]+): after our
  correction 97% of its spectra fall within 4 ppm, 99th percentile 6.6 ppm.
  Worth an estimated 0.005–0.008 of LB, but already inside the harness numbers,
  so it can't explain why the harness overrates us.
- *Tautomer de-duplication.* The engine removes duplicate structures from its
  top 25; our pool keys are already unique. Negligible.

## 8. Our ranker as a second opinion for the public engine

`scripts/blend_harness.py` combines the two harnesses' per-candidate scores
(`--dump` in both) on the same 246 molecules.

**Where they disagree** (each pipeline's top-1 on its own candidates):

| class | both right | engine only | ours only | neither |
|---|---:|---:|---:|---:|
| 1 | 180 | 29 | 15 | 22 |
| 2 | 101 | 48 | 27 | 70 |

Ours is right where the engine is wrong on 42 molecule-simulations: they make
different mistakes, the condition for a blend to help.

**The blend**, `(1 − w)·engine rank + w·our rank` on the engine's candidates:

| w | class 1 | class 2 | predicted LB | vs engine alone |
|---|---:|---:|---:|---:|
| 0 | 0.905 | 0.724 | 0.3058 | |
| **0.1** | 0.914 | 0.735 | **0.3099** | **+0.0040 ± 0.0035** |
| 0.2 | 0.917 | 0.732 | 0.3095 | +0.0036 ± 0.0049 |
| 0.3 | 0.914 | 0.734 | 0.3095 | +0.0037 ± 0.0062 |
| 0.5 | 0.913 | 0.713 | 0.3047 | −0.0012 ± 0.0084 |
| 1 | 0.858 | 0.655 | 0.2831 | −0.0228 ± 0.0131 |

`scripts/make_engine_plus_own.py` builds it (`kaggle/engine_plus_own/`): our
modules are written into the notebook as a real `casmi` package, our pipeline
runs first and frees its memory, then the engine's `make_submissions` is wrapped
to apply the blend at w = 0.1. `scripts/dryrun_engine_notebook.py` executes
engine-style notebooks locally (fake `/kaggle/input`, `%%writefile` honoured):
20 molecules OK.

Submission: **pending**.
