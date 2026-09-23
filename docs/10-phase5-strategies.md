# Phase 5 — every strategy, measured honestly

> **Read docs/12 first.** Every result below was measured on enveda-180, a synthetic
> screening library that does not resemble the test set. The methods (leakage rules,
> skew checks, paired comparisons) stand; the conclusions about which strategies
> help do not transfer. On natural products the learned ranker and fragmentation
> did not help, and the ranker cost the leaderboard 0.244 -> 0.204.

Starting point: LB 0.244 (v2 + 0.5*analog), and a v10 submission at **0.228**
that docs/09 traced to a leak in the *measurement*, not the model.

Every number below comes from molecules the model being evaluated never
trained on (`validate_fpnet.py --unseen-only`), and every comparison between
two variants is **paired** on identical molecules, so molecule difficulty
cancels and the error bars are real (mean +/- 2 standard errors).

## Results at a glance

| strategy | source | result | shipped |
|---|---|---|---|
| per-bit Platt calibration | CSI:FingerID, MSNovelist | **loss** (docs/08) | no |
| in-silico fragmentation | MetFrag | **+0.006 to +0.008**, 3 samples | yes |
| structure-capped training data | our diagnosis | **+0.0135 +/- 0.0081** | yes |
| contrastive fine-tune | Spec2Mol | +0.0056 +/- 0.0064 (likely) | yes |
| learned ranker (MLP) | "ensemble" | **+0.049 +/- 0.009** (on v3) | yes |
| linear ranker | -- | -0.006 (loses to fixed blend) | no |
| cross-ring fragmentation | MetFrag | alone 0.3628 -> 0.3537; blend +0.0009 (noise) | no |
| SMILES decoder (class 3) | MSNovelist | deferred, see end | no |

## 1. In-silico fragmentation (`src/casmi/fragment.py`)

Breaks each candidate's single bonds (up to two at once), computes the masses
of the pieces, and scores the fraction of observed intensity they explain.
It needs no spectral library, so it is the only channel that says anything
about class-3 molecules.

Weak alone (MRR ~0.38-0.41), but independent of the other two channels:

| sample | without | with 0.25*frag | gain |
|---|---:|---:|---:|
| seed 0, n=300 | 0.8298 | 0.8356 | +0.0058 |
| seed 1, n=300 | 0.8506 | 0.8574 | +0.0068 |
| seed 7, n=2300 | 0.8426 | 0.8502 | +0.0076 |

Same sign, similar size, three independent samples. Cheap: ~0.15 s per
molecule thanks to caching (the candidate pool is shared across the test set).

**Cross-ring cleavage did not help** (`RING_BREAKS`, off). Allowing two bonds
of the same aliphatic ring to break roughly doubles the fragment masses for
sugars and saturated cores -- and made the channel *worse* alone (0.3628 ->
0.3537, same 600 molecules), with a noise-level +0.0009 in the blend. More
candidate fragments means more chance matches for the WRONG candidates too, so
the score loses its ability to discriminate. It is the same trade-off that
limits hydrogen shifts to +/-1: permissiveness is not free.

## 2. Structure-capped training data (v4)

Diagnosis from docs/09: on unseen molecules fpnet *loses* to analog search, so
its weakness is generalisation, not capacity. The training sample was the
cause. Sampling spectra uniformly let heavily-measured molecules dominate --
median 4 spectra per structure, but one had 1,438. At 900k spectra that
covered 221,792 of 268,969 structures.

Capping at 4 spectra per structure covers **all 268,969** for 943k spectra --
the same compute, 21% more distinct molecules. It also shifted the mix toward
the test instrument (timsTOF 38% -> 69%), because Orbitrap reference libraries
are the ones that measure each compound at many collision energies.

Result: **+0.0135 +/- 0.0081** over v3, paired on 600 molecules neither saw --
and v4 trained 20 epochs to v3's 30, so the gain is the data, not the time.

## 3. Contrastive fine-tune (v4c, `scripts/train_contrastive.py`)

Predicting fingerprint bits is only indirectly the task. The task is: among
~60 structures sharing this mass, which one? So v4 was fine-tuned on exactly
that -- a spectrum, its real candidate list from the pool (same-mass isomers,
the hardest negatives there are), and a softmax over `cand_fp . logits`,
which is the same score inference ranks by.

Leakage rules: queries only from v4's training structures, and candidates
only from those or from COCONUT structures absent from the spectral library.
Honest-holdout molecules never appear, even as negatives -- "this spectrum is
not X" still teaches the model about X.

Result: **+0.0056 +/- 0.0064** over v4. The interval includes zero, so this
is a probable gain, not a proven one. Two things point the same way: class 2
improved (0.8623 -> 0.8689), and BCE on v4's *own* held-out structures
improved (0.1965 -> 0.1934) even though fine-tuning never saw them.

One bug found by the smoke test: at softmax temperature 1, isomer scores
differing by tens of units made the loss 6.3 -- worse than guessing uniformly
(ln 57 = 4.0). Starting the temperature at 10, with its own learning rate,
brought it to 2.4 and turned a degradation of bit prediction into an
improvement.

## 4. The learned ranker (`src/casmi/ranker.py`, `scripts/train_ranker.py`)

A fixed blend weighs the channels identically for every molecule. A ~1.5k
parameter MLP instead scores each candidate from the channel scores *plus*
context: how good the best spectral relative is, how many candidates there
are, how rich the spectrum is. Trained listwise (softmax over each molecule's
candidates), on 2,300 unseen molecules, 16% simulated as class 1 so it learns
the regime where analog search finds a near-perfect self-match.

Out-of-fold, 5-fold grouped by molecule, on v3 features:

| method | all | class 1 | class 2 |
|---|---:|---:|---:|
| fixed blend | 0.8502 | 0.9015 | 0.8404 |
| linear ranker | 0.8445 | 0.9039 | 0.8331 |
| **MLP ranker** | **0.8988** | **0.9094** | **0.8968** |

**+0.0486 +/- 0.0087** -- the largest single gain of the phase. The linear
ranker *losing* is the informative part: re-weighting the channels does not
help; the gain is entirely the conditional rules only a nonlinear model can
express. The step count (300) was fixed before seeing these numbers.

Retrained on v4c features for shipping (fresh dump, 2,300 molecules unseen by
v4c):

| method | all | class 1 | class 2 |
|---|---:|---:|---:|
| fixed blend | 0.8691 | 0.9194 | 0.8594 |
| linear ranker | 0.8627 | 0.9283 | 0.8501 |
| **MLP ranker** | **0.9208** | 0.9184 | **0.9213** |

**+0.0517 +/- 0.0087.** Class 2 -- where the leaderboard gap lives -- goes
0.859 -> 0.921. Same signature as on v3: linear loses, nonlinear wins.

### What the stronger model changed

On molecules it never saw, fpnet *alone* went from losing to analog search to
beating it:

| channel alone | v3 | v4c |
|---|---:|---:|
| analog | 0.8050 | 0.8151 |
| fpnet | 0.7436 | **0.8533** |

As a result the best *fixed* weight moved back to 0.25*analog (0.8737). That is
the same number the rigged holdout recommended for v3 -- but now for the right
reason: a genuinely stronger model, not a memorising one. It only affects the
fallback path; the learned ranker sets the balance itself.

### A skew bug caught by construction

The ranker's training data was dumped by the validator, which computed the
features inline. Training re-derives every feature with
`ranker_features` -- the function the notebook calls -- and compares. That
check found `z_fr` differing by up to 1.0: fragment scores are float32, many
molecules have every candidate tied, and a float32 mean off by one rounding
step turns an all-tied channel into a z-score of +/-1 instead of 0.

Harmless to the fixed blend (a constant per molecule cannot reorder its
candidates) but real skew for a nonlinear ranker. The fix is structural: the
ranker trains only on features rebuilt by the serving function, so training
and serving cannot disagree.

## 5. Guards added

- every checkpoint carries `train_keys`, so "unseen" survives regenerating
  the training data (it silently broke once already);
- `--exclude-seen-by` so two models are graded on molecules *neither* saw;
- checkpoints carry a `model_tag`; the ranker records the tag of the fpnet
  whose scores it learned; the notebook and the packager both refuse a
  mismatch -- the packager in seconds, before a 25-minute Kaggle run.

## 6. Bugs of my own this phase

- **Quadratic checkpoint save.** `d["keys"][r]` inside a list comprehension:
  `d` is a lazy NpzFile, so every access re-read and unpickled 269k keys, 255k
  times. GPU sat at 1% for the whole first save. Fix: read once.
- **5.7 GB of duplicated labels.** SpectraSet copied a float32 label row per
  spectrum. Storing the index instead cut training from 11 GB to 4 GB, which
  is what let validation run alongside training.
- **`pgrep -f` matching itself**, three times: a waiter whose own command line
  contains the pattern waits forever. Wait on a PID.
- **WSL2 wall-clock jumps**: one epoch "took" 1,055 s while the process had
  existed for less than the sum of its epochs. Time durations with
  `time.monotonic()`, not `time.time()`.

## 7. Deferred: MSNovelist's SMILES decoder

The one strategy that could reach class 3 directly: generate structures token
by token from the predicted fingerprint instead of retrieving them. Deferred,
not rejected:

- MRR@25 needs the *exact* connectivity inside the top 25. MSNovelist's own
  top-1 on easy benchmarks is modest, and class 3 here is novel natural
  products -- the hardest case for a generator.
- It needs a decoder trained from scratch plus beam search inside Kaggle's
  9-hour offline budget.
- Fragmentation already gives class 3 a signal for any generated candidate,
  so a decoder's candidates could later be fed into the existing ranking
  rather than replacing it.
