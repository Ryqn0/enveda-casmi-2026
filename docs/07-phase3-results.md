# Phase 3 results — the fingerprint model

**Public LB 0.244** (submission 56472241), rank **646 / 1227**. Previous best 0.240.

| # | approach | LB |
|---|---|---:|
| 1 | spectral library search | 0.130 |
| 2 | + COCONUT pool + analog propagation | 0.240 |
| 3 | + fingerprint model, wrong weight, broken inputs | **0.196** |
| 4 | + retrained model, right weight, fixed inputs | **0.244** |

## The model works and is worth almost nothing

A 5.5M-parameter set transformer maps a spectrum to 1,576 fingerprint bits;
candidates are ranked by the log-likelihood of their own bits under that
prediction. Trained locally on an RTX 4080: 900k spectra, 256-peak cap, 30
epochs, val loss 0.2217.

On the timsTOF holdout it beats analog propagation on every configuration. On
the leaderboard it contributed **+0.004**.

## Why the validation oversold it

Class-1 contribution is constant at `0.154 x 0.896 = 0.138`, so class-2
contribution went 0.1020 -> 0.1060, a **+3.9%** gain. Against the predictions:

| proxy | predicted | actual |
|---|---:|---:|
| timsTOF **all** (n=300) | +6.0% | +3.9% |
| timsTOF **hard**, best-analog Tanimoto < 0.4 (n=17) | +83.6% | +3.9% |

**The aggregate over 300 molecules is roughly calibrated. The hard bucket is
noise.** n=17 was never enough to read, and it was quoted as the headline
because it was the most encouraging number available.

This inverts the Phase 2 conclusion, where the aggregate over-predicted and the
hard bucket tracked reality — but that holdout was natural products with n=74
in the hard bucket. `enveda-180` is congeneric, so almost nothing in it is hard,
and the same framing stopped meaning anything.

**Rule:** a bucket is only a metric if it has the sample size to be one. Check
n before quoting a subgroup, every time, and prefer the largest subgroup that
still resembles the test set.

## The three bugs that cost submission #3

All were **train/serve skew** — the training pipeline and the inference pipeline
computing the same thing differently. None crashed. All produced confident,
plausible numbers.

| | training | inference |
|---|---|---|
| collision energy | real values, median 32 eV | hardcoded **0.0** |
| peak selection | top-N by **intensity** | first N by **m/z** |
| peak cap | 256 | hardcoded **128** in the validator |

The peak-selection bug was the worst: arrays are stored sorted by m/z, so
`peaks[:128]` kept the *lowest-mass* peaks and discarded every high-mass
fragment, on the 69.8% of test spectra that exceed the cap.

Every fpnet number measured before these fixes was wrong. Fixing the first two
alone — with no retraining — moved fpnet from 0.6035 to 0.7738 on the timsTOF
aggregate.

**Common cause:** the preprocessing was written three times (data prep,
`predict_logits`, the validator). Duplicated logic drifts, silently. There is
now one `select_peaks()`, and `max_peaks` travels inside the checkpoint because
a model's preprocessing config is part of the model.

## The weight, and the wrong-instrument lesson

`fpnet + 0.25*analog` came from a natural-product holdout. On timsTOF — the
test instrument — 0.25 is the single worst choice available, the only weight
that loses to analog alone (0.7510 vs 0.7817). That one decision is most of the
0.240 -> 0.196 regression.

The instrument-matched holdout predicted the direction correctly both times:
it said 0.25 would lose, and it lost; it said 0.5 would win, and it won.
**Direction has been reliable; magnitude has not.**

## Where the remaining gap is

Leader 0.415, us 0.244. Class-1 contribution is capped at ~0.154 for everyone,
so they draw ~0.277 from classes 2-3 against our 0.106 — roughly 2.6x.

Published pipelines at 0.34 run four channels: library search, analog
propagation, **in-silico fragmentation (MetFrag-lite)**, and a **larger
fingerprint model**, fused by a **gradient-boosted ranker** rather than fixed
weights. We have the first two and a small version of the fourth.

Ranked by expected value:

1. **A bigger, longer-trained fingerprint model.** Ours is 5.5M parameters on
   900k of 2.15M available spectra; val loss was still falling at 30 epochs.
2. **MetFrag-lite.** Scores a candidate by how much observed intensity its bond
   breaks explain — needs no library and no relative, so it reaches class 3,
   which nothing we have touches.
3. **A learned ranker.** Fixed weights cannot express "trust the model when the
   analog is distant"; a GBDT over per-candidate features can.
4. **Re-tune the mass window.** It was tuned against the analog-only ranker,
   when fewer candidates helped a weak ranker. A stronger ranker may prefer a
   wider window and the extra recall.
