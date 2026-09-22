# Phase 2 results — candidate pool + analog propagation

**Public LB 0.240** (submission 56465266, 2026-09-22), rank **627 / 1193**.
Up from 0.130 / rank 959.

## What changed

| | Phase 1 | Phase 2 |
|---|---|---|
| Candidates | training structures with spectra (269k) | COCONUT + training, mass-filtered (722k) |
| Ranking | entropy similarity (identity search) | Tanimoto to a spectral analog (hybrid search) |
| Reachable classes | 1 only | 1 and 2 |
| Public LB | 0.130 | **0.240** |

Two findings drove the design, both measured rather than assumed:

- **Retrieval was never the problem.** At 5 ppm the pool returns the answer
  ~95-100% of the time, with a median of 68 candidates. Ranking does all the work.
- **The identity channel was worth deleting.** Hybrid search subsumes it — a
  zero-mass-shift match *is* an identity match — and on class 2 identity is
  actively harmful, because the true answer has no spectrum and scores 0 while
  library-known wrong candidates score above it.

## The important result: our validation was optimistic by ~1.6x

Predicted 0.25-0.40, scored 0.240 — outside the range.

Class-1 contribution is `P(class1) x MRR1 = 0.154 x 0.896 = 0.138`, so class 2
contributed `0.240 - 0.138 = 0.102`.

The leader's score bounds the class-2 share from below: at 0.415, their class-1
contribution is at most 0.154 (an MRR cannot exceed 1.0), so
`P(class2) x MRR2 >= 0.261`, hence **P(class 2) >= 26%**. Therefore:

| if P(class 2) is | our real class-2 MRR |
|---|---:|
| 26% | 0.392 |
| 30% | 0.340 |
| 43% | 0.237 |

**We measured 0.621 locally.** The real value is at most 0.392.

### Why

Every holdout molecule is a **published** compound, so the training library
contains its relatives. The test molecules are **unpublished** — Enveda measured
them because nobody else had. Analog propagation needs a spectral cousin, and
genuinely novel compounds have fewer.

Switching the holdout from the congeneric `enveda-180` library to natural
products caught part of this (0.802 -> 0.645). It was the right instinct and
still insufficient: the residual gap is "published vs unpublished", which no
holdout drawn from published data can reproduce.

### The fix

The per-Tanimoto breakdown was already in the output; the mistake was quoting
the aggregate. Molecules whose best analog is distant (Tanimoto < 0.4) have no
close relative in the library either — the same situation a novel compound is
in — and that bucket read **0.396**, which brackets the real 0.24-0.39.

`scripts/validate_class2.py` now reports the low-Tanimoto subset as the
CALIBRATED ESTIMATE and demotes the aggregate to context. Optimise the hard
bucket, not the mean.

**Transferable rule:** when a validation set is built from data that is
available *because it was published*, and the test set exists *because it was
not*, the two differ in a way no random split can fix. Find the sub-population
of the validation set that resembles the test set, and treat that as the metric.

## What is left

Solving for the class structure, with P(class 2) >= 26% and P(class 1) = 15.4%,
class 3 is plausibly **40-58%** of the test set. Class-3 structures are in no
database, so retrieval scores exactly zero on them by construction. No candidate
pool and no ranker can touch that share.

That is the ceiling this approach is approaching, and the reason the next phase
has to be a model rather than a better lookup.
