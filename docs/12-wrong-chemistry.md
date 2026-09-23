# Right instrument, wrong chemistry — calibrating the holdout against the LB

**LB 0.204** (kernel v11: v4c + fragmentation + learned ranker), after the best
offline number of the project (0.921). The third submission in a row where an
offline improvement lowered the leaderboard.

## What the timsTOF holdout actually contained

Since Phase 3 every decision was made on a "timsTOF holdout" -- chosen because
the test set is 100% timsTOF. But the two timsTOF libraries in the training
data are unrelated chemistry:

| source library | structures | in COCONUT | timsTOF |
|---|---:|---:|---:|
| enveda-180 | 182,941 | **0.0%** | 100% |
| enveda-np-examples | 250 | **99.6%** | 100% |
| riken / mona / msdial / massbank | 9-16k each | 63-81% | 0% |

enveda-180 is a synthetic, congeneric screening library, and it made up 99.9%
of the "timsTOF holdout". Enveda is a natural-products company; the
test-like example set is natural products.

**We matched the instrument and missed the chemistry.**

## The calibration test

Four configurations have leaderboard scores on comparable pipelines. A holdout
is only worth optimising against if it ranks them the way the leaderboard
does. Same 101 natural-product molecules, unseen by every model compared:

| config | NP holdout | timsTOF holdout | LB |
|---|---:|---:|---:|
| v2 + 0.5*analog | **0.5618** | 0.8287 | **0.244** |
| analog only | 0.5572 | 0.7817 | 0.240 |
| v3 + 0.25*analog | 0.5152 | 0.8509 | 0.228 |
| v4c + learned ranker | 0.5103 | 0.921 | 0.204 |

The NP holdout reproduces the LB ordering 4 for 4 (1-in-24 by chance). The
timsTOF holdout ranks them essentially backwards. Error bars at n=101 are wide
(+/- 0.04-0.07 on paired differences), so this is strong evidence, not proof.

## Why each "improvement" hurt

- **The learned ranker**: on natural products its class-1 MRR collapses to
  0.39 (vs 0.73). It learned rules from a congeneric library, where a close
  spectral relative always exists, that do not hold for natural products.
  Class 1 is ~15% of the test, so this is most of the drop to 0.204.
- **v4's data cap** raised the timsTOF share of training from 38% to 69% --
  i.e. shifted the model toward enveda-180 and away from the natural-product
  reference libraries, which measure each compound many times and so were
  capped hardest.
- **Weight 0.25**: congeneric molecules always have close analogs, so on
  enveda-180 the fingerprint model looks stronger relative to analog search
  than it is on natural products.

## What this corrects

- **docs/11 (PubChem) is withdrawn.** "0.2% of timsTOF molecules are in
  COCONUT" described enveda-180, not the test. The test-like natural products
  are 99.6% in COCONUT, so the pool was the right database all along, and
  PubChem would mostly add thousands of wrong competitors per query.
- **docs/10's numbers** are measurements on enveda-180 chemistry. The
  mechanisms (leakage rules, skew checks, paired comparisons) stand; the
  conclusions about which strategies help do not transfer.
- **Phase 2 had it right.** Its natural-product holdout was the one whose
  predictions tracked the LB; it was replaced for "instrument match".

## The lesson

Two different mismatches can break a validation set, and fixing one can
hide the other:

1. **leakage** -- the holdout overlaps training (docs/09)
2. **distribution** -- the holdout is not the population being scored

A holdout earns trust by one test only: **does it reproduce the leaderboard's
ordering of submissions you already have?** Run that check before optimising
against anything, and re-run it whenever the holdout changes.

## Re-tuned on natural products (984 molecules unseen by v4c, fair ties)

Scoring note: earlier scripts counted rank as "1 + candidates scored strictly
higher", which puts the truth first among its ties. Ties are common (many
candidates score exactly 0 on analog similarity), so that flattered tie-heavy
methods. All numbers here average over random tie order (`expected_rr`).

| v4c + fixed blend | MRR |
|---|---:|
| analog only | 0.5628 |
| fpnet + 0.5*analog | 0.5865 |
| fpnet + 0.75*analog | 0.5919 |
| fpnet + 1.0*analog | 0.5897 |
| any of the above + 0.25*frag | within +/- 0.002 |

- The fingerprint model earns its place: +0.024 +/- 0.013 over analog alone.
- The analog weight is a plateau between 0.5 and 1.0.
- Fragmentation adds nothing measurable on natural products.
- The learned ranker, retrained on natural products, ties the blend
  (+0.006 +/- 0.009 at best); the enveda-180 one collapsed class 1.

**Next submission:** v4c + 0.5*analog, no fragments, no ranker -- one variable
changed (the model) versus the 0.244 configuration.

## Result: 0.234 (submission 56500746)

v4c + 0.5*analog -- one change versus the 0.244 configuration -- scored
**0.234**. The code review found no defect in the shipped path; the prediction
(+0.024 +/- 0.043 at n=101) simply could not resolve a difference this small.

Two things to carry forward:

- **The leaderboard is noisy too.** ~400 test molecules, perhaps a subset on
  the public board: a paired difference between similar configurations has a
  standard error around 0.01-0.015. 0.234 vs 0.244 is ~1 SE -- indistinguishable.
  0.204 vs 0.244 (~3 SE) was a real loss. Do not chase differences below ~0.02.
- **The unseen-only holdout has a class-1 blind spot.** Test class-1 molecules
  have public spectra, so they are probably IN the training data, and
  memorising them is a legitimate advantage on the leaderboard. A holdout of
  molecules the model never saw cannot measure it. v2 trained on uniformly
  sampled spectra, which over-represent the natural-product reference
  libraries; v4 capped those in favour of enveda-180. That likely costs class 1.

Best confirmed configuration remains **v2 + 0.5*analog (0.244)**.
