# Phase 6 — building on the public engine

## Baseline: 0.341, reproduced

`kaggle/public_engine/` is an unmodified, credited copy of Jiachen Li's
[CASMI26 Next Direct W088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088),
itself built on public work by prvsiyan, haideptry and megayak. Run under our
account it scored **0.341** -- exactly its published score. Best result of the
project; up from 0.244.

## Why it beats our pipeline (summary; see the conversation notes)

Same candidate pool, and each evidence channel on its own scores the same as
ours on natural products (analog ~0.52, fingerprint ~0.49, fragmentation
~0.25). The gap is the **combiner**:

- gradient-boosted rankers over 31-51 raw features, instead of a fixed
  z-score blend that discards how strong the evidence is in absolute terms
- a dedicated library channel (best, mean, adduct-shifted, count) -- we dropped
  identity search in Phase 2, which stopped being harmless once fpnet was added
- rankers trained on 2,250 simulated molecules from 7 natural-product-heavy
  libraries, both class-1 and class-2 simulations
- validated on enveda-np-examples from the start (we spent Phases 3-5 on
  enveda-180 -- docs/09, docs/12)

## Experiment 1: leak-free fingerprints for ranker B -- did not help

Ranker B (megayak's, 51 features) is trained and served with fingerprints
switched off in W088, because the public fingerprint model has seen most
library structures. megayak published leak-free models and matching rows.
`scripts/make_engine_fpb.py` wires them in; `scripts/cv_public_ranker.py`
measures ranker B offline.

| | offline, ranker B alone (np-examples CV) | leaderboard, full blend |
|---|---|---:|
| W088 (no fingerprint for B) | class 2 0.655, predicted 0.286 | **0.341** |
| + leak-free fingerprint, w_pv 0.88 | class 2 0.734, predicted 0.299 | 0.332 |
| + leak-free fingerprint, w_pv 0.50 | same | 0.330 |

Each LB difference is ~1 standard error, but both point the same way.

**Most likely reason: ensemble diversity.** B's value in the blend came from
being *different* from ranker A, which uses the fingerprint model. With
fingerprints, B is stronger alone but more correlated with A, so the blend
gains less -- and giving B more weight (0.50) scored lowest.

**Lesson: evaluate what you ship, not a component.** The offline harness scored
ranker B in isolation because ranker A's training rows are not aligned with
B's, so the blend itself could not be evaluated. A component improving is not
evidence that the ensemble improves.

## A false alarm handled correctly

The run logged the served fingerprint features next to the training ones.
Six columns (25-30) looked shifted (~0.99 vs 0.6-0.8). They are
library x fingerprint *agreement* features, and a notebook run's
`test.parquet` is the placeholder built from training rows -- every molecule
has a perfect library match that the fingerprint model also ranks first. A
population effect, not a train/serve skew; the pure fingerprint columns
matched. Worth knowing: **diagnostics computed on the placeholder test set
describe the placeholder, not the real test.**
