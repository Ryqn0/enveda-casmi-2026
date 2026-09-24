# Phase 7 — a harness for the whole public engine, and a tie bug

Goal: push the public engine (LB 0.341) toward 0.35. docs/13's lesson was that
scoring one component (ranker B alone) did not predict the leaderboard. So
before changing anything, build a harness that scores **what would be
submitted**: channels, both rankers, the rank blend, the de-duplicated top 25.

## The harness (`scripts/engine_harness.py`)

```bash
uv run python scripts/engine_harness.py channels             # ~19 min, once
uv run python scripts/engine_harness.py score                # ~2 min with cached rankers
uv run python scripts/engine_harness.py score --nb kaggle/engine_ties/casmi26-engine-ties.ipynb
```

- **Runs the engine's own code.** The `%%writefile` modules are extracted
  from the notebook at run time, so the harness can't drift from the submission.
  `--nb` points it at a variant notebook, and then the measurement **is** the
  variant.
- **Queries:** the 250 structures of `enveda-np-examples` (1,184 timsTOF
  spectra), the set the public calibration uses: predicted LB =
  `0.162·class1 + 0.220·class2`. All 250 also have spectra in other libraries,
  so each molecule is simulated twice, **one molecule at a time**:
  - class 1: hide its own query spectra;
  - class 2: hide every spectrum of its structure, and drop the structure from
    the analog results.
- **Two stages.** `channels` (library, pool, index, per-molecule evidence) is
  slow and saved to `data/harness/`; `score` (rankers, blend, metric) reuses
  it, so each experiment costs minutes. Fitted rankers are cached by name.
- **Scoring** mirrors `make_submissions`: stable sort, tautomer-canonical
  de-duplication, first hit in the top 25. Every table has a paired ±2 SE
  column against the first blend.
- **Known optimism:** the public fingerprint models (ranker A's input) were
  trained on library spectra that include these structures, so absolute
  class-2 numbers are high. Changes that don't touch those models compare fairly.

## Getting the simulation right: three corrections

Each was found because a number didn't match the authors' published ones.

| version | ranker B class 1 / class 2 | what was wrong |
|---|---|---|
| B trained without the 250 molecules | 0.743 / 0.560 | they are B's **only timsTOF rows**; dropping them handicaps it |
| + 5-fold CV over the 250 | 0.848 / 0.559 | class 1 fixed; class 2 still far from the published 0.655 |
| + one molecule hidden at a time | 0.844 / 0.572 | hiding all 250 at once also removed each molecule's isomers and relatives, which are in the library on the real test |

**Cross-validation needed a mapping** from B's anonymous training groups to
molecules. `match_np_groups` rebuilds each true candidate's feature row and
takes the nearest of their groups: median distance 0.002, 249/250 matched
one-to-one for class 1.

## The finding: ties decide a rank feature, differently in training and serving

Class 2 still didn't match, so the feature rows were compared **column by
column** (true candidates, mine vs theirs). Three columns were far apart, all
"rank of the library score among this molecule's candidates":

```python
def _rank_norm(x):
    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)
```

`argsort` orders **tied** values by their position in the array. In class 2
the true answer has no library spectrum, so it ties at score 0 with a median of
**36** other candidates, and its "rank" is decided by where it sits in the list:

| truth's library rank, class 2 (0 = best) | engine (position ties) | ties averaged |
|---|---:|---:|
| authors' training rows | **0.468** | 0.629 |
| serving (the harness = the submission) | **0.762** | 0.641 |

With ties averaged the two agree, so the simulation is right. The difference is
**list order**: in the training rows the truth tends to come early among its
ties; at serving, candidates are in mass order and it comes late. The rankers
learned "among the zeros, the truth is near the front", which is false at
serving time. It's train/serve skew one layer deeper than usual: the same
function, fed lists in a different order.

**Fix:** average ties everywhere. At serving, `_rank_norm` uses
`scipy.stats.rankdata(..., method="average")`. In training, each rank column of
both rankers' rows is **recomputed from the raw column next to it**, per
candidate list. `retie` first checks the pairing: where a list has no ties, old
and new values must agree. That check caught that ranker A's lists are
`(G, M)` pairs, not `G`.

### Result (250 molecules, paired, predicted LB)

| config | before | after | paired change | class-2 molecules better / worse |
|---|---:|---:|---:|---|
| ranker B | 0.2625 | 0.2763 | **+0.0138 ± 0.0057** | 72 / 14 |
| ranker A | 0.3079 | 0.3079 | +0.0000 ± 0.0037 | 24 / 24 |
| blend 0.88 (ships) | 0.3061 | 0.3104 | **+0.0044 ± 0.0045** | 36 / 23 |

B improves clearly; the shipped blend by about two standard errors. On the
leaderboard that's ~+0.005, below its ±0.01–0.015 noise, so a submission can't
confirm it by itself. The harness evidence is what supports it.

`scripts/make_engine_ties.py` builds the variant (`kaggle/engine_ties/`), the
same way as the earlier variants: exact-match text patches, credit kept.

## Lessons

- **Reproduce the published numbers before experimenting.** Every mismatch
  here was a simulation mistake of ours, until the last one, which was a bug in
  the engine.
- **Compare feature rows column by column** when two pipelines disagree. The
  aggregate said "B is worse"; the columns said which feature and why.
- **Ties are data.** Whenever a rank is computed, decide what ties mean. Here
  "position in the list" silently became a feature.
