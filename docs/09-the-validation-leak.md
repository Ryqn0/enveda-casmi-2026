# The validation leak — why the holdout lost to the leaderboard twice

**LB 0.228** (kernel v10, fpnet v3 + 0.25*analog), against 0.244 at weight 0.5.

Two things changed in that submission — the model AND the blend weight — which
is a confound I should not have shipped. But the diagnosis is unambiguous, and
it invalidates more than one submission's worth of measurement.

## The pattern that gave it away

| submission | model | weight | LB |
|---|---|---|---:|
| Phase 3 | v2 (broken inputs) | 0.25 | 0.196 |
| Phase 3b | v2 | **0.5** | **0.244** |
| Phase 4 | v3 | 0.25 | 0.228 |

Every time weight 0.25 shipped, the score fell. Every time the holdout was
consulted, it recommended 0.25. A measurement that disagrees with reality once
is noise; twice, in the same direction, is a broken instrument.

## The cause

`validate_fpnet.py` sampled its holdout from the whole spectral library. fpnet
was trained on a 95% structure-level split of that same library. So:

    holdout structures sampled:  300
      IN fpnet's TRAINING split: 241  (80.3%)   <-- memorised
      in fpnet's val split:       20
      never seen by fpnet:        39

**80.3% of the holdout was molecules fpnet had already been trained on.**

The comparison was rigged in a way that was invisible, because the two channels
cheat differently:

- **analog search** is measured honestly. `score_components(..., exclude=...)`
  removes the query molecule's own spectra from the library at search time, so
  it genuinely has to generalise.
- **fpnet** cannot be excluded that way. Its knowledge of the molecule is baked
  into its weights. There is no runtime flag that makes a trained network
  forget.

So fpnet was graded on an open book and analog on a closed one.

## What honest numbers look like

Same script, same 300-molecule budget, same candidates — only the sampling pool
changed, to the 11,669 structures fpnet never trained on (`--unseen-only`):

| channel | rigged (80% memorised) | **honest (0% memorised)** |
|---|---:|---:|
| analog alone | 0.7817 | **0.7979** |
| fpnet alone | 0.8147 | **0.7321** |
| fpnet + 0.25*analog | **0.8509** | 0.8139 |
| fpnet + 0.5*analog | 0.8287 | **0.8298** |
| fpnet + 1*analog | 0.8162 | 0.8245 |

Three things flip at once:

1. **fpnet falls 0.8147 -> 0.7321.** Most of its apparent strength was recall,
   not generalisation.
2. **analog RISES slightly** (0.7817 -> 0.7979), because it was never cheating
   and the unseen sample happens to be marginally kinder to it.
3. **fpnet alone now LOSES to analog alone.** The learned model is the weaker
   of the two channels on molecules it has not seen.

And the weight ordering inverts to agree with the leaderboard: **0.5 is best**,
0.25 is clearly worse. The honest holdout reproduces the LB verdict that the
rigged one contradicted twice.

Weight reverted to **0.5** in `rank.py`.

## The nuance worth keeping

The rigged holdout is not meaningless — it is a fair proxy for **class 1**,
where the test molecule genuinely does have public spectra and plausibly *is* in
fpnet's training data. Memorisation is a legitimate advantage there.

But class 1 contributes at most ~0.154 for everyone. Classes 2 and 3 — where
the molecule is definitely not in training — are ~85% of the remaining value,
and `--unseen-only` is the right proxy for them. Tune the blend for the regime
that dominates the score.

## What this changes going forward

**Every fpnet number measured before this is inflated.** docs/07 and docs/08
tables should be read as class-1-like estimates, not overall ones. In
particular the docs/08 claim that v3's engineered features gave +0.024 was
measured on the rigged holdout; the honest re-measurement is still owed.

More importantly, the priority order changes. fpnet losing to analog on unseen
molecules means the model is **memorising rather than generalising**, so the
lever is generalisation — more of the 2.15M available spectra (we train on
900k), augmentation, regularisation — not a bigger network or more epochs.

## The transferable lesson

Excluding the test item from the *index* is not the same as excluding it from
the *model*. When one channel is a lookup and another is a trained network,
"hold out the query" means two completely different things, and the network
gets a silent advantage unless the holdout is drawn from its own validation
split.

**Rule: a holdout must be unseen by every component being compared, not just
the ones where hiding it is easy.** Before trusting any comparison between a
retrieval channel and a learned one, print the overlap between the holdout and
the model's training set. It is three lines of code and it was worth two
submissions.
