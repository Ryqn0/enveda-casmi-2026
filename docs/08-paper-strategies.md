# Phase 4 — strategies from the literature

Source papers: **MSNovelist** (Stravs 2022), **Mass2SMILES**, **Spec2Mol**,
and a review of LC-MS/MS structure-elucidation strategies.

Two ideas were implementable without a new architecture. Both were measured on
the 300-molecule timsTOF holdout — the test instrument — reading the **all**
column, because docs/07 established the hard bucket (n=17) is noise.

## 1. Engineered peak features — WORKED, shipped

Mass2SMILES and Spec2Mol both feed the encoder more than raw (m/z, intensity).
Two features were added per peak, both derivable in principle from inputs the
model already had, which is why the outcome was genuinely uncertain:

- `log1p(intensity * 1000) / 7` — peak heights span orders of magnitude and
  skew hard toward near-zero noise. On a linear scale almost every peak looks
  identical to the network.
- `mz / precursor`, clamped to [0, 1] — whether a fragment is a small piece or
  nearly the whole molecule. Absolute mass was already encoded; this is the
  scale-invariant view that transfers across molecules of different size.

| | v2 | **v3 (features)** |
|---|---:|---:|
| val loss | 0.2217 | **0.2187** |
| fpnet alone | 0.7905 | **0.8147** |
| fpnet + 0.25*analog | 0.8289 | **0.8509** |
| fpnet + 0.5*analog | 0.8287 | 0.8388 |

Mid-training v3 looked *behind* v2 at matched epochs and the call was nearly
made to drop it. It overtook v2 around epoch 26 of 30. **A feature that is
redundant in principle can still be worth real accuracy**, because it changes
what the optimiser can reach easily, not what the function class can express.

## 2. Modified Platt score — FAILED, not shipped

CSI:FingerID and MSNovelist rank candidates by a *calibrated* score, not a raw
likelihood. This should have been our best free win: `pos_weight` deliberately
biases the model toward predicting 1s, and it measurably does — mean predicted
P(bit=1) 0.1328 against an actual rate of 0.0278, **over-predicting 4.77x**.

Per-bit Platt scaling (`p = sigmoid(A*z + B)`, fitted on the held-out split)
fixed the probabilities completely:

    log-loss 0.2187 -> 0.0732, mean predicted 0.0278 vs actual 0.0278

And made the ranking worse at every blend weight:

| scoring | fpnet alone | +0.25*analog |
|---|---:|---:|
| raw (shipped) | **0.8147** | **0.8509** |
| Platt A-only | 0.8095 | 0.8461 |
| Platt full (v2) | 0.6409 | 0.7864 |

### Why — the useful part

Our score is `cand_fp @ lp + (1-cand_fp) @ lq`. Expand it:

    = cand_fp @ (lp - lq) + sum(lq)

`sum(lq)` is identical for every candidate, so it cannot affect ranking. And
`lp - lq = log(p/(1-p))` is just the logit. **The ranking is exactly
`cand_fp @ z`** — a plain dot product with the raw logits.

Calibration turns that into `cand_fp @ (A*z + B)`. With B ~ -3.2 across all
bits, that adds `-3.2 * (number of bits the candidate sets)`: a blanket
complexity penalty on larger, more-decorated molecules. That is a prior about
molecules, not evidence from the spectrum, and it swamps the real signal.

Stripping B and keeping only the per-bit reliability weights A — the part of
the idea that is actually about the model — got back most of the loss but still
did not beat raw. On v2, A-only was a small *gain* (0.7905 -> 0.7988); on the
stronger v3 it was a small loss. The reliability reweighting is worth roughly
nothing once the model is good.

**Transferable lesson: better probabilities are not better rankings.** Log-loss
improved 3x while the metric we actually care about fell. Calibration is the
right tool when a probability will be *thresholded, or compared across
examples*. Ranking within one example only sees relative order, and a
transformation that is monotone per-bit is not monotone on a sum over bits.

The coefficients are still saved in the checkpoint and `CombinedRanker` still
accepts `use_platt`, defaulted **off**, so the negative result stays
reproducible rather than being deleted.

## 3. Not attempted

- **MSNovelist's RNN decoder** (fingerprint -> SMILES, token by token). The only
  idea here that reaches **class 3** — structures in no database, which our
  pool-based pipeline cannot touch at all. Biggest remaining upside; needs a
  decoder trained from scratch, so it was out of scope for one submission slot.
- **Spec2Mol's contrastive embedding** (rank by cosine in a joint
  spectrum/molecule space). A genuinely different ranking signal, so it should
  ensemble well with bit-likelihood rather than duplicate it.

## What shipped

**fpnet v3 + 0.25 * analog, no calibration.** Holdout 0.8509. Kernel v10,
submission 56474944.

The blend weight moved 0.5 -> 0.25, which is the value that once cost
0.240 -> 0.196. That earlier choice came from a natural-product holdout on the
**wrong instrument**; on timsTOF/v2 it only ever tied 0.5 on the aggregate.
Here, on the test instrument, 0.25 wins **both** columns with a real +0.012
aggregate margin. The mechanism is coherent: v3 is stronger alone, so it needs
less help from the analog channel.

## Guards added

v2 and v3 are both 256-peak models, so the notebook's existing
`max_peaks == 256` assert could **not** tell them apart — a stale dataset mount
would have run v2 silently and simply scored worse. The notebook now also
asserts `cfg["extra_feats"] is True` and `val_loss < 0.220`, and that
`ranker.platt is None`. Same failure class as every other bug in this project:
nothing crashes, the numbers just quietly come out wrong.
