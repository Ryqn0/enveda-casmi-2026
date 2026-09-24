# Part 6 — `rank.py`: combining the evidence for one molecule

**Written:** Phase 2 (analog only), extended in Phase 3 (fpnet), Phase 5
(fragmentation, learned ranker, Platt switch), and simplified back in Phase 5
(fragments off, ranker off). **Used by:** the Kaggle notebook and
`validate_combined.py` (`CombinedRanker`); `validate_fpnet.py` imports its
constants `PPM`, `MZ_OFFSET_PPM`, `TOP_ANALOGS`.

This is the top of our pipeline: it imports from every other module and turns
"one test molecule" into "25 ranked guesses".

---

## Imports (lines 18–28)

```python
from casmi.analog import rank_by_analogy
from casmi.fpnet import candidate_bits, predict_logits, score_candidates
from casmi.fragment import score_spectra_frag
from casmi.ranker import ranker_features, score_with_ranker
from casmi.library import Library
from casmi.pool import ADDUCT_MASS, CandidatePool
```

Everything from Parts 2–5 and 7 meets here. (In the Kaggle notebook these
`from casmi…` lines are stripped by the embedder, because all seven modules are
pasted into one namespace in dependency order.)

## Constants (lines 29–77) — every number, and where it came from

| constant | value | measured how |
|---|---|---|
| `FPNET_ANALOG_WEIGHT` | 0.5 | natural-product holdout: a plateau from 0.5 to 1.0; kept at the leaderboard-confirmed 0.5 |
| `FRAG_WEIGHT` | 0.0 | fragmentation added nothing measurable on natural products |
| `ANALOG_WEIGHT` | 0.5 | legacy identity-vs-analog weight; no longer affects ranking |
| `TOP_ANALOGS` | 100 | 100 cousins beat 50 by +0.012 |
| `PPM` | 4.0 | 4 ppm kept 100% recall with 11% fewer candidates than 5 ppm (on training molecules — see docs/11 for why that recall number was optimistic) |
| `MZ_OFFSET_PPM` | 0.98 | timsTOF calibration bias |
| `K` | 25 | the competition's guess limit |

The long comment above `FPNET_ANALOG_WEIGHT` is a **decision record**: the
table of measurements, which holdout they came from, and why that holdout is
trusted. Keeping the evidence next to the constant means nobody changes it
without seeing why it is what it is.

## `_z(x)` (lines 56–59)

```python
def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)
```

A **z-score**: subtract the mean, divide by the standard deviation. After this,
every channel is "how many standard deviations above average is this
candidate?", so channels on different scales (log-likelihoods around −900,
analog scores around 0.1) can be added. If every value is the same (`std ≈ 0`),
return zeros: a channel that doesn't vary carries no ranking information.

> **Its limitation (the key lesson of Phase 6):** a z-score only says how a
> candidate compares to the *other candidates of the same molecule*. It throws
> away how strong the evidence is in absolute terms — an exact library match at
> 0.95 and a weak best match at 0.30 can end up with the same z. The public
> engine's learned rankers use raw values too, which is part of why they win.

(A float64 twin lives in `ranker.py`; see Part 7 for why float64 mattered there.)

---

## `class CombinedRanker`

### `__init__` (lines 83–126)

Stores what every query needs, computed **once**:

- `lib`, `pool`: the spectral library (Part 2) and candidate pool (Part 3).
- `fpnet`, `fpnet_ck`: the model and its checkpoint, **optional** — without a
  model the ranker falls back to analog-only (the 0.240 configuration).
- From the checkpoint: `bit_sel` (which fingerprint bits the model predicts),
  `adducts` (the adduct → id list it was trained with), `max_peaks` (256).
  **Read from the model file, not hard-coded** — hard-coding 128 was a Phase 3 bug.
- `platt`: calibration coefficients, **only if** `use_platt=True` (default off).
- `ranker`, `ranker_ck`: optional learned ranker (default off).
- `_lib_row_of_key`: a dict `{InChIKey14: library row}`.
  ```python
  self._lib_row_of_key = {k: i for i, k in enumerate(lib.structures.inchikey14.to_numpy())}
  ```
  **Why:** the library numbers molecules one way, the pool another. The key is
  the only thing they share, so this dict **bridges the two numberings**. Built
  once in `__init__` rather than per query: it has 268,932 entries.

### `score_components(spectra, adduct, ppm, exclude)` (lines 128–185)

Computes the expensive evidence for one molecule. Kept separate from `rank()`
so validation can compute it **once** and try many blend weights on it.

**Input:** `spectra` = `[(precursor_mz, peaks), ...]` — all measurements of one
molecule. **Output:** `(cand, ident, analog, an_conf)`.

1. **Candidates** (lines 145–153): the pool's mass window around the first
   spectrum's precursor. Unknown adduct or empty window → empty result.
   ```python
   cand = (self.pool.query(...) if adduct in ADDUCT_MASS else np.empty(0, np.int64))
   ```
2. **Channel 1 — identity** (lines 155–169): average `score_spectrum` over the
   molecule's spectra (Part 2), then look up each candidate's score through
   `_lib_row_of_key`. A candidate with no library spectrum scores 0 — correct:
   no reference spectrum, no identity evidence.
   > ⚠ **Computed but not used** by `rank()` (since Phase 2, hybrid search was
   > judged to subsume it). It still costs a search per spectrum. Phase 6 showed
   > an explicit library channel *does* help a learned combiner — so this is
   > both wasted work today and a hint for the future.
3. **Channel 2 — analog** (lines 171–184):
   ```python
   best = np.zeros(self.lib.n_structures, dtype=np.float32)
   for q_mz, peaks in spectra:
       idx, sc = self.lib.analog_structures(q_mz, peaks, top_n=TOP_ANALOGS, exclude=exclude)
       np.maximum.at(best, idx, sc)
   ```
   Find cousins for **each** spectrum and keep every structure's **best** score
   across spectra (`maximum.at`, Part 2). Then the top 100 cousins overall
   (`argpartition` + `argsort`, the top-k pattern), and
   `rank_by_analogy` (Part 4) scores each candidate against them.
4. **`an_conf = best[a_idx[0]]`**: the score of the single best cousin — how
   much the analog channel deserves trust for *this* molecule. Only the
   learned ranker uses it.

`exclude` passes through to the library searches: validation uses it to hide
the query molecule's own spectra (the class-1/class-2 simulations).

### `rank(spectra, adduct, k, ppm, exclude, collision_energies)` (lines 187–266)

```python
        cand, ident, analog, an_conf = self.score_components(spectra, adduct, ppm, exclude)
        if len(cand) == 0:
            return cand, np.empty(0, dtype=np.float32)
```

Nothing in the mass window → return empty (the notebook then pads, see below).

```python
        if self.fpnet is None:
            total = analog
        else:
            logits = predict_logits(self.fpnet, spectra, adduct, self.adducts,
                                    instrument_id_=0, device=self.device,
                                    max_peaks=self.max_peaks,
                                    collision_energies=collision_energies)
            fp_score = score_candidates(logits, candidate_bits(self.pool.smiles[cand], self.bit_sel),
                                        platt=self.platt)
```

- No model → analog only.
- Otherwise: the model's prediction for this molecule (Part 5), averaged over
  its spectra, with the **real collision energies** passed through (a Phase 3
  bug was passing none). `instrument_id_=0` = timsTOF, because the test set is
  100% timsTOF.
- `fp_score`: each candidate's log-likelihood under the prediction.

```python
            needs_frag = self.ranker is not None or FRAG_WEIGHT != 0
            frag = (score_spectra_frag(...) if needs_frag else np.zeros(len(cand), dtype=np.float32))
```

Fragmentation (Part 7) is slow, so it's **skipped** when nothing uses it — with
`FRAG_WEIGHT = 0` and no ranker, that's always.

```python
            if self.ranker is not None:
                feats = ranker_features(fp_score, analog, frag, an_conf,
                                        n_peaks=max(len(pk) for _, pk in spectra))
                total = score_with_ranker(self.ranker, self.ranker_ck, feats)
            else:
                total = (_z(fp_score) + FPNET_ANALOG_WEIGHT * _z(analog) + FRAG_WEIGHT * _z(frag))
```

- **Learned ranker path:** features built by the **same function** that built
  its training data (Part 7) — the train/serve-skew defence.
- **Fixed blend path** (what ships): `z(fp) + 0.5 · z(analog)`.

```python
        n = min(k, len(cand))
        top = np.argpartition(-total, n - 1)[:n]
        top = top[np.argsort(-total[top])]
        out, scores = cand[top], total[top]
        if len(out) < k:
            out, scores = self._pad(out, scores, spectra[0][0], adduct, ppm, k)
        return out, scores
```

The top-k pattern again (Part 2), then map positions in `cand` back to pool
positions (`cand[top]`). If fewer than 25, pad.

### `_pad(out, scores, mz, adduct, ppm, k)` (lines 268–296)

A tight window sometimes returns only 3 candidates. The metric charges nothing
for a wrong guess, so 22 empty slots are 22 free chances thrown away.

```python
        wide = self.pool.query(mz, adduct, ppm=ppm * 20, offset_ppm=self.offset_ppm)
        extra = wide[np.isin(wide, out, invert=True)]
        extra = extra[np.argsort(np.abs(self.pool.mass[extra] - target))]
        extra = extra[: k - len(out)]
        return np.concatenate([out, extra]), np.concatenate([scores, np.zeros(len(extra), dtype=np.float32)])
```

1. Re-query with a **20× wider** window (80 ppm).
2. `np.isin(a, b, invert=True)`: keep elements of `a` **not** in `b` — drop
   candidates already ranked.
3. Sort the rest by closeness in mass to the target.
4. Take just enough to reach 25, and **append after** the ranked ones (with
   score 0), so padding can never push a real candidate down.

---

## How the notebook uses it

The real loop from `kaggle/submit_notebook.py`:

```python
for mid, grp in test.groupby("molecule_id", sort=True):
    spectra = [(float(r.precursor_mz), peaks_of(r)) for r in grp.itertuples()]
    adduct = grp.adduct.iloc[0]
    ces = [
        float(np.mean(r.collision_energy_ev))
        if r.collision_energy_ev is not None and len(r.collision_energy_ev)
        else 32.0
        for r in grp.itertuples()
    ]
    cand, scores = ranker.rank(spectra, adduct, k=K, collision_energies=ces)
    if len(cand) == 0:
        no_cand += 1
        smiles = list(pool.smiles[:K])
    else:
        smiles = list(pool.smiles[cand])
    rows.append({"molecule_id": mid, "smiles": ";".join(smiles)})
```

- `test.groupby("molecule_id")` yields `(id, rows of that molecule)` — one
  molecule at a time, all its spectra together.
- A spectrum's collision energy can be a **list** (a *stepped* acquisition at
  several energies); its mean is used, as in training, and 32 eV when missing.
- One `rank()` call per molecule; pool positions → SMILES → `";".join` → one
  line of `submission.csv`.
- **Never an empty row:** if nothing matched the mass, submit 25 arbitrary
  structures. A wrong guess costs nothing, while an empty field is rejected.

---

## Try it (needs the built index, ~6 GB RAM)

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
from casmi.library import Library
from casmi.pool import CandidatePool
from casmi.fpnet import load
from casmi.rank import CombinedRanker
lib, pool = Library(Path('data/index')), CandidatePool(Path('data/pool.parquet'))
m, ck = load('data/fpnet/fpnet_v4c.pt')
r = CombinedRanker(lib, pool, fpnet=m, fpnet_ck=ck)
peaks = np.array([[110.06, .15], [123.0441, .31], [138.0662, 1.0], [195.0877, .42]], np.float32)
cand, s = r.rank([(195.087652, peaks)], '[M+H]+')
print(pool.smiles[cand[:3]])
"
```
