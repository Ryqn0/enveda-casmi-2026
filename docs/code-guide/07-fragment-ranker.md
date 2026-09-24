# Part 7 — `fragment.py` and `ranker.py`: the Phase 5 channels

Both were written in Phase 5 and are **switched off** in the shipped pipeline:
on natural products, fragmentation added nothing measurable and the learned
ranker didn't beat the fixed blend (and the version trained on the wrong
chemistry collapsed class 1). They're kept because the code — and the reasons
it's off — are part of the record, and both would be building blocks for a
future learned combiner.

**Used by:** `score_spectra_frag` → `rank.py`, `validate_fpnet.py`;
`ranker_features`, `score_with_ranker` → `rank.py`, `compare_dumps.py`,
`train_ranker.py`; `expected_rr` → `compare_dumps.py`, `train_ranker.py`,
`cv_public_ranker.py`; `RankNet` → `train_ranker.py`; `load_ranker` → `compare_dumps.py`.

---

# `fragment.py` — in-silico fragmentation

## The idea

A molecule in the collision cell breaks at bonds. If a candidate is right,
the observed peaks should correspond to **pieces of it**. So: break the
candidate's bonds in software, compute the pieces' masses, and measure what
**fraction of the observed intensity** they explain. Unlike the other two
channels, this needs **no library at all** — the only channel that can say
anything about class 3 (though on real data its signal was weak).

## Constants (lines ~46–63)

```python
RDLogger.DisableLog("rdApp.*")
PROTON = 1.007276466
HYDROGEN = 1.007825032
H_SHIFTS = (-1, 0, 1)
RING_BREAKS = False
```

- Silence RDKit: fragmenting hundreds of thousands of candidates would print
  hundreds of MB of warnings (on Kaggle, that bloats the log).
- `PROTON` turns a neutral piece into the ion we'd see (+ for positive mode).
- `HYDROGEN` and `H_SHIFTS`: when a bond breaks, a hydrogen often **moves**
  from one piece to the other, so a fragment is often seen at ±1 H. Allowing
  ±1 is standard; allowing more would let almost anything match.
- `RING_BREAKS` off: measured — more fragment masses also matched **wrong**
  candidates more often, so the channel got less discriminating (docs/10).

## `_fragment_masses(smiles, max_breaks=2, rings=False)` — cached

```python
@lru_cache(maxsize=200_000)
def _fragment_masses(smiles, max_breaks=2, rings=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()
    masses = {ExactMolWt(mol)}
```

- `@lru_cache` with **several arguments**: the cache key is the whole tuple
  `(smiles, max_breaks, rings)`. All arguments must be hashable (strings,
  ints, bools are).
- Returns a **tuple**, not a list: tuples are immutable, so a cached value
  can't be modified by a caller by accident.
- The intact molecule is always a "fragment" (the unbroken precursor ion).
- `{...}` with values is a **set**: duplicate masses collapse automatically.

```python
    breakable = [b.GetIdx() for b in mol.GetBonds()
                 if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
    if rings:
        breakable += [b.GetIdx() for b in mol.GetBonds()
                      if b.GetBondType() == Chem.BondType.SINGLE and b.IsInRing()]
```

- Only **single** bonds (double/aromatic bonds rarely break), and by default
  only **outside rings**: cutting one ring bond leaves the molecule in one
  piece, so it produces no new mass.
- With `rings=True`, ring single bonds are added so **pairs** of them can cut
  a ring open (how sugars fragment). RDKit types aromatic bonds as `AROMATIC`,
  not `SINGLE`, so aromatic rings are never opened.

```python
    for n in range(1, max_breaks + 1):
        for combo in combinations(breakable, n):
            try:
                frag = Chem.FragmentOnBonds(mol, list(combo), addDummies=False)
                pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
            except Exception:
                continue
            for p in pieces:
                try:
                    masses.add(ExactMolWt(p))
                except Exception:
                    continue
    return tuple(sorted(masses))
```

- **`itertools.combinations(items, n)`** yields every way to choose `n` items
  without order: `combinations([a,b,c], 2)` → `(a,b), (a,c), (b,c)`. So `n=1`
  breaks one bond at a time, `n=2` every pair — pairs produce **internal**
  fragments (cut at both ends).
- `FragmentOnBonds(..., addDummies=False)` cuts those bonds without adding
  placeholder atoms (which would add mass); `GetMolFrags(asMols=True)` splits
  the result into separate molecules.
- Errors skip that combination rather than killing the run: one odd candidate
  must not crash scoring for 400 molecules.
- **Cost:** a molecule with 20 breakable bonds → 20 + 190 = 210 cuts. That's
  why the cache matters, and why the channel is skipped when unused.

Caffeine (almost all aromatic rings) yields only **4** fragment masses — a
limit of this simplified approach.

## `explained_intensity(smiles, mzs, intensities, positive, ppm=10.0, max_breaks=2, rings=None)`

```python
    mz = np.asarray(mzs, dtype=np.float64)
    inten = np.asarray(intensities, dtype=np.float64)
    total = inten.sum()
    if total <= 0 or mz.size == 0:
        return 0.0
    frags = _fragment_masses(smiles, max_breaks, RING_BREAKS if rings is None else rings)
    if not frags:
        return 0.0
```

`rings=None` means "use the module default" — a common pattern so callers can
override a setting without the default being copied into every signature.

```python
    sign = 1.0 if positive else -1.0
    base = np.asarray(frags, dtype=np.float64) + sign * PROTON
    predicted = np.concatenate([base + k * HYDROGEN for k in H_SHIFTS])
    predicted.sort()
```

Every m/z we **could** see: each fragment ± a proton (by polarity), each at
−1, 0, +1 hydrogen. Sorted, for binary search.

```python
    pos = np.searchsorted(predicted, mz)
    lo = predicted[np.clip(pos - 1, 0, len(predicted) - 1)]
    hi = predicted[np.clip(pos, 0, len(predicted) - 1)]
    nearest = np.where(np.abs(mz - lo) < np.abs(mz - hi), lo, hi)
```

**The nearest-value trick:** `searchsorted` gives, for each observed peak, the
position where it would be inserted. The nearest predicted value must be just
**before** or just **at** that position — so compare those two and keep the
closer (`np.where(cond, a, b)` picks element-wise). `np.clip` keeps the
positions inside the array at both ends. All peaks at once, O(log n) each,
instead of comparing every peak with every fragment.

```python
    matched = np.abs(mz - nearest) <= mz * ppm * 1e-6
    return float(inten[matched].sum() / total)
```

A peak is explained if its nearest predicted mass is within 10 ppm (looser
than the 4 ppm precursor window, because our hydrogen bookkeeping is
simplified). The score is the **fraction of total intensity** explained: 0..1.

## `score_candidates_frag(smiles_list, mzs, intensities, adduct, ...)`

```python
    positive = not adduct.rstrip().endswith("-")
    return np.array([explained_intensity(s, mzs, intensities, positive, ppm, max_breaks, rings)
                     for s in smiles_list], dtype=np.float32)
```

The adduct string's last character tells the polarity (`[M+H]+` vs `[M-H]-`).
One score per candidate.

## `score_spectra_frag(smiles_list, spectra, adduct, ...)` — the one callers use

```python
    _, peaks = max(spectra, key=lambda s: len(s[1]))
```

- A molecule has several spectra; this uses the **richest** (most peaks).
  `max(items, key=f)` returns the item with the largest `f(item)`; a **lambda**
  is a tiny anonymous function (`lambda s: len(s[1])` = "the number of peaks").
- **Why not average:** the score is a *fraction*. A 3-peak spectrum is
  "fully explained" by almost any candidate, so averaging it in would pull
  every candidate toward the same score.
- **Why this function exists:** it's the **single place** that decides which
  spectrum is used — shared by `rank.py`, the validator, and the notebook.

## Self-check

Caffeine's own fragments are ~100% explained; an unrelated molecule explains
less; empty input and a bad SMILES return 0 (not a crash); switching polarity
changes the score; ring breaking adds fragments to glucose but leaves an
aromatic-only molecule (toluene) unchanged.

---

# `ranker.py` — the learned ranker

## The idea

The fixed blend weighs the channels identically for every molecule. A small
neural network can instead learn **conditional** rules — "trust the analog
channel only when the best cousin is strong". On the (wrong-chemistry)
enveda-180 data it gained +0.05; on natural products it tied the blend. A
*linear* ranker always lost: the value, when there was some, came entirely from
the conditional rules.

## `FEATURES` — the 12 columns, in a fixed order

```python
FEATURES = ["z_fp", "z_an", "z_fr",       # channels, z-scored within the molecule
            "rk_fp", "rk_an", "rk_fr",    # the candidate's rank in each channel, 0 = top
            "an_raw", "fr_raw",           # absolute strength (z-scoring discards it)
            "fp_gap",                     # fpnet log-likelihood minus the best one
            "an_conf",                    # how good the best spectral cousin is
            "log_ncand",                  # how crowded the candidate list is
            "log_npk"]                    # how rich the richest spectrum is
```

Order matters: a saved model reads column 3 as `rk_fp` forever. So the list is
stored **in the checkpoint** and checked on load.

## `_z(x)` and `_rank01(x)`

```python
def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

def _rank01(x):
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / len(x)
```

- `_z` is the same z-score as in `rank.py`, but here always fed **float64**.
  That mattered: in float32, a column where every value is tied had a mean off
  by one rounding step, so `std` wasn't 0 and the "z-score" came out ±1 instead
  of 0 — harmless in a fixed blend (a constant for every candidate), but a
  real train/serve difference for a neural net.
- `_rank01`: **inverting a permutation.** `order[j]` = which candidate is
  `j`-th best; writing `r[order] = arange(n)` gives each candidate its position.
  Divided by `n` → 0 (best) … just under 1 (worst). `kind="stable"` breaks ties
  the same way every time.

## `ranker_features(s_fpnet, s_analog, s_frag, an_conf, n_peaks)` — THE ONE RULE

Builds the `(n_candidates, 12)` matrix for one molecule, in `FEATURES` order.

```python
    full = lambda v: np.full(n, v, dtype=np.float64)     # a per-molecule constant
    cols = [_z(fp), _z(an), _z(fr),
            _rank01(fp), _rank01(an), _rank01(fr),
            an, fr,
            fp - fp.max(),
            full(an_conf), full(np.log(n)), full(np.log(max(n_peaks, 1)))]
    assert len(cols) == len(FEATURES)
    return np.stack(cols, axis=1).astype(np.float32)
```

- `np.full(n, v)` repeats one value `n` times — per-molecule context copied
  onto every candidate row.
- `fp - fp.max()` makes the log-likelihood **relative** to the best candidate
  (its absolute level is arbitrary).
- `np.log(n)`: counts spanning 10 to 500 become 2.3 to 6.2 — easier for a network.
- **This is the only place features are computed.** Training rebuilds every
  feature through it from raw dumped values, and the notebook calls it too — so
  training and serving *cannot* disagree. That rule is how the float32 bug
  above was caught: the training script compared its rebuilt features with the
  dumped ones and found `z_fr` off by up to 1.0.

## `class RankNet(n_features, hidden=32)`

```python
        if hidden == 0:
            self.net = nn.Linear(n_features, 1)
        else:
            self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)
```

- One score per candidate row. ~1.5k parameters at `hidden=32` — deliberately
  tiny: there are only ~1–2k training molecules, and a big network would memorise them.
- `hidden=0` → a **linear** model (a learned weighted sum) — the comparison
  that showed the non-linear rules were where any gain came from.
- `ReLU` = "negative → 0", the classic non-linearity.
- `.squeeze(-1)`: `(…, 1)` → `(…)`.

**How it's trained** (`train_ranker.py`, Part 8): **listwise softmax** — for
each molecule, softmax over its candidates' scores and maximise the probability
of the true one (cross-entropy). That directly rewards putting the right answer
**first**, which is what MRR mostly measures.

## `expected_rr(scores, truth, k=25)` — fair ties

```python
    s = np.asarray(scores, dtype=np.float64)
    higher = int((s > s[truth]).sum())
    ties = int((s == s[truth]).sum()) - 1
    ranks = np.arange(higher + 1, higher + ties + 2)
    return float(np.where(ranks <= k, 1.0 / ranks, 0.0).mean())
```

- Counting only candidates **strictly** higher puts the truth **first** among
  every candidate it ties with — flattering methods that produce many ties
  (many candidates score exactly 0 on analog similarity).
- On the leaderboard a tie's order is arbitrary, so this **averages** the
  reciprocal rank over every position the truth could take among its ties.
  Example: truth tied with one other at the top → ranks 1 or 2 → (1 + ½) / 2 = 0.75.
- `(s > x).sum()` counts `True`s — a neat numpy idiom (True = 1).

## `load_ranker(path)` and `score_with_ranker(net, ck, feats)`

```python
    assert list(ck["features"]) == FEATURES, ...
    ...
    x = (feats - ck["mean"]) / ck["std"]
    with torch.no_grad():
        return net(torch.as_tensor(x, dtype=torch.float32)).numpy()
```

- Refuse a checkpoint trained on a different feature list — it would silently
  read the wrong columns.
- **Standardise with the training statistics** (`mean`, `std` stored in the
  checkpoint), never recomputed at serving time — the same values the network
  was trained on.

## Self-check

Shape and finiteness; **shifting the fpnet scores by a constant changes
nothing** (only differences carry information — and it's what lets training
rebuild features from the dumped `fp_gap`); rank 0 is the best candidate; a
single-candidate molecule doesn't produce NaN.

---

## Try it

```bash
uv run python src/casmi/fragment.py
uv run python src/casmi/ranker.py
uv run python -c "
import sys; sys.path.insert(0, 'src')
from casmi.ranker import expected_rr
print(expected_rr([1, 1], 0))           # 0.75 -- tied with one other at the top
print(expected_rr([3, 2, 1], 2))        # 0.333... -- third place
"
```
