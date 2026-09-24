# Part 4 — `analog.py`: fingerprints and similarity to spectral cousins

**Written:** Phase 2. **Used by:** `morgan_fp` → `fpnet.py`,
`prepare_fpnet_data.py`, `prepare_contrastive.py`, `validate_class2.py`,
`validate_fpnet.py`; `rank_by_analogy` → `rank.py`, `validate_class2.py`,
`validate_fpnet.py`.

---

## The chain of reasoning (the module docstring, lines 1–18)

A class-2 molecule has no reference spectrum, so direct search finds nothing.
But natural products come in families. So:

1. `Library.analog_structures` (Part 2) finds library spectra matching the
   query **up to a constant mass shift** — the signature of a chemical
   modification. Those molecules are **cousins** of the unknown.
2. `CandidatePool.query` (Part 3) lists every structure of the right mass.
3. **This file:** prefer candidates that *look like* the cousins.

The unknown is never identified directly; it's located by triangulation from
known neighbours.

---

## Morgan fingerprints (lines 30–34)

```python
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
```

A **fingerprint** turns a molecule into 2,048 yes/no bits. For every atom, the
Morgan algorithm looks at its **circular neighbourhood** — the atom itself, its
neighbours (radius 1), their neighbours (radius 2) — and hashes each
neighbourhood pattern to one of the 2,048 bit positions. A bit is on if
*some* atom in the molecule has that neighbourhood pattern.

- Molecules sharing substructures share bits.
- It's a **hash**, so two different patterns can occasionally land on the same
  bit (a *collision*). 2,048 bits keeps that rare.
- Radius 2 is the standard "ECFP4" choice (diameter 4) for similarity.
- `_GEN` is built **once** at import; making a generator per molecule would be slow.

Real numbers: caffeine sets **25** bits, theophylline **26**, glucose **17**.

## `DEFAULT_POWER = 4.0` (lines 36–39)

How sharply to favour strong cousins. An analog's match score `s` is raised to
the 4th power before use:

| analog match `s` | `s⁴` |
|---:|---:|
| 0.9 | 0.656 |
| 0.7 | 0.240 |
| 0.5 | 0.0625 |
| 0.3 | 0.008 |

So one good cousin (0.9) outweighs many mediocre ones. The value was tuned on
a holdout and independently confirmed by a public notebook's sweep, whose
score rose 0.498 → 0.521 → 0.525 as the power increased.

## `morgan_fp(smiles)` (lines 42–50)

```python
@lru_cache(maxsize=2_000_000)
def morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _GEN.GetFingerprint(mol)
```

- Parse, then fingerprint; `None` if the SMILES is invalid.
- `@lru_cache` again (see Part 1): candidate lists of similar masses overlap
  heavily, and parsing dominates the cost. Caching 2M fingerprints is the
  single biggest speed-up in validation.
- `return A if cond else B` — a conditional expression in one line.

## Tanimoto similarity

The similarity used everywhere in this project:

```
Tanimoto(A, B) = |bits on in both| / |bits on in either|  =  |A ∩ B| / |A ∪ B|
```

1.0 = identical fingerprints, 0.0 = nothing shared. For caffeine vs
theophylline (caffeine minus one methyl group): **16 shared bits / 35 in the
union = 0.4571**. Caffeine vs glucose: **0.0** — nothing in common.

## `rank_by_analogy(candidate_smiles, analog_smiles, analog_scores, power)` (lines 53–102)

```
score(candidate) = max over analogs of  analog_score**power × Tanimoto(candidate, analog)
```

Line by line:

```python
    scores = np.zeros(len(candidate_smiles), dtype=np.float32)
    if len(candidate_smiles) == 0 or len(analog_smiles) == 0:
        return scores
```
One score per candidate, all zero to start. No candidates or no cousins → all zeros (a valid "nothing matched").

```python
    fps, keep = [], []
    for i, smi in enumerate(candidate_smiles):
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            keep.append(i)
    if not fps:
        return scores
    keep = np.asarray(keep)
```
Fingerprint each candidate once. `keep` remembers **which positions** parsed,
because the bulk function below can't accept `None`, and unparseable
candidates must keep their slot (score 0) so everything stays aligned.

```python
    for a_smi, a_score in zip(analog_smiles, analog_scores):
        weight = float(a_score) ** power
        if weight <= 0:
            continue
        a_fp = morgan_fp(a_smi)
        if a_fp is None:
            continue
```
- `zip(a, b)` walks two sequences in parallel: (1st cousin, its score), (2nd, …).
- `**` is exponentiation.

```python
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(a_fp, fps), dtype=np.float32)
        scores[keep] = np.maximum(scores[keep], weight * sims)
```
- `BulkTanimotoSimilarity(one, many)`: Tanimoto of one cousin to **all**
  candidates in a single C++ call — much faster than a Python loop.
- `np.maximum(a, b)`: element-wise larger of two arrays. So each candidate
  keeps its **best** (cousin weight × similarity) over all cousins — `max`, not
  `sum`: one convincing cousin is strong evidence, while a sum would reward
  being vaguely similar to many unrelated hits.

### The bug this line avoids (read the comment on line 97)

```python
np.maximum(scores[keep], weight * sims, out=scores[keep])   # WRONG — silently does nothing
scores[keep] = np.maximum(scores[keep], weight * sims)      # right
```

`scores[keep]` with an **integer array** index (*fancy indexing*) returns a
**copy**, not a view. So `out=scores[keep]` writes into a temporary copy that
is then thrown away — every score stays 0. The plain assignment writes back
into `scores`. The self-check caught this during development.

> Slices (`a[2:5]`) are views; integer-array indexing (`a[[2,3,4]]`) and
> boolean masks (`a[a > 0]`) are copies. Assigning *to* `a[idx] = …` always
> writes into `a`; it's reading `a[idx]` that makes a copy.

## Worked example (the self-check's molecules, real numbers)

Cousin: caffeine. Candidates: glucose, theophylline.

| analog match score | glucose | theophylline |
|---:|---:|---:|
| 1.0 (weight 1.0⁴ = 1) | 0 | **0.4571** (1 × 0.4571) |
| 0.5 (weight 0.5⁴ = 0.0625) | 0 | **0.0286** (0.0625 × 0.4571) |

Theophylline ranks above glucose, and a weaker cousin gives uniformly weaker
scores — exactly the two things `_self_check` asserts.

## `_self_check()` (lines 105–132)

| assertion | promise |
|---|---|
| `morgan_fp("not a molecule") is None` | invalid input doesn't crash |
| `s[1] > s[0]` | a caffeine-like cousin ranks theophylline above glucose |
| `weak[1] < s[1]` | weaker cousins give weaker scores |
| `s2[0] == 0.0 and s2[1] > 0.0` | a bad SMILES keeps its slot, alignment intact |
| no cousins → `[0.0, 0.0]` | "nothing matched" is valid, not an error |

---

## Try it

```bash
uv run python src/casmi/analog.py
uv run python -c "
import sys; sys.path.insert(0, 'src')
from rdkit import DataStructs
from casmi.analog import morgan_fp
a, b = morgan_fp('CN1C=NC2=C1C(=O)N(C)C(=O)N2C'), morgan_fp('Cn1c(=O)c2[nH]cnc2n(C)c1=O')
print(DataStructs.TanimotoSimilarity(a, b))   # 0.4571...
"
```
