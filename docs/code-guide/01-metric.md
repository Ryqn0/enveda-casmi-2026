# Part 1 — `metric.py` and `official_metric.py`: how answers are scored

**Written:** Phase 0, the very first code in the project.
**Why first:** every later decision is "did the score go up?". If the scorer
is wrong, every decision is wrong. So the scorer came before any model.

**Used by:** `metric.inchikey14` → `predict.py`; `metric.reciprocal_rank` →
`validate.py`; `official_metric.score` → only `metric.py`'s self-check.
The InChIKey14 idea itself is used everywhere (pool keys, labels, holdouts).

---

## The two files and why there are two

- `official_metric.py` is the **host's own scoring code**, copied unchanged
  from their public notebook (*vendored*). We never edit it.
- `metric.py` is **ours**: a thin wrapper that adds what the host's code
  doesn't expose — a reciprocal rank *per molecule*, so we can see which
  molecules failed, not just the average.

> **Pattern: vendor, then wrap.** When correctness is defined by someone
> else's code, copy their code and build on top of it. Re-implementing it
> invites a subtle difference that silently mis-grades every experiment.

---

## `official_metric.py`

### Module docstrings (lines 1–28)

```python
"""VENDORED verbatim from the competition metric notebook. Do not edit.
...
Pulled: 2026-09-22 via `kaggle kernels pull metric/casmi-mean-reciprocal-rank`
...
"""
"""Kaggle scoring for CASMI: Mean Reciprocal Rank @ max_rank (MRR@25 by default).
...
"""
```

- A string as the first statement of a file is its **docstring** — Python
  stores it as documentation (`help(module)` shows it).
- The **first** docstring is ours: it records *where* and *when* the file was
  copied, so anyone can re-pull it if the host updates it. The second is the
  host's original. (Only the first one is the "real" docstring to Python; the
  second is just an unused string — harmless.)
- The host's text defines the rules exactly: one row per molecule, guesses
  joined by `;`, score `1/position` of the first match.

### Imports and constants (lines 30–53)

```python
import importlib
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
```

- Standard-library modules first, then third-party (`numpy`, `pandas`) — the
  usual Python convention (PEP 8).
- `importlib` refreshes Python's import caches after installing a package;
  `subprocess` runs `pip`; `sys.executable` is the path of the running Python.

```python
EXPECTED_RDKIT_VERSION = '2026.03.3'
RDKIT_REQUIREMENT = 'rdkit==2026.3.3'
RDKIT_WHEEL_DIR = '/kaggle/input/datasets/metric/rdkit-2026-3-3-wheel'
```

- **Why pin RDKit:** "same molecule" is decided by RDKit's tautomer
  canonicalisation, which changes between releases. A different version can
  turn a correct answer into a non-match. The comment says it bluntly: the pin
  *decides which submissions match*.
- Two spellings of one version: PyPI normalises `2026.03.3` to `2026.3.3`
  (PEP 440 drops leading zeros), but `rdkit.__version__` reports `2026.03.3`.
- `RDKIT_WHEEL_DIR`: where Kaggle mounts the RDKit installer (a *wheel*, a
  pre-built package file), because scoring runs **offline**.

```python
Chem = None
_tautomer_canonicalizer = None
```

- Module-level placeholders, filled in later by `install_dependencies()`.
  Importing this file must work **before** RDKit is installed, so it cannot
  `import rdkit` at the top. This is a *lazy import* pattern.
- The leading underscore in `_tautomer_canonicalizer` means "private to this
  module" — a convention, not enforced.

### `class HostVisibleError(Exception)` / `ParticipantVisibleError`

```python
class HostVisibleError(Exception):
    pass
```

- Defines a new exception type. `pass` = "no extra behaviour"; it inherits
  everything from `Exception`.
- Two types so Kaggle can decide *who sees the message*: a
  `ParticipantVisibleError` (your submission is malformed) is shown to you; a
  `HostVisibleError` (their answer file is broken) only to the host.

### `install_dependencies()` (lines 60–121)

Purpose: make RDKit importable, installing it offline if needed. Called once;
cheap on repeat calls.

```python
    global Chem, _tautomer_canonicalizer
    if Chem is not None:
        return
```

- `global` lets the function **assign** the module-level variables (without
  it, `Chem = …` inside the function would create a new local variable).
- Early return makes it **idempotent**: calling it twice does no extra work.

```python
    try:
        import rdkit
    except ImportError:
        if not os.path.isdir(RDKIT_WHEEL_DIR):
            raise HostVisibleError(...)
```

- Try the import; only if RDKit is missing do we install. If the wheel folder
  isn't mounted either, fail with a message that says exactly what to attach.

```python
        command = [sys.executable, '-m', 'pip', 'install',
                   '--no-deps', '--no-index', f'--find-links={RDKIT_WHEEL_DIR}',
                   RDKIT_REQUIREMENT]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
```

Each piece of the pip command:

| part | meaning |
|---|---|
| `sys.executable -m pip` | run pip *of this exact Python*, not whatever `pip` is on the PATH |
| `install` | install packages |
| `--no-deps` | don't install dependencies (RDKit only needs numpy and Pillow, already present) |
| `--no-index` | never contact PyPI — there is no internet |
| `--find-links=DIR` | look for installers in this folder instead |
| `rdkit==2026.3.3` | exactly this version |

- `capture_output=True, text=True`: keep pip's messages as strings so they can
  be put **into** the error message. `check=False`: don't raise automatically —
  the code builds a far more useful error itself (Python version, folder
  contents, pip's last 1,500 characters of output).

```python
        importlib.invalidate_caches()
        try:
            import rdkit
        except ImportError as err:
            raise HostVisibleError(...)
```

- After installing, Python's cached view of installed packages is stale;
  `invalidate_caches()` refreshes it, then the import is retried.

```python
    from rdkit import Chem as _Chem
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog('rdApp.*')
    if rdkit.__version__ != EXPECTED_RDKIT_VERSION:
        warnings.warn(...)
    Chem = _Chem
    _tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
```

- `DisableLog('rdApp.*')` silences RDKit's warnings about odd molecules
  (thousands of them would flood the log).
- A version mismatch only **warns** (it may still work), it doesn't crash.
- Finally the placeholders are filled. The `TautomerEnumerator` is built once
  and reused: creating it per molecule would be slow.

### `canonicalize_smiles_to_ik14(smiles)` (lines 124–139) — the heart of the metric

```python
def canonicalize_smiles_to_ik14(smiles: str) -> str:
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_mol = _tautomer_canonicalizer.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        return inchikey[:14]
    except (ValueError, RuntimeError, TypeError):
        return None
```

Line by line:

1. `smiles: str -> str` are **type hints**: documentation for readers and
   tools; Python doesn't enforce them.
2. Not a string (e.g. `NaN` from a table) → `None` = "not a valid molecule".
3. `MolFromSmiles` parses the text into a molecule object; returns `None` if
   the text isn't valid chemistry.
4. `Canonicalize(mol)`: **tautomers** are forms of the same molecule that
   differ only by where a hydrogen sits (e.g. 2-pyridone vs
   2-hydroxypyridine). Canonicalising picks one standard form, so both count
   as the same answer.
5. `MolToInchiKey` produces a 27-character hash like
   `RYYVLZVUVIJVGH-UHFFFAOYSA-N`. The **first 14 characters** encode the
   connectivity (which atoms bond to which); the rest encode stereochemistry
   and charge, which the competition ignores.
6. The `except` catches every error RDKit can raise, so **one malformed
   SMILES can never crash the whole scoring run** — it just doesn't match.

### `heavy_atom_composition(smiles)` (lines 142–161) — a speed trick

```python
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))
```

- Counts non-hydrogen atoms, e.g. caffeine → `(('C', 8), ('N', 4), ('O', 2))`.
- `counts.get(sym, 0)` returns the current count, or 0 if unseen.
- Returned as a sorted **tuple**, because tuples are *hashable* (usable as dict
  keys and in sets); lists aren't.
- **Why:** canonicalisation is slow. Two molecules with the same InChIKey14
  must have the same heavy atoms, so if the counts differ we can skip the slow
  step. Hydrogens and charge are excluded because tautomers and charged forms
  differ there but still match.

### `score(solution, submission, row_id_column_name, max_rank=25)` (lines 168–324)

This is what Kaggle runs on your submission. In order:

1. **`install_dependencies()`** — make sure RDKit is ready.
2. **Validate the submission** — correct columns, not empty, no missing values,
   each `molecule_id` exactly once. Each failure raises a
   `ParticipantVisibleError` you'd see on Kaggle.
3. **Parse guesses** (lines 265–273):
   ```python
   parsed = [candidate.strip() for candidate in str(raw_guesses).split(';')]
   parsed = [candidate for candidate in parsed if candidate]
   ```
   These are **list comprehensions**: `[expression for item in iterable if condition]`
   builds a list in one line. The first splits on `;` and trims spaces; the
   second drops empty strings (e.g. from a trailing `;`). More than
   `max_rank` guesses is an error — you can't win by submitting 1,000.
4. **Gate keys** (line 276) — `heavy_atom_composition` of every guess, computed
   once up front. The `{... for ... for ...}` is a **dict comprehension** with
   two loops.
5. **Solution keys** (lines 288–301) — for each answer, its accepted InChIKey14
   set and heavy-atom composition. The answer is canonicalised **with the same
   function** as your guesses (one source of truth).
6. **Memoised canonicalisation** (lines 304–309):
   ```python
   canon_cache = {}
   def canon(smiles):
       if smiles not in canon_cache:
           canon_cache[smiles] = canonicalize_smiles_to_ik14(smiles)
       return canon_cache[smiles]
   ```
   A function **defined inside** another function, remembering results in a
   dict — so a SMILES guessed for many molecules is canonicalised once.
7. **The scoring loop** (lines 311–322):
   ```python
   for rank, guess in enumerate(guesses_by_molecule.get(mol_id, []), 1):
       if gate_keys[guess] not in accept_keys:
           continue
       if canon(guess) in acceptable:
           ap = 1.0 / rank
           break
   ```
   - `enumerate(items, 1)` yields `(1, first), (2, second), …` — counting from 1.
   - `continue` skips to the next guess (cheap reject); `break` stops at the
     **first** match — only the first correct guess counts.
8. `np.mean(scores)` — the average over molecules. Molecules you didn't submit
   get 0 because `.get(mol_id, [])` returns an empty list.

---

## `metric.py` (ours)

### Imports and setup (lines 12–22)

```python
from __future__ import annotations
from functools import lru_cache
from casmi import official_metric as _official

_official.install_dependencies()
K = 25
```

- `from __future__ import annotations` makes type hints like `str | None`
  work on older Pythons, by not evaluating them at run time. We put it at the
  top of every module.
- `import … as _official`: a short, private alias.
- `install_dependencies()` runs **at import time** here (unlike the host's
  file), because locally RDKit is already installed — it just binds `Chem`.
- `K = 25` — the one place the "25 guesses" rule lives.

### `inchikey14(smiles)` (lines 25–32)

```python
@lru_cache(maxsize=1_000_000)
def inchikey14(smiles: str) -> str | None:
    return _official.canonicalize_smiles_to_ik14(smiles)
```

- `@lru_cache` is a **decorator**: it wraps the function so repeated calls with
  the same argument return a stored answer instead of recomputing. "LRU" = least
  recently used entries are dropped when the cache holds 1,000,000.
- Validation canonicalises the same library SMILES thousands of times — this
  one line saves minutes.
- `1_000_000`: underscores in numbers are just for readability.
- `str | None`: returns a string, or `None` for invalid SMILES.

### `reciprocal_rank(predictions, truth, k=K)` (lines 35–48)

```python
    target = inchikey14(truth)
    if target is None:
        raise ValueError(f"unparseable ground-truth SMILES: {truth!r}")
    for rank, smi in enumerate(predictions[:k], start=1):
        if inchikey14(smi) == target:
            return 1.0 / rank
    return 0.0
```

- A bad **answer** is a bug in our data, so it raises (loud) instead of
  scoring 0 (silent).
- `predictions[:k]`: **slicing** — only the first 25 count.
- `f"...{truth!r}"`: an **f-string**; `!r` shows the value with quotes, so an
  empty string is visible as `''`.
- An unparseable prediction simply doesn't match but **keeps its slot** — the
  same as the host.

### `mrr(predictions, truths, k=K)` (lines 51–57)

```python
    return sum(
        reciprocal_rank(predictions.get(mid, []), smi, k) for mid, smi in truths.items()
    ) / len(truths)
```

- `sum(... for ...)`: a **generator expression** — like a list comprehension
  but computed one item at a time without building a list.
- `.items()` yields `(key, value)` pairs from a dict.

### `parse_submission(path)` (lines 60–68)

- Reads a `submission.csv` into `{molecule_id: [smiles, ...]}` so a written
  file can be re-scored locally. `import pandas as pd` sits *inside* the
  function so importing `metric.py` stays light.

### `_self_check()` (lines 71–118) — the contract, as code

Each `assert` is one promise:

| assertion | what it guarantees |
|---|---|
| `inchikey14(caffeine) == inchikey14(caffeine_alt)` | two SMILES of one molecule match |
| stereo pair equal | stereochemistry is ignored |
| pyridone == hydroxypyridine | tautomers collapse |
| `reciprocal_rank(["@@bad@@", caffeine], caffeine) == 0.5` | junk keeps its slot |
| `[aspirin]*25 + [caffeine]` → 0 | the 26th guess never counts |
| `abs(ours - theirs) < 1e-12` | **our mean equals the host's score** |

The last one is the reason the file exists: if the host's code and ours ever
disagreed, this check would fail.

### `if __name__ == "__main__":`

- `__name__` is `"__main__"` only when the file is **run directly**
  (`python src/casmi/metric.py`), not when imported. So running the file runs
  the self-check; importing it doesn't.

---

## Try it

```bash
uv run python src/casmi/metric.py
uv run python -c "
import sys; sys.path.insert(0, 'src')
from casmi.metric import inchikey14, reciprocal_rank
print(inchikey14('CN1C=NC2=C1C(=O)N(C)C(=O)N2C'))                     # RYYVLZVUVIJVGH
print(reciprocal_rank(['CCO', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'], 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C'))  # 0.5
"
```
