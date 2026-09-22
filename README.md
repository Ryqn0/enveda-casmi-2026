# CASMI 2026 — molecule ID from mass spectra

Predicting 2D chemical structures (SMILES) from LC-MS/MS spectra for the
[Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

New to mass spectrometry? Start with **[docs/00-competition-primer.md](docs/00-competition-primer.md)**.

**Metric:** MRR@25 on tautomer-canonical InChIKey14 — 25 ranked SMILES per molecule,
score `1/rank` of the first correct one.

## Setup

```bash
uv sync --extra dev
cp .env.example .env      # fill in KAGGLE_USERNAME / KAGGLE_KEY
```

Credentials come from [kaggle.com/settings](https://www.kaggle.com/settings) →
API → Create New Token.

## Layout

```
docs/       competition primer, notes per phase
src/casmi/  library code (metric, data loading, models)
scripts/    reproducible entry points
notebooks/  exploration only — findings get promoted into src/
app/        the deployed demo
```

## Verify

```bash
uv run python src/casmi/metric.py     # scorer self-check (asserts agreement with the host)
```

## Status

- [x] Phase 0 — offline metric
- [x] Phase 1 — library-search baseline — **public LB 0.130** (local class-1 MRR 0.845, class 3 0.000)
- [x] Phase 2 — candidate pool + analog propagation — **public LB 0.240**, rank 627/1193
- [x] Phase 3 — fingerprint model — **public LB 0.244**, rank 646/1227 (+0.004 over analog-only)
- [x] Phase 4 — rank fusion — fpnet + 0.5*analog, weight swept on the test instrument
- [ ] Phase 5 — deployed demo — app/app.py works locally, not yet on HF Spaces

Notebook runs on Kaggle in ~8 min (see docs/05-kaggle-runtime-notes.md); submission is a notebook re-run, not a CSV upload.
