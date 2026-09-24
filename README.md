# CASMI 2026 — molecule ID from mass spectra

Predicting 2D chemical structures (SMILES) from LC-MS/MS spectra for the
[Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

New to mass spectrometry? Start with **[docs/00-competition-primer.md](docs/00-competition-primer.md)**.
Want the whole story in one read? **[docs/14-walkthrough.md](docs/14-walkthrough.md)** — every
phase, every file, every command, and the lessons.

**Metric:** MRR@25 on tautomer-canonical InChIKey14 — 25 ranked SMILES per molecule,
score `1/rank` of the first correct one.

**Best public LB: 0.341** — the public "W088" engine reproduced under this account
(`kaggle/public_engine/`, credited below). Our own pipeline peaked at **0.244**.

## Setup

```bash
uv sync --extra dev
cp .env.example .env      # fill in KAGGLE_API_TOKEN and HF_TOKEN
```

The Kaggle CLI (2.x) reads the token from `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`
(kaggle.com/settings → API → Create New Token).

## Layout

```
docs/                 primer, then one note per phase (read in order)
src/casmi/            our pipeline: pool, analog search, library, fpnet, fragmentation, ranker
scripts/              training, validation and packaging entry points
kaggle/               our submission notebook + the public engine and its variants
app/                  Gradio demo
```

## Verify

```bash
uv run python src/casmi/metric.py            # scorer self-check (asserts agreement with the host)
uv run python scripts/dryrun_notebook.py     # runs the real Kaggle notebook cells locally
```

## History

| phase | what | public LB |
|---|---|---:|
| 1 | spectral library search | 0.130 |
| 2 | candidate pool (COCONUT + train) + analog propagation | 0.240 |
| 3 | fingerprint model (fpnet) + fixed blend | **0.244** |
| 4 | engineered features, weight picked on a leaky holdout | 0.228 |
| 5 | structure-capped data, contrastive fine-tune, fragmentation, learned ranker | 0.204 / 0.234 |
| 6 | public W088 engine, reproduced | **0.341** |
| 6 | + leak-free fingerprints for its second ranker | 0.332 / 0.330 |

## What the docs teach

- **[09](docs/09-the-validation-leak.md)** — a holdout that overlaps the model's training
  data grades memory, not skill. Exclude the model's *training set*, not just the query.
- **[12](docs/12-wrong-chemistry.md)** — right instrument, wrong chemistry. Trust a holdout
  only if it reproduces the leaderboard's ordering of submissions you already have.
- **[13](docs/13-public-engine.md)** — why the public engine wins with channels no better
  than ours (a learned combiner, a kept library channel, the right training chemistry),
  and why a stronger component can make an ensemble worse.
- **[15](docs/15-engine-harness.md)** — an offline harness for the whole engine; reproduce
  published numbers before experimenting; a rank feature whose ties were broken by list
  position, differently in training and serving.

## Credits

`kaggle/public_engine/` is an unmodified copy of
[CASMI26 Next Direct W088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088)
by **Jiachen Li**, built on public work by **prvsiyan** (Analog Propagation),
**haideptry** (4-channel retrieval + neural fingerprint ranker) and **megayak**
(Two Rankers, One Engine; leak-free fingerprint models). Apache 2.0, as public
Kaggle notebooks. Variants in `kaggle/engine_fpb_*` are ours, built on it.
