# Retracted: the "data leak" was the documented placeholder test set

**Claimed:** 2026-09-22 · **Retracted:** same day · Kept as a record of the mistake.

## The claim

`scripts/check_leak.py` found that all 1213 spectra in `test.parquet` appear in
`train.parquet` as bitwise-identical arrays, labelled with their structures, and
that the Phase 1 baseline therefore scored MRR 1.0000 against them. I wrote this
up as an unintended data leak.

The measurements were all correct. Every number reproduces. The conclusion was
wrong.

## Why it was wrong

The competition's **data description page** says so directly:

> **test.parquet** — the spectra to identify. [...] **Note**: the file you see is
> comprised of examples from the training dataset; it will be replaced by the
> hidden test set during re-run, and will be approximately the same size.

This is a **Notebooks-only code competition**. The `test.parquet` shipped with
the public data is a deliberate placeholder built from training rows so that the
notebook has something correctly-shaped to run against during development. At
scoring time Kaggle swaps in the real hidden test set.

So "the test spectra are in the training file" is not a leak. It is the
documented design, and finding it proves nothing except that the placeholder is
what it says it is.

Corroborating signs that were visible and that I did not weigh:

- The official tutorial notebook filters `ingest_lib != 'enveda-180'` — exactly
  the library the placeholder is drawn from — so that development validation is
  not trivially perfect.
- The public leaderboard sat at 0.415 while my analysis implied 1.0 was free.
  I treated that tension as evidence that 1169 teams had missed something,
  which should have been the least likely explanation, not the working one.
- The data page states the real test set spans monoisotopic masses **157–1159 Da
  across 10 adducts**. The placeholder spans 245–460 m/z across 7. The dummy is
  visibly not the described test set.

## Root cause

I inferred the dataset's structure from the files instead of reading the data
description page first. Every subsequent check — exact float equality, the
precursor-mass consistency check at 2 ppm — was rigorous, and all of it was
rigour applied to a question that the documentation had already answered.

Confirmation without falsification: I kept building evidence *for* the leak and
never asked what else would produce identical spectra in both files.

## What this invalidated

- The leak claim, and the MRR 1.0000 attached to it. That number was the
  baseline retrieving training rows from training data.
- The Phase 0 EDA's characterisation of "the test set" — it profiled the
  placeholder. See `docs/01-data-notes.md`, which now carries a warning.
- The Phase 1 library filter, which was derived from the placeholder's mass
  range and adducts and so was far too narrow for the real test set.

## What survived

- `src/casmi/metric.py` — independently cross-checked against the official
  scoring notebook, agrees to 1e-9.
- `src/casmi/library.py` and `scripts/validate.py` — validation holds out
  *training* structures and never referenced `test.parquet`, so its numbers
  stand.
- The `rdkit==2026.3.3` pin, which came from reading the official metric.

## The rule going forward

Read the competition's data description and code requirements **before** running
analysis on the files. A surprising empirical finding that contradicts 1169
other teams is a prompt to re-read the documentation, not to write it up.
