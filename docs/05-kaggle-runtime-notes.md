# Kaggle runtime notes

Measured facts from the first successful notebook run (v2, 2026-09-22). Written
down because none of it is guessable, and each wrong guess costs a round trip of
several minutes.

## Mount paths — not what the docs imply

```
/kaggle/input/
    competitions/enveda-CASMI26-molecule-id-mass-spectra/    <- train/test parquet
    datasets/ryqn00/casmi26-deps/                            <- attached dataset
```

Both are **one or two levels deeper** than the commonly-cited
`/kaggle/input/<slug>`. Version 1 of the notebook hardcoded
`/kaggle/input/casmi26-deps` and `/kaggle/input/enveda-CASMI26-...` and was
wrong about both.

**Do not hardcode. Discover:**

```python
INPUT = Path("/kaggle/input")
WHEELS = sorted({p.parent for p in INPUT.rglob("*.whl")})[0]
TRAIN = next(INPUT.rglob("train.parquet"))
```

And print `sorted(INPUT.glob("*"))` before anything else, so the log diagnoses
the next failure without needing another run.

## Environment

| | |
|---|---|
| Python | **3.12** (`/usr/bin/python3`, dist-packages under `python3.12`) |
| numpy | 2.0.2 preinstalled |
| RDKit | **not installed** — ships as a wheel if needed |
| ms_entropy | not installed — we ship cp310–cp313 wheels |
| CPUs | 4 vCPU (vs 32 on the dev machine) |
| Notebook runner | papermill; a cell exception aborts the whole run |

Shipping wheels for several Python versions costs ~1 MB and removes the risk of
a base-image upgrade breaking the submission silently.

## Timings, and what they mean

| stage | Kaggle (4 vCPU) | local (32 vCPU) |
|---|---:|---:|
| index build, 2.15M spectra | **439 s** | 367 s |
| rank 400 molecules / 1213 spectra | **31 s** | ~100 s |
| total run | **~8 min** | — |

439 s on 4 cores versus 367 s on 32 is only 20% slower for 8× fewer cores, which
means **the index build is I/O-bound, not CPU-bound**. Parallelising it would
buy close to nothing; if it ever needs to be faster, the target is the parquet
read and the per-row Python loop, not thread count.

Against the 9-hour cap, ~8 minutes leaves room for a far heavier Phase 2
pipeline — candidate-pool retrieval and a fingerprint model included.

## Submission mechanics

This is a **re-run** code competition. Two distinct steps:

1. `kaggle kernels push -p kaggle` uploads the notebook and runs it against the
   *placeholder* test set. This is a smoke test; its `submission.csv` is not
   graded.
2. `kaggle competitions submit <comp> -k <user>/<kernel> -v <version> -m "..."`
   makes Kaggle **re-run the notebook against the hidden test set** and score
   that output.

So the notebook must not assume anything about the placeholder — not its 400
molecules, not its 245–460 m/z range, not its 7 adducts. The real test set spans
157–1159 Da across 10 adducts.

## Gotcha: dataset processing is asynchronous

`kaggle datasets create` returns while the dataset is still being processed. A
kernel pushed immediately afterwards starts before the dataset mounts and fails
with a misleading pip error:

```
WARNING: Location '...' is ignored: it is either a non-existing path or lacks a specific scheme.
ERROR: No matching distribution found for ms_entropy
```

Wait for `kaggle datasets status <ref>` to report `ready` before pushing.
