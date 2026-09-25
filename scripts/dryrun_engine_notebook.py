"""Execute an ENGINE-style notebook locally on a slice of the test set.

    uv run python scripts/dryrun_engine_notebook.py kaggle/engine_plus_own/casmi26-engine-plus-own.ipynb -n 20

Like dryrun_notebook.py (which runs our own notebook), but for notebooks built
on the public engine: its modules arrive as `%%writefile` cells and it reads
seven attached datasets. So this script
  - builds a fake /kaggle/input tree of symlinks to the local copies,
  - writes %%writefile cells to disk instead of executing them (as Jupyter does),
  - execs every other cell in ONE shared namespace, one compile per cell,
  - cuts test.parquet and sample_submission.csv to the first N molecules
    (the engine asserts 25 guesses for every sample row, so both must shrink).
Substituted, because they cannot work offline: pip installs (the packages are
present) and our index build (reuses data/index).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "public"
DATASETS = {   # mount path under /kaggle/input/datasets -> local files
    "prvsiyan/casmi26-fp-models-v2": sorted((PUB / "engine").glob("fp_*.pt")),
    "prvsiyan/casmi26-ranker-features": [PUB / "engine/rank_train.npz"],
    "prvsiyan/coconut-casmi26-candidates": [PUB / f"engine/{f}" for f in
                                            ("coco_fp.npy", "coco_mass.npy", "coco_meta.pkl", "fp_bits.npy")],
    "megayak/casmi26-simulated-ranker-rows": [PUB / "sim_rank_rows_nofp.npz", PUB / "sim_rank_rows_fppair.npz"],
    "ryqn00/casmi26-deps": [p for p in (ROOT / "data/kaggle_deps").iterdir()
                            if p.name != "dataset-metadata.json"],
}


def fake_input(root: Path, n: int) -> None:
    comp = root / "competitions" / "enveda-casmi26"
    comp.mkdir(parents=True)
    (comp / "train.parquet").symlink_to(ROOT / "data/raw/train.parquet")
    te = pd.read_parquet(ROOT / "data/raw/test.parquet")
    keep = sorted(te.molecule_id.unique())[:n]
    te[te.molecule_id.isin(keep)].to_parquet(comp / "test.parquet")
    s = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    s[s.molecule_id.isin(keep)].to_csv(comp / "sample_submission.csv", index=False)
    for mount, files in DATASETS.items():
        d = root / "datasets" / mount
        d.mkdir(parents=True)
        for f in files:
            assert f.exists(), f"missing local copy: {f}"
            (d / f.name).symlink_to(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()
    cells = [c for c in json.loads(args.notebook.read_text())["cells"] if c["cell_type"] == "code"]

    with tempfile.TemporaryDirectory() as t:
        inp, work = Path(t) / "input", Path(t) / "working"
        work.mkdir()
        fake_input(inp, args.n)
        subs = [  # (old, new, must appear exactly once?)
            ("/kaggle/input", str(inp), False),
            ("/kaggle/working", str(work), False),
            ("subprocess.run(", "(lambda *a, **k: None)(", False),
            ('OWN_INDEX = Path("/tmp/own_index")', f'OWN_INDEX = Path("{ROOT / "data/index"}")', True),
            ('build(Path(COMP) / "train.parquet", OWN_INDEX, TEST_MZ_RANGE, set(TEST_ADDUCTS))', "pass", True),
        ]
        srcs = ["".join(c["source"]) for c in cells]
        for old, new, once in subs:
            hits = sum(s.count(old) for s in srcs)
            assert hits >= 1 and (hits == 1 or not once), f"{old!r} found {hits}x"
            srcs = [s.replace(old, new) for s in srcs]

        t0 = time.monotonic()
        ns = {"__name__": "__main__"}
        for i, s in enumerate(srcs):
            if s.startswith("%%writefile "):
                head, body = s.split("\n", 1) if "\n" in s else (s, "")
                # Jupyter refuses an empty body ("UsageError: ... the cell body
                # is empty"); a dry run that allowed it passed a notebook that
                # then died on Kaggle.
                assert body.strip(), f"code cell {i}: {head} has an empty body (Jupyter rejects it)"
                # Relative to the notebook's working directory (cell 6 chdirs there).
                p = Path(head.split()[1])
                p = p if p.is_absolute() else Path.cwd() / p
                p.write_text(body)
                continue
            exec(compile(s, f"{args.notebook.name}[code cell {i}]", "exec"), ns)
        import casmi
        print(f"\n(casmi imported from {casmi.__file__})")
        out = work / "submission.csv"
        sub = pd.read_csv(out)
        assert len(sub) == args.n and sub.smiles.str.count(";").eq(24).all()
        print(f"DRY RUN OK: {args.n} molecules in {(time.monotonic() - t0) / 60:.1f} min -> {out.name}")


if __name__ == "__main__":
    main()
