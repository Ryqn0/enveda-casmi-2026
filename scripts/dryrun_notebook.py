"""Execute the GENERATED Kaggle notebook locally, on a slice of the test set.

    uv run python scripts/dryrun_notebook.py            # 20 molecules
    uv run python scripts/dryrun_notebook.py -n 50

Why: kernel v7 died eight minutes into a Kaggle run on an import that only
fired when a function was CALLED. Syntax checks and `ast.parse` cannot catch
that; running the cells can. This runs the real .ipynb code cells -- not the
source modules -- so it tests exactly what ships, embedding and all.

Only three things are substituted, each because it cannot work locally:
  - the offline `pip install` (uv venvs have no pip; the packages are present)
  - the index BUILD (reuses data/index instead of a 20-minute rebuild)
  - the test set is cut to the first N molecules
Everything else, including path discovery, runs unmodified: a fake
/kaggle/input tree of symlinks makes find_one() behave as it does on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle" / "casmi-library-baseline.ipynb"


def fake_input(tmp: Path) -> Path:
    """Mirror Kaggle's mount layout with symlinks to the local files."""
    comp = tmp / "competitions" / "enveda-casmi26"
    deps = tmp / "datasets" / "ryqn00" / "casmi26-deps"
    comp.mkdir(parents=True)
    deps.mkdir(parents=True)
    for f in ("train.parquet", "test.parquet"):
        (comp / f).symlink_to(ROOT / "data" / "raw" / f)
    for f in (ROOT / "data" / "kaggle_deps").iterdir():
        if f.name != "dataset-metadata.json":
            (deps / f.name).symlink_to(f)
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()

    cells = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"]
             if c["cell_type"] == "code"]

    with tempfile.TemporaryDirectory() as t:
        tmp = fake_input(Path(t))
        out = Path(t) / "submission.csv"
        subs = [
            ('INPUT = Path("/kaggle/input")', f'INPUT = Path("{tmp}")'),
            ("subprocess.run(", "(lambda *a, **k: None)("),
            ('INDEX = Path("/tmp/index")', f'INDEX = Path("{ROOT / "data" / "index"}")'),
            ("build(TRAIN, INDEX, TEST_MZ_RANGE, set(TEST_ADDUCTS))", "pass"),
            ('OUT = Path("/kaggle/working/submission.csv")', f'OUT = Path("{out}")'),
            ('test = pd.read_parquet(COMP / "test.parquet")',
             'test = pd.read_parquet(COMP / "test.parquet")\n'
             f'test = test[test.molecule_id.isin(test.molecule_id.unique()[:{args.n}])]'),
        ]
        for old, new in subs:
            # Each substitution must hit exactly once across the notebook; if
            # it changes shape, fail loudly rather than silently run something
            # else.
            hits = sum(c.count(old) for c in cells)
            assert hits == 1, f"substitution target found {hits}x: {old!r}"
            cells = [c.replace(old, new) for c in cells]

        t0 = time.monotonic()
        # One shared namespace, one compile PER CELL -- exactly what Jupyter
        # does. Concatenating the cells instead breaks on the embedded
        # modules' `from __future__` imports, which are only legal at the top.
        ns = {"__name__": "__main__"}
        for i, cell in enumerate(cells):
            exec(compile(cell, f"{NB.name}[cell {i}]", "exec"), ns)
        print(f"\nDRY RUN OK: {args.n} molecules in {time.monotonic() - t0:.0f}s "
              f"-> {sum(1 for _ in out.open()) - 1} submission rows")


if __name__ == "__main__":
    main()
