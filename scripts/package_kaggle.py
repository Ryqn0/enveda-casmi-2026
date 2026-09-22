"""Package everything Kaggle needs: a deps dataset and the submission notebook.

This is a Notebooks-only competition with internet disabled, so every
dependency must arrive as an attached dataset. Rather than pasting the library
code into the notebook (where it would drift from the repo), we ship the
`casmi` package itself alongside the wheels and import it.

    uv run python scripts/package_kaggle.py

Then push, in order (the notebook depends on the dataset existing):

    kaggle datasets create -p data/kaggle_deps --dir-mode zip   # first time
    kaggle datasets version -p data/kaggle_deps -m "..."        # updates
    kaggle kernels push -p kaggle
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "ryqn00"
DATASET = f"{USER}/casmi26-deps"
KERNEL = f"{USER}/casmi-library-baseline"

STAGE = ROOT / "data" / "kaggle_deps"
NOTEBOOK = ROOT / "kaggle" / "casmi-library-baseline.ipynb"
SOURCE = ROOT / "kaggle" / "submit_notebook.py"


def build_dataset() -> None:
    """Stage wheels + the casmi package into one attachable dataset."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "wheels").mkdir(parents=True)

    wheels = sorted((ROOT / "data" / "wheels").glob("*.whl"))
    if not wheels:
        raise SystemExit("no wheels in data/wheels — see scripts/fetch_wheels")
    for w in wheels:
        shutil.copy2(w, STAGE / "wheels" / w.name)

    # Only the runtime modules. official_metric is scoring-side, not needed here.
    pkg = STAGE / "casmi"
    pkg.mkdir()
    for name in ("__init__.py", "library.py"):
        shutil.copy2(ROOT / "src" / "casmi" / name, pkg / name)

    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "casmi26-deps", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        )
    )
    print(f"staged {len(wheels)} wheels + casmi package -> {STAGE.relative_to(ROOT)}")


def to_notebook() -> None:
    """Convert the `# %%`-delimited .py source into an .ipynb."""
    text = SOURCE.read_text()
    # Drop the module docstring: it documents the source file, not the notebook.
    body = text.split('"""', 2)[-1].lstrip("\n")

    cells = []
    for chunk in body.split("\n# %%"):
        chunk = chunk.lstrip()
        if chunk.startswith("# %%"):
            chunk = chunk[4:].lstrip()
        is_md = chunk.startswith("[markdown]")
        if is_md:
            chunk = chunk[len("[markdown]") :].lstrip("\n")
            # strip the leading comment marker from every markdown line
            chunk = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in chunk.splitlines()
            )
        src = chunk.strip("\n")
        if not src:
            continue
        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
            if is_md
            else {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(True),
            }
        )

    NOTEBOOK.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
        )
    )

    (ROOT / "kaggle" / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": KERNEL,
                "title": "CASMI Library Baseline",
                "code_file": NOTEBOOK.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                # Must stay false: the competition requires internet disabled.
                "enable_internet": False,
                "dataset_sources": [DATASET],
                "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
                "kernel_sources": [],
            },
            indent=2,
        )
    )
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    print(f"wrote {NOTEBOOK.relative_to(ROOT)} — {len(cells)} cells "
          f"({n_md} markdown, {len(cells) - n_md} code)")


if __name__ == "__main__":
    build_dataset()
    to_notebook()
    print("\nnext:")
    print(f"  kaggle datasets create -p {STAGE.relative_to(ROOT)} --dir-mode zip")
    print("  kaggle kernels push -p kaggle")
