"""Package everything Kaggle needs: a deps dataset and the submission notebook.

This is a Notebooks-only competition with internet disabled, so third-party
packages must arrive as an attached dataset — hence the wheels. The library code
does NOT ship that way: a `# %%EMBED <path>` marker expands into the notebook, so
the notebook is self-contained and readable while `src/casmi/library.py` remains
the single source of truth.

Usage — always push with --push, which regenerates first. Pushing separately is
how v3 shipped a notebook that predated the fix it was meant to contain:

    uv run kaggle datasets version -p data/kaggle_deps -m "..."   # if deps changed
    uv run python scripts/package_kaggle.py --push

The dataset must exist and report `ready` before the kernel is pushed; a kernel
that starts while the dataset is still processing fails with a misleading pip
error. First time only:

    uv run kaggle datasets create -p data/kaggle_deps
"""

from __future__ import annotations

import json
import subprocess
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
    """Stage the offline wheels into an attachable dataset.

    Only wheels: the library code is embedded directly into the notebook, so
    there is no package to ship and nothing that can drift out of step with it.
    """
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    wheels = sorted((ROOT / "data" / "wheels").glob("*.whl"))
    if not wheels:
        raise SystemExit("no wheels in data/wheels — re-download them first")
    # Flat, at the dataset root. A subdirectory would need `--dir-mode zip` on
    # upload, which turns it into wheels.zip on Kaggle and breaks --find-links.
    for w in wheels:
        shutil.copy2(w, STAGE / w.name)

    # The candidate pool ships pre-built: rebuilding it on Kaggle would mean
    # uploading the 577 MB COCONUT dump to save 34 MB. Reproducible from source
    # with scripts/build_pool.py.
    pool = ROOT / "data" / "pool.parquet"
    if not pool.exists():
        raise SystemExit("data/pool.parquet missing — run scripts/build_pool.py")
    shutil.copy2(pool, STAGE / "pool.parquet")

    # Trained fingerprint model. torch is already in the Kaggle base image, so
    # only the weights need shipping.
    ckpt = ROOT / "data" / "fpnet" / "fpnet.pt"
    if not ckpt.exists():
        raise SystemExit("data/fpnet/fpnet.pt missing — run scripts/train_fpnet.py")
    shutil.copy2(ckpt, STAGE / "fpnet.pt")

    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "casmi26-deps", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        )
    )
    print(f"staged {len(wheels)} wheels -> {STAGE.relative_to(ROOT)}")


def to_notebook() -> None:
    """Convert the `# %%`-delimited .py source into an .ipynb."""
    text = SOURCE.read_text()
    # Drop the module docstring: it documents the source file, not the notebook.
    body = text.split('"""', 2)[-1].lstrip("\n")

    cells = []
    for chunk in body.split("\n# %%"):
        chunk = chunk.lstrip()

        # The very first cell is not preceded by a newline, so split() left its
        # "# %%" marker attached. Strip it the same way split removed the others.
        if chunk.startswith("# %%"):
            chunk = chunk[4:].lstrip()

        # `# %%EMBED <path>` becomes a code cell holding that file verbatim, so
        # the notebook carries the real implementation instead of importing a
        # second copy that could fall out of step with the repo.
        if chunk.startswith("EMBED "):
            target = ROOT / chunk.split(maxsplit=1)[1].strip().splitlines()[0]
            text = target.read_text().strip("\n")
            # Embedded modules share one notebook namespace, so their
            # `from casmi.x import y` lines refer to a package that does not
            # exist there — and the symbols are already defined by an earlier
            # cell. Drop those imports; embed order supplies the dependency.
            text = "\n".join(
                ln for ln in text.splitlines() if not ln.startswith("from casmi.")
            )
            chunk = text
            print(f"embedded {target.relative_to(ROOT)} ({len(chunk.splitlines())} lines)")

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


def push() -> None:
    """Regenerate then push, in one step.

    v3 was pushed from a notebook generated 45 seconds BEFORE the source change
    it was supposed to contain, so the fix shipped was the one from the previous
    edit. Nothing detected it — the notebook was valid, it just held stale code,
    and a 17-minute Kaggle run was spent proving it.

    Separating "generate" from "push" makes that failure available every single
    time. Doing both here removes the gap rather than relying on remembering.
    """
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(ROOT / "kaggle")], check=True
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true",
                    help="regenerate and push the kernel in one step")
    args = ap.parse_args()

    build_dataset()
    to_notebook()

    if args.push:
        push()
    else:
        print("\nnext:")
        print(f"  kaggle datasets version -p {STAGE.relative_to(ROOT)} -m '...'")
        print("  uv run python scripts/package_kaggle.py --push")
