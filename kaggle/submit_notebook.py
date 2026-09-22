"""Source for the Kaggle submission notebook.

Kept as a .py so it diffs and lints like code; `scripts/package_kaggle.py`
converts it to .ipynb. Cells are split on `# %%`.

This is a Notebooks-only competition: internet is disabled, the run must finish
in 9 hours, and `test.parquet` is swapped for the hidden test set at re-run. So
nothing here may download anything, and nothing may assume the shape of the
placeholder test file.
"""

# %% [markdown]
# # CASMI 2026 — spectral library baseline
#
# Ranks candidate structures by entropy similarity against a Flash Entropy index
# built from the training spectra.
#
# **What this does well:** competition class 1 — molecules with public reference
# spectra. An exact library match is usually rank 1.
#
# **What this cannot do:** classes 2 and 3 score ~0 by construction. The
# candidate pool here is *training structures only*, and a class-2 molecule is
# defined as having no public spectra — so it is not in the pool at all, and no
# amount of ranking can retrieve it. Reaching class 2 needs a structure database
# (PubChem / COCONUT) as the candidate pool, which is the next phase.

# %%
import subprocess
import sys

# Internet is disabled during scoring, so ms_entropy installs from wheels
# attached as a dataset. --no-index stops pip reaching for PyPI; --find-links
# points it at the local directory. Wheels for cp310-cp313 are included so this
# survives a Kaggle base-image Python bump.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links=/kaggle/input/casmi26-deps/wheels", "ms_entropy",
    ],
    check=True,
)

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The casmi package ships in the same dataset, so the notebook and the local
# repo run identical code rather than a copy that drifts.
sys.path.insert(0, "/kaggle/input/casmi26-deps")
from casmi.library import TEST_ADDUCTS, TEST_MZ_RANGE, Library, build  # noqa: E402

COMP = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
# /tmp, not /kaggle/working: the index is >2GB and working/ is collected as the
# notebook's output.
INDEX = Path("/tmp/index")
OUT = Path("/kaggle/working/submission.csv")
K = 25

# %%
t0 = time.time()
build(COMP / "train.parquet", INDEX, TEST_MZ_RANGE, set(TEST_ADDUCTS))
print(f"index built in {time.time() - t0:.0f}s")

# %%
lib = Library(INDEX)
test = pd.read_parquet(COMP / "test.parquet")
print(f"test: {len(test):,} spectra / {test.molecule_id.nunique():,} molecules")
print(f"library: {len(lib.meta):,} spectra / {lib.n_structures:,} structures")


def peaks_of(row) -> np.ndarray:
    return np.stack(
        [
            np.asarray(row.ms2_mzs, dtype=np.float32),
            np.asarray(row.ms2_normalized_intensities, dtype=np.float32),
        ],
        axis=1,
    )


# %%
t0 = time.time()
rows, no_match = [], 0
for mid, grp in test.groupby("molecule_id", sort=True):
    queries = [(float(r.precursor_mz), peaks_of(r)) for r in grp.itertuples()]
    top, scores = lib.rank_molecule(queries, k=K)
    if scores[0] == 0:
        no_match += 1
    rows.append({"molecule_id": mid, "smiles": ";".join(lib.smiles[top])})

sub = pd.DataFrame(rows)
print(f"ranked {len(sub):,} molecules in {time.time() - t0:.0f}s")
print(f"  molecules with no spectral match: {no_match} ({no_match / len(sub):.1%})")

# %%
# Validate the submission contract before writing. The host rejects a file that
# repeats a molecule_id, contains nulls, or gives more than 25 guesses; failing
# loudly here is far cheaper than a rejected submission.
assert not sub.molecule_id.duplicated().any(), "duplicate molecule_id"
assert sub.smiles.notna().all() and sub.molecule_id.notna().all(), "null value"
counts = sub.smiles.str.count(";") + 1
assert counts.max() <= K, f"{counts.max()} guesses for one molecule, max {K}"
assert counts.min() >= 1, "a molecule has no guesses"
assert set(sub.molecule_id) == set(test.molecule_id), "molecule_id set mismatch"

sub.to_csv(OUT, index=False)
print(f"wrote {OUT} — {len(sub):,} rows, {counts.min()}-{counts.max()} guesses each")
print(sub.head(3).to_string())
