"""Source for the Kaggle submission notebook.

Kept as .py so it diffs and lints like code; `scripts/package_kaggle.py` turns it
into .ipynb. Cells split on `# %%`. The line `# %%EMBED <path>` is replaced with
the full contents of that file, so the notebook ships the real library code
rather than importing a copy that could drift.
"""

# %% [markdown]
# # CASMI 2026 — spectral library baseline
#
# **Goal:** given tandem mass spectra of an unknown molecule, output 25 candidate
# structures as SMILES, best guess first.
#
# ## How this notebook works, end to end
#
# 1. **Install `ms_entropy`** from wheels attached as a dataset (internet is off).
# 2. **Build a searchable index** of ~2.1M training spectra.
# 3. **For each test molecule**, score every known structure against its spectra.
# 4. **Write `submission.csv`** — 25 SMILES per molecule, semicolon-separated.
#
# ## The core idea
#
# A mass spectrometer smashes a molecule and weighs the pieces. The result is a
# list of `(fragment mass, height)` pairs — a fingerprint of how the molecule
# breaks apart. Two measurements of the *same* molecule produce similar lists.
#
# So: find the training spectrum our unknown most resembles, and return that
# molecule's structure. That is all "library search" means.
#
# ## What this can and cannot do
#
# The competition sorts test molecules into three classes:
#
# | Class | Meaning | This notebook |
# |---|---|---|
# | 1 | Public reference spectra exist | **Works well** — local MRR 0.845 |
# | 2 | Structure known, but no public spectra | **Scores 0** |
# | 3 | Structure unknown to chemistry | **Scores 0** |
#
# Classes 2 and 3 score zero *by construction*, not because the similarity
# function is weak. This notebook can only return structures that are already in
# the training set, and those classes are defined by being absent from it.
# Reaching class 2 needs a structure **database** (PubChem/COCONUT) as the
# candidate pool — that is the next phase, not a tuning problem.

# %% [markdown]
# ## Step 1 — install the dependency, offline
#
# **On `!pip install`:** it does work in a Kaggle notebook *while you are
# editing*, because you can switch internet on in the sidebar. But this
# competition requires **internet disabled** for submission, and with the network
# off `pip` cannot reach PyPI — `!pip install ms_entropy` would fail during
# scoring, after the run had already burned its time.
#
# The fix is to ship the package as a file and point pip at it. This is exactly
# what the competition's own metric notebook does for RDKit.
#
# One more detail: we call pip through `sys.executable` rather than `!pip`.
# `!pip` runs whichever pip is first on `PATH`, which is not guaranteed to belong
# to the Python actually running the notebook. `sys.executable -m pip` installs
# into *this* interpreter, always.

# %%
import subprocess  # lets Python run another program and wait for it to finish
import sys         # sys.executable is the full path to the running Python

subprocess.run(
    [
        sys.executable,      # this exact Python, not whatever is on PATH
        "-m", "pip",         # "-m" runs a module as a script: python -m pip
        "install",
        # --no-index: never contact PyPI. Without it pip would try the network,
        # hang, and fail once internet is disabled.
        "--no-index",
        # --find-links: look for packages in this local folder instead. It holds
        # wheels for Python 3.10-3.13, so this survives Kaggle upgrading its
        # base image.
        "--find-links=/kaggle/input/casmi26-deps",
        "ms_entropy",        # the package we want
    ],
    check=True,  # raise immediately if pip fails, rather than failing later
                 # with a confusing ImportError
)

# %% [markdown]
# ## Step 2 — the library code
#
# The whole pipeline is below, commented line by line. It is embedded rather than
# imported so this notebook is self-contained and readable on its own.
#
# The three things worth understanding before you read it:
#
# - **Entropy similarity** compares two peak lists and returns 0–1. It weights
#   peaks by how informative they are, so a handful of distinctive fragments
#   count for more than many common ones.
# - **The index sorts spectra by precursor mass**, so a query only gets compared
#   against spectra that could plausibly be the same molecule. This is what makes
#   searching 2.1M spectra fast enough to be practical.
# - **Scores collapse from spectra to structures.** A famous molecule may have
#   hundreds of library spectra; it is judged by its single best match, not its
#   average, so being well-studied is not a penalty.

# %%EMBED src/casmi/library.py

# %% [markdown]
# ## Step 3 — build the index
#
# This reads `train.parquet` in 50,000-row batches, keeps the rows that could
# match a test molecule, and builds the search structure.
#
# Expect roughly **6 minutes and ~10 GB of RAM** locally. Kaggle gives 4 CPUs
# (versus 32 here) so it will be slower, but the budget is 9 hours — there is
# ample headroom.
#
# The index is written to `/tmp`, **not** `/kaggle/working`: it is over 2 GB, and
# everything in `working/` is collected and stored as the notebook's output.

# %%
import time                 # to report how long each stage took
from pathlib import Path    # Path objects handle "/" joining across platforms

import numpy as np
import pandas as pd

# Kaggle mounts competition data read-only at this fixed location.
COMP = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
INDEX = Path("/tmp/index")                        # scratch space, not collected
OUT = Path("/kaggle/working/submission.csv")      # the graded file
K = 25                                            # candidates per molecule

t0 = time.time()
build(
    COMP / "train.parquet",
    INDEX,
    TEST_MZ_RANGE,        # (130.0, 1220.0) — the real test mass window
    set(TEST_ADDUCTS),    # the 10 adducts the real test set uses
)
print(f"index built in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Step 4 — rank candidates for every test molecule
#
# A molecule may have been measured several times at different collision
# energies. The competition scores **per molecule**, so those spectra must be
# fused into one ranked list — `rank_molecule` averages their scores.

# %%
lib = Library(INDEX)                        # load the index we just wrote
test = pd.read_parquet(COMP / "test.parquet")

print(f"test:    {len(test):,} spectra / {test.molecule_id.nunique():,} molecules")
print(f"library: {len(lib.meta):,} spectra / {lib.n_structures:,} structures")


def peaks_of(row) -> np.ndarray:
    """Turn one test row's two parallel arrays into the (N, 2) shape we need."""
    return np.stack(
        [
            np.asarray(row.ms2_mzs, dtype=np.float32),
            np.asarray(row.ms2_normalized_intensities, dtype=np.float32),
        ],
        axis=1,  # stack side by side as columns: [[mz, intensity], ...]
    )


# %%
t0 = time.time()
rows = []       # one entry per molecule, becomes the submission
no_match = 0    # molecules where nothing in the library resembled the query

# groupby gathers all spectra sharing a molecule_id. sort=True makes the output
# order deterministic, so two runs produce identical files.
for mid, grp in test.groupby("molecule_id", sort=True):
    # Every measurement of this molecule, as (precursor mass, peaks).
    queries = [(float(r.precursor_mz), peaks_of(r)) for r in grp.itertuples()]

    # top = indices of the 25 best structures; scores = their fused similarities.
    top, scores = lib.rank_molecule(queries, k=K)

    # A top score of exactly 0 means no library spectrum shared any peaks — very
    # likely a class 2 or 3 molecule. Counting these tells us how much of the
    # test set is out of this approach's reach.
    if scores[0] == 0:
        no_match += 1

    # The submission format joins candidates with ";" into one field.
    rows.append({"molecule_id": mid, "smiles": ";".join(lib.smiles[top])})

sub = pd.DataFrame(rows)
print(f"ranked {len(sub):,} molecules in {time.time() - t0:.0f}s")
print(f"  no spectral match at all: {no_match} ({no_match / len(sub):.1%})")

# %% [markdown]
# ## Step 5 — check the submission before writing it
#
# The host rejects a file that repeats a `molecule_id`, contains nulls, or gives
# more than 25 guesses. A rejection costs a submission slot and a day, so every
# rule is asserted here first — failing loudly now is much cheaper.

# %%
# Each molecule must appear exactly once.
assert not sub.molecule_id.duplicated().any(), "duplicate molecule_id"

# Neither column may contain nulls.
assert sub.smiles.notna().all() and sub.molecule_id.notna().all(), "null value"

# Guess count = number of ";" separators + 1.
counts = sub.smiles.str.count(";") + 1
assert counts.max() <= K, f"{counts.max()} guesses for one molecule, max {K}"
assert counts.min() >= 1, "a molecule has no guesses"

# Every test molecule predicted, and nothing invented.
assert set(sub.molecule_id) == set(test.molecule_id), "molecule_id set mismatch"

# index=False stops pandas writing its row numbers as an extra unnamed column,
# which would break the expected two-column format.
sub.to_csv(OUT, index=False)

print(f"wrote {OUT}")
print(f"  {len(sub):,} rows, {counts.min()}-{counts.max()} guesses each")
print(sub.head(3).to_string())
