"""Source for the Kaggle submission notebook.

Kept as .py so it diffs and lints like code; `scripts/package_kaggle.py` turns it
into .ipynb. Cells split on `# %%`. The line `# %%EMBED <path>` is replaced with
the full contents of that file (with its `from casmi.` imports stripped, since
every embedded module shares one namespace), so the notebook ships the real
library code rather than a copy that could drift.
"""

# %% [markdown]
# # CASMI 2026 — analog propagation over a candidate pool
#
# **Goal:** given tandem mass spectra of an unknown molecule, output 25 candidate
# structures as SMILES, best guess first.
#
# ## Why this version exists
#
# The previous submission scored **0.130** using spectral library search. That
# approach can only return a structure that already has a reference spectrum,
# and the score told us exactly how rare that is:
#
#     P(class 1) = LB / class-1 MRR = 0.130 / 0.845 = 15.4%
#
# So **~85% of the test set was unreachable**. This version targets it.
#
# ## The three difficulty classes
#
# | Class | Meaning | Reachable how |
# |---|---|---|
# | 1 | Public reference spectra exist | library search |
# | 2 | Structure known (PubChem/COCONUT), no spectra | **candidate pool + analogy** |
# | 3 | Structure unknown to chemistry | neither — in no database |
#
# ## The method
#
# 1. **Candidate pool.** Subtract the adduct from the measured precursor to get
#    the neutral mass, then take every structure of that mass from COCONUT plus
#    the training set. ~34–70 candidates, and the answer is among them 95% of
#    the time.
# 2. **Find an analog.** `hybrid_search` matches spectra that differ by a
#    *constant mass shift* — the signature of a chemical modification. Add a
#    methyl group and every fragment containing it moves by the same 14 Da. This
#    finds a known relative even when the molecule itself was never measured.
# 3. **Rank by chemical resemblance.** Score each candidate by Tanimoto
#    similarity to the analog, weighted by how well the analog matched.
#
# The unknown is never identified directly. It is located by triangulation from
# a known neighbour.
#
# ## Measured on a 196-molecule natural-product holdout
#
# | strategy | class 1 | class 2 |
# |---|---:|---:|
# | identity only (previous submission) | 0.8765 | 0.0921 |
# | linear blend w=4 | 0.8994 | 0.5320 |
# | gated t=0.7 | 0.8986 | 0.5977 |
# | **analog only (this notebook)** | 0.8955 | **0.6207** |
#
# Ranking is analog-only because **hybrid search subsumes identity search**: a
# zero-mass-shift hybrid match *is* an identity match, with self-Tanimoto 1.0.
# The identity channel adds nothing on class 1 and actively harms class 2, where
# the true answer has no spectrum and scores 0 while wrong candidates that happen
# to be in the library score above it.

# %% [markdown]
# ## Step 1 — install dependencies offline
#
# **On `!pip install`:** it works while you are *editing* a notebook, because
# internet can be switched on in the sidebar. But this competition scores with
# **internet disabled**, so `pip` cannot reach PyPI and the install would fail
# mid-run. Dependencies must therefore be shipped as files.
#
# RDKit is pinned to **2026.3.3**, the version the host scores with. The metric
# notebook states the pin "decides which submissions match; it is not a
# convenience" — tautomer canonicalization changes between releases, so a
# different version can produce a correct answer that fails to match.
#
# Paths are **discovered, not hardcoded**: inputs mount at
# `/kaggle/input/datasets/<user>/<name>` and `/kaggle/input/competitions/<slug>`,
# both deeper than the commonly-cited `/kaggle/input/<name>`.

# %%
import subprocess  # runs another program and waits for it to finish
import sys         # sys.executable is the path of the running Python
from pathlib import Path

INPUT = Path("/kaggle/input")

# Log what is actually mounted before assuming anything. If a later cell fails,
# this is the first thing to read.
print("mounted under /kaggle/input:")
for p in sorted(INPUT.glob("*")):
    print("  ", p.name)

# Locate the wheels wherever the dataset landed. rglob searches subdirectories.
wheel_dirs = sorted({p.parent for p in INPUT.rglob("*.whl")})
assert wheel_dirs, (
    "no .whl under /kaggle/input — is the casmi26-deps dataset attached, and had "
    "it finished processing before this notebook was run?"
)
WHEELS = wheel_dirs[0]
print("wheels:", WHEELS)

subprocess.run(
    [
        sys.executable,              # this interpreter, not whatever is on PATH
        "-m", "pip", "install",
        "--no-index",                # never contact PyPI
        f"--find-links={WHEELS}",    # resolve from the local directory instead
        # RDKit's only real needs are numpy and Pillow, both already present.
        # Without --no-deps the resolver would hunt for them and, with
        # --no-index, find nowhere to look.
        "--no-deps",
        "ms_entropy", "rdkit",
    ],
    check=True,  # fail here with pip's message, not later with an ImportError
)

# %% [markdown]
# ## Step 2 — the candidate pool
#
# Converts a measured ion back to the neutral mass of the molecule, then looks
# that mass up in a sorted table of 722,737 structures (COCONUT + training).

# %%EMBED src/casmi/pool.py

# %% [markdown]
# ## Step 3 — chemical similarity
#
# Morgan fingerprints turn a structure into a bit vector of its substructures;
# Tanimoto measures overlap. This is what "chemically resembles" means
# numerically.

# %%EMBED src/casmi/analog.py

# %% [markdown]
# ## Step 4 — the spectral library
#
# Builds a Flash Entropy index over the training spectra and exposes both
# searches: `score_spectrum` (identity) and `analog_structures` (hybrid).

# %%EMBED src/casmi/library.py

# %% [markdown]
# ## Step 5 — the fingerprint model
#
# A set transformer that reads peaks and predicts which substructures the
# molecule contains. Unlike analogy it needs no relative in the library, which
# is why it wins where analogy fails: on a 290-molecule natural-product
# holdout, MRR 0.4246 -> 0.6307 on molecules whose nearest spectral cousin is
# chemically distant.
#
# Trained locally on an RTX 4080: 800k spectra, 224,744 structures, 20 epochs,
# 5.5M parameters. The weights ship in the deps dataset.

# %%EMBED src/casmi/fpnet.py

# %% [markdown]
# ## Step 6 — the ranker
#
# Ties everything together: pool for candidates, the model for substructure
# evidence, hybrid search for analogy.
#
# The blend is `fpnet + 0.25*analog`, both z-scored. Swept on the hard bucket:
#
# | strategy | all | hard |
# |---|---:|---:|
# | analog alone | 0.6427 | 0.4246 |
# | fpnet alone | 0.6946 | 0.6307 |
# | **fpnet + 0.25*analog** | **0.7424** | **0.6567** |
# | fpnet + 1.0*analog | 0.7105 | 0.5489 |
#
# An equal weight drowns the model exactly where the model is the only thing
# that works.

# %%EMBED src/casmi/rank.py

# %% [markdown]
# ## Step 7 — build the index
#
# ~7 minutes and ~10 GB on Kaggle's 4 vCPUs, against a 9-hour cap. Written to
# `/tmp`, not `/kaggle/working`, because it exceeds 2 GB and `working/` is
# collected as the notebook's output.

# %%
import time

import numpy as np
import pandas as pd

def find_one(name: str) -> Path:
    """Locate a required input by filename, or fail with a usable message.

    `next(INPUT.rglob(name))` raises a bare StopIteration that says nothing
    about what went wrong. v6 died that way: the kernel had mounted the
    PREVIOUS version of the deps dataset, which predated fpnet.pt. The dataset
    reported `ready` before the push, so `ready` is not sufficient — the
    version takes longer to reach the kernel runtime.

    Listing what IS present turns a one-word exception into a diagnosis.
    """
    hits = sorted(INPUT.rglob(name))
    if not hits:
        have = sorted(p.name for p in INPUT.rglob("*") if p.is_file())[:40]
        raise FileNotFoundError(
            f"{name!r} not found under /kaggle/input. If this is a dataset file "
            f"just uploaded, the kernel may have mounted an older version — "
            f"re-push after the new version propagates. Present: {have}"
        )
    return hits[0]


# Find inputs by filename rather than assuming the mount path.
TRAIN = find_one("train.parquet")
COMP = TRAIN.parent
POOL = find_one("pool.parquet")
FPNET = find_one("fpnet.pt")
print("competition data: ", COMP)
print("candidate pool:   ", POOL)
print("fingerprint model:", FPNET)

INDEX = Path("/tmp/index")
OUT = Path("/kaggle/working/submission.csv")
K = 25

t0 = time.time()
build(TRAIN, INDEX, TEST_MZ_RANGE, set(TEST_ADDUCTS))
print(f"index built in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Step 8 — rank every test molecule
#
# A molecule may have been measured several times at different collision
# energies. Scoring is per molecule, so `rank` fuses them, keeping each
# structure's best analog score across the measurements.

# %%
import torch

lib = Library(INDEX)
pool = CandidatePool(POOL)

# torch ships in the Kaggle base image, so no wheel is needed for it.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
fpnet, fpnet_ck = load(FPNET, device=DEV)
print(f"fpnet: {fpnet_ck['cfg']['n_bits']} bits, "
      f"val F1 {fpnet_ck.get('val_f1', float('nan')):.3f}, running on {DEV}")
ranker = CombinedRanker(lib, pool, fpnet=fpnet, fpnet_ck=fpnet_ck, device=DEV)

test = pd.read_parquet(COMP / "test.parquet")
print(f"test:    {len(test):,} spectra / {test.molecule_id.nunique():,} molecules")
print(f"library: {len(lib.meta):,} spectra / {lib.n_structures:,} structures")
print(f"pool:    {len(pool):,} structures")


def peaks_of(row) -> np.ndarray:
    """One test row's two parallel arrays as the (N, 2) shape ms_entropy wants."""
    return np.stack(
        [
            np.asarray(row.ms2_mzs, dtype=np.float32),
            np.asarray(row.ms2_normalized_intensities, dtype=np.float32),
        ],
        axis=1,
    )


# %%
t0 = time.time()
rows, no_cand, short = [], 0, 0

# sort=True keeps the output order deterministic between runs.
for mid, grp in test.groupby("molecule_id", sort=True):
    spectra = [(float(r.precursor_mz), peaks_of(r)) for r in grp.itertuples()]
    adduct = grp.adduct.iloc[0]

    cand, scores = ranker.rank(spectra, adduct, k=K)

    if len(cand) == 0:
        # No candidate of the right mass — an unrecognised adduct, or a mass
        # outside the pool. Never leave the row empty: a wrong guess costs
        # nothing beyond the rank it occupies, and an empty field is rejected.
        no_cand += 1
        smiles = list(pool.smiles[:K])
    else:
        smiles = list(pool.smiles[cand])

    if len(smiles) < K:
        short += 1
    rows.append({"molecule_id": mid, "smiles": ";".join(smiles)})

sub = pd.DataFrame(rows)
print(f"ranked {len(sub):,} molecules in {time.time() - t0:.0f}s")
print(f"  no mass-matched candidates: {no_cand} ({no_cand / len(sub):.1%})")
print(f"  fewer than {K} candidates:  {short} ({short / len(sub):.1%})")

# %% [markdown]
# ## Step 9 — validate the submission contract
#
# The host rejects a file that repeats a `molecule_id`, contains nulls, or gives
# more than 25 guesses. A rejection costs a submission slot and a day, so assert
# every rule before writing.

# %%
assert not sub.molecule_id.duplicated().any(), "duplicate molecule_id"
assert sub.smiles.notna().all() and sub.molecule_id.notna().all(), "null value"

counts = sub.smiles.str.count(";") + 1   # guesses = separators + 1
assert counts.max() <= K, f"{counts.max()} guesses for one molecule, max {K}"
assert counts.min() >= 1, "a molecule has no guesses"
assert set(sub.molecule_id) == set(test.molecule_id), "molecule_id set mismatch"

# index=False stops pandas writing row numbers as a third, unnamed column.
sub.to_csv(OUT, index=False)

print(f"wrote {OUT}")
print(f"  {len(sub):,} rows, {counts.min()}-{counts.max()} guesses each")
print(sub.head(3).to_string())
