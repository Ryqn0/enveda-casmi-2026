"""Source for the Kaggle submission notebook.

Kept as .py so it diffs and lints like code; `scripts/package_kaggle.py` turns it
into .ipynb. Cells split on `# %%`. The line `# %%EMBED <path>` is replaced with
the full contents of that file (with its `from casmi.` imports stripped, since
every embedded module shares one namespace), so the notebook ships the real
library code rather than a copy that could drift.
"""

# %% [markdown]
# # CASMI 2026 — three evidence channels over a candidate pool
#
# **Goal:** given tandem mass spectra of an unknown molecule, output 25 candidate
# structures as SMILES, best guess first. Scored by MRR@25 on InChIKey
# connectivity (the first 14 characters).
#
# ## The three difficulty classes
#
# | Class | Meaning | What can reach it |
# |---|---|---|
# | 1 | Public reference spectra exist | library / analog search |
# | 2 | Structure known (PubChem/COCONUT), no spectra | candidate pool + any channel |
# | 3 | Structure unknown to chemistry | only channels needing no relative |
#
# Class 1 is ~15% of the test set, so ~85% of the score lives in classes 2-3.
#
# ## The method
#
# 1. **Candidate pool.** Subtract the adduct from the measured precursor to get
#    the neutral mass, then take every structure of that mass (within 4 ppm)
#    from COCONUT plus the training set. For classes 1-2 the answer is in the
#    pool essentially every time -- so **ranking, not retrieval, is the problem**.
# 2. **Rank each candidate with three independent channels:**
#    - **analog** -- `hybrid_search` finds library spectra that differ from the
#      query by a constant mass shift (the signature of a chemical
#      modification), then scores candidates by Tanimoto to those relatives.
#    - **fpnet** -- a set transformer predicts which substructures the molecule
#      contains; each candidate is scored by how well its fingerprint fits.
#    - **fragment** -- break each candidate's bonds and measure how much of the
#      observed peak intensity its own pieces explain. Needs no library at all.
# 3. **Blend** the z-scored channels: `fpnet + 0.5 * analog`. The fragment
#    channel and a learned ranker are included but switched off -- see Step 7.
#
# ## Leaderboard history
#
# | version | method | LB |
# |---|---|---:|
# | 1 | spectral library search | 0.130 |
# | 2 | + candidate pool + analog propagation | 0.240 |
# | 3 | + fingerprint model (buggy inputs, weight 0.25) | 0.196 |
# | 3b | fixed inputs, weight 0.5 | 0.244 |
# | 4 | engineered features, weight 0.25 | 0.228 |
# | 5 | v4c model + fragmentation + learned ranker | 0.204 |
# | 6 | v4c model, same blend as version 3b | 0.234 |
# | 7 | **this notebook** -- learned ranker over raw features, fpnet v5 (Step 7b) | ? |
#
# Versions 3, 4 and 5 each looked better offline and scored worse. Two causes,
# both in the VALIDATION, not the model: the holdout overlapped the model's
# training data, and -- the bigger one -- it was built from a synthetic
# screening library instead of natural products, which is what the test looks
# like. Version 6 changes exactly one thing relative to the best submission so
# far (3b, 0.244): the fingerprint model. Step 7 explains.

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
# molecule contains (1,583 Morgan fingerprint bits). Each candidate is then
# scored by how well its own fingerprint fits that prediction.
#
# This checkpoint ("v4c") differs from earlier versions in two ways, each
# measured on 600 molecules no model version ever saw:
#
# | change | MRR | paired gain |
# |---|---:|---|
# | v3: engineered peak features | 0.8554 | -- |
# | v4: at most 4 spectra per structure | 0.8689 | +0.0135 +/- 0.0081 |
# | v4c: + contrastive fine-tune | 0.8745 | +0.0056 +/- 0.0064 (likely, not proven) |
#
# - **Data diversity.** Sampling spectra uniformly let a few heavily-measured
#   molecules (one had 1,438 spectra) dominate training, which taught
#   memorisation. Capping at 4 per structure covers all 268,969 structures for
#   the same compute.
# - **Contrastive fine-tune.** Bit prediction is only indirectly the task. The
#   real task is picking the right structure among ~60 same-mass isomers, so
#   the model was fine-tuned on exactly that, with those isomers as negatives.
#
# 5.5M parameters, trained locally on an RTX 4080. Weights ship in the deps
# dataset.

# %%EMBED src/casmi/fpnet.py

# %% [markdown]
# ## Step 6 — in-silico fragmentation
#
# Breaks each candidate's bonds and asks what fraction of the observed peak
# intensity its own pieces can explain. The other two channels both need a
# spectral relative in the library; this one needs nothing but the candidate
# structure, so it is the only signal that still works for class-3 molecules
# that exist in no database.
#
# Weak on its own (MRR 0.41) but independent of the other two, which is what
# makes it worth adding.

# %%EMBED src/casmi/fragment.py

# %% [markdown]
# ## Step 6b — the learned ranker
#
# Instead of one fixed weight per channel, a small neural network (~1.5k
# parameters) scores each candidate from the three channel scores AND context
# about the molecule -- how good its best spectral relative is, how many
# candidates there are, how rich the spectrum is. That lets it learn rules a
# fixed blend cannot express, like "ignore the analog channel when the best
# analog is weak".
#
# **Not used in this version.** On a synthetic screening library it beat the
# fixed blend by +0.052, but that library turned out not to resemble the test.
# On natural products it is no better than the blend (+0.006 +/- 0.009), and
# the version trained on the screening library collapsed class-1 accuracy from
# 0.63 to 0.39 -- most of why version 5 scored 0.204. Kept here, switched off,
# because the code is the record of what was tried.

# %%EMBED src/casmi/ranker.py

# %% [markdown]
# ## Step 7 — the ranker
#
# Ties everything together: pool for candidates, then `fpnet + 0.5*analog`.
#
# **How to trust a holdout.** Four past submissions have leaderboard scores.
# A holdout is only worth optimising against if it ranks them the same way.
# Natural-product molecules the models never saw (COCONUT structures from the
# public reference libraries) do; the earlier timsTOF holdout, which was 99.9%
# a synthetic screening library, ranked them backwards:
#
# | submission | natural-product holdout | leaderboard |
# |---|---:|---:|
# | v2 + 0.5*analog | 0.535 | 0.244 |
# | analog only | 0.524 | 0.240 |
# | v4c + learned ranker | 0.510 | 0.204 |
# | v3 + 0.25*analog | 0.495 | 0.228 |
#
# On that holdout, the analog weight is flat between 0.5 and 1.0 and fragments
# add nothing, so this version keeps the leaderboard-confirmed 0.5 and changes
# only the model: v4c + 0.5*analog scored 0.559 vs 0.535 for the 0.244
# configuration (+0.024 +/- 0.043 -- promising, not proven).

# %%EMBED src/casmi/rank.py

# %% [markdown]
# ## Step 7b — features and the learned ranker (Phase 8)
#
# The fixed blend above threw away how STRONG each piece of evidence is. The
# public notebooks beat it with channels no better than ours by feeding raw
# evidence to gradient-boosted trees (docs/13). This is that recipe with our own
# components: identity, an adduct-shifted self match, library coverage, analog,
# fingerprint (v5, trained WITHOUT the molecules the ranker learns from), fragments
# and mass error -- raw, plus rank / gap / z-score within the molecule, with tied
# ranks averaged (docs/15).
#
# Measured on the 250 enveda-np-examples molecules, class 1 / class 2 simulated
# one at a time (docs/16): predicted LB 0.283, vs 0.276 analog only and 0.255
# for the fixed blend with the same model.

# %%EMBED src/casmi/features.py

# %% [markdown]
# ## Step 8 — build the index
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
RANKER_ROWS = find_one("ranker_rows.npz")
print("competition data: ", COMP)
print("candidate pool:   ", POOL)
print("fingerprint model:", FPNET)
print("ranker rows:      ", RANKER_ROWS)

INDEX = Path("/tmp/index")
OUT = Path("/kaggle/working/submission.csv")
K = 25

t0 = time.time()
build(TRAIN, INDEX, TEST_MZ_RANGE, set(TEST_ADDUCTS))
print(f"index built in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Step 9 — train the ranker, then rank every test molecule
#
# The ranker is trained HERE, from shipped simulation rows (~1 minute), rather
# than shipped pickled: a pickled scikit-learn model can break on a version
# change, rows cannot. A molecule may have been measured several times, at
# different collision energies and even as different adducts; its evidence
# fuses all of them.

# %%
import torch

lib = Library(INDEX)
pool = CandidatePool(POOL)

# torch ships in the Kaggle base image, so no wheel is needed for it.
DEV = "cuda" if torch.cuda.is_available() else "cpu"
fpnet, fpnet_ck = load(FPNET, device=DEV)
print(f"fpnet: {fpnet_ck['cfg']['n_bits']} bits, "
      f"max_peaks {fpnet_ck.get('max_peaks')}, "
      f"val loss {fpnet_ck.get('val_loss', float('nan')):.4f}, "
      f"val F1 {fpnet_ck.get('val_f1', float('nan')):.3f}, running on {DEV}")

# Kaggle can mount a STALE dataset version — v6 failed exactly that way, with
# the listing showing the new file while the kernel had the old one. Assert the
# model is the one we meant to ship, so a stale mount fails in seconds instead
# of quietly scoring an 18-minute run with the wrong weights.
assert fpnet_ck.get("max_peaks") == 256, (
    f"expected the 256-peak model, got max_peaks={fpnet_ck.get('max_peaks')}. "
    f"Kaggle has probably mounted an older version of casmi26-deps; re-push."
)
# max_peaks alone cannot tell one 256-peak model from another, and neither
# can val_loss thresholds -- those broke the moment the validation set changed.
# Every shipped checkpoint now carries a model_tag; check it exactly.
EXPECTED_FPNET_TAG = "v5"
assert fpnet_ck.get("model_tag") == EXPECTED_FPNET_TAG, (
    f"mounted fpnet is {fpnet_ck.get('model_tag')!r}, expected "
    f"{EXPECTED_FPNET_TAG!r}. Kaggle mounted a stale casmi26-deps; re-push."
)

# The ranker's rows hold fpnet features computed by ONE model; serving them
# another model's scores is the train/serve skew this project kept paying for.
R = np.load(RANKER_ROWS, allow_pickle=True)
assert str(R["fpnet_tag"]) == EXPECTED_FPNET_TAG, f"ranker rows built with fpnet {R['fpnet_tag']}"
assert list(R["feature_names"]) == FEATURE_NAMES, "ranker rows use a different feature list"
t0 = time.time()
ranker_models = fit_ranker(R["X"], R["y"], R["c1"])
print(f"ranker: {len(ranker_models)} models on {len(R['X']):,} rows in {time.time() - t0:.0f}s")

# CombinedRanker is kept only for its padding rule (_pad): fewer than 25
# candidates in the window -> fill from a 20x wider window, after the ranked ones.
padder = CombinedRanker(lib, pool)

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
    adducts = list(grp.adduct)

    # Real collision energies, averaged over a stepped acquisition exactly as
    # training did. Passing 0 instead cost the previous submission: the model
    # saw a median of 32 eV in training and only 14.5% zeros.
    ces = [
        float(np.mean(r.collision_energy_ev))
        if r.collision_energy_ev is not None and len(r.collision_energy_ev)
        else 32.0
        for r in grp.itertuples()
    ]

    # instrument 0 = timsTOF: the whole test set is timsTOF.
    ev = molecule_evidence(lib, pool, spectra, adducts, instrument_id=0,
                           collision_energies=ces)
    cand = ev["cand"]
    if len(cand):
        ev["fp"] = fp_evidence(fpnet, fpnet_ck, ev, pool, device=DEV)
        s = ranker_scores(ranker_models, candidate_features(ev))
        top = np.argsort(-s, kind="stable")[:K]
        cand = cand[top]
    if len(cand) < K:
        # Also when the window is EMPTY: nearest masses from a 20x wider
        # window beat 25 arbitrary structures.
        cand, _ = padder._pad(cand, np.zeros(len(cand), np.float32), spectra[0][0],
                              adducts[0], PPM, K)

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
# ## Step 10 — validate the submission contract
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
