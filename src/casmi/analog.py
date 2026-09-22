"""Analog propagation: rank candidates by resemblance to a spectral cousin.

THE PROBLEM THIS SOLVES
    A competition class-2 molecule has no reference spectrum anywhere, so
    identity search returns nothing and scores exactly zero. But chemistry is
    not random: an unknown natural product almost always has a *relative* that
    has been measured.

THE CHAIN OF REASONING
    1. Hybrid search finds library spectra that match apart from a constant mass
       shift. A constant shift is the signature of a chemical modification.
    2. Those hits identify a known molecule the unknown resembles — the analog.
    3. The pool gives every structure of the right mass.
    4. Prefer candidates that look chemically like the analog.

    The unknown is never identified directly. It is located by triangulation
    from a known neighbour.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

# Morgan fingerprints (a.k.a. ECFP4 at radius 2). Each bit records the presence
# of one circular atom neighbourhood, so two molecules sharing substructures
# share bits. radius=2 means "each atom plus everything within two bonds",
# which is the standard choice for similarity work.
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# How sharply to favour strong analogs. score**4 means an analog matching at
# 0.9 counts ~2.5x more than one at 0.7, and a weak 0.3 match is suppressed to
# nearly nothing. Without this, a crowd of poor analogs outvotes one good one.
DEFAULT_POWER = 4.0


@lru_cache(maxsize=2_000_000)
def morgan_fp(smiles: str):
    """Fingerprint for a SMILES, or None if RDKit cannot parse it.

    Cached because candidate pools overlap heavily between queries — molecules
    of similar mass keep reappearing — and parsing dominates the cost.
    """
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _GEN.GetFingerprint(mol)


def rank_by_analogy(
    candidate_smiles: np.ndarray,
    analog_smiles: np.ndarray,
    analog_scores: np.ndarray,
    power: float = DEFAULT_POWER,
) -> np.ndarray:
    """Score each candidate by its weighted similarity to the best analog.

    score(candidate) = max over analogs of  analog_similarity**power
                                            * Tanimoto(candidate, analog)

    `max` rather than a sum: one convincing analog is strong evidence, whereas
    summing would reward a candidate that is vaguely similar to many unrelated
    hits. Returns one score per candidate, aligned to `candidate_smiles`.
    """
    scores = np.zeros(len(candidate_smiles), dtype=np.float32)
    if len(candidate_smiles) == 0 or len(analog_smiles) == 0:
        return scores

    # Fingerprint the candidates once, keeping track of which survived parsing.
    # BulkTanimotoSimilarity cannot accept a None in its list.
    fps, keep = [], []
    for i, smi in enumerate(candidate_smiles):
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            keep.append(i)
    if not fps:
        return scores
    keep = np.asarray(keep)

    for a_smi, a_score in zip(analog_smiles, analog_scores):
        # A weak analog cannot lift any candidate above what a strong one
        # already gave, so skip it rather than paying for the comparison.
        weight = float(a_score) ** power
        if weight <= 0:
            continue
        a_fp = morgan_fp(a_smi)
        if a_fp is None:
            continue
        # One C++ call for all candidates instead of a Python loop.
        sims = np.asarray(
            DataStructs.BulkTanimotoSimilarity(a_fp, fps), dtype=np.float32
        )
        # Plain assignment, NOT np.maximum(..., out=scores[keep]): indexing with
        # an integer array returns a copy, so an `out=` write would land in a
        # temporary and be discarded, leaving every score at zero.
        scores[keep] = np.maximum(scores[keep], weight * sims)

    return scores


def _self_check() -> None:
    caffeine = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    theophylline = "Cn1c(=O)c2[nH]cnc2n(C)c1=O"   # caffeine minus one methyl
    glucose = "OCC1OC(O)C(O)C(O)C1O"               # unrelated

    assert morgan_fp("not a molecule") is None
    assert morgan_fp(caffeine) is not None

    # A caffeine-like analog must rank theophylline above glucose.
    cands = np.array([glucose, theophylline])
    s = rank_by_analogy(cands, np.array([caffeine]), np.array([1.0]))
    assert s[1] > s[0], s

    # A weaker analog must produce uniformly weaker scores.
    weak = rank_by_analogy(cands, np.array([caffeine]), np.array([0.5]))
    assert weak[1] < s[1]

    # Unparseable candidates keep their slot at score 0 rather than shifting
    # the alignment of everything after them.
    s2 = rank_by_analogy(
        np.array(["@@bad@@", theophylline]), np.array([caffeine]), np.array([1.0])
    )
    assert s2[0] == 0.0 and s2[1] > 0.0, s2

    # No analogs at all is a valid state (nothing matched); score everything 0.
    assert rank_by_analogy(cands, np.array([]), np.array([])).tolist() == [0.0, 0.0]

    print("analog self-check passed")


if __name__ == "__main__":
    _self_check()
