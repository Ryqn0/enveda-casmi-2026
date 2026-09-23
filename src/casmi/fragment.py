"""In-silico fragmentation: score a candidate by how much of the spectrum its
own bonds can explain.

WHY THIS CHANNEL EXISTS
    Our other two channels both need a RELATIVE in the spectral library:

      library search   needs the molecule itself to have a reference spectrum
      analog search    needs something structurally similar to have one

    Class 3 molecules (in no database at all) have neither, so both channels
    score every class-3 candidate on essentially nothing. This channel needs no
    library: it takes the candidate structure, breaks its bonds, and asks
    whether the resulting pieces have the masses we actually observed.

    That makes it the only signal we have that works when the molecule is
    genuinely novel -- which is where the leaderboard gap is.

THE IDEA
    A molecule in a collision cell breaks at its weakest bonds. If a candidate
    is correct, most of the observed peaks should correspond to connected
    sub-pieces of it. If it is wrong, the masses will not line up.

    So: enumerate sub-fragments, compute their exact masses, and measure the
    fraction of total peak intensity that some fragment explains.

DELIBERATE SIMPLIFICATIONS
    Real MetFrag does a bond-energy-weighted tree search over multiple bond
    breakages. This breaks at most `max_breaks` single bonds at once, which
    covers the large majority of real fragmentations at a small fraction of the
    cost. By default only ACYCLIC bonds break; RING_BREAKS additionally allows
    cross-ring cleavage of aliphatic rings (two ring bonds at once), which is
    how sugars and saturated cores fragment. Aromatic rings are never opened.

    ponytail: depth-limited breaking, no bond energies. Upgrade to a scored
    tree search if this channel earns more of its place.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

# RDKit complains loudly about candidate structures it dislikes. We already
# handle failures by returning 0.0, so the log noise is pure cost -- and on
# Kaggle it can produce hundreds of MB of output.
RDLogger.DisableLog("rdApp.*")

# Masses that convert a neutral fragment into the ion we actually detect.
PROTON = 1.007276466      # H+ : an H nucleus, i.e. H minus one electron
HYDROGEN = 1.007825032    # a whole neutral H atom, for rearrangement counting

# Collision-induced dissociation routinely moves a hydrogen from one piece to
# the other as the bond breaks, so a fragment is commonly observed at +/- 1 H
# from its nominal mass. Allowing a 1-H window is standard practice; allowing
# more would match almost anything and destroy the score's discriminating power.
H_SHIFTS = (-1, 0, 1)

# Whether to also try cross-ring cleavage of aliphatic rings. Off until the
# honest holdout says it helps -- see _fragment_masses.
RING_BREAKS = False


@lru_cache(maxsize=200_000)
def _fragment_masses(smiles: str, max_breaks: int = 2,
                     rings: bool = False) -> tuple[float, ...]:
    """Every neutral sub-fragment mass this structure can produce.

    Cached because the SAME candidate appears for many different query
    spectra -- the pool is shared across the whole test set, so without the
    cache the identical molecule is re-fragmented hundreds of times.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ()

    # The intact molecule is always a valid "fragment": the precursor ion
    # itself is usually present, and some spectra are dominated by it.
    masses = {ExactMolWt(mol)}

    # Only single bonds that are NOT in a ring. Breaking one ring bond leaves
    # the molecule in one connected piece, so it yields no new fragment mass --
    # it would just cost time and return the parent again.
    breakable = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
    ]
    if rings:
        # Cross-ring cleavage: two single bonds of the same ALIPHATIC ring
        # breaking together cut the ring into two pieces. That is how sugars
        # and saturated terpenoid cores fragment, and acyclic-only breaking
        # cannot produce those masses at all. Aromatic bonds are excluded
        # (RDKit types them AROMATIC, not SINGLE): aromatic rings rarely open
        # under collision-induced dissociation.
        breakable += [
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBondType() == Chem.BondType.SINGLE and b.IsInRing()
        ]
    if not breakable:
        return tuple(sorted(masses))

    # Try breaking 1 bond, then 2 at once, etc. Breaking n bonds simultaneously
    # is what produces internal fragments (pieces with both ends cut off).
    for n in range(1, max_breaks + 1):
        for combo in combinations(breakable, n):
            try:
                # addDummies=False: we want the real masses of the pieces, not
                # pieces decorated with placeholder atoms that would add mass.
                frag = Chem.FragmentOnBonds(mol, list(combo), addDummies=False)
                pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
            except Exception:
                # A candidate RDKit cannot fragment is not worth failing the
                # whole run over; it simply contributes no fragment masses.
                continue
            for p in pieces:
                try:
                    masses.add(ExactMolWt(p))
                except Exception:
                    continue

    return tuple(sorted(masses))


def explained_intensity(smiles: str, mzs, intensities, positive: bool,
                        ppm: float = 10.0, max_breaks: int = 2,
                        rings: bool | None = None) -> float:
    """Fraction of total peak intensity this structure can account for.

    Returns a value in [0, 1]. A correct structure typically explains most of
    the intensity; an incorrect isomer explains noticeably less.

    `ppm` is deliberately looser than the 4 ppm used for precursor matching.
    Precursor m/z is measured precisely, but fragment assignment also absorbs
    the error of our simplified chemistry (which H ended up on which piece), so
    a tight window would reject correct assignments.
    """
    mz = np.asarray(mzs, dtype=np.float64)
    inten = np.asarray(intensities, dtype=np.float64)
    total = inten.sum()
    if total <= 0 or mz.size == 0:
        return 0.0

    frags = _fragment_masses(smiles, max_breaks,
                             RING_BREAKS if rings is None else rings)
    if not frags:
        return 0.0

    # Convert neutral fragment masses into the m/z values we would SEE.
    # Positive mode adds a proton, negative mode removes one; the H_SHIFTS
    # then cover hydrogen rearrangement during the break.
    sign = 1.0 if positive else -1.0
    base = np.asarray(frags, dtype=np.float64) + sign * PROTON
    predicted = np.concatenate([base + k * HYDROGEN for k in H_SHIFTS])
    predicted.sort()

    # For each observed peak, find the nearest predicted m/z. searchsorted gives
    # the insertion point, so the nearest value is at that index or the one
    # before it -- checking both is what makes this O(log n) instead of an
    # O(peaks x fragments) full comparison.
    pos = np.searchsorted(predicted, mz)
    lo = predicted[np.clip(pos - 1, 0, len(predicted) - 1)]
    hi = predicted[np.clip(pos, 0, len(predicted) - 1)]
    nearest = np.where(np.abs(mz - lo) < np.abs(mz - hi), lo, hi)

    # Tolerance in ppm scales with m/z: a fixed Da window would be far too
    # tight at low mass and far too loose at high mass.
    matched = np.abs(mz - nearest) <= mz * ppm * 1e-6
    return float(inten[matched].sum() / total)


def score_candidates_frag(smiles_list, mzs, intensities, adduct: str,
                          ppm: float = 10.0, max_breaks: int = 2,
                          rings: bool | None = None) -> np.ndarray:
    """Explained-intensity score for every candidate, as an array."""
    positive = not adduct.rstrip().endswith("-")
    return np.array(
        [explained_intensity(s, mzs, intensities, positive, ppm, max_breaks,
                             rings)
         for s in smiles_list],
        dtype=np.float32,
    )


def score_spectra_frag(smiles_list, spectra, adduct: str,
                       ppm: float = 10.0, max_breaks: int = 2,
                       rings: bool | None = None) -> np.ndarray:
    """Fragment score for a molecule measured as SEVERAL spectra.

    THE SINGLE SOURCE OF TRUTH for which spectrum the fragment channel uses --
    shared by the ranker, the validator and the notebook. Deciding this in each
    caller is how this project's train/serve skew bugs happened.

    Uses the richest spectrum (most peaks) rather than averaging all of them.
    The score is an explained-intensity FRACTION, and fractions over spectra
    of very different richness are not comparable: a 3-peak spectrum is
    trivially "fully explained" by almost any candidate, so averaging it in
    would drag every candidate toward the same score.

    `spectra` is a list of (precursor_mz, peaks) with peaks an (n, 2) array of
    (m/z, intensity) -- the same shape CombinedRanker.rank receives.
    """
    _, peaks = max(spectra, key=lambda s: len(s[1]))
    pk = np.asarray(peaks, dtype=np.float64)
    if pk.ndim != 2 or len(pk) == 0:
        return np.zeros(len(smiles_list), dtype=np.float32)
    return score_candidates_frag(smiles_list, pk[:, 0], pk[:, 1], adduct,
                                 ppm, max_breaks, rings)


def _self_check() -> None:
    # Caffeine, and a peak list built from masses its own bonds can produce.
    caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    frags = _fragment_masses(caffeine)
    assert len(frags) > 1, "no fragments enumerated for caffeine"

    # Build a synthetic spectrum FROM caffeine's own fragments, so a correct
    # structure must explain ~all of it.
    peaks = np.array([f + PROTON for f in frags[:6]])
    inten = np.ones(len(peaks))
    hit = explained_intensity(caffeine, peaks, inten, positive=True)
    assert hit > 0.99, f"caffeine failed to explain its own fragments: {hit}"

    # A structure of a totally different mass must explain much less of it.
    other = explained_intensity("CCCCCCCCCCCCCCCCCC(=O)O", peaks, inten, True)
    assert other < hit, f"unrelated molecule scored as well as the truth: {other}"

    # Empty and unparseable inputs must return 0.0, not raise: the pool
    # contains structures RDKit refuses, and one of them must not kill a run.
    assert explained_intensity(caffeine, [], [], True) == 0.0
    assert explained_intensity("not a smiles", peaks, inten, True) == 0.0

    # Negative mode must shift the other way, so a spectrum built for positive
    # mode should NOT score identically in negative mode.
    neg = explained_intensity(caffeine, peaks, inten, positive=False)
    assert neg < hit, "adduct polarity is being ignored"

    # Cross-ring cleavage: a glucose-like ring gains fragments; benzene-only
    # structures (aromatic bonds are not SINGLE) must be unaffected.
    sugar = "OCC1OC(O)C(O)C(O)C1O"
    assert len(_fragment_masses(sugar, 2, True)) > len(_fragment_masses(sugar, 2, False)), \
        "ring breaking added no fragments to an aliphatic ring"
    tol = "Cc1ccccc1"
    assert _fragment_masses(tol, 2, True) == _fragment_masses(tol, 2, False), \
        "ring breaking touched an aromatic ring"

    print(f"fragment self-check passed ({len(frags)} caffeine fragments)")


if __name__ == "__main__":
    _self_check()
