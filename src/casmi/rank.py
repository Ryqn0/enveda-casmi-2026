"""Combine identity search (class 1) and analog propagation (class 2).

The two channels answer different questions and are strong in different places:

  identity  "is this exact molecule in the spectral library?"  -> class 1
  analog    "does it resemble something in the library?"       -> class 2

Both produce a score per candidate, but not on the same footing. A correct
identity match sits near 1.0 and everything else near 0, so the channel is
almost binary. Analog scores are soft and rarely exceed ~0.5, because they are
`hybrid_similarity**4 * Tanimoto` and both factors are below 1.

Adding them with a weight below 1 keeps that asymmetry honest: a confident
identity match always wins, and analog ordering decides everything identity
leaves at zero — which is ~85% of the test set.
"""

from __future__ import annotations

import numpy as np

from casmi.analog import rank_by_analogy
from casmi.library import Library
from casmi.pool import ADDUCT_MASS, CandidatePool

# Chosen so a confident identity match (~1.0) cannot be displaced by the best
# realistic analog score (~0.5 * 0.5 = 0.25). Validated by checking that class-1
# MRR does not move when the analog channel is switched on.
ANALOG_WEIGHT = 0.5

# Swept on a 200-molecule timsTOF holdout (scripts/validate_class2.py):
#   5 ppm, no offset : 64 candidates, 100.0% recall, MRR 0.7121
#   4 ppm, +0.98 ppm : 57 candidates, 100.0% recall, MRR 0.7163  <- chosen
#   3 ppm, +0.98 ppm : 52 candidates,  99.5% recall, MRR 0.7165
# 4 ppm keeps full recall while cutting candidates 11%; 3 ppm trades 0.5% of
# recall for no measurable gain, so it is not worth the risk on unseen data.
TOP_ANALOGS = 100   # 100 beat 50 by +0.012 on the calibrated metric
PPM = 4.0
# timsTOF reads +0.98 ppm high against the mass implied by the labelled
# structure. Correcting it centres the search window instead of straddling.
MZ_OFFSET_PPM = 0.98
K = 25


class CombinedRanker:
    """Ranks pool candidates using both spectral channels."""

    def __init__(self, lib: Library, pool: CandidatePool,
                 analog_weight: float = ANALOG_WEIGHT,
                 offset_ppm: float = MZ_OFFSET_PPM):
        self.lib = lib
        self.pool = pool
        self.analog_weight = analog_weight
        # Held on the instance rather than threaded through every call: it is a
        # property of the instrument, constant for a whole run.
        self.offset_ppm = offset_ppm

        # The two channels live in different index spaces: identity scores are
        # per LIBRARY structure, candidates are per POOL row. InChIKey14 is the
        # only thing they share, so build the bridge once rather than per query.
        self._lib_row_of_key = {
            k: i for i, k in enumerate(lib.structures.inchikey14.to_numpy())
        }

    def score_components(
        self,
        spectra: list[tuple[float, np.ndarray]],
        adduct: str,
        ppm: float = PPM,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pool positions, identity scores, analog scores).

        Kept separate from `rank` so validation can sweep the blend weight
        without paying for the spectral searches again — the analog channel is
        ~40x slower than identity and dominates the cost.
        """
        mz = spectra[0][0]
        cand = (
            self.pool.query(mz, adduct, ppm=ppm, offset_ppm=self.offset_ppm)
            if adduct in ADDUCT_MASS
            else np.empty(0, np.int64)
        )
        empty = np.empty(0, dtype=np.float32)
        if len(cand) == 0:
            return cand, empty, empty

        # --- channel 1: identity ------------------------------------------
        # Mean across the molecule's spectra, matching Phase 1.
        ident_struct = np.zeros(self.lib.n_structures, dtype=np.float32)
        for q_mz, peaks in spectra:
            ident_struct += self.lib.score_spectrum(q_mz, peaks, exclude=exclude)
        ident_struct /= len(spectra)

        # Project library-structure scores onto the candidate list. A candidate
        # absent from the library simply scores 0 here, which is correct: no
        # reference spectrum means no identity evidence.
        ident = np.zeros(len(cand), dtype=np.float32)
        for i, key in enumerate(self.pool.key[cand]):
            row = self._lib_row_of_key.get(key)
            if row is not None:
                ident[i] = ident_struct[row]

        # --- channel 2: analog --------------------------------------------
        best = np.zeros(self.lib.n_structures, dtype=np.float32)
        for q_mz, peaks in spectra:
            idx, sc = self.lib.analog_structures(
                q_mz, peaks, top_n=TOP_ANALOGS, exclude=exclude
            )
            np.maximum.at(best, idx, sc)
        n = min(TOP_ANALOGS, self.lib.n_structures)
        a_idx = np.argpartition(-best, n - 1)[:n]
        a_idx = a_idx[np.argsort(-best[a_idx])]

        analog = rank_by_analogy(
            self.pool.smiles[cand], self.lib.smiles[a_idx], best[a_idx]
        )
        return cand, ident, analog

    def rank(
        self,
        spectra: list[tuple[float, np.ndarray]],
        adduct: str,
        k: int = K,
        ppm: float = PPM,
        exclude: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pool positions, scores) for the k best candidates.

        Ranking is ANALOG-ONLY. Measured on a 196-molecule natural-product
        holdout (scripts/validate_combined.py):

            strategy        class 1   class 2   projected LB
            identity only    0.8765    0.0921         0.1746
            linear w=4       0.8994    0.5320         0.3673
            gated t=0.7      0.8986    0.5977         0.3954
            analog only      0.8955    0.6207         0.4048   <-

        Hybrid search subsumes identity search: a zero-mass-shift hybrid match
        IS an identity match, and a molecule's Tanimoto to itself is 1.0. So
        identity adds almost nothing on class 1 (0.8986 -> 0.8955, a rounding
        error) while costing a great deal on class 2, where the true answer has
        no spectrum and scores 0 but wrong candidates that happen to be in the
        library score above it.

        `self.analog_weight` is retained so score_components can be re-tested if
        a future channel changes the balance, but it no longer affects ranking.
        """
        cand, ident, analog = self.score_components(spectra, adduct, ppm, exclude)
        if len(cand) == 0:
            return cand, np.empty(0, dtype=np.float32)

        total = analog

        n = min(k, len(cand))
        top = np.argpartition(-total, n - 1)[:n]
        top = top[np.argsort(-total[top])]
        out, scores = cand[top], total[top]

        if len(out) < k:
            out, scores = self._pad(out, scores, spectra[0][0], adduct, ppm, k)
        return out, scores

    def _pad(self, out, scores, mz, adduct, ppm, k):
        """Fill unused slots with the next mass-nearest structures.

        A tight mass window can return fewer than k candidates — sometimes as
        few as 3. The metric charges nothing for a wrong guess beyond the rank
        it occupies, so leaving 22 slots empty forfeits 22 free chances. Widen
        the tolerance and append by mass proximity.

        These are appended AFTER the ranked candidates, so they can never
        displace a scored one.
        """
        target = self.pool.neutral_mass(mz, adduct, self.offset_ppm)
        if target is None:
            return out, scores
        wide = self.pool.query(mz, adduct, ppm=ppm * 20, offset_ppm=self.offset_ppm)
        if len(wide) == 0:
            return out, scores

        # np.isin(..., invert=True) drops the ones already ranked.
        extra = wide[np.isin(wide, out, invert=True)]
        if len(extra) == 0:
            return out, scores

        extra = extra[np.argsort(np.abs(self.pool.mass[extra] - target))]
        extra = extra[: k - len(out)]
        return (
            np.concatenate([out, extra]),
            np.concatenate([scores, np.zeros(len(extra), dtype=np.float32)]),
        )
