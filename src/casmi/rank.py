"""Rank a molecule's candidates by combining three evidence channels.

  analog    does the spectrum resemble a library spectrum of something
            structurally similar? (hybrid search + Tanimoto propagation)
  fpnet     do the candidate's substructures match what the model predicts
            from the peaks?
  fragment  can the candidate's own bond breaks explain the observed peaks?
            (needs no library at all -- the only channel for class 3)

By default the channels are combined by a fixed blend of z-scores,
`fpnet + FPNET_ANALOG_WEIGHT * analog + FRAG_WEIGHT * fragment`, with the
fragment weight currently 0 (no measurable gain on natural products). A small
learned ranker (casmi/ranker.py) can replace the blend, but is not used: on
natural products it did not beat the blend (docs/12).

Identity search (exact library match) is still computed by score_components
for analysis, but not used for ranking: hybrid search subsumes it, since a
zero-mass-shift hybrid match IS an identity match.
"""

from __future__ import annotations

import numpy as np

from casmi.analog import rank_by_analogy
from casmi.fpnet import candidate_bits, predict_logits, score_candidates
from casmi.fragment import score_spectra_frag
from casmi.ranker import ranker_features, score_with_ranker
from casmi.library import Library
from casmi.pool import ADDUCT_MASS, CandidatePool

# Blend weights, measured on the NATURAL-PRODUCT holdout (docs/12): 984
# molecules in COCONUT that fpnet v4c never trained on, from the reference
# libraries, ties broken fairly. This is the only holdout found so far that
# reproduces the leaderboard's ordering of past submissions; the earlier
# "timsTOF holdout" was 99.9% enveda-180, a synthetic screening library, and
# ranked them backwards.
#
#   fpnet v4c +              MRR
#   0.00*analog only      0.5628   (analog alone)
#   0.50*analog           0.5865   <- chosen: the LB-best weight (0.244)
#   0.75*analog           0.5919
#   1.00*analog           0.5897
#   1.50*analog           0.5853
#
# 0.5-1.0 is a plateau (differences <= 0.006, within noise), so the weight is
# held at the value the leaderboard already confirmed, and the next submission
# changes ONE thing -- the model -- versus the 0.244 configuration.
FPNET_ANALOG_WEIGHT = 0.5
# In-silico fragmentation (casmi/fragment.py). Off: on natural products it adds
# nothing measurable (+/- 0.001 at every weight tried), and it costs runtime.
# It helped on enveda-180 only (docs/10), which is not the test chemistry.
FRAG_WEIGHT = 0.0


def _z(x: np.ndarray) -> np.ndarray:
    """Z-score, so two differently-scaled channels can be added."""
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

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
                 offset_ppm: float = MZ_OFFSET_PPM,
                 fpnet=None, fpnet_ck=None, device: str = "cpu",
                 use_platt: bool = False, ranker=None, ranker_ck=None):
        self.lib = lib
        self.pool = pool
        self.analog_weight = analog_weight
        # Optional so the pipeline still runs without a trained model.
        self.fpnet = fpnet
        self.device = device
        # Optional learned ranker (casmi/ranker.py). When absent, the fixed
        # blend below is used -- which is also the fallback if it misbehaves.
        self.ranker, self.ranker_ck = ranker, ranker_ck
        if fpnet is not None:
            self.bit_sel = np.asarray(fpnet_ck["bit_sel"])
            self.adducts = list(fpnet_ck["adduct_list"])
            # The cap the model was TRAINED at. Guessing 128 for a 256-peak
            # model silently truncates every rich spectrum, and 69.8% of test
            # spectra exceed 128 peaks.
            self.max_peaks = int(fpnet_ck.get("max_peaks", 128))
            # Per-bit Platt calibration (CSI:FingerID / MSNovelist). The
            # checkpoint CARRIES it but it is OFF by default, because measured
            # on the timsTOF holdout it is a large loss:
            #     v3 fpnet alone   raw 0.8147 | A-only 0.8095 | full 0.6409(v2)
            #     v3 +0.25*analog  raw 0.8509 | A-only 0.8461
            # It improves the probabilities a lot (log-loss 0.2187 -> 0.0732)
            # and the RANKING not at all. Ranking reduces to cand_fp @ (A*z+B),
            # so B ~ -3.2 acts as a -3.2 * (bits set) complexity penalty: a
            # prior about molecules, not evidence from the spectrum.
            # Better probabilities != better ranking. Left wired for the record.
            a, b = fpnet_ck.get("platt_a"), fpnet_ck.get("platt_b")
            self.platt = ((np.asarray(a), np.asarray(b))
                          if use_platt and a is not None else None)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return (pool positions, identity scores, analog scores, an_conf).

        an_conf is the hybrid-search score of the single best spectral
        relative: how much the analog channel deserves to be trusted for THIS
        molecule. The learned ranker uses it; the fixed blend ignores it.

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
            return cand, empty, empty, 0.0

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
        return cand, ident, analog, float(best[a_idx[0]])

    def rank(
        self,
        spectra: list[tuple[float, np.ndarray]],
        adduct: str,
        k: int = K,
        ppm: float = PPM,
        exclude: np.ndarray | None = None,
        collision_energies: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pool positions, scores) for the k best candidates.

        With a model loaded, ranking is `fpnet + 0.5*analog` (see
        FPNET_ANALOG_WEIGHT). Without one it falls back to analog-only, which
        is the configuration that scored 0.240.

        `collision_energies` must be the real per-spectrum values; the model was
        trained with them and defaults to the training median otherwise.

        The analog-only choice was itself measured, on a 196-molecule
        natural-product holdout (scripts/validate_combined.py):

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
        cand, ident, analog, an_conf = self.score_components(
            spectra, adduct, ppm, exclude)
        if len(cand) == 0:
            return cand, np.empty(0, dtype=np.float32)

        if self.fpnet is None:
            total = analog
        else:
            # Weights and the measurements behind them: see FPNET_ANALOG_WEIGHT.
            logits = predict_logits(
                self.fpnet, spectra, adduct, self.adducts,
                instrument_id_=0,   # the test set is 100% timsTOF
                device=self.device,
                max_peaks=self.max_peaks,
                collision_energies=collision_energies,
            )
            fp_score = score_candidates(
                logits, candidate_bits(self.pool.smiles[cand], self.bit_sel),
                platt=self.platt,
            )
            # Fragmentation is the slowest channel; skip it when nothing uses it.
            needs_frag = self.ranker is not None or FRAG_WEIGHT != 0
            frag = (score_spectra_frag(self.pool.smiles[cand], spectra, adduct)
                    if needs_frag else np.zeros(len(cand), dtype=np.float32))
            if self.ranker is not None:
                # Same function, same inputs as the training dump -- n_peaks is
                # the RAW peak count of the richest spectrum, as in validation.
                feats = ranker_features(
                    fp_score, analog, frag, an_conf,
                    n_peaks=max(len(pk) for _, pk in spectra))
                total = score_with_ranker(self.ranker, self.ranker_ck, feats)
            else:
                total = (_z(fp_score) + FPNET_ANALOG_WEIGHT * _z(analog)
                         + FRAG_WEIGHT * _z(frag))

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
