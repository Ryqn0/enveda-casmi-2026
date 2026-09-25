"""Per-candidate evidence and features for a gradient-boosted ranker (Phase 8).

The public engine beat our pipeline with channels no better than ours: it fed
RAW evidence to a learned combiner, kept a library channel, and trained on
natural-product simulations (docs/13). This module is our version of that
recipe, built only from our own components. It is the single implementation of
the features: the harness that trains and evaluates the ranker and the Kaggle
notebook that serves it both call it, so they cannot drift (docs/07, docs/15).

Two steps, because the spectral searches are slow and the model is not:
    ev = molecule_evidence(lib, pool, spectra, adduct, exclude=...)   # searches
    ev["fp"] = fp_evidence(model, ck, ev)                             # fpnet
    X = candidate_features(ev)                                        # cheap

Ties inside a molecule are AVERAGED when ranking (rank01): breaking them by
list position was a train/serve skew in the public engine (docs/15).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from casmi.analog import rank_by_analogy
from casmi.fpnet import candidate_bits, predict_logits, score_candidates
from casmi.fragment import score_spectra_frag
from casmi.library import Library
from casmi.pool import ADDUCT_MASS, CandidatePool
from casmi.rank import MZ_OFFSET_PPM, PPM, TOP_ANALOGS

# Per-candidate channels; each becomes 4 columns (raw, rank, gap, z).
CHANNELS = ["id_max", "id_mean", "hyb_self", "n_lib", "analog", "fp", "frag"]
CONTEXT = ["ppm", "abs_ppm", "log_n_cand", "an_conf", "id_top", "hyb_top",
           "n_spectra", "log_max_peaks"]
FEATURE_NAMES = ([f"{c}_{t}" for c in CHANNELS for t in ("raw", "rank", "gap", "z")]
                 + CONTEXT)


def rank01(x: np.ndarray) -> np.ndarray:
    """0 = best, 1 = worst within the molecule; tied values share their average."""
    x = np.asarray(x, dtype=np.float64)
    return ((rankdata(-x, method="average") - 1) / max(1, len(x) - 1)).astype(np.float32)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return ((x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)).astype(np.float32)


def _row_of_key(lib: Library) -> dict:
    """{InChIKey14: library structure index}, built once per Library."""
    if not hasattr(lib, "_row_of_key"):
        lib._row_of_key = {k: i for i, k in enumerate(lib.structures.inchikey14.to_numpy())}
    return lib._row_of_key


def molecule_evidence(lib: Library, pool: CandidatePool, spectra, adducts,
                      exclude: np.ndarray | None = None, ppm: float = PPM,
                      offset_ppm: float = MZ_OFFSET_PPM, instrument_id: int = 0,
                      collision_energies=None) -> dict:
    """Everything the searches say about one molecule's candidates.

    spectra: [(precursor_mz, peaks (n, 2)), ...], all measurements of the
    molecule; adducts: one per spectrum -- a molecule is often measured as
    several ([M+H]+ and [M-H]-: 160 of the 250 enveda-np-examples structures).
    The candidate window is centred on the MEDIAN neutral mass over spectra.
    exclude: library positions to hide (the class-1/2 simulations).
    Returns a dict of per-candidate arrays plus the query itself (for fpnet).
    """
    masses = [pool.neutral_mass(mz, a, offset_ppm) for (mz, _), a in zip(spectra, adducts)]
    masses = [x for x in masses if x is not None and x > 0]
    m = float(np.median(masses)) if masses else None
    if m is None:
        cand = np.empty(0, np.int64)
    else:
        lo, hi = np.searchsorted(pool.mass, [m * (1 - ppm * 1e-6), m * (1 + ppm * 1e-6)])
        cand = np.arange(lo, hi, dtype=np.int64)
    ev = {"cand": cand, "spectra": spectra, "adducts": list(adducts),
          "instrument_id": instrument_id, "collision_energies": collision_energies}
    if len(cand) == 0:
        return ev

    # Identity (same precursor) and hybrid (any precursor shift) per spectrum,
    # each collapsed to one score per library structure.
    ident = np.zeros((len(spectra), lib.n_structures), np.float32)
    hyb = np.zeros(lib.n_structures, np.float32)
    for j, (q_mz, peaks) in enumerate(spectra):
        ident[j] = lib.score_spectrum(q_mz, peaks, exclude=exclude)
        h = lib.hybrid_scores(q_mz, peaks, exclude)
        if h is not None:
            np.maximum(hyb, h, out=hyb)

    # Library spectra each structure still has after hiding.
    n_lib = np.diff(lib._starts).astype(np.float32)
    if exclude is not None and len(exclude):
        n_lib -= np.bincount(lib.struct_idx[exclude], minlength=lib.n_structures)

    # Map every per-structure array onto the candidates (-1 = not in library).
    rok = _row_of_key(lib)
    row = np.array([rok.get(k, -1) for k in pool.key[cand]])
    has = row >= 0

    def on_cand(per_struct):
        v = np.zeros(len(cand), np.float32)
        v[has] = per_struct[row[has]]
        return v

    ev["id_max"] = on_cand(ident.max(0))
    ev["id_mean"] = on_cand(ident.mean(0))
    ev["hyb_self"] = on_cand(hyb)
    ev["n_lib"] = np.log1p(on_cand(n_lib))

    # Analog: the top cousins by hybrid score, propagated by Tanimoto -- the
    # same channel as rank.py. A candidate's own entry is included here as in
    # rank.py; hyb_self lets the ranker tell "is a cousin" from "is itself".
    n = min(TOP_ANALOGS, lib.n_structures)
    a_idx = np.argpartition(-hyb, n - 1)[:n]
    a_idx = a_idx[np.argsort(-hyb[a_idx])]
    ev["analog"] = rank_by_analogy(pool.smiles[cand], lib.smiles[a_idx], hyb[a_idx])
    ev["an_conf"] = float(hyb[a_idx[0]])

    # score_spectra_frag uses the richest spectrum; give it that one's adduct.
    j = max(range(len(spectra)), key=lambda i: len(spectra[i][1]))
    ev["frag"] = score_spectra_frag(pool.smiles[cand], [spectra[j]], adducts[j]).astype(np.float32)
    ev["ppm"] = ((pool.mass[cand] - m) / m * 1e6).astype(np.float32)
    return ev


def fp_evidence(model, ck, ev: dict, pool: CandidatePool, device: str = "cpu") -> np.ndarray:
    """fpnet log-likelihood of each candidate (Part 5), for one molecule.

    predict_logits takes one adduct, so spectra are grouped by adduct and the
    group logits averaged, weighted by group size -- the same per-spectrum
    mean predict_logits takes within a group.
    """
    if len(ev["cand"]) == 0:
        return np.zeros(0, np.float32)
    ces = ev["collision_energies"] or [None] * len(ev["spectra"])
    logits, n = 0.0, 0
    for a in sorted(set(ev["adducts"])):
        idx = [i for i, x in enumerate(ev["adducts"]) if x == a]
        sub_ces = [ces[i] for i in idx]
        z = predict_logits(model, [ev["spectra"][i] for i in idx], a, list(ck["adduct_list"]),
                           instrument_id_=ev["instrument_id"], device=device,
                           max_peaks=int(ck["max_peaks"]),
                           collision_energies=None if None in sub_ces else sub_ces)
        logits, n = logits + z * len(idx), n + len(idx)
    bits = candidate_bits(pool.smiles[ev["cand"]], np.asarray(ck["bit_sel"]))
    return score_candidates(logits / n, bits).astype(np.float32)


def candidate_features(ev: dict) -> np.ndarray:
    """[n_candidates, len(FEATURE_NAMES)] float32, in FEATURE_NAMES order.

    Raw values keep how STRONG the evidence is (a fixed z-score blend threw
    that away); rank/gap/z say how a candidate compares with its competitors.
    A channel that was not computed (e.g. no fpnet) is all zeros.
    """
    nc = len(ev["cand"])
    cols = []
    for c in CHANNELS:
        v = np.asarray(ev.get(c, np.zeros(nc)), np.float32)
        cols += [v, rank01(v), v - (v.max() if nc else 0.0), _z(v)]
    ppm = ev["ppm"]
    ctx = [ppm, np.abs(ppm),
           np.full(nc, np.log(max(nc, 1))),
           np.full(nc, ev["an_conf"]),
           np.full(nc, ev["id_max"].max()),
           np.full(nc, ev["hyb_self"].max()),
           np.full(nc, len(ev["spectra"])),
           np.full(nc, np.log1p(max(len(p) for _, p in ev["spectra"])))]
    return np.column_stack(cols + ctx).astype(np.float32)


# --- the ranker ---------------------------------------------------------------
# Gradient-boosted trees, the public engine's settings. One classifier per
# (class-1 weight, seed), averaged: class-1 and class-2 simulation rows pull in
# opposite directions on "has library spectra" (docs/16), and the weight sets
# the balance. Trained in the notebook from shipped rows, so no pickled model
# has to survive a scikit-learn version change.
GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80,
           l2_regularization=1.0)
W_A = (0.35, 0.55)
SEEDS = (0, 1)


def fit_ranker(X: np.ndarray, y: np.ndarray, is_c1: np.ndarray, monotonic_cst=None,
               w_a=W_A) -> list:
    """Train the ranker on simulation rows (y: is the true candidate).

    w_a: weights of the class-1 rows (class-2 rows get 1 - w), one model each.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return [HistGradientBoostingClassifier(random_state=sd, monotonic_cst=monotonic_cst, **GBM)
            .fit(X, y, sample_weight=np.where(is_c1, wA, 1.0 - wA))
            for wA in w_a for sd in SEEDS]


def ranker_scores(models: list, X: np.ndarray) -> np.ndarray:
    """Mean probability of being the true candidate; higher ranks first."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def _self_check() -> None:
    # Ties share their rank: two zeros among three values both get 0.75.
    r = rank01(np.array([1.0, 0.0, 0.0]))
    assert np.allclose(r, [0.0, 0.75, 0.75]), r
    assert len(FEATURE_NAMES) == 4 * len(CHANNELS) + len(CONTEXT)
    ev = {"cand": np.arange(3), "spectra": [(200.0, np.zeros((5, 2)))],
          "ppm": np.array([0.1, -1.0, 2.0], np.float32), "an_conf": 0.4,
          "id_max": np.array([0.9, 0.0, 0.0], np.float32),
          "hyb_self": np.array([0.9, 0.2, 0.0], np.float32)}
    X = candidate_features(ev)
    assert X.shape == (3, len(FEATURE_NAMES)) and np.isfinite(X).all()
    print("features self-check OK")


if __name__ == "__main__":
    _self_check()
