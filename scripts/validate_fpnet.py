"""Does the fingerprint model beat analog propagation where analogy fails?

Runs the same class-2 simulation as validate_class2.py — every spectrum of the
structure hidden, the structure still in the candidate pool — and scores the
same candidates three ways:

    analog   Tanimoto to a spectral cousin          (current pipeline)
    fpnet    log-likelihood of the candidate's bits (the new model)
    both     sum of the two, each z-scored

The number that matters is the low-Tanimoto bucket. Analog propagation needs a
spectral relative; the model does not, so if it helps anywhere it should help
most where the nearest analog is distant. An aggregate would hide exactly that.

    uv run python scripts/validate_fpnet.py -n 200 --source gnps,massbank,mona
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_fpnet_data import instrument_id  # noqa: E402
from validate_class2 import fetch_raw, sample_structures  # noqa: E402

from casmi.analog import morgan_fp, rank_by_analogy  # noqa: E402
from casmi.fpnet import load, predict_logits, score_candidates  # noqa: E402
from casmi.library import Library  # noqa: E402
from casmi.pool import ADDUCT_MASS, CandidatePool  # noqa: E402
from casmi.rank import MZ_OFFSET_PPM, PPM, TOP_ANALOGS  # noqa: E402

K = 25
FP_SIZE = 2048


def candidate_bits(smiles_list, bit_sel: np.ndarray) -> np.ndarray:
    """(n_cand, n_bits) 0/1 matrix on the SAME bit selection the model used.

    If this selection ever disagreed with training, every score would be
    computed against the wrong bits and the channel would look useless for a
    reason that has nothing to do with the model.
    """
    out = np.zeros((len(smiles_list), len(bit_sel)), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        fp = morgan_fp(smi)
        if fp is None:
            continue
        arr = np.zeros(FP_SIZE, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr[bit_sel]
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """Put two differently-scaled score channels on comparable footing."""
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default=None)
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "fpnet" / "fpnet.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(args.model, device=dev)
    bit_sel = np.asarray(ck["bit_sel"])
    adducts = list(ck["adduct_list"])
    # Read the cap the model was TRAINED with. v1 predates this field, so
    # fall back to 128, which is what it used.
    MAX_PEAKS = int(ck.get("max_peaks", 128))
    print(f"model: {ck['cfg']['n_bits']} bits, max_peaks {MAX_PEAKS}, "
          f"val loss {ck.get('val_loss', float('nan')):.4f}, "
          f"val F1 {ck.get('val_f1', float('nan')):.3f}")

    lib = Library(ROOT / "data" / "index")
    pool = CandidatePool(ROOT / "data" / "pool.parquet")

    sources = set(args.source.split(",")) if args.source else None
    holdout = sample_structures(lib, pool, args.n, args.seed, sources)
    raw = fetch_raw({int(r) for h in holdout for r in h["query_rows"]})
    print(f"holdout {len(holdout)} structures\n")

    keys = lib.structures.inchikey14.to_numpy()
    res = {}
    tans = []
    t0 = time.time()

    for h in holdout:
        s = h["struct_idx"]
        truth = keys[s]
        rows = [r for r in h["query_rows"] if r in raw]
        if not rows:
            continue
        mz0, adduct, _, _, _ = raw[rows[0]]
        if adduct not in ADDUCT_MASS:
            continue
        cand = pool.query(mz0, adduct, ppm=PPM, offset_ppm=MZ_OFFSET_PPM)
        if len(cand) == 0:
            continue
        ck_keys = pool.key[cand]
        exclude = lib.spectra_of(s)          # class 2: no spectra of this molecule

        # --- analog channel ------------------------------------------------
        best = np.zeros(lib.n_structures, dtype=np.float32)
        for r in rows:
            q_mz, _, peaks, _, _ = raw[r]
            idx, sc = lib.analog_structures(q_mz, peaks, top_n=TOP_ANALOGS,
                                            exclude=exclude)
            np.maximum.at(best, idx, sc)
        a_idx = np.argpartition(-best, TOP_ANALOGS - 1)[:TOP_ANALOGS]
        a_idx = a_idx[np.argsort(-best[a_idx])]
        s_analog = rank_by_analogy(pool.smiles[cand], lib.smiles[a_idx], best[a_idx])

        # how close is the best analog, for the bucket breakdown
        tfp = morgan_fp(lib.smiles[s])
        bt = 0.0
        if tfp is not None:
            for a in a_idx:
                afp = morgan_fp(lib.smiles[a])
                if afp is not None:
                    bt = max(bt, DataStructs.TanimotoSimilarity(tfp, afp))
        tans.append(bt)

        # --- fpnet channel --------------------------------------------------
        # Average the predicted bit probabilities across the molecule's spectra
        # in LOGIT space, which is the natural place to pool log-odds.
        # ONE implementation, shared with the ranker and the notebook. The
        # inline copy that used to live here is how the peak-selection and
        # collision-energy bugs survived: duplicated preprocessing drifts.
        mean_logit = predict_logits(
            model,
            [(raw[r][0], raw[r][2]) for r in rows],
            adduct, adducts,
            instrument_id_=instrument_id(raw[rows[0]][3]),
            device=dev,
            max_peaks=MAX_PEAKS,
            collision_energies=[raw[r][4] for r in rows],
        )
        s_fpnet = score_candidates(mean_logit, candidate_bits(pool.smiles[cand], bit_sel))

        # --- score all three -------------------------------------------------
        # A single blend weight has to serve two regimes. When a close analog
        # exists, analogy is strong evidence; when it does not, analog scores
        # are noise and averaging them in demotes the right answer. Sweep the
        # weight rather than assume, and read the HARD column — that is the
        # regime the real test set is mostly in.
        za, zf = zscore(s_analog), zscore(s_fpnet)
        combos = {"analog": s_analog, "fpnet": s_fpnet}
        for w in (0.25, 0.5, 1.0):
            combos[f"fpnet+{w:g}*analog"] = zf + w * za
        for name, sc in combos.items():
            order = np.argsort(-sc)[:K]
            ranked = list(ck_keys[order])
            res.setdefault(name, []).append(1.0 / (ranked.index(truth) + 1) if truth in ranked else 0.0)

    dt = time.time() - t0
    tans = np.array(tans)
    n = len(res["analog"])
    print(f"{n} molecules in {dt:.0f}s ({dt/max(n,1):.1f}s each)\n")

    hard = tans < 0.4
    print("| channel | all | CALIBRATED (analog Tanimoto < 0.4) |")
    print("|---|---:|---:|")
    for name in res:
        a = np.array(res[name])
        h = a[hard].mean() if hard.sum() else float("nan")
        print(f"| {name} | {a.mean():.4f} | {h:.4f} |")
    print(f"\n(hard bucket n={hard.sum()} of {n}; the calibrated column is the one "
          f"that tracked the leaderboard)")


if __name__ == "__main__":
    main()
