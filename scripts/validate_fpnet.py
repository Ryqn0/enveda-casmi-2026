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
import pandas as pd
import torch
from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_fpnet_data import instrument_id  # noqa: E402
from validate_class2 import fetch_raw, sample_structures  # noqa: E402

from casmi.analog import morgan_fp, rank_by_analogy  # noqa: E402
from casmi.fpnet import load, predict_logits, score_candidates  # noqa: E402
from casmi.fragment import score_spectra_frag  # noqa: E402
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
    ap.add_argument("--ppm", type=float, default=None,
                    help="override the mass window; default uses rank.PPM")
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "fpnet" / "fpnet.pt")
    # A/B switch for per-bit Platt calibration. Default on when the checkpoint
    # has it; --no-platt reproduces the raw-sigmoid scoring that shipped at LB
    # 0.244, so the two are comparable on identical candidates.
    ap.add_argument("--no-platt", action="store_true")
    # Restrict the holdout to structures fpnet never trained on. Without this,
    # 80.3% of the sample is memorised and every fpnet number is inflated.
    ap.add_argument("--unseen-only", action="store_true")
    # Comparing two models? Each must be graded on molecules NEITHER trained
    # on, or they are sitting different exams. Pass the other checkpoint here
    # (repeatable) and its train_keys are excluded too.
    ap.add_argument("--exclude-seen-by", type=Path, action="append", default=[])
    # Natural-product holdout: only structures present in COCONUT.
    ap.add_argument("--coconut-only", action="store_true")
    # In-silico fragmentation is the slowest channel by far, so it is opt-in.
    ap.add_argument("--frag", action="store_true")
    # Also allow cross-ring cleavage of aliphatic rings (casmi/fragment.py).
    ap.add_argument("--frag-rings", action="store_true")
    # Write per-candidate features + label for training a learned ranker.
    ap.add_argument("--dump", type=Path, default=None)
    # Fraction of queries simulated as CLASS 1: hide only the query spectra and
    # leave the molecule's other spectra in the library. The default holdout
    # is pure class 2, but ~15% of the test set is class 1, where the analog
    # channel finds a near-perfect self-match. A ranker that never saw that
    # regime could mishandle the easiest points on the leaderboard.
    ap.add_argument("--class1-frac", type=float, default=0.0)
    # Keep the per-bit reliability weights A, drop the bias B. Full calibration
    # ranks by cand_fp @ (A*z + B); with B ~ -3.25 that is dominated by a
    # -3.25 * (bits set) complexity penalty, which is a prior about molecules,
    # not evidence from the spectrum. A alone is the part that reweights bits
    # by how well the model actually predicts them.
    ap.add_argument("--platt-a-only", action="store_true")
    args = ap.parse_args()
    assert not args.dump or args.frag, "--dump needs --frag: the ranker uses all three channels"
    rng_c1 = np.random.default_rng(args.seed + 1000)
    dump_rows = []

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(args.model, device=dev)
    a, b = ck.get("platt_a"), ck.get("platt_b")
    platt = ((np.asarray(a),
              np.zeros_like(np.asarray(b)) if args.platt_a_only else np.asarray(b))
             if a is not None and not args.no_platt else None)
    mode = "OFF" if platt is None else ("A-ONLY" if args.platt_a_only else "FULL")
    print(f"platt calibration: {mode}")
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
    # Prefer the checkpoint's own record of what it trained on.
    seen = None
    if args.unseen_only and ck.get("train_keys") is not None:
        seen = set(np.asarray(ck["train_keys"]).tolist())
        for other in args.exclude_seen_by:
            ock = torch.load(other, map_location="cpu", weights_only=False)
            assert ock.get("train_keys") is not None, f"{other} lacks train_keys"
            seen |= set(np.asarray(ock["train_keys"]).tolist())
            print(f"also excluding structures seen by {other.name}")
    holdout = sample_structures(lib, pool, args.n, args.seed, sources,
                                unseen_only=args.unseen_only, seen_keys=seen,
                                coconut_only=args.coconut_only)
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
        cand = pool.query(mz0, adduct, ppm=(args.ppm or PPM),
                          offset_ppm=MZ_OFFSET_PPM)
        if len(cand) == 0:
            continue
        ck_keys = pool.key[cand]
        is_c1 = bool(rng_c1.random() < args.class1_frac)
        exclude = (lib.positions_of_rows(h["query_rows"]) if is_c1
                   else lib.spectra_of(s))   # class 2: no spectra of this molecule

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
        s_fpnet = score_candidates(
            mean_logit, candidate_bits(pool.smiles[cand], bit_sel), platt=platt)

        # --- fragment channel -----------------------------------------------
        # Needs no library and no relative, so unlike the other two it still
        # says something when the molecule is genuinely novel (class 3).
        # Which spectrum it reads is decided inside score_spectra_frag, the
        # same function the ranker and the notebook call.
        s_frag = (score_spectra_frag(pool.smiles[cand],
                                     [(raw[r][0], raw[r][2]) for r in rows],
                                     adduct, rings=args.frag_rings)
                  if args.frag else None)

        # --- score all three -------------------------------------------------
        # A single blend weight has to serve two regimes. When a close analog
        # exists, analogy is strong evidence; when it does not, analog scores
        # are noise and averaging them in demotes the right answer. Sweep the
        # weight rather than assume, and read the HARD column — that is the
        # regime the real test set is mostly in.
        za, zf = zscore(s_analog), zscore(s_fpnet)
        combos = {"analog": s_analog, "fpnet": s_fpnet}
        for w in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            combos[f"fpnet+{w:g}*analog"] = zf + w * za
        if s_frag is not None:
            zg = zscore(s_frag)
            combos["frag"] = s_frag
            # Added on top of the two blends worth caring about, at a few
            # weights. If fragmentation carries independent information these
            # beat the same blend without it; if it does not, they will not.
            for wf in (0.25, 0.5, 1.0):
                combos[f"fpnet+0.5*analog+{wf:g}*frag"] = zf + 0.5 * za + wf * zg
                combos[f"fpnet+1*analog+{wf:g}*frag"] = zf + 1.0 * za + wf * zg
        for name, sc in combos.items():
            order = np.argsort(-sc)[:K]
            ranked = list(ck_keys[order])
            res.setdefault(name, []).append(1.0 / (ranked.index(truth) + 1) if truth in ranked else 0.0)

        if args.dump:
            # EVERY feature here must be computable at test time from the
            # spectrum and the candidate list alone. `bt` above is NOT: it is
            # the Tanimoto of the TRUE answer to the analogs, and using it as
            # a feature would hand the ranker the label.
            n = len(cand)
            rank_of = lambda x: np.argsort(np.argsort(-x)) / n   # 0 = top
            dump_rows.append(pd.DataFrame({
                "qid": s, "is_c1": is_c1,
                # which fpnet produced these features: a ranker trained on one
                # model's features must never be served with another model
                "fpnet_tag": ck.get("model_tag") or args.model.stem,
                "label": (ck_keys == truth).astype(np.int8),
                # the three channels, on a per-query common scale
                "z_fp": zf, "z_an": za, "z_fr": zg,
                # where the candidate sits in each channel's own ordering
                "rk_fp": rank_of(s_fpnet), "rk_an": rank_of(s_analog),
                "rk_fr": rank_of(s_frag),
                # absolute evidence strength, which z-scoring throws away
                "an_raw": s_analog, "fr_raw": s_frag,
                "fp_gap": s_fpnet - s_fpnet.max(),
                # query-level context: how much to trust each channel HERE
                "an_conf": float(best[a_idx[0]]),
                "log_ncand": np.log(n),
                "log_npk": np.log(max(len(raw[r][2]) for r in rows)),
            }))

    dt = time.time() - t0
    tans = np.array(tans)
    n = len(res["analog"])
    print(f"{n} molecules in {dt:.0f}s ({dt/max(n,1):.1f}s each)\n")

    hard = tans < 0.4
    print("| channel | all | hard (analog Tanimoto < 0.4) |")
    print("|---|---:|---:|")
    for name in res:
        a = np.array(res[name])
        h = a[hard].mean() if hard.sum() else float("nan")
        print(f"| {name} | {a.mean():.4f} | {h:.4f} |")
    print(f"\n(hard bucket n={hard.sum()} of {n} -- too small to read; use ALL. "
          f"Without --unseen-only, ~80% of the sample is memorised by fpnet "
          f"and every fpnet number is inflated. See docs/09.)")

    if args.dump:
        out = pd.concat(dump_rows, ignore_index=True)
        out.to_parquet(args.dump, index=False)
        print(f"\ndumped {len(out):,} candidate rows / {out.qid.nunique():,} "
              f"queries ({out.groupby('qid').is_c1.first().mean():.0%} class 1) "
              f"-> {args.dump}")


if __name__ == "__main__":
    main()
