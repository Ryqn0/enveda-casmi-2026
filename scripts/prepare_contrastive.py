"""Build candidate lists for contrastive (ranking) fine-tuning of fpnet.

WHY
    fpnet is trained to predict fingerprint BITS (binary cross-entropy per bit).
    But at test time its job is different: tell apart the ~60 isomers that share
    the query's mass. Predicting bits well is only indirectly related to that.

    Spec2Mol-style training closes the gap: show the model a spectrum together
    with its REAL candidate list -- same-mass isomers, the hardest negatives
    that exist -- and train it to put the true structure first. The ranking
    score `cand_fp @ logits` is differentiable, so this optimises the exact
    task the leaderboard grades.

LEAKAGE RULES (docs/09)
    Honest evaluation samples molecules from the spectral library that the base
    model never trained on. Fine-tuning must not see those either -- not even
    as NEGATIVES, since "this spectrum is not X" still teaches the model about
    X's fingerprint. So:

      queries    : only spectra of the base model's TRAINING structures
      candidates : base-model training structures, plus pool structures that
                   are NOT in the spectral library (pure COCONUT). The honest
                   holdout is drawn from the library, so it can never appear.

    uv run python scripts/prepare_contrastive.py --base data/fpnet/fpnet_v4.pt
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
sys.path.insert(0, str(ROOT / "src"))

from casmi.analog import morgan_fp  # noqa: E402
from casmi.pool import ADDUCT_MASS, CandidatePool  # noqa: E402

DATA = ROOT / "data" / "fpnet"
FP_SIZE = 2048
MAX_NEG = 63      # negatives kept per query, so each list is <= 64 long


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True,
                    help="checkpoint whose train_keys define what may be used")
    ap.add_argument("--data", type=Path, default=DATA / "train.npz",
                    help="the npz the base model was trained on")
    ap.add_argument("--n-queries", type=int, default=200_000)
    ap.add_argument("--ppm", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA / "contrastive.npz")
    args = ap.parse_args()
    t0 = time.time()

    ck = torch.load(args.base, map_location="cpu", weights_only=False)
    assert ck.get("train_keys") is not None, "base checkpoint lacks train_keys"
    train_keys = set(np.asarray(ck["train_keys"]).tolist())

    d = np.load(args.data, allow_pickle=True)
    keys = d["keys"]                       # ONE read: d is a lazy NpzFile
    y_row, prec, add_id = d["y_row"], d["prec"], d["add_id"]
    adduct_list = list(d["adduct_list"])
    # Guard against pairing a checkpoint with the wrong npz: every one of its
    # training structures must exist in this file.
    assert train_keys <= set(keys.tolist()), "base model was not trained on --data"

    lib_keys = set(pd.read_parquet(ROOT / "data/index/structures.parquet")
                   .inchikey14.tolist())
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    # A pool structure may appear as a candidate iff it cannot be in any honest
    # holdout: it trained the base model, or it is not in the library at all.
    allowed = np.array([(k in train_keys) or (k not in lib_keys)
                        for k in pool.key])
    print(f"pool {len(pool.key):,} structures, {allowed.sum():,} allowed as candidates")

    # Queries: spectra whose structure the base model trained on.
    is_train = np.array([keys[r] in train_keys for r in y_row])
    rng = np.random.default_rng(args.seed)
    q = rng.permutation(np.flatnonzero(is_train))[: args.n_queries]
    print(f"{is_train.sum():,} training spectra, using {len(q):,} as queries")

    q_keep, lists, true_pos = [], [], []
    skipped = {"adduct": 0, "truth_missing": 0, "no_negatives": 0}
    for r in q:
        adduct = adduct_list[add_id[r]]
        if adduct not in ADDUCT_MASS:
            skipped["adduct"] += 1
            continue
        # No timsTOF offset: training spectra come from every instrument, and
        # the list only needs the truth plus realistic same-mass negatives.
        cand = pool.query(float(prec[r]), adduct, ppm=args.ppm, offset_ppm=0.0)
        # Match the truth by KEY, not by pool row. If the pool held two rows
        # with the true InChIKey14, a row-based test would leave the duplicate
        # among the negatives -- training the model that the right answer is
        # wrong. Key-based matching removes every copy.
        tkey = keys[y_row[r]]
        is_truth = pool.key[cand] == tkey
        if not is_truth.any():
            skipped["truth_missing"] += 1
            continue
        truth = cand[is_truth][0]
        neg = cand[~is_truth & allowed[cand]]
        if len(neg) == 0:
            # Nothing to contrast against; the BCE loss still covers these.
            skipped["no_negatives"] += 1
            continue
        if len(neg) > MAX_NEG:
            neg = rng.choice(neg, MAX_NEG, replace=False)
        lst = np.concatenate([[truth], neg])
        rng.shuffle(lst)                 # the truth must not always be first
        q_keep.append(r)
        lists.append(lst)
        true_pos.append(int(np.flatnonzero(lst == truth)[0]))

    print(f"kept {len(q_keep):,} queries; skipped {skipped}")
    sizes = np.array([len(l) for l in lists])
    print(f"list length: median {np.median(sizes):.0f}, mean {sizes.mean():.1f}, "
          f"max {sizes.max()}")

    # Fingerprint each distinct candidate ONCE, stored packed at full width so
    # the trainer can apply whichever bit selection the base model uses.
    uniq, inv = np.unique(np.concatenate(lists), return_inverse=True)
    print(f"fingerprinting {len(uniq):,} distinct candidates...")
    # Packed as we go: unpacked, ~1M candidates x 2048 bytes would be 2 GB,
    # on a machine already holding training and a validation run.
    fp_packed = np.zeros((len(uniq), FP_SIZE // 8), dtype=np.uint8)
    buf = np.zeros(FP_SIZE, dtype=np.uint8)
    for i, prow in enumerate(uniq):
        fp = morgan_fp(pool.smiles[prow])
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, buf)
            fp_packed[i] = np.packbits(buf)
    ptr = np.concatenate([[0], np.cumsum(sizes)])

    np.savez_compressed(
        args.out,
        q_rows=np.array(q_keep, dtype=np.int64),   # rows of --data
        ptr=ptr.astype(np.int64),                  # CSR offsets into cand
        cand=inv.astype(np.int32),                 # indices into fp_packed
        true_pos=np.array(true_pos, dtype=np.int16),
        fp_packed=fp_packed,
        base=str(args.base), data=str(args.data),
    )
    print(f"-> {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
