"""Harness for OUR pipeline with a learned ranker (Phase 8).

    uv run python scripts/own_harness.py evidence            # spectral searches, once (~30 min)
    uv run python scripts/own_harness.py fp --model data/fpnet/fpnet_v5.pt
    uv run python scripts/own_harness.py score --model fpnet_v5

Same protocol as the public-engine harness (docs/15), so the numbers compare:
the 250 enveda-np-examples structures, each simulated one molecule at a time
  c1  hide its query spectra (the enveda-np-examples rows)       -> class 1
  c2  hide every spectrum of its structure                       -> class 2
and predicted LB = 0.162*c1 + 0.220*c2 (the public calibration).

The ranker trains on 2,000 OTHER natural products (data/own/holdout.json, 400
from each of 5 libraries), simulated the same way. Neither set was seen by
fpnet v5 (prepare_fpnet_data.py --exclude-keys), so the fingerprint features
are as honest in training and evaluation as on the real test.

Features come from casmi.features -- the module the notebook will use too.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from casmi.features import (FEATURE_NAMES, _row_of_key, _z, candidate_features,  # noqa: E402
                            fit_ranker, fp_evidence, molecule_evidence, ranker_scores)
from casmi.library import Library  # noqa: E402
from casmi.pool import CandidatePool  # noqa: E402
from casmi.ranker import expected_rr  # noqa: E402

OWN = ROOT / "data" / "own"
C1_W, C2_W, K = 0.162, 0.220, 25
EVAL_LIB = "enveda-np-examples"
MAX_QUERY = 4

LIB = POOL = RAW = QUERIES = None        # set before forking, shared by workers


def build_queries(lib: Library) -> list[dict]:
    """One dict per held-out molecule: its library positions and query positions."""
    h = json.loads((OWN / "holdout.json").read_text())
    src = pq.read_table(ROOT / "data/raw/train.parquet", columns=["ingest_lib"]).column(0)
    src = np.asarray(src.to_numpy(zero_copy_only=False))[lib.meta.row_id.to_numpy()]
    rok = _row_of_key(lib)
    rng = np.random.default_rng(0)
    out = []
    todo = [("eval", EVAL_LIB, k) for k in h["eval"]]
    todo += [("train", L, k) for L, keys in h["train"].items() for k in keys]
    for split, L, key in todo:
        s = rok.get(key)
        if s is None:
            continue
        pos = lib.spectra_of(s)
        inL = pos[src[pos] == L]
        # eval: all its enveda-np-examples spectra, like a test molecule.
        # train: up to 4 from its library. Both must leave a spectrum behind,
        # or class 1 silently becomes class 2 (the Phase 1 degenerate holdout).
        q = inL if split == "eval" else rng.permutation(inL)[:min(MAX_QUERY, len(pos) - 1)]
        # Class 1 hides the molecule's spectra from the QUERY'S OWN library, so
        # what remains are other labs' and instruments' spectra -- as for an
        # eval molecule (its whole enveda-np-examples entry is hidden) and on
        # the real test. Hiding only the queries (first version) left same-lab
        # near-duplicates behind: training class 1 was nearly perfect identity
        # matches, and the ranker learned to expect them (class 1: 0.72 vs
        # 0.90 for plain analog on eval).
        c1_hide = np.sort(inL)
        if 0 < len(q) < len(pos):
            out.append({"split": split, "lib": L, "key": key, "struct": int(s),
                        "q_pos": np.sort(q), "c1_hide": c1_hide, "all_pos": pos,
                        "has_c1": len(c1_hide) < len(pos)})
    return out


def _evidence(job):
    i, sim = job
    q = QUERIES[i]
    rows = [RAW[r] for r in LIB.meta.row_id.to_numpy()[q["q_pos"]]]
    from prepare_fpnet_data import instrument_id
    ev = molecule_evidence(
        LIB, POOL, [(r[0], r[2]) for r in rows], [r[1] for r in rows],
        exclude=q["c1_hide"] if sim == "c1" else q["all_pos"],
        instrument_id=instrument_id(rows[0][3]),
        collision_energies=[r[4] if r[4] else 32.0 for r in rows])
    return (i, sim), ev


def evidence(workers: int, only: str | None = None) -> None:
    """only="train-c1": recompute just those and merge into evidence.pkl."""
    global LIB, POOL, RAW, QUERIES
    from validate_class2 import fetch_raw
    LIB = Library(ROOT / "data" / "index")
    POOL = CandidatePool(ROOT / "data" / "pool.parquet")
    QUERIES = build_queries(LIB)
    rows = {int(r) for q in QUERIES for r in LIB.meta.row_id.to_numpy()[q["q_pos"]]}
    RAW = fetch_raw(rows)
    n_eval = sum(q["split"] == "eval" for q in QUERIES)
    print(f"{len(QUERIES)} molecules ({n_eval} eval), {len(rows):,} query spectra; "
          f"{sum(q['has_c1'] for q in QUERIES)} can be simulated as class 1", flush=True)
    jobs = [(i, s) for i, q in enumerate(QUERIES) for s in ("c1", "c2")
            if s == "c2" or q["has_c1"]]
    out = {}
    if only:
        split, sim = only.split("-")
        prev = pickle.load(open(OWN / "evidence.pkl", "rb"))
        assert len(prev["queries"]) == len(QUERIES)
        out = {k: v for k, v in prev["ev"].items()
               if not (QUERIES[k[0]]["split"] == split and k[1] == sim)}
        jobs = [j for j in jobs if QUERIES[j[0]]["split"] == split and j[1] == sim]
        print(f"recomputing {len(jobs)} {only} simulations; keeping {len(out)}", flush=True)
    with Pool(workers) as p:          # fork: workers share the loaded index
        for n, (k, ev) in enumerate(p.imap_unordered(_evidence, jobs, chunksize=4)):
            out[k] = ev
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    with open(OWN / "evidence.pkl", "wb") as f:
        pickle.dump({"queries": QUERIES, "ev": out}, f)
    print(f"saved {OWN / 'evidence.pkl'}")


def fp(model_path: Path) -> None:
    import torch
    from casmi.fpnet import load
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    d = pickle.load(open(OWN / "evidence.pkl", "rb"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(model_path, device=dev)
    out = {k: fp_evidence(model, ck, ev, pool, device=dev) for k, ev in d["ev"].items()}
    p = OWN / f"fp_{model_path.stem}.pkl"
    pickle.dump(out, open(p, "wb"))
    print(f"saved {p} ({len(out)} molecule-simulations)")


def rows(d, fps, pool, split):
    """(X, y, class-1 flag, molecule-simulation id) for one split."""
    X, y, c1, gid = [], [], [], []
    for (i, sim), ev in d["ev"].items():
        q = d["queries"][i]
        if q["split"] != split or not len(ev["cand"]):
            continue
        ev = dict(ev, fp=fps[(i, sim)]) if fps is not None else ev
        X.append(candidate_features(ev))
        y.append(pool.key[ev["cand"]] == q["key"])
        c1.append(np.full(len(ev["cand"]), sim == "c1"))
        gid.append(np.full(len(ev["cand"]), 2 * i + (sim == "c2")))
    return (np.vstack(X), np.concatenate(y), np.concatenate(c1), np.concatenate(gid))


def score(args) -> None:
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    d = pickle.load(open(OWN / "evidence.pkl", "rb"))
    fps = pickle.load(open(OWN / f"fp_{args.model}.pkl", "rb")) if args.model else None
    Xtr, ytr, c1tr, _ = rows(d, fps, pool, "train")
    Xev, yev, c1ev, gev = rows(d, fps, pool, "eval")
    keep = [i for i, n in enumerate(FEATURE_NAMES) if not any(n.startswith(p) for p in args.drop)]
    print(f"train {len(Xtr):,} rows ({int(ytr.sum())} true), eval {len(Xev):,} rows; "
          f"{len(keep)} features" + (f" (dropped {args.drop})" if args.drop else ""))

    # Monotonic constraints: more evidence can never make a candidate LESS
    # likely. Without them the ranker learned "has library spectra ->
    # suspicious" from the class-2 simulations (where the truth is hidden, so
    # every candidate WITH spectra is wrong), and in class 1 overruled a 0.9
    # identity match for a candidate with no spectra at all (78/244 molecules).
    names = [FEATURE_NAMES[i] for i in keep]
    up = ("id_max_raw", "id_mean_raw", "hyb_self_raw", "analog_raw", "fp_raw", "frag_raw")
    down = ("id_max_rank", "id_mean_rank", "hyb_self_rank", "analog_rank", "fp_rank", "frag_rank")
    mono = [1 if n in up else -1 if n in down else 0 for n in names] if args.monotonic else None

    def fit_predict(Xa, ya, c1a, Xb):
        return ranker_scores(fit_ranker(Xa[:, keep], ya, c1a, mono), Xb[:, keep])

    if args.cv_eval:
        # Also train on the eval molecules' rows -- the only timsTOF ones --
        # 5-fold by MOLECULE (both simulations of a molecule stay together),
        # so each is scored by a model that never saw it.
        mol = gev // 2
        fold = dict(zip(np.unique(mol), np.random.default_rng(0).permutation(len(np.unique(mol))) % 5))
        f = np.array([fold[x] for x in mol])
        p = np.zeros(len(Xev))
        for k in range(5):
            tr, te = f != k, f == k
            p[te] = fit_predict(np.vstack([Xtr, Xev[tr]]), np.concatenate([ytr, yev[tr]]),
                                np.concatenate([c1tr, c1ev[tr]]), Xev[te])
    else:
        p = fit_predict(Xtr, ytr, c1tr, Xev)

    # Fixed-blend baselines on the SAME candidates: the 0.244-style pipeline.
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    configs = {"gbm ranker": p,
               "fixed: z(fp) + 0.5 z(analog)": None,
               "analog only": Xev[:, col["analog_raw"]],
               "fp only": Xev[:, col["fp_raw"]]}
    per = {}
    for name, s in configs.items():
        rr = {"c1": [], "c2": []}
        for g in np.unique(gev):
            sel = gev == g
            if s is None:
                sc = _z(Xev[sel, col["fp_raw"]]) + 0.5 * _z(Xev[sel, col["analog_raw"]])
            else:
                sc = s[sel]
            y = yev[sel]
            rr["c1" if c1ev[sel][0] else "c2"].append(
                expected_rr(sc, int(np.argmax(y)), K) if y.any() else 0.0)
        per[name] = rr
    n_ev = sum(q["split"] == "eval" for q in d["queries"])
    print(f"\n{n_ev} eval molecules; a molecule whose truth is outside the candidate "
          f"window scores 0 (retrieval)\n")
    print("| config | class 1 | class 2 | predicted LB | vs gbm (+/- 2 SE) |")
    print("|---|---:|---:|---:|---:|")
    base = C1_W * np.array(per["gbm ranker"]["c1"]) + C2_W * np.array(per["gbm ranker"]["c2"])
    for name, rr in per.items():
        c1, c2 = np.array(rr["c1"]), np.array(rr["c2"])
        dl = C1_W * c1 + C2_W * c2 - base
        se = dl.std(ddof=1) / np.sqrt(len(dl))
        print(f"| {name} | {c1.mean():.4f} | {c2.mean():.4f} | {C1_W * c1.mean() + C2_W * c2.mean():.4f} "
              f"| {dl.mean():+.4f} +/- {2 * se:.4f} |")


def export(args) -> None:
    """All simulation rows -> data/own/ranker_rows.npz, which the notebook ships
    and trains the ranker from. Train AND eval molecules: the eval ones are the
    only timsTOF rows, worth +0.07 on class 1 (--cv-eval)."""
    import torch
    pool = CandidatePool(ROOT / "data" / "pool.parquet")
    d = pickle.load(open(OWN / "evidence.pkl", "rb"))
    fps = pickle.load(open(OWN / f"fp_{args.model}.pkl", "rb"))
    parts = [rows(d, fps, pool, s) for s in ("train", "eval")]
    X, y, c1 = (np.concatenate([p[i] for p in parts]) for i in range(3))
    tag = torch.load(ROOT / "data" / "fpnet" / f"{args.model}.pt", map_location="cpu",
                     weights_only=False).get("model_tag")
    assert tag, "the fpnet checkpoint has no model_tag; the notebook checks it"
    out = OWN / "ranker_rows.npz"
    np.savez_compressed(out, X=X, y=y, c1=c1, feature_names=np.array(FEATURE_NAMES),
                        fpnet_tag=np.array(tag))
    print(f"wrote {out}: {len(X):,} rows x {X.shape[1]} features, {int(y.sum())} true, fpnet {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["evidence", "fp", "score", "export"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="evidence: e.g. train-c1, merged into the existing file")
    ap.add_argument("--model", default=None,
                    help="fp: checkpoint path; score: its stem (fp_<stem>.pkl), or omit for no fpnet")
    ap.add_argument("--drop", nargs="*", default=[], help="score: drop features by name prefix")
    ap.add_argument("--monotonic", action="store_true",
                    help="score: evidence features may only raise a candidate's score")
    ap.add_argument("--cv-eval", action="store_true",
                    help="score: also train on the eval molecules, 5-fold by molecule")
    args = ap.parse_args()
    if args.stage == "evidence":
        evidence(args.workers, args.only)
    elif args.stage == "fp":
        fp(Path(args.model))
    elif args.stage == "export":
        export(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
