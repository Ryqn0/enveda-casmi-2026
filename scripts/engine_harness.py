"""Offline harness for the WHOLE public engine (kaggle/public_engine, LB 0.341).

    uv run python scripts/engine_harness.py channels      # slow, once: library, pool, channels
    uv run python scripts/engine_harness.py score         # fast: rankers, blend, the table

Why: docs/13's lesson was "evaluate what you ship, not a component". The earlier
experiment scored ranker B alone and it did not predict the leaderboard. This
harness runs the engine's own code end to end -- channels, both rankers, the
rank blend, the de-duplicated top 25 -- so a change is judged on the output
that would be submitted.

Queries: the 250 structures of enveda-np-examples (1,184 timsTOF spectra), the
set the public calibration uses (predicted LB = 0.162*class1 + 0.220*class2).
Every one of them also has spectra in other libraries, so each is simulated
twice, one molecule at a time (the other 249 stay in the library, as they
would on the real test):
  c1  hide the molecule's own query spectra                        -> class 1
  c2  hide every spectrum of its structure                         -> class 2
The pool keeps all structures: class 2 means "known structure, no spectra".
Ranker B's training rows include these 250 molecules (its only timsTOF rows),
so by default it is cross-validated over them (--b-train, match_np_groups).

Known optimism: the public fingerprint models (ranker A's zlog) were trained on
library spectra, which include other spectra of these structures. Changes that
do not touch ranker A's fingerprint features are compared fairly; absolute
class-2 numbers are high.

The engine code is extracted from the notebook at run time (the %%writefile
cells), so the harness can never drift from what was submitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "kaggle" / "public_engine" / "casmi26-public-engine.ipynb"
PUB = ROOT / "data" / "public"
OUT = ROOT / "data" / "harness"
TRAIN = ROOT / "data" / "raw" / "train.parquet"
QUERY_LIB = "enveda-np-examples"
C1_W, C2_W = 0.162, 0.220          # public calibration: LB ~= 0.162*c1 + 0.220*c2
K = 25


def load_engine(nb: Path = NB):
    """Write the notebook's %%writefile modules to OUT/engine_src and import them."""
    src = OUT / "engine_src"
    src.mkdir(parents=True, exist_ok=True)
    for c in json.loads(nb.read_text())["cells"]:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and s.startswith("%%writefile "):
            head, body = s.split("\n", 1)
            (src / head.split()[1]).write_text(body)
    sys.path.insert(0, str(src))
    import casmi_engine as E
    E.ROOTS = [str(PUB)]            # find() searches here instead of /kaggle/input
    return E


def queries() -> pd.DataFrame:
    """The query spectra, shaped like test.parquet (molecule_id = structure)."""
    te = pd.read_parquet(TRAIN, filters=[("ingest_lib", "==", QUERY_LIB)],
                         columns=["inchikey14", "normalized_smiles", "ms2_mzs",
                                  "ms2_normalized_intensities", "precursor_mz", "adduct",
                                  "ionization_mode", "instrument_type", "collision_energy_ev"])
    return te.assign(molecule_id=te.inchikey14)


def channels(E, workers: int) -> None:
    import torch
    import pv_fp
    L = E.load_library(str(TRAIN))
    E.log(f"library {len(L['prec']):,} spectra")
    E.POOL = E.build_pool(L, workers)
    L["row_p"] = E.POOL["k2i"].reindex(L["ik"]).fillna(-1).values.astype(np.int64)

    te = queries()
    # load_library reads the file row group by row group, in order, so row i of
    # this column is row i of the library.
    lib_src = pq.read_table(TRAIN, columns=["ingest_lib"]).column(0).to_numpy(zero_copy_only=False)
    assert len(lib_src) == len(L["prec"])
    # Rows of each query structure, for the per-molecule class-2 hiding below.
    rows_of = {k: np.flatnonzero(L["ik"] == k) for k in te.inchikey14.unique()}
    E.log(f"{len(rows_of)} query structures; {int((lib_src == QUERY_LIB).sum()):,} query "
          f"spectra; {sum(len(v) for v in rows_of.values()):,} spectra of those structures")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fp_paths = sorted(glob.glob(str(PUB / "engine" / "fp_*.pt")))
    assert len(fp_paths) == 2, fp_paths
    models = pv_fp.load_fp_models(fp_paths, dev)[:2] + (dev,)
    extra = ()
    if fp2_variant(E):
        # The fpb variants feed ranker B from megayak's leak-free models.
        fp2 = sorted(glob.glob(str(PUB / "ours_fpnet_*.pt")))
        assert len(fp2) == 2, fp2
        extra = (pv_fp.load_fp_models(fp2, dev)[:2] + (dev,),)

    # Hide ONE molecule at a time. The other 249 are ordinary training rows: on
    # the real test they ARE in the library, and many are isomers of each
    # other. (The first version hid all 250 at once; the query's candidates then
    # had far fewer library spectra than in the authors' rows -- found by
    # comparing feature columns, docs/15.)
    is_query_row = lib_src == QUERY_LIB
    del L["ik"], L["smi"], lib_src
    I = E.build_index(L)
    # The analog search runs on a prebuilt one-spectrum-per-structure table.
    # Build it without query spectra, so no molecule can meet its own spectrum
    # there; all 250 structures have spectra in other libraries to stand in.
    rows = np.where(np.isfinite(L["nm"]) & (L["row_p"] >= 0) & ~is_query_row)[0]
    I["all"] = E._reps(L, rows)
    I["pos"] = E._reps(L, rows[L["pol"][rows] == 1])
    I["neg"] = E._reps(L, rows[L["pol"][rows] == -1])

    # c1 hides the molecule's query spectra; c2 every spectrum of its structure,
    # and the structure itself from the analog results (row_p = -1 only reaches
    # the library search, which is how build_index already skips rows).
    search = E.analog_search
    recs = {"c1": [], "c2": []}
    for mid, sub in te.groupby("molecule_id", sort=True):
        p = int(E.POOL["k2i"].get(mid, -1))
        for sim, hide in (("c1", rows_of[mid][is_query_row[rows_of[mid]]]), ("c2", rows_of[mid])):
            old = L["row_p"][hide].copy()
            L["row_p"][hide] = -1

            def analog_without_self(R, Q, target, p=p):
                ps, ss = search(R, Q, target)
                return ps[ps != p], ss[ps != p]
            E.analog_search = analog_without_self if sim == "c2" else search
            recs[sim] += E.compute_channels(L, I, models, sub, *extra)[1]
            L["row_p"][hide] = old
    E.analog_search = search
    del L, I
    for sim in recs:
        E.compute_frag(recs[sim], workers)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pool.npz", fp=E.POOL["fp"], mass=E.POOL["mass"],
             keys=E.POOL["keys"].astype(str), smiles=E.POOL["smiles"].astype(str))
    truth = te.groupby("molecule_id").normalized_smiles.first().to_dict()
    with open(recs_path(E), "wb") as f:
        pickle.dump({"recs": recs, "truth": truth}, f)
    E.log(f"saved {recs_path(E).relative_to(ROOT)} and pool.npz")


def fp2_variant(E) -> bool:
    """True for the fpb variants (a second fingerprint model set for ranker B)."""
    return hasattr(E, "FP_COLS")


def recs_path(E) -> Path:
    # Channels differ only when a second model set adds zlog2; the ties
    # variant changes features, not channels, so it shares recs.pkl.
    return OUT / ("recs_fp2.pkl" if fp2_variant(E) else "recs.pkl")


def b_serve(E) -> dict:
    """How ranker B is served: no fingerprints, or the fpb variants' zlog2."""
    return dict(use_fp=True, zkey="zlog2") if fp2_variant(E) else dict(use_fp=False)


def match_np_groups(E, recs) -> dict:
    """Which of ranker B's np training groups belongs to which query molecule.

    Ranker B ships trained on rows simulated from these same 250 molecules --
    the only timsTOF rows it has, so dropping them all handicaps it, and keeping
    them all tests it on its training data. Cross-validation needs the mapping,
    and the row file has only anonymous group ids. So: rebuild the TRUE
    candidate's feature row for each (molecule, simulation) here and take the
    nearest of their groups with the same class. Measured: median distance
    0.002 (c1) / 0.026 (c2), 249/250 and 245/250 groups matched one-to-one.
    Returns {group id: molecule}; unmatched np groups are left out.
    """
    p = OUT / "group_match.json"
    if p.exists():
        return {int(k): v for k, v in json.loads(p.read_text()).items()}
    s = np.load(PUB / "sim_rank_rows_nofp.npz", allow_pickle=True)
    m = s["Q"] == list(s["sets"]).index("np")
    X, Y, S, G = s["X"][m], s["Y"][m], s["S"][m], s["G"][m]
    blocks = [str(b) for b in s["blocks"]]
    out = {}
    for sim, sv in (("c1", 0), ("c2", 1)):
        gs = [g for g in np.unique(G[S == sv])]
        T = np.stack([X[(G == g) & (Y == 1)][0] for g in gs])
        for r in recs[sim]:
            t = np.flatnonzero(E.POOL["keys"][r["cand"]] == r["mid"]) if len(r["cand"]) else []
            if not len(t):
                continue
            rr = dict(r, zlog=None)
            # Features are relative to the molecule's other candidates (rank,
            # gap to best), so build the whole matrix and take the truth's row.
            x = np.hstack([E.BLOCKS[b](rr, E.pool_fps(r["cand"])) for b in blocks])[t[0]]
            out[int(gs[int(np.abs(T - x).mean(1).argmin())])] = r["mid"]
    p.write_text(json.dumps(out))
    return out


# --- tie handling in the rank features ------------------------------------------
# pv._rank_norm ranks with np.argsort, so TIED candidates are ordered by their
# position in the candidate list. In class 2 the truth ties at library score 0
# with ~36 others; its "rank" then depends on list order, which differs between
# the authors' simulation (truth early: mean 0.468) and serving (mass order:
# 0.762). Tie-averaged, both are 0.63-0.64. --ties average removes the position
# signal on BOTH sides: serving (patched _rank_norm) and training rows
# (recomputed from the raw column each rank column was computed from).
# (rank column, raw column): base block, then libx (+31), anapol (+39), fragx (+44).
RANK_COLS_BASE = [(1, 0), (6, 5), (16, 15), (19, 18), (22, 21)]
RANK_COLS_B = RANK_COLS_BASE + [(32, 31), (35, 34), (40, 39), (45, 44), (48, 47)]


def rank_norm_avg(x):
    """pv._rank_norm with ties averaged: 0 = best, 1 = worst."""
    from scipy.stats import rankdata
    x = np.asarray(x)
    return (rankdata(-x, method="average") - 1) / max(1, len(x) - 1)


def retie(X, G, pairs):
    """Recompute rank columns per group, ties averaged. Checks the pairing first:
    where a group has no ties the old and new values must be equal."""
    X = X.copy()
    df = pd.DataFrame({"G": G})
    n = df.groupby("G").G.transform("size").to_numpy()
    for rc, vc in pairs:
        df["v"] = X[:, vc]
        new = (df.groupby("G").v.rank(ascending=False, method="average").to_numpy() - 1) / np.maximum(n - 1, 1)
        untied = df.groupby("G").v.transform("nunique").to_numpy() == n
        assert np.allclose(X[untied, rc], new[untied], atol=1e-5), f"column {rc} is not the rank of {vc}"
        X[:, rc] = new
    return X


def fit_ours(E, exclude_groups=None, keep_np: bool = False, ties: str = "engine",
             np_weight: float = 1.0):
    """Ranker B exactly as fit_rankers, on its rows minus some np groups.

    keep_np: all rows (leaky; what ships). exclude_groups: the np groups to
    leave out. Neither: leave out every np group. np_weight multiplies the
    sample weight of the np rows (the only timsTOF, test-like ones).
    """
    z = np.load(PUB / "sim_rank_rows_nofp.npz", allow_pickle=True)
    is_np = z["Q"] == list(z["sets"]).index("np")
    if keep_np:
        keep = np.ones(len(z["Y"]), bool)
    elif exclude_groups is None:
        keep = ~is_np
    else:
        keep = ~is_np | ~np.isin(z["G"], list(exclude_groups))
    X, Y, S = z["X"][keep], z["Y"][keep], z["S"][keep]
    if fp2_variant(E):
        # As the fpb engine's fit_rankers: fingerprint columns from the
        # leak-free rows (same candidates, same order -- checked there).
        zf = np.load(PUB / "sim_rank_rows_fppair.npz", allow_pickle=True)
        assert np.array_equal(zf["G"], z["G"]) and np.array_equal(zf["Y"], z["Y"])
        X = X.copy()
        X[:, E.FP_COLS] = zf["X"][keep][:, E.FP_COLS]
    if hasattr(E, "retie"):             # a variant notebook that reties itself
        X = E.retie(X, z["G"][keep], E.RANK_COLS_B)
    elif ties == "average":
        X = retie(X, z["G"][keep], RANK_COLS_B)
    boost = np.where(is_np[keep], np_weight, 1.0)
    rankers = [E._hgb(X, Y, np.where(S == 0, wA, 1.0 - wA) * boost, sd, E.RANK.GBM)
               for wA in E.RANK.W_A for sd in E.RANK.SEEDS]
    return rankers, [str(b) for b in z["blocks"]]


def variant_tag(args) -> str:
    """Cache-name suffix: fitted models depend on the notebook, tie mode, np weight."""
    return (("" if args.ties == "engine" else "_tiesavg")
            + ("" if args.nb.resolve() == NB.resolve() else f"_{args.nb.stem}")
            + ("" if args.np_weight == 1.0 else f"_npw{args.np_weight:g}"))


def ours_scores(E, recs, args) -> dict:
    """{sim: {molecule: ranker-B scores}} under the chosen training regime."""
    tag = variant_tag(args)
    if args.b_train in ("nonp", "all"):
        rk, blocks = cached(f"ours_{args.b_train}{tag}",
                            lambda: fit_ours(E, keep_np=args.b_train == "all", ties=args.ties,
                                             np_weight=args.np_weight))
        return {sim: E.score_molecules(recs[sim], rk, blocks, **b_serve(E)) for sim in recs}
    # cv: molecule-level folds; a fold's model never sees that fold's groups.
    g2m = match_np_groups(E, recs)
    s = np.load(PUB / "sim_rank_rows_nofp.npz", allow_pickle=True)
    np_groups = set(np.unique(s["G"][s["Q"] == list(s["sets"]).index("np")]).tolist())
    mids = sorted({r["mid"] for r in recs["c1"]})
    fold = dict(zip(mids, np.random.default_rng(0).permutation(len(mids)) % 5))
    out = {sim: {} for sim in recs}
    for k in range(5):
        # Excluded = this fold's groups + every np group we could not match.
        allowed = {g for g, m in g2m.items() if fold[m] != k}
        rk, blocks = cached(f"ours_cv{k}{tag}",
                            lambda: fit_ours(E, exclude_groups=np_groups - allowed, ties=args.ties,
                                             np_weight=args.np_weight))
        for sim in recs:
            part = [r for r in recs[sim] if fold[r["mid"]] == k]
            out[sim].update(E.score_molecules(part, rk, blocks, **b_serve(E)))
    return out


def cached(name: str, make):
    """Fitting takes minutes; the fitted models only change when their inputs do."""
    p = OUT / f"{name}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    obj = make()
    pickle.dump(obj, open(p, "wb"))
    return obj


def reciprocal_ranks(E, recs, scores, truth, ckey) -> pd.Series:
    """What the metric would give each molecule for these scores.

    Same order and de-duplication as make_submissions: stable sort, then skip a
    candidate whose tautomer-canonical key was already listed.
    """
    out = {}
    for r in recs:
        mid, rr = r["mid"], 0.0
        if len(r["cand"]) and mid in scores:
            order = r["cand"][np.argsort(-scores[mid], kind="mergesort")][: K + 15]
            seen = []
            for c in order:
                c = int(c)
                if c not in ckey:
                    ckey[c] = E.canon_key(E.POOL["smiles"][c]) or E.POOL["keys"][c]
                if ckey[c] not in seen:
                    seen.append(ckey[c])
                if len(seen) == K:
                    break
            if truth[mid] in seen:
                rr = 1.0 / (seen.index(truth[mid]) + 1)
        out[mid] = rr
    return pd.Series(out)


def score(E, args) -> None:
    z = np.load(OUT / "pool.npz", allow_pickle=True)
    E.POOL = {k: z[k] for k in ("fp", "mass", "keys", "smiles")}
    d = pickle.load(open(recs_path(E), "rb"))
    recs = d["recs"]
    truth = {m: E.canon_key(s) for m, s in d["truth"].items()}

    if args.ties == "average":
        z = np.load(PUB / "engine" / "rank_train.npz")
        p = OUT / "rank_train_tiesavg.npz"
        if not p.exists():
            # In these rows G is the molecule and M its simulation (class 1/2):
            # a candidate list is a (G, M) pair.
            groups = z["G"] * 2 + z["M"].astype(int)
            np.savez(p, X=retie(z["X"], groups, RANK_COLS_BASE), Y=z["Y"], M=z["M"], G=z["G"])
        pvr, _ = cached("pv_tiesavg", lambda: E.fit_pv_rankers(str(p)))
        import pv
        pv._rank_norm = rank_norm_avg      # serving side; E._rel reads it from pv too
    else:
        # np_weight only concerns ranker B, so it must not force a refit of A.
        pvr, _ = cached(f"pv{variant_tag(args).split('_npw')[0]}",
                        lambda: E.fit_pv_rankers(str(PUB / "engine" / "rank_train.npz")))
    s_ours_all = ours_scores(E, recs, args)

    ckey, per = {}, {}
    for sim in ("c1", "c2"):
        s_ours = s_ours_all[sim]
        s_pv = E.score_molecules(recs[sim], pvr, ["base"], use_fp=True)
        sets = {"pv": s_pv, "ours": s_ours}
        for w in args.w_pv:
            sets[f"blend {w:g}"] = E.blend_scores(s_pv, s_ours, w)
        for w in args.prob_blend:
            # Probabilities instead of within-molecule ranks: keeps how SURE
            # each ranker is, which the rank blend throws away.
            sets[f"prob {w:g}"] = {m: w * s_pv[m] + (1 - w) * s_ours[m] for m in s_pv}
        for name, sc in sets.items():
            per[(name, sim)] = reciprocal_ranks(E, recs[sim], sc, truth, ckey)

    names = list(dict.fromkeys(n for n, _ in per))
    lb = {n: C1_W * per[(n, "c1")] + C2_W * per[(n, "c2")] for n in names}
    base = lb[f"blend {args.w_pv[0]:g}"] if len(args.w_pv) else lb[names[0]]
    print(f"\n{len(truth)} molecules | ranker B training: {args.b_train} | ties: {args.ties}\n")
    print("| config | class 1 | class 2 | predicted LB | vs first blend (+/- 2 SE) |")
    print("|---|---:|---:|---:|---:|")
    for n in names:
        dlt = (lb[n] - base).to_numpy()
        se = dlt.std(ddof=1) / np.sqrt(len(dlt))
        print(f"| {n} | {per[(n, 'c1')].mean():.4f} | {per[(n, 'c2')].mean():.4f} | "
              f"{lb[n].mean():.4f} | {dlt.mean():+.4f} +/- {2 * se:.4f} |")
    pd.DataFrame({f"{n}|{s}": v for (n, s), v in per.items()}).to_parquet(OUT / "last_score.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["channels", "score"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--w-pv", type=float, nargs="*", default=[0.88, 0.5, 0.7, 0.95])
    ap.add_argument("--b-train", choices=["cv", "nonp", "all"], default="cv",
                    help="ranker B: 5-fold CV over the query molecules (default), "
                         "no query-set rows, or all rows (leaky; what ships)")
    ap.add_argument("--ties", choices=["engine", "average"], default="engine",
                    help="rank features: the engine's position-broken ties, or averaged ties")
    ap.add_argument("--prob-blend", type=float, nargs="*", default=[],
                    help="also score probability blends w*A + (1-w)*B")
    ap.add_argument("--np-weight", type=float, default=1.0,
                    help="multiply the sample weight of ranker B's timsTOF (np) rows")
    ap.add_argument("--nb", type=Path, default=NB,
                    help="engine notebook to test (default: the unmodified public engine)")
    args = ap.parse_args()
    E = load_engine(args.nb)
    channels(E, args.workers) if args.stage == "channels" else score(E, args)


if __name__ == "__main__":
    main()
