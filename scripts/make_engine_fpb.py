"""Build engine variants: ranker B gets leak-free fingerprint features.

    uv run python scripts/make_engine_fpb.py 0.88 kaggle/engine_fpb_w088 casmi26-engine-fpb-w088
    uv run python scripts/make_engine_fpb.py 0.50 kaggle/engine_fpb_w050 casmi26-engine-fpb-w050

Starts from kaggle/public_engine (unmodified W088, LB 0.341) and changes ONE
component: ranker B ("ours", megayak's 51-feature ranker).

  before  trained on sim_rank_rows_nofp.npz -- its 12 fingerprint columns are
          zero -- and served with fingerprints switched off
  after   the same rows with those 12 columns filled from
          sim_rank_rows_fppair.npz, served with logits from megayak's
          LEAK-FREE fingerprint models (ours_fpnet_single_16k / merged_26k),
          never the public ones ranker A uses

Offline (scripts/cv_public_ranker.py, enveda-np-examples, 5-fold grouped CV):
ranker B class 2 0.655 -> 0.734, predicted LB +0.013 for B alone.

Train/serve guards added to the engine:
  - the two row files must be aligned, and must differ in exactly FP_COLS
  - the served fingerprint features are logged next to the training ones, so
    a scale or definition mismatch is visible before submitting
"""

import json
import sys
from pathlib import Path

W_PV, OUT_DIR, SLUG = float(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle/public_engine"

nb = json.loads((SRC / "casmi26-public-engine.ipynb").read_text())
cells = nb["cells"]


def src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = s.splitlines(keepends=True)


def sub(s, old, new):
    assert s.count(old) == 1, f"expected exactly one match for:\n{old}"
    return s.replace(old, new)


# ---- the engine module (the %%writefile casmi_engine.py cell) ----------------
eng = [c for c in cells if src(c).startswith("%%writefile casmi_engine.py")]
assert len(eng) == 1
e = src(eng[0])

e = sub(e, "T0 = time.time()\n", '''T0 = time.time()

# --- leak-free fingerprints for ranker B (ryqn00 variant) ---------------------
# Columns of ranker B's 51-feature rows that are computed from fingerprint
# logits. Zero in sim_rank_rows_nofp.npz; filled in sim_rank_rows_fppair.npz by
# megayak's leak-free models. Verified offline: the two files hold the same
# candidates in the same order and differ in exactly these columns.
FP_COLS = [15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30]
FP_TRAIN_STATS = None      # (mean, std) of FP_COLS in the training rows
FP_SERVE_ROWS = []         # FP_COLS as served at test time, for the comparison
''')

e = sub(e, "def compute_channels(L, I, models, te):",
        "def compute_channels(L, I, models, te, models2=None):")
e = sub(e, '''                                         ces, modes)
            u, lmax, lmean, lsh, lcnt = lib''',
        '''                                         ces, modes)
            # Second set of logits from the leak-free models, for ranker B
            # only. Same inputs, same function -- only the weights differ.
            zlog2 = (pv_fp.molecule_logits(models2, specs, sub.precursor_mz.values.astype(np.float32),
                                           list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),
                                           ces, modes) if models2 is not None else None)
            u, lmax, lmean, lsh, lcnt = lib''')
e = sub(e, "rec.update(target=target, cand=cand, zlog=None if zlog is None else zlog.astype(np.float32), specs=specs,",
        "rec.update(target=target, cand=cand, zlog=None if zlog is None else zlog.astype(np.float32),\n"
        "                       zlog2=None if zlog2 is None else zlog2.astype(np.float32), specs=specs,")

e = sub(e, '''def fit_rankers(rank_train_path):
    """Our ranker: simulation rows from 7 query sets (S = 0 class-1 sim, 1 = class-2 sim)."""
    z = np.load(rank_train_path, allow_pickle=True)
    X, Y, S = z["X"], z["Y"], z["S"]''',
        '''def fit_rankers(rank_train_path, fp_rows_path=None):
    """Our ranker: simulation rows from 7 query sets (S = 0 class-1 sim, 1 = class-2 sim).

    With fp_rows_path, the 12 fingerprint columns are filled from the leak-free
    fppair rows. Both files must describe the same candidates in the same order.
    """
    global FP_TRAIN_STATS
    z = np.load(rank_train_path, allow_pickle=True)
    X, Y, S = z["X"], z["Y"], z["S"]
    if fp_rows_path:
        zf = np.load(fp_rows_path, allow_pickle=True)
        for k in ("Y", "S", "G", "Q"):
            assert np.array_equal(z[k], zf[k]), f"fppair rows not aligned with nofp rows on {k}"
        assert np.all(X[:, FP_COLS] == 0), "nofp fingerprint columns are not all zero"
        diff = [j for j in range(zf["X"].shape[1]) if not np.allclose(X[:, j], zf["X"][:, j])]
        assert diff == FP_COLS, f"fppair differs from nofp in columns {diff}, expected {FP_COLS}"
        X = X.copy(); X[:, FP_COLS] = zf["X"][:, FP_COLS]
        FP_TRAIN_STATS = (X[:, FP_COLS].mean(0), X[:, FP_COLS].std(0))
        log("our ranker: fingerprint columns filled from leak-free fppair rows")''')

e = sub(e, '''def score_molecules(recs, rankers, blocks, use_fp):
    out = {}
    for r in recs:
        if not len(r["cand"]): continue
        rr = r if use_fp else dict(r, zlog=None)
        cfp = pool_fps(r["cand"])
        Xm = np.hstack([BLOCKS[b](rr, cfp) for b in blocks]).astype(np.float32)''',
        '''def score_molecules(recs, rankers, blocks, use_fp, zkey="zlog"):
    out = {}
    for r in recs:
        if not len(r["cand"]): continue
        rr = dict(r, zlog=r.get(zkey)) if use_fp else dict(r, zlog=None)
        cfp = pool_fps(r["cand"])
        Xm = np.hstack([BLOCKS[b](rr, cfp) for b in blocks]).astype(np.float32)
        if zkey == "zlog2" and use_fp:
            FP_SERVE_ROWS.append(Xm[:, FP_COLS])''')

e = sub(e, "def main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88):",
        "def main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88,\n"
        "         fp2_model_paths=None, fp_rows_path=None):")
e = sub(e, '''    single, merged, _ = pv_fp.load_fp_models(fp_model_paths, dev)
    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}")''',
        '''    single, merged, nb1 = pv_fp.load_fp_models(fp_model_paths, dev)
    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}")
    models2 = None
    if fp2_model_paths:
        s2, m2, nb2 = pv_fp.load_fp_models(fp2_model_paths, dev)
        # Features are dot products with the POOL's fingerprints, so the
        # leak-free models must predict the same bit layout.
        assert nb2 == nb1, f"leak-free models predict {nb2} bits, public ones {nb1}"
        assert len(s2) == 1 and len(m2) == 1, f"expected 1 single + 1 merged leak-free model, got {len(s2)} + {len(m2)}"
        models2 = (s2, m2, dev)
        log(f"leak-free fp models for ranker B: {len(s2)} single + {len(m2)} merged")''')
e = sub(e, "    mols, recs = compute_channels(L, I, (single, merged, dev), te)",
        "    mols, recs = compute_channels(L, I, (single, merged, dev), te, models2)")
e = sub(e, '''    ours, blocks = fit_rankers(our_rank_path)
    s_ours = score_molecules(recs, ours, blocks, use_fp=False)''',
        '''    ours, blocks = fit_rankers(our_rank_path, fp_rows_path if models2 is not None else None)
    s_ours = score_molecules(recs, ours, blocks, use_fp=models2 is not None, zkey="zlog2")
    if FP_SERVE_ROWS and FP_TRAIN_STATS is not None:
        # Train/serve check: the same 12 features, as trained on and as served.
        # Different populations (simulation vs test), so they will not match
        # exactly -- but a wrong definition shows up as a different SCALE.
        served = np.vstack(FP_SERVE_ROWS)
        tm, ts = FP_TRAIN_STATS
        for j, c in enumerate(FP_COLS):
            log(f"  fp col {c:2d}: train mean {tm[j]:9.3f} std {ts[j]:8.3f} | served mean {served[:, j].mean():9.3f} std {served[:, j].std():8.3f}")''')
set_src(eng[0], e)

# ---- the run cell -------------------------------------------------------------
run = [c for c in cells if "subs, recs = E.main(" in src(c)]
assert len(run) == 1
r = src(run[0])
r = sub(r, "print('fp models:', fp_models)",
        "print('fp models:', fp_models)\n"
        "fp2_models = sorted(glob.glob('/kaggle/input/**/ours_fpnet_*.pt', recursive=True))\n"
        "print('leak-free fp models for ranker B:', fp2_models)\n"
        "assert len(fp2_models) == 2, fp2_models")
r = sub(r, "fp_models, workers=os.cpu_count(), w_pv=0.88)",
        f"fp_models, workers=os.cpu_count(), w_pv={W_PV},\n"
        "                    fp2_model_paths=fp2_models, fp_rows_path=find('sim_rank_rows_fppair.npz'))")
set_src(run[0], r)

last = [c for c in cells if "EXPERIMENT_PUBLIC_RANKER_WEIGHT" in src(c)]
assert len(last) == 1
set_src(last[0], sub(src(last[0]), "print('EXPERIMENT_PUBLIC_RANKER_WEIGHT=0.88')",
                     f"print('EXPERIMENT_PUBLIC_RANKER_WEIGHT={W_PV}; ranker B: leak-free fingerprints')"))

# ---- description cell ------------------------------------------------------------
cells[0]["source"] = [
    f"# CASMI26 engine — leak-free fingerprints for ranker B (w_pv = {W_PV})\n",
    "\n",
    "Built on [CASMI26 Next Direct W088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088)\n",
    "by **Jiachen Li** (public LB 0.341), itself built on public work by **prvsiyan**,\n",
    "**haideptry** and **megayak**. All credit for the engine belongs to them.\n",
    "\n",
    "**One change:** ranker B (megayak's 51-feature ranker) is trained with its\n",
    "fingerprint columns filled from megayak's own leak-free fingerprint models\n",
    "(`sim_rank_rows_fppair.npz`, `ours_fpnet_*.pt`, from his public dataset)\n",
    "and served with those same models. W088 trains and serves it with the\n",
    "fingerprint switched off.\n",
    "\n",
    "Offline, on enveda-np-examples (5-fold grouped CV): ranker B class 2\n",
    "0.655 -> 0.734.\n",
]

OUT_DIR = ROOT / OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
fname = f"{SLUG}.ipynb"
(OUT_DIR / fname).write_text(json.dumps(nb, indent=1))
meta = json.loads((SRC / "kernel-metadata.json").read_text())
meta.update(id=f"ryqn00/{SLUG}", title=SLUG.replace("-", " "), code_file=fname)
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT_DIR / fname} (w_pv={W_PV})")
