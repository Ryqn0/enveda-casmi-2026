"""Build the engine variant with tie-averaged rank features.

    uv run python scripts/make_engine_ties.py

Starts from kaggle/public_engine (unmodified W088, LB 0.341) and changes ONE
thing: how candidates that TIE on a feature are ranked (docs/15).

pv._rank_norm ranks with np.argsort, so tied candidates are ordered by their
position in the candidate list. In class 2 the true answer ties at library
score 0 with ~36 other candidates, and its rank then depends on list order,
which differs between the rankers' training rows (truth early) and serving
(mass order, truth late). The rankers learned a position signal that is wrong
at serving time. Fix, on both sides:
  serving   _rank_norm averages ties
  training  every rank column of both rankers' rows is recomputed, ties
            averaged, from the raw column it was computed from (checked: where a
            list has no ties, old and new values must agree)

Offline (scripts/engine_harness.py, 250 enveda-np-examples molecules, paired):
ranker B +0.0138 +/- 0.0057, shipped blend +0.0044 +/- 0.0045 (predicted LB).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle/public_engine"
OUT_DIR = ROOT / "kaggle/engine_ties"
SLUG = "casmi26-engine-ties"

nb = json.loads((SRC / "casmi26-public-engine.ipynb").read_text())
cells = nb["cells"]


def src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = s.splitlines(keepends=True)


def sub(s, old, new):
    assert s.count(old) == 1, f"expected exactly one match for:\n{old}"
    return s.replace(old, new)


def cell(prefix):
    hits = [c for c in cells if src(c).startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]


# ---- pv.py: serving-side rank features --------------------------------------------
pv = cell("%%writefile pv.py")
set_src(pv, sub(src(pv),
    "def _rank_norm(x):\n"
    "    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)\n",
    "def _rank_norm(x):\n"
    "    # ryqn00 variant: ties AVERAGED. The original ordered tied candidates by\n"
    "    # their position in the list, which differs between training and serving.\n"
    "    from scipy.stats import rankdata\n"
    "    x = np.asarray(x)\n"
    "    return (rankdata(-x, method='average') - 1) / max(1, len(x) - 1)\n"))

# ---- casmi_engine.py: training rows --------------------------------------------------
eng = cell("%%writefile casmi_engine.py")
e = src(eng)
e = sub(e, "T0 = time.time()\n", '''T0 = time.time()

# --- tie-averaged rank columns in the training rows (ryqn00 variant) ------------
# (rank column, raw column it ranks). Base block = prvsiyan's 31 features; ranker
# B appends libx (+31), anapol (+39), fragx (+44).
RANK_COLS_BASE = [(1, 0), (6, 5), (16, 15), (19, 18), (22, 21)]
RANK_COLS_B = RANK_COLS_BASE + [(32, 31), (35, 34), (40, 39), (45, 44), (48, 47)]


def retie(X, G, pairs):
    """Recompute rank columns per candidate list (G), ties averaged, as serving now
    does. Where a list has no ties the stored and new values must agree -- that
    proves each pair really is (rank of, raw)."""
    X = np.array(X, np.float32, copy=True)
    df = pd.DataFrame({"G": G})
    n = df.groupby("G").G.transform("size").to_numpy()
    for rc, vc in pairs:
        df["v"] = X[:, vc]
        new = (df.groupby("G").v.rank(ascending=False, method="average").to_numpy() - 1) / np.maximum(n - 1, 1)
        untied = df.groupby("G").v.transform("nunique").to_numpy() == n
        assert np.allclose(X[untied, rc], new[untied], atol=1e-5), f"column {rc} is not the rank of {vc}"
        X[:, rc] = new
    return X
''')
e = sub(e, '''    X, Y, S = z["X"], z["Y"], z["S"]
    rankers = [_hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, RANK.GBM) for wA in RANK.W_A for sd in RANK.SEEDS]''',
        '''    X, Y, S = z["X"], z["Y"], z["S"]
    X = retie(X, z["G"], RANK_COLS_B)
    log("our ranker: rank columns recomputed with averaged ties")
    rankers = [_hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, RANK.GBM) for wA in RANK.W_A for sd in RANK.SEEDS]''')
e = sub(e, '''    z = np.load(pv_rank_path)
    rankers = [_hgb(z["X"], z["Y"], np.where(z["M"] == 0, w1, 1.0 - w1), sd, CFG.GBM) for w1 in priors for sd in seeds]''',
        '''    z = np.load(pv_rank_path)
    # In these rows a candidate list is (molecule G, simulation M).
    X = retie(z["X"], z["G"] * 2 + z["M"].astype(int), RANK_COLS_BASE)
    log("prvsiyan ranker: rank columns recomputed with averaged ties")
    rankers = [_hgb(X, z["Y"], np.where(z["M"] == 0, w1, 1.0 - w1), sd, CFG.GBM) for w1 in priors for sd in seeds]''')
set_src(eng, e)

last = [c for c in cells if "EXPERIMENT_PUBLIC_RANKER_WEIGHT" in src(c)]
assert len(last) == 1
set_src(last[0], sub(src(last[0]), "print('EXPERIMENT_PUBLIC_RANKER_WEIGHT=0.88')",
                     "print('EXPERIMENT_PUBLIC_RANKER_WEIGHT=0.88; rank features: ties averaged')"))

cells[0]["source"] = [
    "# CASMI26 engine — tie-averaged rank features\n",
    "\n",
    "Built on [CASMI26 Next Direct W088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088)\n",
    "by **Jiachen Li** (public LB 0.341), itself built on public work by **prvsiyan**,\n",
    "**haideptry** and **megayak**. All credit for the engine belongs to them.\n",
    "\n",
    "**One change:** candidates that tie on a feature get the average rank. The\n",
    "original ordered ties by list position, which differs between the rankers'\n",
    "training rows and serving -- in class 2 the true answer ties at library score 0\n",
    "with ~36 others. Applied at serving and to both rankers' training rows.\n",
    "\n",
    "Offline (250 enveda-np-examples molecules, paired): ranker B +0.014, blend +0.004\n",
    "predicted LB.\n",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
fname = f"{SLUG}.ipynb"
(OUT_DIR / fname).write_text(json.dumps(nb, indent=1))
meta = json.loads((SRC / "kernel-metadata.json").read_text())
meta.update(id=f"ryqn00/{SLUG}", title=SLUG.replace("-", " "), code_file=fname)
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT_DIR / fname}")
