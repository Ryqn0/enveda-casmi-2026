"""Build the combined notebook: public engine (tie fix) + a share of our ranker.

    uv run python scripts/make_engine_plus_own.py

Starts from kaggle/engine_ties (LB 0.349) and changes ONE thing: the engine's
final ranking gets a small share of OUR pipeline's rank (docs/16):

    final = (1 - W_OWN) * engine rank + W_OWN * our rank,   W_OWN = 0.1

Offline (scripts/blend_harness.py, 246 enveda-np-examples molecules, paired):
+0.0040 +/- 0.0035 predicted LB; the two pipelines are right on different
molecules (ours alone is top-1 where the engine is wrong in 15 class-1 and
27 class-2 of 246).

Layout of the notebook:
  engine cells (unchanged)
  + install ms_entropy from ryqn00/casmi26-deps
  + our src/casmi modules written as a REAL package (%%writefile casmi/*.py),
    so their `from casmi.x import y` lines work as-is
  + our pipeline on the test set -> OWN = {molecule: (keys, scores)}, then its
    memory is released before the engine loads its own library
  + the run cell wraps E.make_submissions to apply the blend
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kaggle/engine_ties"
OUT_DIR = ROOT / "kaggle/engine_plus_own"
SLUG = "casmi26-engine-plus-own"
W_OWN = 0.1
MODULES = ["pool", "analog", "library", "fpnet", "fragment", "ranker", "rank", "features"]

nb = json.loads((BASE / "casmi26-engine-ties.ipynb").read_text())
cells = nb["cells"]


def src(c):
    return "".join(c["source"])


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def sub(s, old, new):
    assert s.count(old) == 1, f"expected exactly one match for:\n{old}"
    return s.replace(old, new)


run_i = [i for i, c in enumerate(cells) if "subs, recs = E.main(" in src(c)]
assert len(run_i) == 1
run_i = run_i[0]

ours = [md(
    "## 3b · Our pipeline (ryqn00)\n"
    "\n"
    "Our own library search, candidate pool, fingerprint model (v5) and learned ranker\n"
    "(github docs/16). It runs first, keeps only a per-candidate score per test\n"
    "molecule, and frees its memory before the engine starts.\n"),
    code(
    "import os, glob, subprocess, sys\n"
    "# ms_entropy (the Flash Entropy library search) ships as a wheel in casmi26-deps.\n"
    "deps = sorted({os.path.dirname(p) for p in glob.glob('/kaggle/input/**/ms_entropy-*.whl', recursive=True)})\n"
    "assert deps, 'no ms_entropy wheel: is ryqn00/casmi26-deps attached?'\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index',\n"
    "                f'--find-links={deps[0]}', '--no-deps', 'ms_entropy'], check=True)\n"
    "os.makedirs('casmi', exist_ok=True)   # %%writefile does not create directories\n")]
# Jupyter rejects %%writefile with an EMPTY body ("the cell body is empty") --
# v1 of this notebook died on exactly that. Give the package marker a line.
ours.append(code('%%writefile casmi/__init__.py\n"""Our pipeline (github: Ryqn0/enveda-casmi-2026), src/casmi."""\n'))
for m in MODULES:
    ours.append(code(f"%%writefile casmi/{m}.py\n" + (ROOT / "src/casmi" / f"{m}.py").read_text()))
ours.append(code('''import gc, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from casmi.library import Library, build, TEST_MZ_RANGE, TEST_ADDUCTS
from casmi.pool import CandidatePool
from casmi.fpnet import load
from casmi.features import (FEATURE_NAMES, candidate_features, fit_ranker, fp_evidence,
                            molecule_evidence, rank01, ranker_scores)

t_own = time.time()
OWN_INDEX = Path("/tmp/own_index")
build(Path(COMP) / "train.parquet", OWN_INDEX, TEST_MZ_RANGE, set(TEST_ADDUCTS))
lib, pool = Library(OWN_INDEX), CandidatePool(Path(find("pool.parquet")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
fpnet, fck = load(find("fpnet.pt"), device=dev)
assert fck.get("model_tag") == "v5", f"stale casmi26-deps: fpnet {fck.get('model_tag')}"
R = np.load(find("ranker_rows.npz"), allow_pickle=True)
assert str(R["fpnet_tag"]) == "v5" and list(R["feature_names"]) == FEATURE_NAMES
own_models = fit_ranker(R["X"], R["y"], R["c1"])

OWN = {}   # molecule_id -> (candidate InChIKey14s, our ranker's scores)
te_own = pd.read_parquet(Path(COMP) / "test.parquet")
for mid, grp in te_own.groupby("molecule_id", sort=True):
    spectra = [(float(r.precursor_mz),
                np.stack([np.asarray(r.ms2_mzs, np.float32),
                          np.asarray(r.ms2_normalized_intensities, np.float32)], 1))
               for r in grp.itertuples()]
    ces = [float(np.mean(r.collision_energy_ev))
           if r.collision_energy_ev is not None and len(r.collision_energy_ev) else 32.0
           for r in grp.itertuples()]
    ev = molecule_evidence(lib, pool, spectra, list(grp.adduct), instrument_id=0,
                           collision_energies=ces)
    if len(ev["cand"]):
        ev["fp"] = fp_evidence(fpnet, fck, ev, pool, device=dev)
        OWN[mid] = (pool.key[ev["cand"]].astype(str),
                    ranker_scores(own_models, candidate_features(ev)))
print(f"our pipeline: {len(OWN)}/{te_own.molecule_id.nunique()} molecules scored "
      f"in {(time.time() - t_own) / 60:.1f} min")
del lib, pool, fpnet, own_models, R, te_own
gc.collect()
torch.cuda.empty_cache()
'''))

run = src(cells[run_i])
run = sub(run, "subs, recs = E.main(", f'''# ---- blend our ranker into the engine's final ranking (ryqn00) ----------------
W_OWN = {W_OWN}
_make_submissions = E.make_submissions
BLEND_STATS = {{"blended": 0, "engine only": 0}}

def make_submissions_with_own(mols, recs, score_sets, *a, **k):
    """final = (1 - W_OWN) * engine rank + W_OWN * our rank, within each molecule.
    Engine candidates outside our window get our worst rank (-1 < any probability)."""
    cand = {{r["mid"]: r["cand"] for r in recs if len(r["cand"])}}
    out = {{}}
    for mid, s in score_sets["blend"].items():
        if mid not in OWN:
            out[mid] = s
            BLEND_STATS["engine only"] += 1
            continue
        ok = dict(zip(*OWN[mid]))
        so = np.array([ok.get(key, -1.0) for key in E.POOL["keys"][cand[mid]].astype(str)])
        out[mid] = (1 - W_OWN) * (1 - rank01(s)) + W_OWN * (1 - rank01(so))
        BLEND_STATS["blended"] += 1
    return _make_submissions(mols, recs, dict(score_sets, blend=out), *a, **k)

E.make_submissions = make_submissions_with_own
subs, recs = E.main(''')
run = sub(run, "subs['blend'].to_csv('submission.csv', index=False)",
          "print('blend:', BLEND_STATS)\nsubs['blend'].to_csv('submission.csv', index=False)")
cells[run_i]["source"] = run.splitlines(keepends=True)

last = [c for c in cells if "EXPERIMENT_PUBLIC_RANKER_WEIGHT" in src(c)]
assert len(last) == 1
last[0]["source"] = sub(src(last[0]), "rank features: ties averaged",
                        f"rank features: ties averaged; + {W_OWN} x ryqn00 ranker").splitlines(keepends=True)

cells[run_i:run_i] = ours
cells[0]["source"] = [
    f"# CASMI26 engine + our ranker (share {W_OWN})\n",
    "\n",
    "Built on [CASMI26 Next Direct W088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088)\n",
    "by **Jiachen Li**, itself built on public work by **prvsiyan**, **haideptry** and\n",
    "**megayak**. All credit for the engine belongs to them.\n",
    "\n",
    "Two changes of ours: (1) tie-averaged rank features (LB 0.341 -> 0.349);\n",
    f"(2) the final ranking mixes in our own pipeline's rank with weight {W_OWN} --\n",
    "a separate library search, candidate pool, fingerprint model and learned ranker\n",
    "that is right on different molecules than the engine.\n",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
fname = f"{SLUG}.ipynb"
(OUT_DIR / fname).write_text(json.dumps(nb, indent=1))
meta = json.loads((BASE / "kernel-metadata.json").read_text())
meta.update(id=f"ryqn00/{SLUG}", title=SLUG.replace("-", " "), code_file=fname,
            dataset_sources=meta["dataset_sources"] + ["ryqn00/casmi26-deps"])
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT_DIR / fname}: {len(cells)} cells")
