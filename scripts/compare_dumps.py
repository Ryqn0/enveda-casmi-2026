"""Paired comparison of full ranking CONFIGURATIONS from validate_fpnet.py dumps.

    uv run python scripts/compare_dumps.py \
        /tmp/v2.parquet@fp=1,an=0.5 \
        /tmp/v3.parquet@fp=1,an=0.25 \
        /tmp/v4c.parquet@ranker=data/ranker_v4c.pt \
        /tmp/v2.parquet@fp=0,an=1

Each argument is `dump[@spec]`. The spec is either blend weights over the
z-scored channels (fp, an, fr; default fp=1,an=0.5,fr=0.25) or a learned
ranker checkpoint (ranker=path). The same dump may appear several times with
different specs.

All dumps must come from the SAME molecules (same --seed and -n, and
--exclude-seen-by so every model is graded on molecules none of them saw).
That is checked: the analog and fragment channels do not depend on fpnet, so
they must be identical across dumps.

The point of this script is calibration as much as comparison. A holdout is
only worth optimising against if it reproduces the leaderboard's ORDERING of
configurations we have already submitted (docs/12).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.ranker import (FEATURES, expected_rr, load_ranker,  # noqa: E402
                          ranker_features, score_with_ranker)

K = 25


def parse(arg: str):
    path, _, spec = arg.partition("@")
    cfg = {"fp": 1.0, "an": 0.5, "fr": 0.25}
    ranker = None
    if spec:
        for kv in spec.split(","):
            k, v = kv.split("=")
            if k == "ranker":
                ranker = load_ranker(v)
            else:
                cfg[k] = float(v)
    return Path(path), spec or "fp=1,an=0.5,fr=0.25", cfg, ranker


def per_molecule_rr(df: pd.DataFrame, cfg, ranker) -> pd.Series:
    out = {}
    for q, g in df.groupby("qid", sort=True):
        f = ranker_features(g.fp_gap.to_numpy(), g.an_raw.to_numpy(),
                            g.fr_raw.to_numpy(), float(g.an_conf.iloc[0]),
                            int(round(np.exp(g.log_npk.iloc[0]))))
        if ranker is not None:
            s = score_with_ranker(*ranker, f)
        else:
            s = (cfg["fp"] * f[:, FEATURES.index("z_fp")]
                 + cfg["an"] * f[:, FEATURES.index("z_an")]
                 + cfg["fr"] * f[:, FEATURES.index("z_fr")])
        lab = g.label.to_numpy()
        if not lab.any():
            out[q] = 0.0
            continue
        out[q] = expected_rr(s, int(lab.argmax()), K)
    return pd.Series(out)


def main() -> None:
    specs = [parse(a) for a in sys.argv[1:]]
    cache = {}
    for p, *_ in specs:
        if p not in cache:
            cache[p] = pd.read_parquet(p)
    ref = cache[specs[0][0]]
    for p, d in cache.items():
        same = (set(d.qid) == set(ref.qid) and len(d) == len(ref)
                and np.allclose(np.sort(d.an_raw), np.sort(ref.an_raw))
                and np.allclose(np.sort(d.fr_raw), np.sort(ref.fr_raw)))
        assert same, f"{p.name} was not run on the same molecules"
    c1 = ref.groupby("qid").is_c1.first().sort_index()
    print(f"{ref.qid.nunique()} molecules ({c1.sum()} simulated class 1), "
          f"identical across dumps\n")

    rr = [per_molecule_rr(cache[p], cfg, rk) for p, _, cfg, rk in specs]
    base = rr[0]
    print("| config | all | class 1 | class 2 | paired vs first (+/- 2 SE) |")
    print("|---|---:|---:|---:|---:|")
    for (p, spec, _, _), r in zip(specs, rr):
        d = (r - base).to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        diff = "--" if r is base else f"{d.mean():+.4f} +/- {2 * se:.4f}"
        print(f"| {p.stem} @ {spec} | {r.mean():.4f} | {r[c1].mean():.4f} | "
              f"{r[~c1].mean():.4f} | {diff} |")


if __name__ == "__main__":
    main()
