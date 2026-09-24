"""Build the in-browser candidate pool: COCONUT structures, sharded by mass.

    uv run python scripts/build_web_pool.py

Writes web/pool/<floor(mass)>.json, each {"m": [...], "s": [...], "k": [...]}
(neutral monoisotopic mass, SMILES, InChIKey14), sorted by mass. A query
downloads only the one or two shards its mass window touches.

COCONUT only (CC0, open data): the full pool also contains structures from the
competition's train.parquet, which should not be redistributed outside Kaggle.
Masses come from pool.parquet, i.e. the same formula_mass() the pipeline uses.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "pool"


def main() -> None:
    pool = pd.read_parquet(ROOT / "data" / "pool.parquet",
                           columns=["inchikey14", "smiles", "exact_mass"])
    coconut = set(np.load(ROOT / "data" / "coconut_keys.npy", allow_pickle=True).tolist())
    # build_pool labels a structure found in BOTH sources as "train", so select
    # by key membership, not by the source label.
    p = pool[pool.inchikey14.isin(coconut)].sort_values("exact_mass", kind="stable")
    print(f"COCONUT structures in the pool: {len(p):,} of {len(pool):,}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    n = 0
    for b, g in p.groupby(np.floor(p.exact_mass).astype(int)):
        (OUT / f"{b}.json").write_text(json.dumps({
            "m": [round(float(x), 6) for x in g.exact_mass],
            "s": g.smiles.tolist(),
            "k": g.inchikey14.tolist(),
        }, separators=(",", ":")))
        n += 1
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"wrote {n} shards, {size:.1f} MB total, "
          f"mass {p.exact_mass.min():.1f}-{p.exact_mass.max():.1f} Da")


if __name__ == "__main__":
    main()
