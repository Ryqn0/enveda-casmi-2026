"""Choose the structures our fingerprint model must never see (Phase 8).

    uv run python scripts/own_holdout.py

Writes data/own/holdout.json:
  eval   the 250 enveda-np-examples structures -- where every number is measured
         (the public calibration set: predicted LB = 0.162*c1 + 0.220*c2)
  train  {library: [keys]} -- 400 natural products from each of 5 libraries,
         whose simulated queries become the learned ranker's training rows
and data/own/exclude_keys.json, the flat list for prepare_fpnet_data.py.

Why both are held out of the fingerprint model: v4c had trained on 240 of the
250 eval structures (through their spectra in other libraries), so its class-2
scores there are inflated. Ranker training rows need the same honesty, or the
ranker learns to over-trust a fingerprint channel that is weaker on the test.

Eligible train structures: in COCONUT (natural products, like the test), not
eval, at least 2 spectra in the test mass range with a test adduct (so one can
be the query and another stays for the class-1 simulation), at least one of
them in that library.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casmi.library import TEST_ADDUCTS, TEST_MZ_RANGE  # noqa: E402

OUT = ROOT / "data" / "own"
LIBS = ["gnps", "riken", "mona", "msdial", "massbank"]
PER_LIB = 400


def main() -> None:
    t = pd.read_parquet(ROOT / "data/raw/train.parquet",
                        columns=["ingest_lib", "inchikey14", "adduct", "precursor_mz"])
    lo, hi = TEST_MZ_RANGE
    t = t[t.precursor_mz.between(lo, hi) & t.adduct.isin(TEST_ADDUCTS)]
    coconut = set(np.load(ROOT / "data/coconut_keys.npy", allow_pickle=True).tolist())
    ev = sorted(set(t[t.ingest_lib.eq("enveda-np-examples")].inchikey14))

    n_spec = t.groupby("inchikey14").size()
    ok = set(n_spec[n_spec >= 2].index) & coconut - set(ev)
    rng = np.random.default_rng(0)
    taken, train = set(), {}
    for lib in LIBS:
        pool = sorted(set(t[t.ingest_lib.eq(lib)].inchikey14) & ok - taken)
        pick = rng.choice(pool, size=PER_LIB, replace=False).tolist()
        train[lib] = sorted(pick)
        taken |= set(pick)
        print(f"{lib:9s} {len(pool):6,} eligible -> {len(pick)}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "holdout.json").write_text(json.dumps({"eval": ev, "train": train}))
    excl = sorted(set(ev) | taken)
    (OUT / "exclude_keys.json").write_text(json.dumps(excl))
    print(f"eval {len(ev)}, train {len(taken)}, excluded from fpnet {len(excl)}")


if __name__ == "__main__":
    main()
