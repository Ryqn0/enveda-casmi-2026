"""Golden outputs from the Python pipeline, for testing the JavaScript port.

    uv run python scripts/make_golden.py

Writes web/test/golden.json: for real test spectra plus the caffeine example,
every intermediate the browser must reproduce -- selected peaks, model logits,
candidate list, candidate fingerprint bits, scores and top-25 ranking.
web/test/golden.mjs then runs the JavaScript version on the same inputs and
fails on any disagreement.

Candidates are read from web/pool (the shards the browser loads), so the test
compares LOGIC, not two copies of the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi.fpnet import (candidate_bits, load, predict_logits,  # noqa: E402
                         score_candidates, select_peaks)
from casmi.pool import ADDUCT_MASS  # noqa: E402

META = json.loads((ROOT / "web" / "meta.json").read_text())
PPM, OFFSET, K = 5.0, META["offset_ppm"], 25


def shard_query(m: float, tol: float):
    """Candidates with m - tol <= mass < m + tol, read from the web shards --
    the same half-open window CandidatePool.query gets from searchsorted."""
    ms, ss, ks = [], [], []
    for b in range(int(np.floor(m - tol)), int(np.floor(m + tol)) + 1):
        f = ROOT / "web" / "pool" / f"{b}.json"
        if f.exists():
            d = json.loads(f.read_text())
            ms += d["m"]; ss += d["s"]; ks += d["k"]
    ms = np.array(ms)
    keep = (ms >= m - tol) & (ms < m + tol)
    return [s for s, k_ in zip(ss, keep) if k_], [k for k, k_ in zip(ks, keep) if k_]


def main() -> None:
    model, ck = load(ROOT / "data" / "fpnet" / "fpnet_v4c.pt", device="cpu")
    adducts, max_peaks = list(ck["adduct_list"]), int(ck["max_peaks"])
    bit_sel = np.asarray(ck["bit_sel"])

    t = pq.read_table(ROOT / "data" / "raw" / "test.parquet").to_pandas()
    rows = [(np.stack([np.asarray(r.ms2_mzs, np.float32),
                       np.asarray(r.ms2_normalized_intensities, np.float32)], 1),
             float(r.precursor_mz), r.adduct) for r in t.head(30).itertuples()]
    caffeine = np.array([[110.0600, 0.15], [123.0441, 0.31], [138.0662, 1.00],
                         [195.0877, 0.42]], np.float32)
    rows.append((caffeine, 195.087652, "[M+H]+"))

    cases = []
    for peaks, prec, adduct in rows:
        if adduct not in ADDUCT_MASS:
            continue
        sel_mz, sel_it = select_peaks(peaks[:, 0], peaks[:, 1], max_peaks)
        logits = predict_logits(model, [(prec, peaks)], adduct, adducts,
                                instrument_id_=0, max_peaks=max_peaks)
        m = (prec - ADDUCT_MASS[adduct]) / (1.0 + OFFSET * 1e-6)
        smiles, keys = shard_query(m, m * PPM * 1e-6)
        bits = candidate_bits(smiles, bit_sel) if smiles else np.zeros((0, len(bit_sel)))
        scores = score_candidates(logits, bits) if smiles else np.zeros(0)
        order = np.argsort(-scores, kind="stable")[:K]
        cases.append({
            "peaks": peaks.tolist(), "prec": prec, "adduct": adduct,
            "sel_mz": sel_mz.tolist(), "sel_it": sel_it.tolist(),
            # Full precision: rounding here showed up as ~1e-4 score noise
            # summed over 1,583 bits, masking real disagreements.
            "logits": [float(x) for x in logits],
            "cand_keys": keys,
            "cand_bits": ["".join(str(int(b)) for b in row) for row in bits],
            "scores": [float(s) for s in scores],
            "top_keys": [keys[i] for i in order],
        })
    out = ROOT / "web" / "test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden.json").write_text(json.dumps({"ppm": PPM, "k": K, "cases": cases}))
    n_c = sum(len(c["cand_keys"]) for c in cases)
    print(f"wrote {len(cases)} golden cases, {n_c} candidates in total")


if __name__ == "__main__":
    main()
