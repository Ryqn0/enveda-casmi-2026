"""Export the fingerprint model to ONNX for the in-browser demo, and verify it.

    uv run --with onnx --with onnxruntime --with onnxscript python scripts/export_onnx.py

Writes web/model.onnx and web/meta.json. The export uses the exact fixed input
shape predict_logits builds (one spectrum, padded to max_peaks), so the browser
never has to handle dynamic shapes.

The export is only trusted after a numeric check: ONNX Runtime and PyTorch are
run on the same real spectra, and the logits must agree. A model that exports
without errors but computes something slightly different is exactly the kind
of silent train/serve skew this project kept paying for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from casmi.fpnet import load, predict_logits, select_peaks  # noqa: E402
from casmi.pool import ADDUCT_MASS  # noqa: E402

MODEL = ROOT / "data" / "fpnet" / "fpnet_v4c.pt"
OUT = ROOT / "web"


def inputs_for(peaks, prec, adduct, adducts, max_peaks, ce=32.0):
    """The same tensors predict_logits builds, as numpy arrays."""
    mz_s, it_s = select_peaks(peaks[:, 0], peaks[:, 1], max_peaks)
    n = len(mz_s)
    mz = np.zeros((1, max_peaks), np.float32)
    it = np.zeros((1, max_peaks), np.float32)
    mz[0, :n], it[0, :n] = mz_s, it_s
    pad = np.ones((1, max_peaks), bool)
    pad[0, :n] = False
    return {
        "mz": mz, "inten": it, "pad": pad,
        "prec": np.array([prec], np.float32),
        "add_id": np.array([adducts.index(adduct) if adduct in adducts else 0], np.int64),
        "ins_id": np.array([0], np.int64),          # timsTOF, as in the app
        "ce": np.array([ce], np.float32),
    }


def main() -> None:
    import onnxruntime as ort
    import pyarrow.parquet as pq

    model, ck = load(MODEL, device="cpu")
    max_peaks = int(ck["max_peaks"])
    adducts = list(ck["adduct_list"])
    OUT.mkdir(exist_ok=True)

    # --- export ---------------------------------------------------------------
    names = ["mz", "inten", "pad", "prec", "add_id", "ins_id", "ce"]
    example = inputs_for(np.array([[100.0, 1.0], [150.0, 0.5]], np.float32),
                         200.0, adducts[0], adducts, max_peaks)
    args = tuple(torch.as_tensor(example[n]) for n in names)
    torch.onnx.export(model, args, OUT / "model.onnx", input_names=names,
                      output_names=["logits"], opset_version=17, dynamo=False)
    size = (OUT / "model.onnx").stat().st_size / 1e6
    print(f"exported web/model.onnx ({size:.1f} MB)")

    # --- verify on real spectra -----------------------------------------------
    sess = ort.InferenceSession(str(OUT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    t = pq.read_table(ROOT / "data/raw/test.parquet").to_pandas().head(60)
    worst = 0.0
    for r in t.itertuples():
        peaks = np.stack([np.asarray(r.ms2_mzs, np.float32),
                          np.asarray(r.ms2_normalized_intensities, np.float32)], 1)
        ref = predict_logits(model, [(float(r.precursor_mz), peaks)], r.adduct,
                             adducts, instrument_id_=0, max_peaks=max_peaks)
        x = inputs_for(peaks, float(r.precursor_mz), r.adduct, adducts, max_peaks)
        got = sess.run(None, x)[0][0]
        worst = max(worst, float(np.abs(got - ref).max()))
    print(f"ONNX vs PyTorch on {len(t)} real spectra: max |logit diff| = {worst:.2e}")
    assert worst < 1e-3, "ONNX export does not reproduce the PyTorch model"

    # --- metadata the browser needs, generated from the Python source ---------
    meta = {
        "model_tag": ck.get("model_tag"),
        "max_peaks": max_peaks,
        "adducts": adducts,
        "bit_sel": np.asarray(ck["bit_sel"]).tolist(),
        "fp": {"radius": 2, "nBits": 2048},
        "adduct_mass": ADDUCT_MASS,
        "offset_ppm": 0.98,
        "default_ce": 32.0,
    }
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(f"wrote web/meta.json ({len(meta['bit_sel'])} bits, {len(adducts)} adducts)")


if __name__ == "__main__":
    main()
