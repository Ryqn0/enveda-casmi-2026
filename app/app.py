"""Gradio demo: paste a mass spectrum, get ranked candidate structures.

WHY THIS USES ONLY THE MODEL, NOT THE FULL PIPELINE
    The competition pipeline also runs a Flash Entropy index over 2.1M training
    spectra — 2.3 GB on disk and ~6 GB to build. A free Hugging Face Space has
    16 GB RAM and a cold start budget measured in seconds, so shipping the index
    would make the demo slow and fragile.

    The fingerprint model needs neither. 22 MB of weights plus a 35 MB candidate
    pool is the whole dependency, and it is the part that is genuinely ours:
    spectrum -> substructures -> ranked candidates. Analog propagation is left
    out, which costs accuracy on molecules that have a close spectral relative,
    and nothing on the ones that do not.

Run locally:
    uv run python app/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from casmi.fpnet import candidate_bits, load, predict_logits, score_candidates
from casmi.pool import ADDUCT_MASS, CandidatePool

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
# On a Space these are downloaded from the Hub at startup; locally they are the
# files training and build_pool produced.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "data" / "fpnet" / "fpnet.pt"))
POOL_PATH = Path(os.environ.get("POOL_PATH", ROOT / "data" / "pool.parquet"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, CK = load(MODEL_PATH, device=DEVICE)
BIT_SEL = np.asarray(CK["bit_sel"])
ADDUCTS = list(CK["adduct_list"])
POOL = CandidatePool(POOL_PATH)

EXAMPLE = """110.0600 0.15
123.0441 0.31
138.0662 1.00
195.0877 0.42"""


def parse_peaks(text: str) -> np.ndarray:
    """Read `m/z intensity` pairs, one per line. Commas and tabs also work."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise gr.Error("No peaks found. Use one `m/z intensity` pair per line.")
    a = np.array(rows, dtype=np.float32)
    # Normalise to a base peak of 1.0, as the model saw in training.
    if a[:, 1].max() > 0:
        a[:, 1] /= a[:, 1].max()
    return a[np.argsort(a[:, 0])]


def identify(peak_text: str, precursor_mz: float, adduct: str, ppm: float, top_k: int):
    peaks = parse_peaks(peak_text)
    if precursor_mz <= 0:
        raise gr.Error("Precursor m/z must be positive.")

    # +0.98 ppm is the systematic timsTOF calibration bias measured from the
    # training data; correcting it centres the search window on the truth.
    cand = POOL.query(precursor_mz, adduct, ppm=ppm, offset_ppm=0.98)
    if len(cand) == 0:
        raise gr.Error(
            f"No structure in the pool within {ppm} ppm of that neutral mass. "
            f"Check the precursor and adduct."
        )

    logits = predict_logits(
        MODEL, [(precursor_mz, peaks)], adduct, ADDUCTS,
        instrument_id_=0, device=DEVICE,
    )
    scores = score_candidates(logits, candidate_bits(POOL.smiles[cand], BIT_SEL))
    order = np.argsort(-scores)[:top_k]

    mols, captions = [], []
    for rank, j in enumerate(order, start=1):
        smi = POOL.smiles[cand[j]]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        captions.append(f"#{rank}  {POOL.key[cand[j]]}")

    grid = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(220, 180),
                                legends=captions)
    table = [
        [r + 1, POOL.smiles[cand[j]], f"{POOL.mass[cand[j]]:.4f}", f"{scores[j]:.1f}"]
        for r, j in enumerate(order)
    ]
    summary = (
        f"**{len(cand)} candidates** matched {precursor_mz:.4f} {adduct} "
        f"within ±{ppm} ppm, out of {len(POOL):,} structures. "
        f"Showing the top {len(table)} ranked by predicted fingerprint."
    )
    return summary, grid, table


with gr.Blocks(title="CASMI 2026 — molecule ID from mass spectra") as demo:
    gr.Markdown(
        "# Molecule identification from a mass spectrum\n"
        "A transformer reads the fragment peaks and predicts which **substructures** "
        "the molecule contains. Candidates of the right neutral mass are pulled from "
        "a 722k-structure pool (COCONUT + reference libraries) and ranked by how well "
        "their true fingerprint matches the prediction.\n\n"
        "Trained on 800k spectra. It never generates a structure — it recognises one."
    )
    with gr.Row():
        with gr.Column(scale=2):
            peaks_in = gr.Textbox(EXAMPLE, lines=10, label="Peaks — `m/z intensity` per line")
        with gr.Column(scale=1):
            prec_in = gr.Number(195.0877, label="Precursor m/z")
            add_in = gr.Dropdown(sorted(ADDUCT_MASS), value="[M+H]+", label="Adduct")
            ppm_in = gr.Slider(1, 20, value=5, step=0.5, label="Mass tolerance (ppm)")
            k_in = gr.Slider(5, 25, value=10, step=5, label="Candidates to show")
            go = gr.Button("Identify", variant="primary")

    summary_out = gr.Markdown()
    img_out = gr.Image(label="Ranked candidates", type="pil")
    tbl_out = gr.Dataframe(headers=["rank", "SMILES", "neutral mass", "score"],
                           label="Details", wrap=True)

    gr.Markdown(
        "_The example is caffeine ([M+H]+ at 195.0877). Its true structure is_ "
        "`CN1C=NC2=C1C(=O)N(C)C(=O)N2C`._"
    )
    go.click(identify, [peaks_in, prec_in, add_in, ppm_in, k_in],
             [summary_out, img_out, tbl_out])

if __name__ == "__main__":
    demo.launch()
