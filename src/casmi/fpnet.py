"""Spectrum -> molecular fingerprint: a set transformer over peaks.

WHY THIS SHAPE OF MODEL
    A spectrum is an unordered SET of (m/z, intensity) pairs, not a sequence and
    not an image. Transformer self-attention is permutation-equivariant, so it
    fits naturally: no positional encoding, peaks attend to each other, and the
    order they arrive in cannot matter.

WHY PREDICT A FINGERPRINT RATHER THAN A STRUCTURE
    Generating a SMILES string is an open-ended search over a combinatorial
    space. Predicting a fingerprint is a fixed-size multi-label classification
    with millions of labelled examples. The fingerprint is then used to RANK
    candidates that already exist, so the model never has to invent anything.

WHY IT SHOULD BEAT ANALOG PROPAGATION WHERE WE ARE WEAK
    Analogy needs a spectral cousin to exist. A fingerprint model learns the
    general mapping from fragmentation patterns to substructures, so it scores a
    molecule with no relative in the library at all — which is exactly the
    distant-analog bucket where we measured MRR 0.396.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SinEmb(nn.Module):
    """Sinusoidal embedding of a continuous value, log-spaced.

    A raw m/z of 372.14 is a poor network input: one number spanning three
    orders of magnitude. Projecting it onto sines and cosines of many
    wavelengths gives a smooth, high-dimensional code where nearby masses are
    nearby vectors — the same trick as positional encoding, applied to a
    physical quantity rather than an index.

    Wavelengths are log-spaced from 10^-2 to 10^3.2 Da so the code resolves both
    a 0.01 Da isotope gap and a 1000 Da precursor.
    """

    def __init__(self, dim: int, lo: float = -2.0, hi: float = 3.2):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, torch.linspace(lo, hi, n))
        # register_buffer: saved with the model and moved to GPU with .to(),
        # but not a trainable parameter.
        self.register_buffer("inv", (2 * math.pi) / wav)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class Block(nn.Module):
    """Pre-norm transformer block with masked self-attention."""

    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        y = self.n1(x)
        # One projection for queries, keys and values, then split.
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        # pad is True for padding slots; the mask must be True where attention
        # IS allowed, hence the inversion.
        mask = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    """Predicts fingerprint-bit logits from a spectrum plus its metadata."""

    def __init__(self, n_bits: int, n_adducts: int, n_instruments: int = 4,
                 d: int = 256, layers: int = 4, heads: int = 8, drop: float = 0.1):
        super().__init__()
        self.cfg = dict(n_bits=n_bits, n_adducts=n_adducts,
                        n_instruments=n_instruments, d=d, layers=layers,
                        heads=heads, drop=drop)

        self.mz_emb = SinEmb(d)
        # Neutral loss = precursor - fragment. The SAME chemical group falling
        # off two different molecules gives different fragment masses but the
        # same neutral loss, so this view is translation-invariant and exposes
        # structure the raw m/z hides.
        self.nl_emb = SinEmb(d)
        self.peak = nn.Linear(2 * d + 1, d)   # +1 for intensity

        self.prec_emb = SinEmb(d)
        self.adduct = nn.Embedding(n_adducts, d)
        self.instrument = nn.Embedding(n_instruments, d)
        self.glob = nn.Linear(d + 1, d)       # +1 for collision energy

        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        # Mean and max pooling concatenated: mean captures the overall pattern,
        # max captures whether any single diagnostic peak is present.
        self.head = nn.Sequential(
            nn.Linear(2 * d, 1024), nn.GELU(), nn.Dropout(drop),
            nn.Linear(1024, n_bits),
        )

    def forward(self, mz, inten, pad, prec, add_id, ins_id, ce):
        # Neutral losses, floored at 0 — a fragment heavier than its precursor
        # is noise, and a negative loss has no physical meaning.
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.peak(torch.cat([self.mz_emb(mz), self.nl_emb(nl),
                                 inten.unsqueeze(-1)], dim=-1))

        # One global token carrying everything that is true of the whole
        # measurement rather than of any single peak.
        g = self.glob(torch.cat([self.prec_emb(prec),
                                 (ce / 100.0).unsqueeze(-1)], dim=-1))
        g = g + self.adduct(add_id) + self.instrument(ins_id)

        x = torch.cat([g.unsqueeze(1), p], dim=1)
        # The global token is never padding, so prepend a False column.
        pad = torch.cat([torch.zeros(len(pad), 1, dtype=torch.bool,
                                     device=pad.device), pad], dim=1)
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)

        # Exclude padding from pooling: -inf before max, and a masked mean.
        m = (~pad).unsqueeze(-1).float()
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        mx = x.masked_fill(pad.unsqueeze(-1), float("-inf")).max(1).values
        return self.head(torch.cat([mean, mx], dim=-1))


def score_candidates(logits: np.ndarray, cand_fp: np.ndarray) -> np.ndarray:
    """Rank candidates by the log-likelihood of their fingerprint.

    `logits` is the model's output for one spectrum, `cand_fp` is (n_cand,
    n_bits) of 0/1. The score is

        sum over bits of [ fp*log(p) + (1-fp)*log(1-p) ]

    i.e. how probable this exact bit pattern is under the model. A plain dot
    product would only reward bits the candidate HAS, so a molecule with many
    bits set would win by having more chances to agree; this also penalises
    bits the model expects and the candidate lacks.
    """
    p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    eps = 1e-6
    lp, lq = np.log(p + eps), np.log1p(-p + eps)
    return cand_fp @ lp + (1 - cand_fp) @ lq


def save(model: FPNet, path, **extra) -> None:
    torch.save({"cfg": model.cfg, "state": model.state_dict(), **extra}, path)


def load(path, device="cpu") -> tuple[FPNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FPNet(**ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def _self_check() -> None:
    torch.manual_seed(0)
    B, N, BITS = 3, 16, 64
    # eval(), not train(): dropout is stochastic, so the invariance checks below
    # would compare two different random masks and fail for the wrong reason.
    net = FPNet(n_bits=BITS, n_adducts=10, d=64, layers=2, heads=4).eval()

    mz = torch.rand(B, N) * 400
    it = torch.rand(B, N)
    pad = torch.zeros(B, N, dtype=torch.bool)
    pad[0, 10:] = True          # first spectrum has only 10 real peaks
    prec = torch.rand(B) * 400 + 100
    with torch.no_grad():
        out = net(mz, it, pad, prec, torch.zeros(B, dtype=torch.long),
              torch.zeros(B, dtype=torch.long), torch.full((B,), 30.0))
    assert out.shape == (B, BITS), out.shape
    assert torch.isfinite(out).all(), "non-finite logits"

    # Padding must not change the answer: the padded slots of spectrum 0 hold
    # arbitrary values, and masking should make them irrelevant.
    mz2, it2 = mz.clone(), it.clone()
    mz2[0, 10:] = 999.0
    it2[0, 10:] = 5.0
    out2 = net(mz2, it2, pad, prec, torch.zeros(B, dtype=torch.long),
               torch.zeros(B, dtype=torch.long), torch.full((B,), 30.0))
    assert torch.allclose(out[0], out2[0], atol=1e-5), "padding leaked into output"

    # Permuting real peaks must not change the answer either — a spectrum is a
    # set, and the architecture is only correct if it is permutation-invariant.
    perm = torch.randperm(N)
    out3 = net(mz[1:2, perm], it[1:2, perm], pad[1:2, perm], prec[1:2],
               torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
               torch.full((1,), 30.0))
    assert torch.allclose(out[1], out3[0], atol=1e-5), "not permutation invariant"

    # score_candidates must prefer the candidate matching the model's beliefs.
    logits = np.array([5.0, -5.0, 5.0, -5.0], dtype=np.float32)
    cands = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
    s = score_candidates(logits, cands)
    assert s[0] > s[1], s

    print("fpnet self-check passed")


if __name__ == "__main__":
    _self_check()
