# Part 5 — `fpnet.py`: the spectrum → fingerprint neural network

**Written:** Phase 3 (features added in Phase 4, `select_peaks` made
deterministic in the wrap-up). **Model sizes:** 5,509,167 parameters (v4c).
**Used by:** `FPNet` → `train_fpnet.py`; `save` → the three training scripts;
`load`, `predict_logits`, `score_candidates`, `candidate_bits` → `rank.py`,
`app.py`, validation and the web-demo scripts; `select_peaks` → data
preparation, ONNX export and golden tests.

This is the only file with deep learning in it, so PyTorch concepts are
explained from scratch as they appear.

---

## What the model does

```
spectrum (up to 256 peaks) + precursor mass + adduct + instrument + collision energy
        │
        ▼   FPNet
1,583 numbers ("logits"), one per fingerprint bit: "how likely is this substructure present?"
        │
        ▼   score_candidates
one score per candidate structure: "how well does its real fingerprint fit that prediction?"
```

**Why predict a fingerprint instead of a structure** (docstring, lines 9–13):
writing a SMILES from scratch is an open-ended search; predicting 1,583 yes/no
bits is a fixed-size problem with millions of training examples. The prediction
is then used to **rank candidates that already exist** — the model never has
to invent a molecule.

---

## PyTorch in five ideas (read this first)

1. **Tensor** = a numpy-like array that can live on a GPU and remember how it
   was computed (so gradients can flow back during training).
2. **Shapes are written `(B, N, D)`:** `B` = batch (how many spectra at once),
   `N` = peaks per spectrum (padded to the same length), `D` = numbers
   describing each peak (256 here). Following shapes is the key to reading any network.
3. **`nn.Module`** = a building block. `__init__` **creates** its layers (and
   their trainable weights); `forward` **uses** them to compute an output.
   Calling `model(x)` runs `forward(x)`.
4. **`nn.Linear(a, b)`** = multiply by a learned `a × b` matrix and add a bias:
   turns `a` numbers into `b` numbers. Almost every layer here is one of these.
5. **`.eval()` vs `.train()`, `torch.no_grad()`:** some layers (dropout) behave
   randomly during training; `.eval()` switches that off for prediction.
   `no_grad()` skips remembering the computation, since no training happens —
   faster and less memory.

---

## `class SinEmb` (lines 32–55) — turning a mass into a vector

A raw m/z like 372.14 is a poor input for a network: one number spanning
three orders of magnitude. `SinEmb` turns it into 256 numbers: sines and
cosines of the mass at 128 different wavelengths.

```python
    def __init__(self, dim: int, lo: float = -2.0, hi: float = 3.2):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, torch.linspace(lo, hi, n))
        self.register_buffer("inv", (2 * math.pi) / wav)
```

- `super().__init__()`: run the parent class's setup first (required for every `nn.Module`).
- `dim // 2`: **integer division** (256 // 2 = 128) — half sines, half cosines.
- `torch.linspace(-2, 3.2, 128)`: 128 evenly spaced numbers from −2 to 3.2;
  `10 ** those` = wavelengths from **0.01 Da to ~1,585 Da**, spaced
  *logarithmically*. Short wavelengths resolve tiny mass differences (0.01 Da,
  an isotope gap); long ones distinguish a 200 Da from a 1,000 Da molecule.
- `register_buffer`: stored with the model and moved to the GPU with it, but
  **not trained** — these wavelengths are fixed maths, not learned.

```python
    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)
```

- `x` has shape `(B, N)` (one mass per peak). `unsqueeze(-1)` adds a last
  dimension → `(B, N, 1)`; multiplying by `inv` of shape `(128,)`
  **broadcasts** → `(B, N, 128)` (every mass times every frequency).
- `sin` and `cos` → two `(B, N, 128)`; `cat(..., dim=-1)` → `(B, N, 256)`.
- Nearby masses give nearby vectors — the same trick as the "positional
  encoding" in language transformers, applied to a physical quantity.

---

## `class Block` (lines 58–83) — one transformer layer

The core of the network, stacked 4 times. Each peak looks at every other peak
(**self-attention**), then each peak is processed on its own (**feed-forward**).

### `__init__` — the parts

| attribute | what it is | shape (d = 256) |
|---|---|---|
| `n1`, `n2` | `LayerNorm`: rescale each peak's 256 numbers to mean 0, std 1 | — |
| `qkv` | one `Linear` producing Query, Key and Value together | 256 → 768 (= 3 × 256) |
| `o` | output projection after attention | 256 → 256 |
| `ff` | feed-forward: `Linear` → `GELU` → `Dropout` → `Linear` | 256 → 1024 → 256 |
| `drop` | `Dropout(0.1)`: randomly zero 10% of values during training (fights overfitting) | — |

`nn.Sequential(a, b, c)` chains layers: its output is `c(b(a(x)))`. **GELU** is
the non-linearity (a smooth version of "negative → 0"); without non-linear
functions, stacked Linear layers would collapse into a single one.

### `forward(x, pad)` — attention, step by step

```python
        B, N, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
```

- **Pre-norm:** normalise *before* attention (more stable training).
- **Attention in words:** each peak asks a *question* (Query), advertises what
  it holds (Key), and carries information (Value). A peak's output is a
  weighted mix of all peaks' Values, weighted by how well its Query matches
  their Keys. So a peak at m/z 138 can "notice" that a peak exactly 57 Da
  heavier exists — the kind of pattern chemists look for.
- **Multi-head (8 heads):** 8 independent attention patterns of 32 numbers
  each (256 / 8), so different heads can track different relationships.
- The reshaping: `qkv(y)` is `(B, N, 768)` → `.view(B, N, 3, 8, 32)` splits it
  into 3 roles × 8 heads × 32 → `.permute(2, 0, 3, 1, 4)` reorders axes to
  `(3, B, 8, N, 32)` → unpacking gives `q`, `k`, `v`, each `(B, 8, N, 32)`.
  (`view` reinterprets the same memory with a new shape; `permute` swaps axes.)

```python
        mask = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

- `pad` is `True` for **padding slots** (fake peaks added so every spectrum
  has the same length). `~` flips it: the mask is `True` where attention **is
  allowed**. `[:, None, None, :]` inserts two axes → `(B, 1, 1, N)`, which
  broadcasts across heads and query positions. Result: **no peak ever attends
  to padding.**
- `scaled_dot_product_attention` is PyTorch's optimised attention:
  `softmax(q·kᵀ / √32) · v`.

```python
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))
```

- `transpose(1, 2)` + `reshape` glue the 8 heads back into 256 numbers per peak.
- **`x = x + …` is a residual connection:** the block *adds* a correction to
  its input instead of replacing it. This is what lets deep networks train.
- Then the same pattern for the feed-forward part.

---

## `class FPNet` (lines 86–166) — the whole model

### `__init__` — all the layers (real sizes from the v4c checkpoint)

| attribute | layer | shape | purpose |
|---|---|---|---|
| `mz_emb` | `SinEmb(256)` | m/z → 256 | encode each fragment mass |
| `nl_emb` | `SinEmb(256)` | loss → 256 | encode each **neutral loss** (precursor − fragment) |
| `peak` | `Linear` | **515 → 256** | mix a peak's features: 256 + 256 + 3 scalars |
| `prec_emb` | `SinEmb(256)` | mass → 256 | encode the precursor mass |
| `adduct` | `Embedding(10, 256)` | id → 256 | a learned vector per adduct type |
| `instrument` | `Embedding(4, 256)` | id → 256 | a learned vector per instrument family |
| `glob` | `Linear` | 257 → 256 | precursor embedding + collision energy |
| `blocks` | 4 × `Block` | 256 → 256 | the transformer |
| `norm` | `LayerNorm(256)` | | final normalisation |
| `head` | `Linear 512→1024`, GELU, Dropout, `Linear 1024→1583` | | pooled vector → 1,583 bit logits |

- **Neutral loss** (lines 103–107): the same chemical group falling off two
  different molecules gives different fragment masses but the **same loss**
  (e.g. water = 18.011). Giving the model both views exposes structure the raw
  masses hide.
- **`nn.Embedding(n, d)`** is a learned lookup table: row `i` is the vector for
  category `i`. It turns "adduct #3" into 256 learned numbers.
- **`nn.ModuleList`**: a list of layers PyTorch knows about (so their weights
  are saved and trained). A plain Python list would hide them.
- **`self.cfg`** records the constructor arguments, so `load()` can rebuild the
  exact same architecture from a file (lines 93–99).

### `forward(mz, inten, pad, prec, add_id, ins_id, ce)` — shapes for one spectrum

Inputs (B = 1 spectrum, N = 256 peak slots): `mz`, `inten`, `pad` are `(1, 256)`;
`prec`, `add_id`, `ins_id`, `ce` are `(1,)`.

| line | code | shape |
|---|---|---|
| 129 | `nl = (prec[:, None] - mz).clamp(min=0)` — neutral loss per peak, floored at 0 | (1, 256) |
| 136 | `log_int = log1p(inten × 1000) / 7` — log intensity, roughly 0..1 | (1, 256) |
| 141 | `rel_mz = (mz / prec).clamp(0, 1)` — fragment size as a fraction of the molecule | (1, 256) |
| 143–145 | features per peak: `mz_emb` (256) + `nl_emb` (256) + intensity, log-intensity, rel_mz (3) | (1, 256, 515) |
| 146 | `p = self.peak(cat(...))` | **(1, 256, 256)** |
| 150–152 | global token: `glob(prec_emb ‖ ce/100)` + adduct vector + instrument vector | (1, 256) |
| 154 | `x = cat([g, p])` — the global token becomes "peak 0" | (1, 257, 256) |
| 156–157 | `pad` gets a leading `False` (the global token is never padding) | (1, 257) |
| 158–159 | 4 transformer blocks | (1, 257, 256) |
| 160 | final `LayerNorm` | (1, 257, 256) |
| 163–164 | **masked mean** over real slots | (1, 256) |
| 165 | **masked max** over real slots (padding set to −∞ first) | (1, 256) |
| 166 | `head(cat([mean, max]))` | **(1, 1583)** |

Worth a closer look:

- **Why log-intensity** (line 136): intensities are heavily skewed toward
  tiny noise peaks; on a linear scale most peaks look identical. `log1p`
  (= log(1 + x), safe at 0) spreads out the small values.
- **The global token** (lines 150–154): things true of the whole measurement
  (precursor, adduct, instrument, energy) are packed into one extra "peak", so
  every real peak can attend to it.
- **Masked pooling** (lines 163–165): to get one vector per spectrum, average
  and max over peaks — but padding must not count:
  ```python
  m = (~pad).unsqueeze(-1).float()                       # 1 for real, 0 for padding
  mean = (x * m).sum(1) / m.sum(1).clamp(min=1)          # average of real slots only
  mx = x.masked_fill(pad.unsqueeze(-1), float("-inf")).max(1).values   # padding can never win a max
  ```
  **Mean** captures the overall pattern; **max** captures "is any single
  diagnostic peak present?". Both are concatenated (256 + 256 = 512).
- **Logits** are the output: any real number per bit. `sigmoid(logit)` turns
  it into a probability (0..1). Logit 0 = 50%; +5 ≈ 99%; −5 ≈ 1%.

**How it's trained** (in `train_fpnet.py`, Part 8): binary cross-entropy per bit
with **`pos_weight`** — only ~3% of bits are set, so without extra weight on
the 1s, "always predict 0" would already look good.

---

## `score_candidates(logits, cand_fp, platt=None)` (lines 169–195)

For each candidate: how probable is **its exact bit pattern** under the
model's prediction?

```python
    z = logits.astype(np.float64)
    if platt is not None:
        a, b = platt
        z = a.astype(np.float64) * z + b.astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-6
    lp, lq = np.log(p + eps), np.log1p(-p + eps)
    return cand_fp @ lp + (1 - cand_fp) @ lq
```

1. Work in `float64` for accuracy (sums over 1,583 terms).
2. Optional Platt calibration (off in practice — Phase 4 showed it hurts ranking).
3. `p = sigmoid(z)`: probability each bit is present.
4. `eps` prevents `log(0) = −∞`.
5. `lp` = log-probability "bit present", `lq` = log-probability "bit absent".
6. **`@` is matrix multiplication:** `cand_fp` is `(n_candidates, 1583)` of 0/1,
   `lp` is `(1583,)`, so `cand_fp @ lp` sums `lp` over each candidate's **on**
   bits. `(1 - cand_fp) @ lq` sums `lq` over its **off** bits. Total = the
   candidate's log-likelihood.

**Why not just count matching bits?** A plain dot product would reward
candidates with *many* bits set (more chances to agree). The likelihood also
penalises bits the model expects that the candidate **lacks**.

**What we proved later (Phase 4):** the `Σ lq` part is the same for every
candidate, so for **ranking** this equals `cand_fp @ z` — a dot product with
the logits. That's why Platt's `B` term (a constant per bit) turned into a
"penalty per bit set" and hurt ranking.

---

## `select_peaks(mzs, intensities, max_peaks)` (lines 198–223)

The **single source of truth** for choosing which peaks the model sees.

```python
    mz = np.asarray(mzs, dtype=np.float32)
    it = np.asarray(intensities, dtype=np.float32)
    if mz.size > max_peaks:
        keep = np.sort(np.argsort(-it, kind="stable")[:max_peaks])
        mz, it = mz[keep], it[keep]
    order = np.argsort(mz, kind="stable")
    return mz[order], it[order]
```

1. Cast to `float32` first (the same numbers the model will see).
2. If too many peaks: `argsort(-it, kind="stable")` = positions from most to
   least intense; among **exactly equal** intensities, the earlier peak comes
   first (that's what "stable" guarantees). Keep the first `max_peaks`, and
   `np.sort` them back into original order.
3. Return sorted by m/z.

**History:** (a) inference once used `peaks[:N]` — the **lowest-mass** peaks —
while training used the most intense (the Phase 3 bug); (b) it used
`argpartition`, which picks **arbitrarily** among ties at the cutoff and could
differ between numpy versions; the browser golden test found that, and the
rule is now shared with `web/core.js`.

## `candidate_bits(smiles_list, bit_sel)` (lines 226–245)

```python
    out = np.zeros((len(smiles_list), len(bit_sel)), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        fp = morgan_fp(smi)
        if fp is None:
            continue
        arr = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr[bit_sel]
    return out
```

- Morgan fingerprint (Part 4) → numpy array of 2,048 zeros and ones →
  **`arr[bit_sel]`** keeps only the **1,583 bits the model predicts**.
- **`bit_sel`** was chosen when the training data was prepared
  (`prepare_fpnet_data.py`): bits set in fewer than 0.5% or more than 95% of
  structures carry almost no information, so they were dropped. It's stored in
  the checkpoint; using a different selection here would compare against the
  wrong bits.
- The imports are **inside the function** (lines 233–235) to avoid loading
  RDKit when only the model is needed. (This is the kind of import the Kaggle
  embed-stripper once missed — Phase 3, kernel v7.)
- Invalid SMILES → a row of zeros, keeping alignment.

## `predict_logits(model, spectra, adduct, adducts, instrument_id_, device, max_peaks, collision_energies)` (lines 248–286)

Runs the model on all spectra of one molecule and averages.

```python
    for (mz_val, peaks), ce_val in zip(spectra, collision_energies):
        sel_mz, sel_it = select_peaks(peaks[:, 0], peaks[:, 1], max_peaks)
        n = len(sel_mz)
        mz = np.zeros((1, max_peaks), np.float32)
        it = np.zeros((1, max_peaks), np.float32)
        mz[0, :n], it[0, :n] = sel_mz, sel_it
        pad = np.ones((1, max_peaks), bool)
        pad[0, :n] = False
        with torch.no_grad():
            z = model(... tensors ...)
        outs.append(z[0].float().cpu().numpy())
    return np.mean(outs, axis=0)
```

- `peaks[:, 0]` = all m/z values (column 0), `peaks[:, 1]` = intensities.
- **Padding by hand:** fill the first `n` slots with real peaks; the rest stay
  0 and are marked `pad = True`, so the model ignores them.
- `with torch.no_grad():` — a **context manager**: inside the block, PyTorch
  doesn't record the computation (inference only).
- `.cpu().numpy()`: move the result off the GPU and convert to numpy.
- **Averaging logits** across a molecule's spectra: logits are log-odds, and
  adding log-odds is the natural way to combine independent evidence.
- `adducts.index(adduct) if adduct in adducts else 0`: unknown adduct → id 0.
- **Collision energy default 32 eV** = the training median. Passing 0 (as the
  code once did) asked the model about an energy it almost never saw — one of
  the Phase 3 bugs that took the leaderboard to 0.196.
- **`max_peaks` must equal what the model was trained with** (256). Callers
  read it from the checkpoint; the Gradio app forgot to, and was fixed in the wrap-up.

## `save(model, path, **extra)` / `load(path, device)` (lines 289–303)

```python
def save(model, path, **extra):
    torch.save({"cfg": model.cfg, "state": model.state_dict(), **extra}, path)
```

- `state_dict()` = all trained weights, by name (like `peak.weight`).
- `**extra` collects any extra keyword arguments into a dict, and `**extra`
  inside `{...}` merges them in. That's how checkpoints also carry `bit_sel`,
  `adduct_list`, `max_peaks`, `train_keys`, `model_tag`, `val_loss` — **a model's
  preprocessing settings travel with its weights.**

```python
def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"])
    if "extra_feats" not in cfg:
        cfg["extra_feats"] = ck["state"]["peak.weight"].shape[1] > 2 * cfg["d"] + 1
    model = FPNet(**cfg).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck
```

- `map_location`: load onto CPU or GPU regardless of where it was saved.
- `weights_only=False`: allow non-tensor data (lists, strings) in the file.
  ⚠ This uses Python's *pickle*, which can run code — only load checkpoints you trust.
- **Backward compatibility:** old checkpoints (v2) predate `extra_feats`. Rather
  than guess, it's inferred from the size of the layer that changed: 513 inputs
  (no extra features) vs 515.
- `FPNet(**cfg)` rebuilds the architecture; `load_state_dict` pours the weights in.
- `.eval()` so dropout is off.

## `_self_check()` (lines 306–368)

| check | promise |
|---|---|
| tie rule: keeps the **first** of three tied 1s | `select_peaks` is deterministic |
| identity Platt changes nothing; real Platt changes something | calibration wired correctly |
| output shape `(3, 64)`, all finite | the network runs |
| changing padded slots' values doesn't change the output | **padding never leaks** |
| shuffling peaks doesn't change the output | **order doesn't matter** (a spectrum is a set) |
| the candidate matching the logits scores higher | scoring points the right way |

`torch.manual_seed(0)` makes the random test data identical on every run.
`.eval()` is used so dropout's randomness can't make the comparisons fail.

---

## Try it

```bash
uv run python src/casmi/fpnet.py
uv run python -c "
import sys; sys.path.insert(0, 'src')
import numpy as np
from casmi.fpnet import load, predict_logits
m, ck = load('data/fpnet/fpnet_v4c.pt')
peaks = np.array([[110.06, .15], [123.0441, .31], [138.0662, 1.0], [195.0877, .42]], np.float32)
z = predict_logits(m, [(195.087652, peaks)], '[M+H]+', list(ck['adduct_list']), 0, max_peaks=ck['max_peaks'])
print(z.shape, 'most confident bits:', np.argsort(-z)[:5])
"
```
