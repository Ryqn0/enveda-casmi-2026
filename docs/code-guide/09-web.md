# Part 9 — the in-browser demo: Python → ONNX → JavaScript

**Written:** Phase 6, after the Gradio Space hit 402 (Part 8c). **Live at:**
the private static Space `Ryqn/casmi26-spectrum-to-structure`.

The demo does the **fingerprint channel only** (Part 5) over COCONUT: paste a
spectrum → the model predicts substructure bits → every natural product of the
right mass is scored by how well its fingerprint fits. No server runs any code.
The browser downloads the files and computes everything itself.

**Why this is harder than it looks:** the Python pipeline now has a **second
implementation** in JavaScript. Two implementations of one idea drift apart, and
that's the train/serve-skew lesson of the whole project. So the design is:
**one** JavaScript module (`core.js`), used by both the page and a test, and a
**golden test** that checks it against outputs of the real Python code.

---

## Build order

| step | command | produces |
|---|---|---|
| 1 | `uv run --with onnx --with onnxruntime --with onnxscript python scripts/export_onnx.py` | `web/model.onnx`, `web/meta.json` |
| 2 | `uv run python scripts/build_web_pool.py` | `web/pool/*.json` |
| 3 | `uv run python scripts/make_golden.py` | `web/test/golden.json` |
| 4 | `cd web && npm install && node test/golden.mjs` | pass / fail |
| 5 | `uv run --extra app python scripts/build_web.py --deploy` | `web/vendor/`, the Space |

- `uv run --with X`: add package X **for this one command** without adding it
  to the project's dependencies (the ONNX tools are only needed for export).
- `--extra app`: include the optional dependency group `app` from
  `pyproject.toml`. That group only lists `gradio`, but gradio depends on
  `huggingface_hub`, which is what the deploy step uses.
- `npm install`: download the JavaScript packages listed in
  `web/package.json` into `web/node_modules/` (the npm twin of `uv sync`).

All generated files are **gitignored**: they're rebuilt from the repo, never
edited.

---

## Step 1 — `export_onnx.py`: the model as a portable file

**ONNX** ("Open Neural Network Exchange") is a file format that describes a
network as a graph of standard operations (matrix multiply, softmax, …) plus
the weights. Any ONNX runtime, in Python, C++, Node or a browser, can then run
it without PyTorch. **ONNX Runtime Web** runs it in the browser via
**WebAssembly** (a compact binary format browsers run at near-native speed).

### `inputs_for(peaks, prec, adduct, adducts, max_peaks, ce=32.0)` (lines 34–49)

Builds **exactly** the tensors `predict_logits` builds (Part 5), as numpy
arrays: `select_peaks` (the shared function), zero-padded to 256, the `pad`
mask (`True` = padding), adduct id, instrument 0 (timsTOF), collision energy.
A fixed shape `[1, 256]` means the browser never handles variable shapes.

### `main()` — export, then **prove** the export (lines 52–99)

```python
torch.onnx.export(model, args, OUT / "model.onnx", input_names=names,
                  output_names=["logits"], opset_version=17, dynamo=False)
```

- `torch.onnx.export` runs the model once on example inputs (`args`) and
  records every operation into the graph (this is called *tracing*).
- `input_names`/`output_names`: the names JavaScript uses to feed and read it
  (`mz`, `inten`, …, `logits`).
- `opset_version=17`: which version of the ONNX operator set to target (newer =
  more operations; 17 is widely supported by runtimes).
- `dynamo=False`: use the classic, stable exporter rather than the newer
  experimental one.

Then the check that makes the file trustworthy:

```python
sess = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
for r in t.itertuples():                       # 60 real test spectra
    ref = predict_logits(model, [(float(r.precursor_mz), peaks)], ...)   # PyTorch
    got = sess.run(None, x)[0][0]                                        # ONNX Runtime
    worst = max(worst, float(np.abs(got - ref).max()))
assert worst < 1e-3
```

A model can export without any error and still compute something slightly
different (an operation translated differently). Only running both on real
inputs catches that. `sess.run(None, feeds)`: `None` = return all outputs.

**`meta.json`** (lines 88–98): everything the browser needs **besides** the
weights: `max_peaks`, the adduct list in training order, `bit_sel`, the
fingerprint settings (radius 2, 2048 bits), `ADDUCT_MASS`, the 0.98 ppm offset,
the default collision energy. It's **generated from the Python source and
checkpoint**, so nobody retypes a constant into JavaScript.

## Step 2 — `build_web_pool.py`: the candidates, sharded by mass

- **COCONUT only** (open data, CC0): the full pool also has structures from
  the competition's training file, which shouldn't be redistributed.
- Selected by **key membership** (`isin(coconut)`), not by the `source` label:
  `build_pool` labels a structure found in both sources `"train"`, so
  filtering on the label would drop those COCONUT structures.
- **Sharding** (lines 143–148):
  ```python
  for b, g in p.groupby(np.floor(p.exact_mass).astype(int)):
      (OUT / f"{b}.json").write_text(json.dumps({"m": [...], "s": [...], "k": [...]},
                                                 separators=(",", ":")))
  ```
  One JSON file per dalton: `194.json` holds every structure with mass
  194.000–194.999. A query downloads only the one or two files its window
  touches, tens of KB instead of all 53 MB (2,593 files; `194.json` is 39 KB).
  `separators=(",", ":")` drops the spaces JSON normally adds (smaller files).
  Masses are rounded to 6 decimals (1 µDa, far below the ppm window).

## Step 3 — `make_golden.py`: what the JavaScript must reproduce

For 30 real test spectra plus caffeine, it runs **the real Python functions**
and records every intermediate:

| field | from | checked how |
|---|---|---|
| `sel_mz`, `sel_it` | `select_peaks` | exactly |
| `logits` | `predict_logits` (PyTorch) | within 1e-3 |
| `cand_keys` | `shard_query` on `web/pool` | exactly |
| `cand_bits` | `candidate_bits` (Python RDKit) | exactly |
| `scores` | `score_candidates` | within 1e-6 |
| `top_keys` | `argsort(-scores, kind="stable")[:25]` | exactly, except at ties |

- **`shard_query`** reads the **same shard files the browser loads**, so the
  test compares logic, not two copies of the data. It uses the same half-open
  window `m − tol <= mass < m + tol` that `CandidatePool.query` gets from
  `searchsorted`.
- `PPM = 5.0`, the web page's default (the Kaggle pipeline uses 4).
- **Full-precision logits** (line 79): the first version rounded them, and the
  rounding error summed over 1,583 bits showed up as ~1e-4 score noise, big
  enough to hide real disagreements.
- Bits are stored as strings like `"0100…"`, compact and easy to compare.

---

## Step 4 — `web/core.js`: the port, one function per Python function

JavaScript notes for Python readers: `export function` = a function other files
can `import`; `const` = a variable that won't be reassigned; `let` = one that
will; `Float32Array` / `Uint8Array` = typed arrays, like numpy arrays of a fixed
dtype; `(a, b) => a - b` = a lambda; `async`/`await` = wait for a slow operation
(a download, the model) without freezing the page.

### `parsePeaks(text)` ← `app.py parse_peaks`

- `text.trim().split(/\r?\n/)`: split into lines (`\r?` also handles Windows
  line endings). `/…/` is a regular expression.
- `line.replace(/,/g, " ")`: commas → spaces (`g` = every occurrence), so
  `110.06,7.5` works; `split(/\s+/)` splits on any whitespace, tabs included.
- `Number.isFinite(...)` skips junk lines.
- **`Math.fround(x)`**: round to the nearest **float32**. JavaScript numbers are
  float64, but numpy used float32; rounding at the same point makes the two
  sides see identical values.
- Normalise to the base peak (`/ top`), then sort by m/z.

### `selectPeaks(mzIn, itIn, maxPeaks)` ← `fpnet.select_peaks`

```js
idx.sort((a, b) => it[b] - it[a] || a - b);   // most intense first; ties -> earlier index
idx = idx.slice(0, maxPeaks);
idx.sort((a, b) => mz[a] - mz[b] || a - b);   // then ascending m/z
```

A sort comparator returns negative / zero / positive. `x || y` returns `y`
when `x` is 0, so **ties are broken by original position**. This is the rule
Python now uses too (`np.argsort(-it, kind="stable")`). The golden test found
that the old Python used `np.argpartition`, whose choice among tied peaks isn't
guaranteed, and the Python side was changed to match (Part 5).

### `makeFeeds(ort, sel, precursorMz, adduct, meta, ce)` ← the tensors of `predict_logits`

- `new Float32Array(n)` starts as zeros = padding; `.set(...)` copies the real
  peaks in; `pad.fill(0, 0, count)` marks the real slots.
- `meta.adducts.indexOf(adduct)` returns −1 when unknown; `Math.max(0, …)`
  maps that to 0, as the Python does.
- `new ort.Tensor("int64", BigInt64Array.of(BigInt(addIdx)), [1])`: ONNX wants
  int64, and JavaScript's normal numbers can't represent all 64-bit integers, so
  `BigInt` is used (`0n` is the BigInt zero).
- The `ort` object is passed **in**: in the browser it's onnxruntime-web, in the
  test onnxruntime-node. Same API, so `core.js` works in both.

### `predictLogits(session, ort, peaks, ...)`

Select peaks → build feeds → `await session.run(feeds)` → `out.logits.data`
(a `Float32Array` of 1,583 logits).

### `neutralMass(precursorMz, adduct, meta)` ← `pool.neutral_mass`

`(precursor − adduct_mass) / (1 + 0.98e-6)`: the same formula as Part 3, with
constants from `meta.json`. Returns `null` for an unknown adduct.

### `queryPool(fetchShard, m, ppm)` ← `CandidatePool.query`

Loop over the dalton shards the window touches (`Math.floor(m − tol)` to
`Math.floor(m + tol)`), keep `m − tol <= mass < m + tol`. `fetchShard` is
passed in: the page downloads over HTTP, the test reads from disk.

### `candidateBits(RDKit, smiles, meta)` ← `fpnet.candidate_bits`

- **RDKit.js** is RDKit compiled to WebAssembly, so the fingerprint comes from
  the **same C++ code** as Python's RDKit. That's why 718/718 fingerprint rows
  match exactly.
- `RDKit.get_mol(smiles)` inside `try … catch` (it may throw on bad input);
  invalid → an all-zero row, like Python.
- `mol.get_morgan_fp(JSON.stringify({radius: 2, nBits: 2048}))` returns a
  2048-character string of `0`/`1`. `fp.charCodeAt(b) === 49` checks whether
  character `b` is `"1"` (character code 49), then only the model's `bit_sel`
  bits are kept.
- **`mol.delete()`**: WebAssembly memory isn't garbage-collected; objects from
  RDKit.js must be freed by hand or the page leaks memory. `mol?.delete()`: call
  it only if `mol` isn't null (`?.` = "optional chaining").

### `scoreCandidates(logits, bitRows)` ← `fpnet.score_candidates`

`p = sigmoid(logit)`, then for each candidate the sum of `log(p + ε)` over its
1-bits and `log(1 − p + ε)` over its 0-bits, in `Float64Array` like numpy's
float64. `Math.log1p(x)` = `log(1 + x)`, accurate for small x.

### `rankTop(scores, k)` ← `np.argsort(-scores, kind="stable")[:k]`

JavaScript's `Array.prototype.sort` is guaranteed stable (since ES2019), so
equal scores keep their original order, as in numpy's stable sort.

---

## Step 4b — `web/test/golden.mjs`: the check

```bash
cd web && node test/golden.mjs
```

(`.mjs` = a JavaScript *module* file, which may use `import`/`export`.
`package.json` has `"type": "module"` for the same reason.)

It loads `meta.json`, `golden.json`, the ONNX model through **onnxruntime-node**
and RDKit.js, then for each case compares the six stages from the table above.
Two design choices:

- **Stage 5 scores the golden logits, not the ONNX ones**, so the scoring code
  is tested on its own, isolated from the model's tiny floating-point noise.
- **Stage 6 forgives swaps at ties**: if two candidates' golden scores differ by
  less than 1e-3, their order may legitimately differ. It counts them instead
  of failing.
- Stage 7 tests `parsePeaks` on a messy input (commas, a tab, a junk line, raw
  intensities).
- `process.exit(1)` on failure, so a script or CI can detect it.

Current output (re-run for this guide):

```
cases                 31
max |logit diff|      1.62e-5  (ONNX Runtime Node vs PyTorch)
fingerprint rows      718, mismatched 0  (RDKit.js vs Python RDKit)
max |score diff|      1.93e-12
rank swaps at ties    2

ALL GOLDEN CHECKS PASSED
```

Note: `npm test` is **not** wired up (the `"test"` entry in `package.json` is
npm's placeholder); run the file with `node` as above.

---

## Step 5 — `web/index.html` and `build_web.py`

### The page

- **CSS variables** (`--bg`, `--ink`, …) with a `prefers-color-scheme: dark`
  override: one set of colours per theme; the rest of the CSS only uses the
  variables.
- `<script src="vendor/ort.wasm.min.js">` and `RDKit_minimal.js`: classic
  scripts that define the globals `ort` and `initRDKitModule`.
- `<script type="module">`: the app code, which `import`s from `./core.js`.
  **Top-level `await`** is allowed in modules.

**Loading (lines 102–136)**, the part that took three attempts:

1. `new URL("vendor/", location.href)`: an **absolute** URL. A bare
   `import("vendor/x.mjs")` is read as a *package name*, not a path (first
   error: "Failed to resolve module specifier").
2. On a **private** Space every file request needs the Space's login cookie.
   ONNX Runtime's own dynamic `import()` of its glue file arrived without it and
   got **401**. Fix: download the files with ordinary `fetch()` (which does
   carry it) and hand them over **from memory**:
   - the glue code as a **`blob:` URL** (`URL.createObjectURL(new Blob([...]))`
     = a temporary same-origin URL pointing at data already in memory, so no
     network request at all);
   - the `.wasm` engines as raw bytes (`wasmBinary`), for both ONNX Runtime
     and RDKit.
3. `Promise.all([...])`: start the downloads **in parallel** and wait for all.
4. `ort.env.wasm.numThreads = 1`: multi-threaded WebAssembly needs special
   cross-origin-isolation HTTP headers a static host can't be relied on to send.

**Shard cache** (lines 138–142): a `Map` from shard number to a **promise** of
its contents. Storing the promise (not the result) means two queries for the
same shard share one download. Shards that don't exist resolve to `null`.

**On submit:** `e.preventDefault()` stops the browser's default form behaviour
(reloading the page); then parse → `neutralMass` → `queryPool` →
`predictLogits` → `candidateBits` → `scoreCandidates` → `rankTop`, and draw.
Structures are drawn by RDKit.js (`mol.get_svg(240, 180)`), no server needed.
Text from the data is inserted with **`textContent`**, never `innerHTML`, so a
SMILES string can't inject HTML into the page.

**"Static but dynamic":** *static hosting* means the server only sends files
and never runs code for you. Everything that changes on the page (the adduct
list, the results) is computed **by your browser** from those files.

### `build_web.py`

- **`vendor()`**: copy the five runtime files (ONNX Runtime's loader, glue and
  engine; RDKit's JS and WASM) from `node_modules` into `web/vendor/`. They
  are the **exact files the golden test ran against**, not a CDN copy that
  might be a different version. Then check every site file exists and print
  the total size.
- **`deploy(public)`**: `create_repo(..., space_sdk="static")` (a free static
  Space), upload the README (its YAML header says `sdk: static`), then
  `upload_folder` with **`allow_patterns`**: only the files in `SITE` (the
  page, `core.js`, `meta.json`, the model, `pool/**`, `vendor/**`), never
  `node_modules` or the tests. `pool/**` = everything under `pool/`, at any
  depth.
- `next(... for l in lines if ...)`: the first line that matches, i.e. the
  token (never printed).
- Private by default, like `deploy_space.py`; `--public` after the competition.

---

## `app/app.py` — the Python twin (Gradio)

The same demo as a Python web app, written first. It's what `deploy_space.py`
would have hosted (Part 8c), and it still runs locally:

```bash
uv run --extra app python app/app.py      # then open the printed http://127.0.0.1:7860
```

- **Module level** (lines 35–47): load the model and pool **once**, when the app
  starts, not per request. Paths come from environment variables
  (`os.environ.get("MODEL_PATH", default)`) so the Space can point them at its
  own files; locally the defaults are the repo's files.
  `MAX_PEAKS = int(CK.get("max_peaks", 128))`: read from the checkpoint. The app
  once omitted it and fell back to 128 for a 256-peak model: the same train/serve
  skew as Phase 3, fixed here in Phase 6.
- **`parse_peaks(text)`**: the function `parsePeaks` in `core.js` mirrors.
  `raise gr.Error(...)` shows the message to the user in the page instead of
  crashing.
- **`identify(...)`**: `POOL.query` (Part 3) → `predict_logits` (Part 5) →
  `score_candidates(logits, candidate_bits(...))` → `np.argsort(-scores)[:top_k]`.
  Only the fingerprint channel runs: the spectral index is 2.3 GB and ~6 GB to
  build, too heavy for a demo (the module docstring explains the trade-off).
  `Draw.MolsToGridImage` draws the structures into one image.
- **The UI** (lines 123–152): `gr.Blocks` lays out components (`with gr.Row():`
  / `gr.Column(scale=2)` = side by side, 2:1 width). `go.click(fn, inputs,
  outputs)` wires the button: Gradio reads the input components, calls `fn`
  with their values **in that order**, and puts the returned values into the
  output components.
- `demo.launch()` starts a local web server.

The browser version does exactly these steps in JavaScript. The one difference:
the app's pool includes training-set structures, while the web pool is COCONUT
only, so for some inputs the two can show different candidates.

---

## Why ONNX in the browser instead of a server?

| | Docker/Gradio server | static + ONNX |
|---|---|---|
| cost on Hugging Face | needs PRO (the 402) | free |
| who computes | the server, for everyone | each visitor's browser |
| cold start | an idle Space sleeps and must restart its Python process | none: files only, cached by the browser |
| extra work | none, it's the Python code | a JavaScript port + a golden test |

The extra work is the price, and the golden test is what makes paying it safe.
