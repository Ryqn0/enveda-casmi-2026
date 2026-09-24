# Part 10 — Tutorial: build the in-browser demo from scratch

Part 9 explains **what** each file of the demo does. This part is the **how**: the
order it was built in, the decisions behind it, the commands (every flag
explained), what went wrong and how each problem was found. Follow it top to
bottom and you rebuild the demo yourself.

Every command and output below was run on this project on 2026-09-24.

---

## What you're building

```
                    Hugging Face static Space (only serves files)
   ┌──────────────────────────────────────────────────────────────────┐
   │ index.html  core.js  meta.json  model.onnx  pool/*.json  vendor/ │
   └──────────────────────────────────────────────────────────────────┘
                                   │  downloads (once, then cached)
                                   ▼
   Visitor's browser ── does ALL the computing ─────────────────────────
     peaks ─► select_peaks ─► ONNX model ─► 1,583 bit logits ─┐
     precursor + adduct ─► neutral mass ─► pool shard ─► candidates
                                               RDKit.js ─► their bits ─┤
                                                                       ▼
                                          log-likelihood scores ─► top 25
```

No server runs any of our code. That's what makes it free, and it's also why the
work is harder: the Python pipeline has to be rebuilt in the browser.

**Prerequisites:** the project set up (Parts 1–5), a trained model
(`data/fpnet/fpnet_v4c.pt`), the pool (`data/pool.parquet`) and the COCONUT key
list (`data/coconut_keys.npy`).

---

## Step 0 — decide what the demo does

A demo isn't the competition pipeline. Choose the **smallest piece that shows
the idea**:

| piece | size | in the demo? |
|---|---|---|
| spectral library index (Flash Entropy) | 2.3 GB on disk, ~6 GB RAM to build | **no**: too big to download |
| analog channel | needs that index | **no** |
| fingerprint model | 22 MB as ONNX | **yes** |
| candidate pool | 34 MB parquet | **yes**, COCONUT part only (licensing, Step 5) |

So the demo is the **fingerprint channel only**: spectrum → substructures →
ranked candidates. It's the part that's ours, and it needs no library. It
costs accuracy on molecules with a close spectral relative (the analog
channel's strength), and the page says so.

> **Lesson:** decide scope from sizes and dependencies *before* writing code.
> The index size alone rules out half the pipeline.

---

## Step 1 — the first try: a Python web app (Gradio)

The obvious route is to reuse the Python directly. `app/app.py` does it in ~150
lines (Part 9 walks through it):

```bash
uv run --extra app python app/app.py
```

- `uv run`: run a command inside the project's virtual environment.
- `--extra app`: also install the optional `app` dependency group
  (`gradio`) from `pyproject.toml`.
- Gradio prints a local address (`http://127.0.0.1:7860`); open it in a browser.

It worked locally. Then `scripts/deploy_space.py` tried to put it on a free
Hugging Face Space, and Hugging Face answered **402 Payment Required**: Gradio
and Docker Spaces now need a paid account. Only **static** Spaces (files
only, no code running on the server) are free.

> **Lesson:** check a platform's current pricing and limits before building for
> it. The Gradio app isn't wasted: it's the reference the browser version
> copies, and it still runs locally.

---

## Step 2 — plan the static version

With no server, the browser must do three things Python did:

| problem | Python used | browser answer |
|---|---|---|
| run the neural network | PyTorch | **ONNX Runtime Web**: runs `.onnx` model files in WebAssembly |
| compute candidate fingerprints | RDKit (C++) | **RDKit.js**: the *same* C++ RDKit compiled to WebAssembly |
| find candidates by mass | `CandidatePool` (numpy, in memory) | **static JSON files**, one per dalton, fetched on demand |

**WebAssembly** (`.wasm`) is a compact binary format browsers run at
near-native speed. It's what makes running a transformer and a chemistry toolkit
in a web page realistic.

And one rule, from the whole project's history of train/serve skew:

> **The browser must compute the same numbers as Python, and a test must prove
> it.** That test (the "golden test", Step 9) is designed in from the start,
> not added at the end.

---

## Step 3 — export the model to ONNX

```bash
uv run --with onnx --with onnxruntime --with onnxscript python scripts/export_onnx.py
```

- `--with onnx`: make the `onnx` package available **for this command only**,
  without adding it to the project. Same for `onnxruntime` (to verify the
  export) and `onnxscript` (needed by PyTorch's exporter).

It prints three lines: the file size (22.1 MB), the largest logit difference
between ONNX Runtime and PyTorch on 60 real spectra (the script stops if it's
above 1e-3), and `wrote web/meta.json (1583 bits, 10 adducts)`.

What the script does, in plain terms (code in Part 9):

1. Build **one example input** with the exact shapes `predict_logits` uses.
2. `torch.onnx.export` runs the model on it and **records** every operation into
   a graph file.
3. **Verify:** run 60 real test spectra through both PyTorch and ONNX Runtime and
   require the logits to agree within 1e-3. An export can succeed silently and
   still compute something slightly different; only this comparison catches it.

**Look inside the file yourself:**

```bash
uv run --with onnxruntime python -c "
import onnxruntime as ort
s = ort.InferenceSession('web/model.onnx')
for i in s.get_inputs(): print('input ', i.name, i.shape, i.type)
for o in s.get_outputs(): print('output', o.name, o.shape, o.type)
"
```

```
input  mz [1, 256] tensor(float)
input  inten [1, 256] tensor(float)
input  pad [1, 256] tensor(bool)
input  prec [1] tensor(float)
input  add_id [1] tensor(int64)
input  ins_id [1] tensor(int64)
input  ce [1] tensor(float)
output logits [1, 1583] tensor(float)
```

This is the **contract** between Python and JavaScript: seven named inputs with
fixed shapes and types, one output. The JavaScript must build exactly these.

**Design choice: fixed shapes.** Every spectrum is padded to 256 peaks with a
`pad` mask, as in training. The browser never deals with variable sizes, and
`[1, 256]` is small enough to run in milliseconds.

---

## Step 4 — `meta.json`: never retype a constant

The same script writes `web/meta.json`:

```
['model_tag', 'max_peaks', 'adducts', 'bit_sel', 'fp', 'adduct_mass', 'offset_ppm', 'default_ce']
```

These are every number the browser needs besides the weights: which 1,583 of
the 2,048 fingerprint bits the model predicts (`bit_sel`), the adduct list **in
training order** (the model's adduct ids are positions in it), the adduct
masses, the 0.98 ppm timsTOF offset…

They're **generated from the checkpoint and the Python source**. If you typed
`offset_ppm = 0.98` into JavaScript by hand, it would silently go stale the day
the Python changes. This is the exact bug class of Phase 3 (a hard-coded 128 for
a 256-peak model).

---

## Step 5 — the candidate pool as static files

```bash
uv run python scripts/build_web_pool.py
```

It reports how many pool structures are COCONUT's (473,297 of 722,737) and
writes **2,593 shards, 53 MB in total**, covering 2.0–7,855.7 Da.

**Two decisions:**

1. **COCONUT only.** COCONUT is open data (CC0). The full pool also contains
   structures from the competition's `train.parquet`, which shouldn't be
   redistributed while the competition runs.
2. **Shard by dalton.** 53 MB is too much to download before every search. One
   file per whole-number mass means a query at 194.08 Da downloads only
   `194.json`: **39 KB** instead of 53 MB. A window that crosses a boundary
   (e.g. 194.999 ± tol) fetches two files.

**Look at a shard:**

```bash
python3 -c "
import json; d = json.load(open('web/pool/194.json'))
print(len(d['m']), 'structures in 194.json; first:', d['m'][0], d['s'][0], d['k'][0])
"
```

```
653 structures in 194.json; first: 194.007136 C=CCOS(=O)(=O)SCC=C YLYFNQFCMOUCDJ
```

Three parallel lists: `m` (neutral mass, sorted), `s` (SMILES), `k` (InChIKey
first block). Parallel lists instead of a list of objects avoid repeating the
key names 653 times, so the file is smaller.

---

## Step 6 — install Node.js and the JavaScript packages

The golden test runs the JavaScript **outside** a browser, in Node.js. Node
wasn't installed, and `sudo apt install nodejs` gives an old version, so the
current LTS (long-term support) release went into your home folder:

```bash
V=$(curl -fsSL https://nodejs.org/dist/index.json | python3 -c "import json,sys; print(next(r['version'] for r in json.load(sys.stdin) if r['lts']))")
mkdir -p ~/.local/node
curl -fsSL https://nodejs.org/dist/$V/node-$V-linux-x64.tar.xz | tar -xJ -C ~/.local/node --strip-components=1
grep -q ".local/node/bin" ~/.bashrc || echo 'export PATH=$HOME/.local/node/bin:$PATH' >> ~/.bashrc
```

- `curl -fsSL URL`: download. `-f` fail on HTTP errors, `-s` silent, `-S` but
  still show errors, `-L` follow redirects.
- `index.json` lists every Node release; the Python one-liner prints the first
  (newest) one marked `lts`. `$( … )` captures that output into `V`
  (here `v24.21.0`).
- `mkdir -p`: create the folder, parents too, no error if it exists.
- `| tar -xJ -C DIR --strip-components=1`: unpack the download as it streams:
  `-x` extract, `-J` it's `.xz`-compressed, `-C DIR` into DIR,
  `--strip-components=1` drop the archive's top folder (`node-v24.21.0-linux-x64/`)
  so `bin/` lands directly in `~/.local/node`.
- `grep -q … || echo … >> ~/.bashrc`: add Node to your `PATH` in `~/.bashrc`
  **only if** it isn't already there (`-q` = quiet, just an exit code; `||` =
  run the right side only if the left failed; `>>` = append).
- No `sudo` needed: everything lives in your home folder.

Check: `node --version` → `v24.21.0`, `npm --version` → `11.19.0` (in a new
terminal, so `~/.bashrc` is re-read).

**Create the JavaScript project** in `web/`:

```bash
cd web
npm init -y
npm pkg set type=module name=casmi26-web private=true
npm install -D onnxruntime-node @rdkit/rdkit onnxruntime-web@1.30.0 --no-fund --no-audit
```

- `npm init -y`: create `package.json` with defaults (`-y` = yes to every question).
- `npm pkg set key=value`: edit `package.json` from the command line.
  `type=module` lets files use `import`/`export`; `private=true` stops it ever
  being published to the npm registry by accident.
- `npm install -D pkg…`: download packages into `node_modules/` and record them
  under `devDependencies` (`-D`). `@1.30.0` pins a version: **onnxruntime-web
  and onnxruntime-node must match** so the test and the page run the same
  engine version.
- `--no-fund --no-audit`: skip npm's funding messages and vulnerability report.

You'll see `npm warn install-scripts … approve <pkg>`: npm 11 no longer runs
packages' install scripts without approval. Nothing here needed them (the test
below runs).

Check: `npm ls --depth=0` →

```
├── @rdkit/rdkit@2026.3.6
├── onnxruntime-node@1.30.0
└── onnxruntime-web@1.30.0
```

`web/node_modules/` is gitignored: `npm install` recreates it from `package.json`.

---

## Step 7 — port the logic: one JavaScript function per Python function

This is the core of the work. The method:

1. **List the Python functions on the demo's path**, in order: `parse_peaks`,
   `select_peaks`, the tensors of `predict_logits`, `neutral_mass`, `query`,
   `candidate_bits`, `score_candidates`, `argsort`.
2. **Write one JavaScript function for each**, in one file (`web/core.js`),
   with a comment naming the Python function it mirrors. Nothing else goes in
   that file: no page code, no HTML.
3. **Copy the numerics, not just the idea:**
   - numpy used **float32** for peaks; JavaScript numbers are float64. So values
     are rounded to float32 at the same points (`Math.fround`, `Float32Array`).
   - **Ties** must break the same way. Sorting "most intense first", two peaks of
     equal intensity must come out in the same order in both languages (Step 9
     shows what happened when they didn't).
   - Scores are summed in **float64** (`Float64Array`), like numpy.
4. **Pass in whatever differs between environments.** The page and the test
   use different ONNX Runtime packages and read shards differently (HTTP vs
   disk). So `core.js` never imports them; callers pass `ort` and a
   `fetchShard` function in. The **same** `core.js` then runs in both places.

| Python (`src/casmi/`) | JavaScript (`web/core.js`) |
|---|---|
| `app.parse_peaks` | `parsePeaks` |
| `fpnet.select_peaks` | `selectPeaks` |
| tensors inside `fpnet.predict_logits` | `makeFeeds` + `predictLogits` |
| `pool.CandidatePool.neutral_mass` | `neutralMass` |
| `pool.CandidatePool.query` | `queryPool` |
| `fpnet.candidate_bits` | `candidateBits` |
| `fpnet.score_candidates` | `scoreCandidates` |
| `np.argsort(-s, kind="stable")[:k]` | `rankTop` |

Line-by-line explanations of each are in Part 9.

---

## Step 8 — run it in Node before touching a browser

A browser adds caching, security rules and a UI between you and your bug. Node
runs the same `core.js` from the terminal in seconds. From `web/`:

```bash
node --input-type=module <<'EOF'
import fs from "node:fs";
import ort from "onnxruntime-node";
import initRDKitModule from "@rdkit/rdkit";
import { parsePeaks, predictLogits, neutralMass, queryPool, candidateBits, scoreCandidates, rankTop } from "./core.js";

const meta = JSON.parse(fs.readFileSync("meta.json"));
const session = await ort.InferenceSession.create("model.onnx");
const RDKit = await initRDKitModule();
const fetchShard = async (b) => fs.existsSync(`pool/${b}.json`) ? JSON.parse(fs.readFileSync(`pool/${b}.json`)) : null;

const peaks = parsePeaks("110.0600 0.15\n123.0441 0.31\n138.0662 1.00\n195.0877 0.42");
const m = neutralMass(195.0877, "[M+H]+", meta);
const { smiles, keys } = await queryPool(fetchShard, m, 5);
const { logits } = await predictLogits(session, ort, peaks, 195.0877, "[M+H]+", meta);
const scores = scoreCandidates(logits, smiles.map((s) => candidateBits(RDKit, s, meta)));
console.log("neutral mass", m.toFixed(6), "|", smiles.length, "candidates");
for (const [r, i] of rankTop(scores, 5).entries()) console.log(`#${r + 1}`, keys[i], scores[i].toFixed(1), smiles[i]);
EOF
```

- `node --input-type=module`: read the program from standard input and treat
  it as a module (so `import` and top-level `await` work).
- `<<'EOF' … EOF`: a **heredoc**, which feeds the lines in between as standard
  input. Quoting `'EOF'` stops the shell from touching `$` and backticks inside.

Output:

```
neutral mass 194.080233 | 15 candidates
#1 RYYVLZVUVIJVGH -324.5 Cn1c(=O)c2c(ncn2C)n(C)c1=O
#2 XWJJHUXHXCANGE -325.4 CN1C=NC2=C(C1=O)N(C)C(=O)N2C
#3 NWCPBHZYJHWSEL -332.6 CN1C=NC(=O)C2=C1N(C)C(=O)N2C
#4 SIQPXVQCUCHWDI -332.8 CCCN1C(=O)NC(=O)C2=C1N=CN2
#5 PTEMGJSNACOECE -335.2 CC(=O)NC1=CC(NC(C)=O)=NC=N1
```

`RYYVLZVUVIJVGH` is **caffeine**: rank 1 out of 15 same-mass candidates. #2 and
#3 are its isomers (same atoms, rearranged), which is exactly why ranking is
the hard part. The scores are log-likelihoods: higher (less negative) is better.

"It gives a sensible answer" is not "it matches Python", though. That's the
next step.

---

## Step 9 — the golden test: prove JavaScript = Python

**Idea:** run the real Python on fixed inputs, save **every intermediate
result** to a file (the "golden" answers), then make the JavaScript reproduce
each one.

```bash
uv run python scripts/make_golden.py        # Python -> web/test/golden.json
cd web && node test/golden.mjs              # JavaScript vs golden.json
```

```
cases                 31
max |logit diff|      1.62e-5  (ONNX Runtime Node vs PyTorch)
fingerprint rows      718, mismatched 0  (RDKit.js vs Python RDKit)
max |score diff|      1.93e-12
rank swaps at ties    2

ALL GOLDEN CHECKS PASSED
```

**How to choose what to compare, and how strictly:**

| stage | strictness | why |
|---|---|---|
| selected peaks | exact | pure bookkeeping: no arithmetic, so any difference is a bug |
| logits | 1e-3 | two different engines (PyTorch vs ONNX Runtime) do floating-point math in different orders; tiny differences are normal |
| candidate list | exact | the same files and the same `>=`/`<` window |
| fingerprint bits | exact | the same C++ RDKit, compiled twice |
| scores | 1e-6, **on the golden logits** | feeding Python's logits isolates the scoring code from model noise |
| ranking | exact, except ties | candidates with equal scores may legitimately swap |

Test on **real** inputs (30 test spectra, 718 candidates), not only caffeine:
real spectra have 200+ peaks, tied intensities, and odd adducts.

### The bug the test found, in Python

The first run failed at stage 1, "selected peaks differ", on spectra with more
than 256 peaks. The JavaScript sorted by intensity and broke ties by
position. The Python used `np.argpartition`, which picks the top 256 quickly but
makes **no promise about which of several tied peaks it keeps**. So Python's
choice among ties was arbitrary. The fix went into **Python**
(`np.argsort(-it, kind="stable")`, a documented tie rule), and the JavaScript
matches it.

> **Lesson:** a second implementation plus a comparison test finds bugs in the
> **first** implementation too. Neither side is "the truth" until they agree
> for a stated reason.

A second fix came from the golden file itself: logits were first saved rounded,
and that rounding, summed over 1,583 bits, showed up as ~1e-4 score
differences that could hide real ones. They're now saved at full precision.

---

## Step 10 — the page

`web/index.html` is one file: some CSS, a form, and a module script that
imports `core.js` and wires it to the form (Part 9 goes through it). Two
things to know before you open it:

**You can't just double-click `index.html`.** A page opened as `file://…` isn't
allowed to `fetch()` local files or load modules, so nothing would load.
Serve the folder over HTTP instead:

```bash
uv run python -m http.server -d web 8000
```

- `python -m http.server`: Python's built-in minimal web server (it serves
  files, nothing else, like a static Space).
- `-d web`: serve the `web/` folder. `8000`: the port.
- Open `http://localhost:8000`; stop with Ctrl+C.

From WSL, `localhost` works in the Windows browser too.

**Check what the browser will see**, from a second terminal:

```bash
curl -s -o /dev/null -w "%{http_code} %{content_type}\n" localhost:8000/core.js
curl -s -o /dev/null -w "%{http_code} %{size_download} bytes\n" localhost:8000/pool/194.json
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/pool/99999.json
```

- `-o /dev/null`: throw the body away; `-w FORMAT`: print only what you ask for
  (status code, content type, size).

```
200 text/javascript
200 38931 bytes
404
```

The 404 is expected: no structure weighs 99,999 Da. `fetchShard` turns a
missing shard into `null`, so a window at the edge of the pool doesn't crash.

**The libraries come from `web/vendor/`**, not from a CDN:

```bash
uv run python scripts/build_web.py
```

This copies ONNX Runtime Web's and RDKit.js's files out of `node_modules/` into
`web/vendor/`. The page then runs **the exact files the golden test ran**,
never a newer version a CDN might serve.

---

## Step 11 — debugging in the browser

Open the browser's developer tools with **F12**:

- **Console** tab: JavaScript errors and `console.log` output.
- **Network** tab: every file the page requested, with its status (200, 404,
  401…). Tick "Disable cache" while developing, or you may be testing yesterday's
  file.

Three errors, in the order they appeared:

### Error 1: `Failed to resolve module specifier 'vendor/ort-wasm-simd-threaded.mjs'`

- **Seen:** locally, the page said "Could not load the model: no available
  backend found".
- **Why:** ONNX Runtime loads a helper file with `import("vendor/…")`. In
  JavaScript modules, a path **without** `./` or `/` is read as a *package name*
  (like `import "react"`), not a file path.
- **Fix:** build an absolute URL from the page's own address:
  `new URL("vendor/", location.href).href`.

### Error 2: `Failed to fetch dynamically imported module` (401) on the private Space

- **Seen:** fine locally, broken once deployed privately.
- **Diagnosis:** the failing request was the helper `.mjs`, answered **401
  Unauthorized** (the Network tab shows this as a red row), while an ordinary
  `fetch()` of the very same file worked. A
  private Space needs your login on every request; the dynamic `import()` was
  sent without it.
- **Fix:** don't let ONNX Runtime download it. Download everything with plain
  `fetch()`, then hand it over **from memory**: the helper as a `blob:` URL
  (a temporary same-origin address for data already in memory, so no network
  request at all), and the `.wasm` engines as raw bytes (`wasmBinary`). The same
  treatment for RDKit's `.wasm`.
- **Lesson:** "works locally" and "works deployed" are different tests.
  Private hosting adds authentication, and it breaks things that public hosting
  doesn't.

### Pre-empted: multi-threading

ONNX Runtime can use several threads, but multi-threaded WebAssembly needs
special cross-origin-isolation headers that a static host can't be relied on
to send. `ort.env.wasm.numThreads = 1` avoids that whole failure class; the
model is small enough that one thread is fast.

---

## Step 12 — deploy

**One-time:** create a Hugging Face access token with **write** permission
(huggingface.co → Settings → Access Tokens) and put it in `.env`:

```
HF_TOKEN=hf_…
```

`.env` is gitignored; the scripts read it and never print it.

**Deploy:**

```bash
uv run --extra app python scripts/build_web.py --deploy
```

- `--extra app`: brings in `huggingface_hub` (through gradio).
- `--deploy`: after assembling `web/vendor/`, create the Space if needed
  (`space_sdk="static"`, **private**) and upload **only** the site files:
  `index.html`, `core.js`, `meta.json`, `model.onnx`, `pool/`, `vendor/`.
  Never `node_modules/` or the tests.
- `--public`: add it after the competition ends. Until then the model, trained
  on competition data, stays private.

Then open the Space's URL while logged in to Hugging Face, and use F12 → Network
on the first load (~43 MB, then cached).

---

## When the model changes: the full rebuild

The order matters: each step consumes the previous one's output.

```bash
uv run --with onnx --with onnxruntime --with onnxscript python scripts/export_onnx.py   # model + meta
uv run python scripts/build_web_pool.py                                                  # only if the pool changed
uv run python scripts/make_golden.py                                                     # new golden answers
(cd web && node test/golden.mjs)                                                         # must pass
uv run --extra app python scripts/build_web.py --deploy                                  # ship
```

`( … )` runs the `cd` in a sub-shell, so your terminal stays where it was.

**Never skip the golden test.** It's the only thing standing between "the model
changed" and "the page silently computes something else".

---

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| page stuck on "Loading…" | a file failed to load | F12 → Network: find the red row |
| "no available backend found" | ONNX Runtime couldn't load its wasm | check `web/vendor/` exists (`build_web.py`); check the Console for the real error |
| works locally, 401 on the Space | request sent without the Space's auth | load files with `fetch()` (Step 11, error 2) |
| "No COCONUT structure within ±…" | wrong adduct/precursor, or the molecule isn't in COCONUT | check the neutral mass on the status line; widen the ppm |
| golden: "selected peaks differ" | tie-breaking or float32 rounding differs | compare `select_peaks` and `selectPeaks` rule by rule |
| golden: fingerprint rows differ | RDKit versions differ, or `meta.fp` changed | pin versions; re-export `meta.json` |
| `node: command not found` | new PATH not loaded | open a new terminal, or `source ~/.bashrc` |
| old behaviour after a change | browser cache | F12 → Network → "Disable cache", then reload |

---

## Exercises

1. **Collision energy input.** The page always uses 32 eV (`meta.default_ce`).
   Add a number input and pass its value through `predictLogits` to
   `makeFeeds`. *Hint:* `makeFeeds` already takes `ce`; `predictLogits`
   doesn't pass it on yet. Does the golden test still pass? (It should: the
   golden cases use the default.)
2. **Show the mass error.** The table shows each candidate's neutral mass. Add
   a column with its difference from the query's neutral mass in ppm:
   `(mass − m) / m × 1e6`. Which rows are closest?
3. **Download the results.** Add a button that saves the top-k table as CSV.
   *Hint:* build the text, wrap it in `new Blob([text], {type: "text/csv"})`,
   make an `<a>` with `href = URL.createObjectURL(blob)` and `download =
   "candidates.csv"`, and `.click()` it. (The same `blob:` trick as Step 11.)
4. **Think:** why can't the analog channel be added the same way? Estimate
   what the browser would have to download (Step 0's table), and what you'd
   have to shrink to make it possible.

---

## What you learned

- Scope a demo by **size and dependencies** first.
- **ONNX** makes a PyTorch model runnable anywhere, and an export is only
  trusted after a **numeric comparison**.
- **Generate constants** (`meta.json`) from the source; never retype them.
- **Shard static data** so a query downloads kilobytes, not megabytes.
- Port with **one function per function**, the same float precision and the
  same tie rules, and keep environment-specific things out of the shared code.
- A **golden test** turns "looks right" into "proven equal", and it can find
  bugs in the original too.
- "Works locally" ≠ "works deployed". Debug with the browser's **Network tab**.
