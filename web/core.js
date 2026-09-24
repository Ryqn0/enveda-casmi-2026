// In-browser port of the fingerprint ranking pipeline.
//
// ONE implementation, shared by the web page (index.html) and the Node golden
// test (test/golden.mjs), which checks it against outputs of the Python
// pipeline. Each function names the Python function it mirrors; if you change
// either side, regenerate the golden file (scripts/make_golden.py) and rerun
// the test. Two implementations of one concept drift -- this project learned
// that the hard way (docs/07, docs/09).

// app/app.py parse_peaks: "m/z intensity" per line (commas and tabs work too),
// intensities normalised to a base peak of 1.0 -- as the model saw in
// training -- and sorted by m/z. Returns [[mz, it], ...]; throws on no peaks.
export function parsePeaks(text) {
  const rows = [];
  for (const line of text.trim().split(/\r?\n/)) {
    const parts = line.replace(/,/g, " ").trim().split(/\s+/);
    if (parts.length < 2) continue;
    const mz = Number(parts[0]), it = Number(parts[1]);
    if (Number.isFinite(mz) && Number.isFinite(it)) rows.push([Math.fround(mz), Math.fround(it)]);
  }
  if (!rows.length) throw new Error("No peaks found. Use one `m/z intensity` pair per line.");
  const top = Math.max(...rows.map((r) => r[1]));
  if (top > 0) rows.forEach((r) => { r[1] = Math.fround(r[1] / top); });
  return rows.sort((a, b) => a[0] - b[0]);
}

// casmi.fpnet.select_peaks: the max_peaks most intense peaks, in ascending m/z.
// Values are cast to float32 FIRST, as numpy does, so comparisons and ties
// behave identically.
export function selectPeaks(mzIn, itIn, maxPeaks) {
  const mz = Float32Array.from(mzIn);
  const it = Float32Array.from(itIn);
  let idx = Array.from(mz.keys());
  if (idx.length > maxPeaks) {
    // Most intense first; exact intensity ties broken by position, so the
    // choice is deterministic.
    idx.sort((a, b) => it[b] - it[a] || a - b);
    idx = idx.slice(0, maxPeaks);
  }
  idx.sort((a, b) => mz[a] - mz[b] || a - b);
  return { mz: Float32Array.from(idx, (i) => mz[i]), it: Float32Array.from(idx, (i) => it[i]) };
}

// casmi.fpnet.predict_logits, for one spectrum: the exact tensors it builds.
// `ort` is onnxruntime-web in the browser and onnxruntime-node in the test;
// both expose the same Tensor API.
export function makeFeeds(ort, sel, precursorMz, adduct, meta, ce = meta.default_ce) {
  const n = meta.max_peaks;
  const mz = new Float32Array(n);
  const it = new Float32Array(n);
  const pad = new Uint8Array(n).fill(1);       // 1 = padding slot, ignored by attention
  mz.set(sel.mz.subarray(0, n));
  it.set(sel.it.subarray(0, n));
  pad.fill(0, 0, Math.min(sel.mz.length, n));
  const addIdx = Math.max(0, meta.adducts.indexOf(adduct));   // unknown adduct -> 0, as in Python
  return {
    mz: new ort.Tensor("float32", mz, [1, n]),
    inten: new ort.Tensor("float32", it, [1, n]),
    pad: new ort.Tensor("bool", pad, [1, n]),
    prec: new ort.Tensor("float32", Float32Array.of(precursorMz), [1]),
    add_id: new ort.Tensor("int64", BigInt64Array.of(BigInt(addIdx)), [1]),
    ins_id: new ort.Tensor("int64", BigInt64Array.of(0n), [1]),   // timsTOF, as in the app
    ce: new ort.Tensor("float32", Float32Array.of(ce), [1]),
  };
}

export async function predictLogits(session, ort, peaks, precursorMz, adduct, meta) {
  const sel = selectPeaks(peaks.map((p) => p[0]), peaks.map((p) => p[1]), meta.max_peaks);
  const out = await session.run(makeFeeds(ort, sel, precursorMz, adduct, meta));
  return { sel, logits: out.logits.data };
}

// casmi.pool.CandidatePool.neutral_mass: measured ion m/z -> neutral mass.
export function neutralMass(precursorMz, adduct, meta) {
  const add = meta.adduct_mass[adduct];
  if (add === undefined) return null;
  return (precursorMz - add) / (1 + meta.offset_ppm * 1e-6);
}

// casmi.pool.CandidatePool.query: every structure with m - tol <= mass < m + tol
// (the half-open window numpy.searchsorted gives). The pool is split into one
// file per dalton; only the shards the window touches are fetched.
export async function queryPool(fetchShard, m, ppm) {
  const tol = (m * ppm) / 1e6;
  const smiles = [], keys = [], masses = [];
  for (let b = Math.floor(m - tol); b <= Math.floor(m + tol); b++) {
    const d = await fetchShard(b);             // null when the shard does not exist
    if (!d) continue;
    for (let i = 0; i < d.m.length; i++) {
      if (d.m[i] >= m - tol && d.m[i] < m + tol) {
        smiles.push(d.s[i]); keys.push(d.k[i]); masses.push(d.m[i]);
      }
    }
  }
  return { smiles, keys, masses };
}

// casmi.fpnet.candidate_bits: Morgan radius 2 / 2048 bits (casmi.analog), then
// the model's bit selection. Unparseable SMILES give an all-zero row, as in Python.
export function candidateBits(RDKit, smiles, meta) {
  const bits = new Uint8Array(meta.bit_sel.length);
  let mol = null;
  try { mol = RDKit.get_mol(smiles); } catch { mol = null; }
  if (!mol || !mol.is_valid()) { mol?.delete(); return bits; }
  const fp = mol.get_morgan_fp(JSON.stringify(meta.fp));   // "0101..." of length nBits
  mol.delete();
  meta.bit_sel.forEach((b, j) => { bits[j] = fp.charCodeAt(b) === 49 ? 1 : 0; });
  return bits;
}

// casmi.fpnet.score_candidates: log-likelihood of each candidate's bits under
// the predicted probabilities, in float64 like numpy.
export function scoreCandidates(logits, bitRows) {
  const eps = 1e-6;
  const lp = new Float64Array(logits.length);
  const lq = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const p = 1 / (1 + Math.exp(-logits[i]));
    lp[i] = Math.log(p + eps);
    lq[i] = Math.log1p(-p + eps);
  }
  return bitRows.map((bits) => {
    let s = 0;
    for (let i = 0; i < bits.length; i++) s += bits[i] ? lp[i] : lq[i];
    return s;
  });
}

// numpy.argsort(-scores, kind="stable")[:k]. Array.prototype.sort is stable.
export function rankTop(scores, k) {
  return Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, k);
}
