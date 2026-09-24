// Golden test: does the JavaScript port reproduce the Python pipeline?
//
//   cd web && node test/golden.mjs
//
// Runs core.js -- the same module the web page imports -- on the inputs in
// test/golden.json (written by scripts/make_golden.py from the real Python
// code) and compares every stage. Exits non-zero on any disagreement.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ort from "onnxruntime-node";
import initRDKitModule from "@rdkit/rdkit";
import {
  candidateBits, neutralMass, parsePeaks, predictLogits, queryPool, rankTop,
  scoreCandidates,
} from "../core.js";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const meta = JSON.parse(fs.readFileSync(path.join(WEB, "meta.json")));
const golden = JSON.parse(fs.readFileSync(path.join(WEB, "test", "golden.json")));
const session = await ort.InferenceSession.create(path.join(WEB, "model.onnx"));
const RDKit = await initRDKitModule();
const fetchShard = async (b) => {
  const f = path.join(WEB, "pool", `${b}.json`);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f)) : null;
};

const fail = [];
let worstLogit = 0, worstScore = 0, bitRows = 0, bitMismatch = 0, tieSwaps = 0;

for (const [ci, c] of golden.cases.entries()) {
  // 1. peak selection -- exact, it is pure bookkeeping
  const { sel, logits } = await predictLogits(session, ort, c.peaks, c.prec, c.adduct, meta);
  const same = sel.mz.length === c.sel_mz.length &&
    c.sel_mz.every((v, i) => sel.mz[i] === v && sel.it[i] === c.sel_it[i]);
  if (!same) fail.push(`case ${ci}: selected peaks differ`);

  // 2. model output -- floating point, so a tolerance
  const d = Math.max(...c.logits.map((v, i) => Math.abs(v - logits[i])));
  worstLogit = Math.max(worstLogit, d);

  // 3. candidates from the mass window -- exact
  const m = neutralMass(c.prec, c.adduct, meta);
  const { smiles, keys } = await queryPool(fetchShard, m, golden.ppm);
  if (keys.join() !== c.cand_keys.join()) fail.push(`case ${ci}: candidate lists differ`);

  // 4. fingerprint bits, RDKit.js vs Python RDKit -- exact
  const bits = smiles.map((s) => candidateBits(RDKit, s, meta));
  bits.forEach((b, i) => {
    bitRows++;
    if (b.join("") !== c.cand_bits[i]) bitMismatch++;
  });

  // 5. scores on the GOLDEN logits -- isolates the scoring code from model noise
  const goldenLogits = Float32Array.from(c.logits);
  const s = scoreCandidates(goldenLogits, bits);
  s.forEach((v, i) => { worstScore = Math.max(worstScore, Math.abs(v - c.scores[i])); });

  // 6. ranking -- identical, except candidates whose golden scores tie
  const top = rankTop(s, golden.k).map((i) => keys[i]);
  top.forEach((k, r) => {
    if (k === c.top_keys[r]) return;
    const a = c.scores[keys.indexOf(k)], b = c.scores[keys.indexOf(c.top_keys[r])];
    if (Math.abs(a - b) < 1e-3) tieSwaps++;
    else fail.push(`case ${ci}: rank ${r + 1} is ${k}, expected ${c.top_keys[r]}`);
  });
}

// 7. the page's peak parser, mirroring app.py parse_peaks: normalise to the
// base peak, sort by m/z, skip junk lines, accept commas and tabs.
const parsed = parsePeaks("138.0662 50\n110.06,7.5\njunk line\n123.0441\t15.5\n");
const want = [[110.06, 0.15], [123.0441, 0.31], [138.0662, 1.0]];
if (parsed.length !== 3 || parsed.some((r, i) =>
    Math.abs(r[0] - want[i][0]) > 1e-4 || Math.abs(r[1] - want[i][1]) > 1e-6)) {
  fail.push(`parsePeaks gave ${JSON.stringify(parsed)}`);
}

console.log(`cases                 ${golden.cases.length}`);
console.log(`max |logit diff|      ${worstLogit.toExponential(2)}  (ONNX Runtime Node vs PyTorch)`);
console.log(`fingerprint rows      ${bitRows}, mismatched ${bitMismatch}  (RDKit.js vs Python RDKit)`);
console.log(`max |score diff|      ${worstScore.toExponential(2)}`);
console.log(`rank swaps at ties    ${tieSwaps}`);
if (worstLogit > 1e-3) fail.push(`logits differ by up to ${worstLogit}`);
if (bitMismatch) fail.push(`${bitMismatch} fingerprint rows differ`);
if (worstScore > 1e-6) fail.push(`scores differ by up to ${worstScore}`);
if (fail.length) {
  console.log("\nFAIL\n  " + fail.slice(0, 10).join("\n  "));
  process.exit(1);
}
console.log("\nALL GOLDEN CHECKS PASSED");
