# Part 8b — scripts: training models and evaluating them

The scripts behind Phases 3–5: building training data, training the
fingerprint model, the experiments on top of it, and the evaluation tools that
eventually told us which experiments were real.

---

## `prepare_fpnet_data.py` — the training set (Phase 3, sampling changed in Phase 5)

Produces `data/fpnet/train.npz`: peaks, metadata and fingerprint labels for
~900k spectra.

**Constants** (lines 39–56):
- `MAX_PEAKS = 256` — measured: 69.8% of test spectra exceed 128 peaks.
- `MIN_FREQ, MAX_FREQ = 0.005, 0.95` — which fingerprint bits to keep (Part 5).
- `ADDUCT_LIST = sorted(TEST_ADDUCTS)` — a **fixed order**, because the model's
  adduct embedding table is indexed by it. Reordering between training and
  inference would silently scramble which vector means which adduct. It's saved
  into the data file and the checkpoint for that reason.

**`bits_of(smiles)`** (lines 59–66): `morgan_fp` (Part 4; the same fingerprint
the analog channel uses) → a numpy array of 2,048 zeros and ones.
`DataStructs.ConvertToNumpyArray` is RDKit's copy from its own bit-vector type
into numpy. Returns `None` for a SMILES RDKit can't parse; those molecules are
skipped.

**`instrument_id(s)`** (lines 69–80): the file has 82 free-text instrument
names; they're mapped to 4 families by substring (`timsTOF`, Orbitrap-like,
other TOF, other). `validate_fpnet.py` imports this function so both use the
same mapping.

**Two passes over the 3 GB file:**

1. **Pass 1 — which rows** (lines 110–138): read only 4 small columns, keep rows
   in the test mass range with a test adduct and a structure, then sample:
   - uniform over spectra (`--n-spectra`, the original), or
   - **`--cap-per-structure 4`** (Phase 5): at most 4 spectra per molecule.
     ```python
     d = d.sample(frac=1.0, random_state=args.seed)
     rows = np.sort(d.groupby("key").head(args.cap_per_structure).row.to_numpy())
     ```
     `sample(frac=1.0)` shuffles all rows; `groupby(key).head(4)` keeps the first
     4 of each molecule. Shuffling first makes the kept 4 random (file order
     correlates with source library). Uniform sampling had let molecules with up
     to 1,438 spectra dominate training.
2. **Fingerprint every needed structure once** (lines 141–163), then choose the
   informative bit band (`freq = fp_all.mean(axis=0)` = fraction of molecules
   with each bit on).
3. **Pass 2 — stream the peaks** (lines 166–208): for each kept row,
   `select_peaks` (Part 5 — **the same function inference uses**), then fill
   pre-allocated arrays. Pre-allocating with `np.zeros((n, 256))` and filling
   row by row avoids growing lists of arrays (slow and memory-hungry).
4. **Save** with `np.savez_compressed` (a zip of arrays). Fingerprints are
   stored with **`np.packbits`**: 8 yes/no bits per byte, 8× smaller.
   `keys` maps a structure's row in the fingerprint table to its InChIKey.

## `train_fpnet.py` — training the network

### `class SpectraSet(Dataset)` (lines 28–50)

PyTorch's **`Dataset`** protocol: `__len__` (how many examples) and
`__getitem__(i)` (the i-th example). Here, one spectrum's real peaks and its
label.

```python
        self.fp, self.y_row = fp, d["y_row"][idx]
    ...
    def __getitem__(self, i):
        n = int(self.npk[i])
        return (self.mz[i][:n], self.inten[i][:n], self.prec[i],
                int(self.add_id[i]), int(self.ins_id[i]), self.ce[i],
                self.fp[self.y_row[i]])
```

- Returns only the **real** peaks (`[:n]`); padding is decided per batch.
- **Labels by index** (Phase 5 fix): store `y_row` (which molecule) and look up
  its fingerprint on demand. The old version copied a float32 label row per
  spectrum — 5.7 GB — which made training hold 11 GB and blocked validation.
  Many spectra share a molecule, so an index into a shared table is far smaller.

### `class SortishBatches(Sampler)` (lines 53–88)

A **`Sampler`** decides which examples form each batch.

```python
        idx = self.rng.permutation(len(self.lengths))
        for start in range(0, len(idx), self.pool):
            chunk = idx[start : start + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk) - self.bs + 1, self.bs):
                batches.append(chunk[b : b + self.bs].tolist())
        self.rng.shuffle(batches)
        yield from batches
```

Shuffle everything → cut into pools of 50 batches → **sort each pool by peak
count** → slice batches → shuffle batch order. Each batch then holds spectra of
similar length, so it pads to *their* length, not 256. Measured: mean padded
width 97 instead of 256, attention cost ×0.14 (attention is quadratic in
length). `yield from batches` yields each item of a list in turn.

### `collate(batch)` (lines 91–114)

Combines a list of examples into one batch of tensors.

```python
    mzs, its, prec, add, ins, ce, y = zip(*batch)
    width = max(len(m) for m in mzs)
```

- **`zip(*batch)`** transposes: a list of 8 examples of 7 fields becomes 7
  tuples of 8 values each (`*` unpacks the list into separate arguments).
- Pad each spectrum to the **batch's** longest; mark padding in `pad`.
- Labels are cast to float32 **here** — the only place floats are needed.

### `evaluate(model, loader, dev, loss_fn)` (lines 117–138)

`@torch.no_grad()` as a decorator: the whole function runs without gradient
tracking. Reports loss and bit **precision / recall / F1**:

```python
        pred = logits > 0                  # logit > 0  ⇔  probability > 0.5
        tp += (pred & (y > 0.5)).sum().item()
```

- Plain accuracy would be useless: only ~3% of bits are on, so "always 0"
  scores 97%. F1 on the **positive** bits is informative.
- `.item()` turns a one-number tensor into a Python number.

### `main()` — the training loop

1. **Split by structure** (lines 161–170): 5% of **molecules** go to validation,
   so spectra of one molecule never sit on both sides, and an `assert` proves it.
   (Splitting by spectrum would let the model score well by recognising
   molecules it had already seen.)
2. **Checkpoint metadata computed once** (lines 172–185): `d` is a lazy
   `NpzFile` — every `d["keys"]` re-reads and decompresses the array. The first
   version did that inside a loop (quadratic; hours per save; GPU idle).
3. **`DataLoader`** (lines 187–192): feeds batches, using `num_workers=4` extra
   processes to prepare them while the GPU trains. `persistent_workers=True`
   keeps those processes alive between epochs.
4. **Loss with `pos_weight`** (lines 199–204):
   ```python
   pos_rate = fp.mean()                                   # ~0.03
   pos_weight = torch.full((n_bits,), float((1 - pos_rate) / pos_rate), device=dev)
   loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```
   Binary cross-entropy per bit, with mistakes on the rare 1s weighted ~32× so
   they matter as much as the common 0s. `WithLogits` applies the sigmoid
   inside, in a numerically stable way. Validation uses **unweighted** BCE
   (`eval_loss`) so the number is a true log-loss.
5. **Optimiser and schedule** (lines 206–209):
   - **AdamW**: the standard optimiser; `weight_decay=0.01` gently shrinks
     weights to limit overfitting.
   - **OneCycleLR**: the learning rate warms up for the first 10% of steps
     (`pct_start=0.1`), then decays to near zero by the end. Because the
     schedule is tied to the total number of steps, you can't just stop early
     and get the "finished" model — that's why the best checkpoint is saved
     instead of using early stopping.
6. **The step** (lines 216–234):
   ```python
   opt.zero_grad(set_to_none=True)
   with torch.autocast("cuda", dtype=torch.bfloat16):
       loss = loss_fn(model(...), y)
   loss.backward()
   nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   opt.step()
   sched.step()
   ```
   - `zero_grad`: clear the previous step's gradients (PyTorch accumulates them).
   - **`autocast(bfloat16)`**: compute in 16-bit floats where safe — about 2×
     faster on this GPU; bf16 has the same exponent range as float32.
   - `backward()`: compute gradients; `clip_grad_norm_`: cap their size (avoids
     rare huge updates); `opt.step()`: update weights; `sched.step()`: advance
     the learning-rate schedule.
7. **Save the best epoch** by validation loss, with `ck_meta` (bit selection,
   adducts, `max_peaks`, `train_keys`) — the model file carries everything
   inference needs.

## `calibrate_fpnet.py` — the Platt experiment (Phase 4)

Fits per-bit `sigmoid(A·z + B)` on the model's own validation split.

- Reuses `SpectraSet` and `collate` **from `train_fpnet.py`** so calibration
  runs on exactly the training split's held-out side.
- `fit_platt`: all 1,583 tiny logistic regressions solved as one batched
  problem with **LBFGS** (a second-order optimiser).
  `line_search_fn="strong_wolfe"` was essential: the loss is averaged over
  ~70M values, so raw gradients were ~10⁻⁸, and a fixed step size moved
  nothing — the first run silently returned the starting point. The script now
  has two `assert`s: log-loss must go **down**, and the calibrated mean
  probability must land within 30% of the real bit rate (otherwise "the
  optimiser stalled rather than converged").
- Outcome: probabilities 3× better, ranking worse (Part 5 explains the algebra).

## `validate_fpnet.py` — the main holdout (Phases 3–5)

The most-used evaluation script. For each held-out molecule it computes every
channel on the **same candidates**, then scores many blends.

**Flags, each added by a lesson:**

| flag | what it does |
|---|---|
| `-n`, `--seed` | how many molecules, which ones |
| `--source` | restrict to source libraries (natural products) |
| `--model` | which checkpoint |
| `--unseen-only` | only molecules the model never trained on (reads `train_keys` from the checkpoint) |
| `--exclude-seen-by X.pt` | also exclude molecules **another** model saw — so two models sit the same exam |
| `--coconut-only` | only natural-product structures |
| `--class1-frac 0.15` | simulate 15% as class 1 (hide only the query spectra) |
| `--frag`, `--frag-rings` | add the fragmentation channel |
| `--no-platt`, `--platt-a-only` | calibration switches |
| `--dump FILE` | write per-candidate features for the learned ranker |

**Per molecule** (lines 156–272):
1. Candidates from the pool; class 1 or class 2 exclusion
   (`positions_of_rows` vs `spectra_of`).
2. Analog channel — the same steps as `rank.py`.
3. **`bt`** (lines 185–193): the Tanimoto of the **true answer** to its nearest
   cousin, used only to report "hard" vs "easy" molecules. **It must never become
   a feature** — it uses the answer (the dump's comment says so).
4. fpnet channel through `predict_logits` — the shared function, with the
   spectrum's **real** instrument and collision energies.
5. Fragment channel through `score_spectra_frag` (shared).
6. **A sweep of blends** (lines 229–241): one dict of named score arrays, every
   blend evaluated on the same candidates. `f"{w:g}"` formats a number compactly
   (`0.5`, not `0.500000`).
7. Reciprocal rank per blend: `res.setdefault(name, []).append(...)` — create
   the list the first time a name appears.
8. **`--dump`**: one DataFrame row per candidate with raw inputs, features and
   the label (`ck_keys == truth`). `pd.concat` joins them at the end.

The printed table ranks ties in `argsort` order; `compare_dumps.py` (below) uses
the fair `expected_rr` and is the one to trust for close calls.

## `compare_dumps.py` — paired comparisons of whole configurations (Phase 5)

```bash
uv run python scripts/compare_dumps.py /tmp/v2.parquet@fp=1,an=0.5 /tmp/v4c.parquet@ranker=data/ranker_v4c.pt
```

- **`parse(arg)`**: `str.partition("@")` splits at the first `@` into
  (before, "@", after); the part after is blend weights or a ranker path.
- **Same-molecules check**: the analog and fragment channels don't depend on
  the fingerprint model, so they must be identical across dumps — otherwise the
  runs didn't see the same molecules and the comparison is void:
  ```python
  same = (set(d.qid) == set(ref.qid) and len(d) == len(ref)
          and np.allclose(np.sort(d.an_raw), np.sort(ref.an_raw)) ...)
  ```
  Same molecule ids, same row count, same analog values (sorted, so row order
  doesn't matter).
- **`per_molecule_rr`**: rebuilds features with `ranker_features` (the serving
  function) and scores with `expected_rr` (fair ties).
- **Paired difference** (lines ~87–90):
  ```python
  d = (r - base).to_numpy()
  se = d.std(ddof=1) / np.sqrt(len(d))
  ```
  For each molecule, (this config's score − the first config's score). Molecule
  difficulty cancels out, so real differences show through. The **standard
  error** `std / √n` measures uncertainty; ±2 SE is roughly a 95% interval.
  `ddof=1` = the sample standard deviation (divide by n − 1).

This script is what calibrated the natural-product holdout against the
leaderboard (docs/12).

## `prepare_contrastive.py` — candidate lists for fine-tuning (Phase 5)

For ~200k training spectra: the true structure plus up to 63 same-mass
negatives from the pool.

- **Leakage rules** in code: queries only from the base model's `train_keys`;
  candidates only from those or from pool structures **not in the spectral
  library** (an honest-holdout molecule can't even appear as a negative).
- **The truth's position is shuffled** (`rng.shuffle(lst)`): if it were always
  first, the model could learn "pick position 0" instead of chemistry.
- **Truth matched by key, not row** (lines ~113–122): if the pool held two rows
  with the true key, a row-based check would leave the duplicate among the
  negatives — teaching the model that the right answer is wrong.
- **CSR storage** (`ptr`, `cand`): variable-length lists stored as one flat
  array plus offsets. List `i` is `cand[ptr[i]:ptr[i+1]]` — the layout sparse
  matrices use ("compressed sparse row").
- Fingerprints packed per row as they're computed (unpacked, ~1M × 2048 bytes
  would be 2 GB).

## `train_contrastive.py` — the v4c fine-tune (Phase 5)

Teaches the model the real task: pick the true structure among its isomers.

- **`ContrastSet`** wraps `SpectraSet` (reused from `train_fpnet.py`) and adds
  the candidate list and the truth's position.
- **`make_collate(fp_packed, bit_sel)`** returns a *function* (a **closure**:
  an inner function that remembers the variables of the function that made
  it). It unpacks only the ~8k fingerprints needed per batch.
- **`rank_logits`** (line 92):
  ```python
  s = torch.einsum("bln,bn->bl", C, z.float()) / log_tau.exp()
  ```
  **`einsum`** spells out a tensor product with index letters: for each batch
  item `b` and list position `l`, sum over bits `n` of `C[b,l,n] × z[b,n]` —
  i.e. each candidate's `cand_fp · logits`, the same score inference uses.
  Computed in float32 even under bf16: sums over 1,583 bits need the precision.
- **Temperature `tau`** (lines 163–177): divides the scores before the softmax.
  Started at 10 with **its own learning rate** (an optimizer *parameter group*):
  at 1 the softmax was overconfident (loss 6.3, worse than uniform guessing's 4.0).
- **Loss** = listwise cross-entropy (put the truth first) + the original BCE
  (keep bit predictions meaningful).
- **`sanity(tag)`** reports top-1 and MRR on held-out lists (optimistic:
  training structures) **and** BCE on the base model's own validation molecules
  (honest: never seen). The latter improved 0.1965 → 0.1934.
- `s.gather(1, T[:, None])`: pick, for each row, the score at the truth's position.

## `train_ranker.py` — the learned ranker (Phase 5)

- **`rebuild(df)`**: replaces every dumped feature with the output of
  `ranker_features` — train on exactly what will be served. Also asserts the raw
  inputs agree (the diagnostic that caught the float32 z-score bug).
- **`to_tensors(df)`**: molecules have different candidate counts, so they're
  padded to a dense `[Q, Cmax, F]` tensor with a validity mask. Standardisation
  statistics come from **real** candidates only (padding zeros would drag every
  mean toward 0).
- **`train(...)`**: full-batch listwise training,
  `cross_entropy(scores_with_padding_at_-inf, true_position)`.
- **5-fold cross-validation grouped by molecule**: a molecule's candidates are
  never split across training and scoring folds.
  ```python
  fold_of = dict(zip(qids, rng.permutation(len(qids)) % args.folds))
  ```
  A shuffled 0…Q−1 taken modulo 5 gives each **molecule** a fold number (0–4);
  `df.qid.map(fold_of)` then looks up each row's fold. The fixed blend is scored
  on the same held-out molecules for a paired comparison.
- The final model is trained on everything with a configuration (`hidden=32,
  steps=300`) **picked by hand** from the table, not automatically: automatic
  selection on the same folds would quietly overfit the choice.
- The checkpoint stores `fpnet_tag`: the ranker learned one model's scores and
  must never be served with another's.

## `run_np_calibration.sh` — one holdout, three models (Phase 5)

```bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON=(-n 400 --seed 21 --source "$SRC" --coconut-only --unseen-only
        --no-platt --frag --class1-frac 0.15
        --exclude-seen-by "$F/fpnet_v2_nofeat.pt"
        --exclude-seen-by "$F/fpnet.pt"
        --exclude-seen-by "$F/fpnet_v4c.pt")
for m in fpnet_v2_nofeat fpnet fpnet_v4c; do
  uv run python -u scripts/validate_fpnet.py --model "$F/$m.pt" "${COMMON[@]}" \
      --dump "/tmp/np_$m.parquet" > "/tmp/np_$m.log" 2>&1
  grep -E "holdout|molecules in|excluded|restricted" "/tmp/np_$m.log"
done
```

- `cd "$(dirname "$0")/.."`: `$0` is the script's own path, `dirname` strips the
  file name (→ `scripts`), and `/..` goes up one level to the project root, from
  whatever directory you ran it (the bash twin of `parents[1]` in Part 8a).
- `-u` makes Python print immediately (unbuffered), so the log fills as it runs;
  `> file 2>&1` sends normal output **and** errors (stream 2) into the log;
  `grep -E` then shows only the summary lines.

- **`set -euo pipefail`** (bash "strict mode"): stop on any error (`-e`),
  treat unset variables as errors (`-u`), and fail a pipeline if any part fails
  (`pipefail`).
- A **bash array** `COMMON=(...)` holds flags shared by all runs;
  `"${COMMON[@]}"` expands them with spacing preserved.
- Same seed and the **union** of all three models' training sets excluded →
  the three dumps contain identical molecules, which `compare_dumps.py` verifies.
