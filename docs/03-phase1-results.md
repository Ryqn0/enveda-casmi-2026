# Phase 1 results — spectral library baseline

**Date:** 2026-09-22 · **Reproduce:** `scripts/build_library.py` then `scripts/validate.py -n 250`

## Setup

Flash Entropy index over **2,115,727 training spectra / 268,932 structures**,
filtered to the documented test spec (10 adducts, precursor m/z 130–1220).
Build: 367 s, 10.5 GB peak RSS, 2.3 GB on disk.

Validation holds out 250 timsTOF structures with ≥3 spectra, queries with their
**raw** peaks, and hides either the query measurements (`known`) or every
spectrum of the structure (`novel`).

## Results

| mode | simulates | MRR@25 | top-1 | found in 25 |
|---|---|---:|---:|---:|
| `known` | class 1 — public spectra exist | **0.8446** | 77.2% | 96.4% |
| `novel` | class 3 — structure absent | **0.0000** | 0.0% | 0.0% |

By how much of the structure remains in the library after hiding the queries:

| spectra left | MRR@25 | n |
|---|---:|---:|
| 1 | 0.834 | 171 |
| 2 | 0.769 | 25 |
| 3 | 0.851 | 18 |
| 4+ | 0.944 | 36 |

**A single reference spectrum is almost as good as four.** MRR only climbs from
0.83 to 0.94 across a 4× difference in library depth. Class-1 performance is
therefore driven by *whether* a structure has any public spectrum, not by how
well-studied it is — so there is little to gain from weighting well-represented
structures, and coverage is what matters.

`novel` is exactly 0.0000 at every depth, which is the masking proving itself:
identity search cannot retrieve a structure that is not in the index, and any
leak would show up as a nonzero score here.

## What this implies for the leaderboard

Our candidate pool is training *structures*, which exist only because they have
spectra. Class 2 is defined as "structure known, **no public spectra**" — so
class-2 molecules are not in our pool at all, and class 3 even less so. Expected
leaderboard score is therefore roughly:

    LB  ≈  P(class 1) × 0.845

The class distribution is hidden, but it can be estimated from a published run.
The `haideptry` notebook reports Class-1 MRR 1.000 / Class-2 0.132 at a public
LB of 0.163, which is only consistent with **class 1 being ~12–16% of the test
set**. That puts this baseline at roughly **0.10–0.15** — against a current
leaderboard top of 0.415.

That is the honest floor for a library-only approach, and it is not a
disappointment: it quantifies exactly how much of the competition is *not*
solvable by lookup.

## Decisions taken from measurement

**Tautomer-aware dedupe is not worth shipping.** Candidates are already distinct
by the dataset's `inchikey14`; running the metric's canonicalization on top
frees a mean of **0.09 slots** (median 0, max 1, affecting 9 of 100 molecules).
RDKit is not in Kaggle's base image, so this would have cost a second wheel and
an install step for nothing. Left out.

**No mass-proximity fallback for zero-score molecules.** It would rank 25
training structures that, for a class-2 molecule, provably cannot contain the
answer. Effort with no expected value.

## Next

Phase 2 is a **candidate database**, not a better ranker. COCONUT plus PubChem
formula lookups put class-2 structures in the pool for the first time; only then
does ranking work have anything to rank. The `haideptry` progression — class-2
MRR 0.132 → 0.612 moving the LB 0.163 → 0.336 — is the evidence that this is
where the points are.
