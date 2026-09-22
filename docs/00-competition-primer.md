# CASMI 2026 — the competition, explained from scratch

> Written for: a developer who knows ML but not mass spectrometry.

## 1. The scientific problem

Nature is full of small molecules — plants, microbes, and your own blood contain
thousands of them. A mass spectrometer can *detect* them in bulk, but it cannot
*name* them. Roughly **90% of the signals in a typical LC-MS/MS run are "dark
matter"**: real molecules whose chemical structure nobody has determined.

Enveda is a drug-discovery company that mines those unidentified molecules. If you
can turn a spectrum into a structure, you unlock a huge unexplored chemical space.

## 2. What the instrument actually does

**LC** — *Liquid Chromatography*. A mixture is pushed through a column so the
molecules come out separated in time. This is just the "unmixing" step.

**MS1** — the molecule is ionized (usually by gaining or losing a proton) and
weighed. You get the **precursor m/z**: mass-to-charge ratio of the intact
molecule plus its adduct. From this you can often infer the **molecular formula**.

**MS/MS (MS2)** — the machine isolates that one precursor, smashes it with an
inert gas (*collision-induced dissociation*), and weighs all the pieces. The
result is the **MS2 spectrum**: a list of `(m/z, intensity)` peaks, one per
fragment.

```
intensity
   |          .
   |     .    |         .
   |  .  |    |    .    |      .
   +--+--+----+----+----+------+---> m/z
     91 105  152  179  213   301 (precursor)
```

So a spectrum is a **sparse, unordered set of ~10-500 (mass, intensity) pairs** —
a point cloud, not an image and not a sequence.

## 3. Why this is hard

Fragmentation is a **many-to-one, lossy, non-invertible** process:

- Two different molecules can fragment into near-identical peak sets (isomers are
  the classic trap — same formula, different connectivity).
- The same molecule gives *different* spectra on different instruments, at
  different collision energies, and with different adducts.
- Peaks tell you the *mass* of a fragment, not which atoms it contained. Mass
  `91.0542` could be tropylium (C7H7+) — or something else entirely.
- There is no closed-form inverse. You cannot run the physics backwards.

This is why the field is still 90% unsolved after 40 years.

## 4. The competition task

**Input:** one or more MS2 spectra belonging to the same unknown molecule
(1-16 spectra each, collected at different collision energies), plus the
precursor m/z and adduct.

**Output:** a **SMILES string** — the 2D chemical structure.
*2D only*: connectivity matters, stereochemistry (wedges, E/Z, R/S) does not.

You submit up to **25 ranked candidates** per molecule.

## 5. The metric — MRR@25 on InChIKey14

Three ideas stacked:

**(a) InChIKey14.** An InChIKey looks like
`RYYVLZVUVIJVGH-UHFFFAOYSA-N`. The first 14 characters are the **connectivity
block** — a hash of the 2D skeleton only. Comparing just that block is exactly
"did you get the 2D structure right, ignoring stereo and charge state".

**(b) Tautomer canonicalization.** Some molecules shuffle a hydrogen between two
positions (keto to enol). These are the same compound in practice but hash
differently. RDKit normalizes them to one canonical form first, so you are not
punished for picking the "wrong" tautomer.

**(c) MRR@25** — *Mean Reciprocal Rank*. If the true structure is at rank `r` in
your 25 candidates you score `1/r`; if it is absent you score `0`. Average over
all test molecules.

| rank of correct answer | score |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| 10 | 0.100 |
| 25 | 0.040 |
| not in list | 0.000 |

**The strategic consequence:** getting the right answer *anywhere* in the top 25
is worth a lot more than nothing, and moving it from rank 5 to rank 1 is worth
0.8 points. So the job splits cleanly into two halves:

1. **Retrieval** — get the true structure into the candidate list at all.
2. **Ranking** — push it toward position 1.

Never submit fewer than 25 candidates. Empty slots are free lottery tickets.

## 6. The data

| Split | Size | Notes |
|---|---|---|
| Train | ~2.5M spectra / ~275k unique structures | public reference libraries |
| Test | ~400 molecules / ~1,500-2,500 spectra | **unpublished**, measured on a timsTOF instrument |

The test molecules are deliberately *not* all in public databases. That splits the
test set into three difficulty classes:

- **Class 1 — in the library.** The exact structure appears in training data.
  Solvable by spectral library search. This is the "free" part of the score.
- **Class 2 — analog.** A close relative is known (same scaffold, one extra
  methyl). Solvable by analog search plus editing.
- **Class 3 — de novo.** Nothing like it is known. Requires actually predicting
  structure from physics or learned priors. This is the open research problem.

## 7. Approach families

**A. Spectral library search** (the baseline)
Compare the query spectrum to every training spectrum with a similarity function
(modified cosine, spectral entropy), return the structures of the best matches.
Cheap, strong on class 1, scores exactly 0 on class 3.

**B. Candidate retrieval + learned ranking** (what wins CASMI historically)

1. From the precursor mass, enumerate a **molecular formula**.
2. Pull every structure with that formula from a database (PubChem, COCONUT, the
   training set) — this is the candidate pool, often 10k+ molecules.
3. Score each candidate against the spectrum with a learned model, and rank.

The classic scorer is **CSI:FingerID** style: a model predicts a **molecular
fingerprint** (a ~4000-bit vector of "does this molecule contain substructure
X?") *from the spectrum*, then you rank candidates by similarity between the
predicted fingerprint and each candidate's true fingerprint. `MIST` is the modern
transformer version. This handles class 2 well because fingerprints degrade
gracefully.

**C. De novo generation**
Sequence models that emit SMILES directly from peaks (MassGenie, Spec2Mol), or
diffusion over molecular graphs (DiffMS). Currently weaker than (B) but the only
route to class 3.

**D. Forward simulation**
Predict the spectrum *from* a candidate structure (ICEBERG, CFM-ID, GRAFF-MS) and
rank by agreement with the observed spectrum. The forward direction is much
better-posed than the inverse.

## 8. Our roadmap

| Phase | Goal | Deliverable |
|---|---|---|
| 0 | Understand data, build the metric offline | `src/casmi/metric.py` + EDA notebook |
| 1 | Library-search baseline | first leaderboard score |
| 2 | Candidate pool + formula enumeration | retrieval recall@25 measured |
| 3 | Fingerprint model (MIST-style) trained on Kaggle GPU | learned ranking |
| 4 | Ensemble / rank fusion | leaderboard push |
| 5 | Deploy: upload a spectrum, get ranked structures | Hugging Face Space |

Every phase is a commit and a reproducible script. The deployed app is the
demo layer on top of whatever model is current.

## 9. Vocabulary cheat-sheet

| Term | Meaning |
|---|---|
| **m/z** | mass-to-charge ratio; with charge 1 it is just the mass |
| **precursor** | the intact ionized molecule before fragmentation |
| **adduct** | what the molecule picked up to become an ion, e.g. `[M+H]+`, `[M+Na]+` |
| **neutral loss** | precursor m/z minus a fragment m/z — often more informative than the fragment itself |
| **collision energy (CE)** | how hard the molecule was smashed; low CE = few big pieces, high CE = many small ones |
| **SMILES** | text encoding of a molecular graph, e.g. caffeine = `CN1C=NC2=C1C(=O)N(C)C(=O)N2C` |
| **InChIKey** | 27-char hash of a structure; first 14 chars = 2D connectivity |
| **fingerprint** | fixed-length bit vector of substructure presence |
| **Tanimoto** | Jaccard similarity between two fingerprints; the standard chemical similarity |
| **timsTOF** | the Bruker instrument family used for the test set |
| **COCONUT** | open database of ~700k known natural products — a good candidate pool |

## 10. Sources

- Competition: <https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>
- Timeline: opened 2026-09-14, closes 2026-12-14. $50,000 across the top 5 teams.
