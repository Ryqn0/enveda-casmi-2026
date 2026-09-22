# Phase 2 plan — a candidate database

## Why this is the next step, not a better ranker

Phase 1 scores **0.845 on class 1 and exactly 0 on classes 2 and 3**. That is
not a tuning failure. Our candidate pool is training *structures*, which exist
only because they have spectra; class 2 is *defined* as "structure known, no
public spectra". Those molecules are not in the pool, so no ranking improvement
can reach them.

The published `haideptry` progression is the evidence for where the points are:

| version | change | class-2 MRR | public LB |
|---|---|---:|---:|
| V10–V11 | library + COCONUT precursor fallback | ~0.132 | 0.163 |
| V16 | analog propagation + MetFrag-lite + GBDT | 0.545 | 0.266 |
| V17 | + neural fingerprint | 0.612 | 0.300–0.336 |

Class-2 MRR went 0.132 → 0.612 and the leaderboard roughly doubled.

## The shape of the change

1. **Candidate pool** — every structure whose neutral mass matches the query
   within a few ppm, drawn from COCONUT (natural products) plus training
   structures. Typical published pool size is ~712k structures, narrowing to a
   median of ~56 candidates per molecule after the mass filter.
2. **Scoring without a reference spectrum** — a class-2 candidate has no library
   spectrum to compare against, so similarity is useless. The evidence has to
   come from the structure itself:
   - **in-silico fragmentation** (MetFrag-style): break 1–2 bonds, check how
     much of the observed peak intensity the fragments explain;
   - **predicted fingerprint**: a model maps spectrum → substructure bit vector,
     ranked against each candidate's true fingerprint (the CSI:FingerID idea);
   - **analog similarity**: a mass-shifted match to a known relative, weighted
     by chemical similarity.
3. **Fusion** — a gradient-boosted ranker over those signals plus the Phase 1
   library score.

## Offline data is allowed, and already exists

Internet is disabled at scoring time, so any database must be attached as a
Kaggle dataset. The rules permit "freely & publicly available external data,
including pre-trained models", and the community has published several.
Candidates found 2026-09-22 (titles are other people's data — verify contents
before trusting any of them):

| ref | size | note |
|---|---|---|
| `aidensong123/casmi26-coconut-202609` | 44 MB | COCONUT, Sept 2026 |
| `kiza123123/coconut-20-casmi2026-index` | 26 MB | COCONUT 2.0 index |
| `dariushafshar/casmi26-coconut-scorer-keys-rdkit-2026-03-3` | 115 MB | keys canonicalized with the pinned RDKit |
| `kitopl/casmi-2026-candidate-pool-fingerprints` | 66 MB | pool + precomputed fingerprints |
| `thedevastator/open-source-natural-product-annotations` | 48 MB | original COCONUT dump, 2023 |

The RDKit-2026.03.3 variant matters: candidate keys must be canonicalized with
the same version the host scores with, or matches are missed. We already pin
that version locally.

Building our own pool from a COCONUT dump is the safer route — a third-party
dataset of "scorer keys" is exactly the kind of thing that can be silently
wrong, and we would not find out until the leaderboard disagreed with local
validation.

## Validation must change too

`scripts/validate.py` currently has no class-2 mode, because there was no pool
to retrieve from. Once one exists, add a third mask: hide every spectrum of the
held-out structure **but keep the structure in the candidate pool**. That is
exactly class 2, and it is the number Phase 2 must move.
