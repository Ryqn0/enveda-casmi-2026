# The pool gap — why offline 0.92 and leaderboard ~0.24 disagree

> **WITHDRAWN -- see docs/12.** The 0.2% COCONUT coverage below described enveda-180,
> not the test. Test-like natural products (enveda-np-examples) are 99.6% in COCONUT,
> so the existing pool is the right database and PubChem was not added.

## The circular recall check

Phase 2 reported "100% recall at 5 ppm" for the candidate pool
(`scripts/pool_recall.py`). It sampled TRAINING structures -- and the pool is
COCONUT **plus every training structure**. So the check could only ever say
100%. It proved the mass arithmetic works; it said nothing about whether a real
class-2 answer is in the pool.

Same shape of mistake as docs/09: the test item was guaranteed present in the
thing being tested.

## What the pool actually covers

Class 2 is "structure known to PubChem or COCONUT, no public spectrum". Our
pool has COCONUT, not PubChem. How much of the chemistry we care about is in
COCONUT?

| molecules with spectra in the library | also in COCONUT |
|---|---:|
| all instruments (268,932) | 9.2% |
| **timsTOF -- the test instrument (179,943)** | **0.2%** |

The timsTOF data is not natural products. It is drug-like screening chemistry
(bromophenyls, amides, piperazines -- see any dry-run output), which lives in
PubChem. If test class 2 resembles training timsTOF, **almost none of its
answers are in our pool**, and no amount of ranking can score them.

## What that explains

- Offline MRR ~0.92 vs LB ~0.24. Every offline holdout draws molecules that
  are in the pool by construction; the leaderboard does not.
- Why ranking improvements move the LB so little: they only act on the
  minority of test molecules whose answer is retrievable.

## Rough bound

LB 0.244 ~= 0.154 (class 1 share) x ~0.9 + contribution from classes 2-3. That
leaves ~0.105 from everything else -- consistent with most class-2 answers
being absent from the pool, not merely mis-ranked.

## Implication

The next lever is **retrieval, not ranking**: add PubChem (or the relevant
part of it) to the pool. Costs: a multi-GB download and Kaggle dataset,
thousands of candidates per mass instead of ~60 (so a two-stage ranker: cheap
channels over all, fragmentation + learned ranker over the top few hundred),
and retraining the ranker on realistic list sizes. The upside is that the
offline holdout would finally measure the task the leaderboard grades.
