"""CASMI 2026 metric: MRR@25 on tautomer-canonical InChIKey14.

Canonicalization is delegated to `casmi.official_metric`, vendored verbatim from
the host's scoring notebook, so our keys cannot drift from the leaderboard's.
What this module adds is *per-molecule* reciprocal rank: the host scorer returns
only the mean, and a mean alone hides which molecules failed and why.

`_self_check` asserts our mean equals the host's `score()` on a fixture, so the
two can never silently diverge.
"""

from __future__ import annotations

from functools import lru_cache

from casmi import official_metric as _official

# Binds RDKit into the vendored module. Cheap and idempotent; the host calls it
# at the top of score() for the same reason.
_official.install_dependencies()

K = 25


@lru_cache(maxsize=1_000_000)
def inchikey14(smiles: str) -> str | None:
    """Canonical 2D key for a SMILES, or None if RDKit cannot parse it.

    Cached because validation canonicalizes the same library SMILES thousands of
    times; the host caches within a single score() call for the same reason.
    """
    return _official.canonicalize_smiles_to_ik14(smiles)


def reciprocal_rank(predictions: list[str], truth: str, k: int = K) -> float:
    """1/rank of the first prediction matching `truth`, else 0.

    `predictions` are SMILES in descending confidence; only the first `k` count.
    Unparseable predictions still occupy their slot, matching the host, which
    treats a canonicalization failure as a non-match rather than skipping it.
    """
    target = inchikey14(truth)
    if target is None:
        raise ValueError(f"unparseable ground-truth SMILES: {truth!r}")
    for rank, smi in enumerate(predictions[:k], start=1):
        if inchikey14(smi) == target:
            return 1.0 / rank
    return 0.0


def mrr(predictions: dict[str, list[str]], truths: dict[str, str], k: int = K) -> float:
    """Mean reciprocal rank over molecules. Missing molecule_ids score 0."""
    if not truths:
        raise ValueError("empty ground truth")
    return sum(
        reciprocal_rank(predictions.get(mid, []), smi, k) for mid, smi in truths.items()
    ) / len(truths)


def parse_submission(path) -> dict[str, list[str]]:
    """Read a submission.csv (`molecule_id,smiles` with `;`-joined candidates)."""
    import pandas as pd

    df = pd.read_csv(path)
    return {
        row.molecule_id: [s for s in str(row.smiles).split(";") if s]
        for row in df.itertuples()
    }


def _self_check() -> None:
    import pandas as pd

    caffeine = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    caffeine_alt = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"  # same molecule, aromatic form
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    glucose = "OCC1OC(O)C(O)C(O)C1O"

    assert inchikey14(caffeine) == inchikey14(caffeine_alt)
    assert inchikey14(caffeine) != inchikey14(aspirin)
    assert inchikey14("not a molecule") is None

    # stereochemistry ignored: same skeleton, different wedges
    assert inchikey14("C[C@H](N)C(=O)O") == inchikey14("C[C@@H](N)C(=O)O")
    # tautomers collapse: 2-pyridone / 2-hydroxypyridine
    assert inchikey14("O=c1cccc[nH]1") == inchikey14("Oc1ccccn1")

    assert reciprocal_rank([caffeine], caffeine) == 1.0
    assert reciprocal_rank([aspirin, caffeine_alt], caffeine) == 0.5
    assert reciprocal_rank([aspirin, glucose], caffeine) == 0.0
    assert reciprocal_rank(["@@bad@@", caffeine], caffeine) == 0.5  # junk holds its slot
    assert reciprocal_rank([aspirin] * 24 + [caffeine], caffeine) == 1 / 25
    assert reciprocal_rank([aspirin] * 25 + [caffeine], caffeine) == 0.0  # past k
    assert mrr({}, {"m1": caffeine}) == 0.0  # nothing submitted

    # The contract that matters: our mean must equal the host's score().
    truths = {"m1": caffeine, "m2": aspirin, "m3": glucose}
    preds = {
        "m1": [aspirin, caffeine_alt],          # rank 2 -> 0.5
        "m2": [aspirin],                        # rank 1 -> 1.0
        "m3": [caffeine, aspirin, "@@bad@@"],   # absent -> 0.0
    }
    ours = mrr(preds, truths)
    theirs = _official.score(
        pd.DataFrame({"molecule_id": list(truths), "smiles": list(truths.values())}),
        pd.DataFrame(
            {
                "molecule_id": list(preds),
                "smiles": [";".join(v) for v in preds.values()],
            }
        ),
        "molecule_id",
        max_rank=K,
    )
    assert abs(ours - theirs) < 1e-12, f"diverged from host: {ours} vs {theirs}"
    assert abs(ours - 0.5) < 1e-12, ours

    print(f"metric self-check passed (agrees with host score(): {ours:.6f})")


if __name__ == "__main__":
    _self_check()
