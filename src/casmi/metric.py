"""Offline copy of the CASMI 2026 metric: MRR@25 on tautomer-canonical InChIKey14.

Two molecules count as the same answer when the first 14 characters of their
InChIKey agree after RDKit tautomer canonicalization. That block encodes 2D
connectivity only, so stereochemistry and protonation state are ignored by
construction.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

K = 25

_TAUTOMER = rdMolStandardize.TautomerEnumerator()
# ponytail: default 1000-transform cap; raise if canonicalization silently bails
# on large natural products (check GetStatus() if scores look off).


@lru_cache(maxsize=1_000_000)
def inchikey14(smiles: str) -> str | None:
    """Canonical 2D key for a SMILES, or None if RDKit cannot parse it."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _TAUTOMER.Canonicalize(mol)
    except Exception:
        pass  # keep the untransformed molecule rather than dropping the candidate
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def reciprocal_rank(predictions: list[str], truth: str, k: int = K) -> float:
    """1/rank of the first prediction matching `truth`, else 0.

    `predictions` are SMILES in descending confidence. Only the first `k` count.
    Unparseable predictions occupy their slot (they cannot match), matching how
    a real leaderboard treats junk output.
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
    caffeine = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
    caffeine_alt = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"  # same molecule, aromatic form
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    glucose = "OCC1OC(O)C(O)C(O)C1O"

    assert inchikey14(caffeine) == inchikey14(caffeine_alt)
    assert inchikey14(caffeine) != inchikey14(aspirin)
    assert inchikey14("not a molecule") is None

    # stereochemistry must be ignored: same skeleton, different wedges
    assert inchikey14("C[C@H](N)C(=O)O") == inchikey14("C[C@@H](N)C(=O)O")

    # tautomers must collapse: 2-pyridone / 2-hydroxypyridine
    assert inchikey14("O=c1cccc[nH]1") == inchikey14("Oc1ccccn1")

    assert reciprocal_rank([caffeine], caffeine) == 1.0
    assert reciprocal_rank([aspirin, caffeine_alt], caffeine) == 0.5
    assert reciprocal_rank([aspirin, glucose], caffeine) == 0.0
    assert reciprocal_rank(["@@bad@@", caffeine], caffeine) == 0.5  # junk holds its slot
    assert reciprocal_rank([aspirin] * 24 + [caffeine], caffeine) == 1 / 25
    assert reciprocal_rank([aspirin] * 25 + [caffeine], caffeine) == 0.0  # past k

    score = mrr(
        {"m1": [caffeine], "m2": [caffeine, aspirin]},
        {"m1": caffeine, "m2": aspirin},
    )
    assert abs(score - 0.75) < 1e-9, score

    assert mrr({}, {"m1": caffeine}) == 0.0  # nothing submitted

    print("metric self-check passed")


if __name__ == "__main__":
    _self_check()
