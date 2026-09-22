"""Candidate pool: every structure a test molecule might plausibly be.

WHY THIS EXISTS
    Phase 1 could only ever answer with a structure that had a training
    spectrum. Competition class 2 is defined as "structure is known, but no
    public spectra exist", so those molecules were unreachable — not badly
    ranked, absent. The leaderboard confirmed it: we scored 0.130, which implies
    class 1 is only ~15% of the test set.

    A candidate pool fixes the retrieval half of that. We take every structure
    from COCONUT (natural products) plus the training set, and for a query we
    keep the ones whose neutral mass matches the measured precursor. Those are
    the molecules it *could* be. Ranking them is a separate problem.

HOW A MASS BECOMES A FILTER
    The instrument measures the ion, not the molecule: an [M+H]+ ion weighs the
    molecule plus one proton. Subtract the adduct's contribution and you have
    the neutral mass, which is a property of the structure alone and can be
    compared against a database.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Monoisotopic masses: the mass of each element's MOST ABUNDANT isotope, not the
# periodic-table average. A mass spectrometer resolves individual isotopes, so
# using the average weight (e.g. Cl = 35.45) would be wrong by a third of a Dalton.
MONOISOTOPIC = {
    "C": 12.000000, "H": 1.0078250319, "N": 14.0030740052, "O": 15.9949146221,
    "S": 31.97207069, "P": 30.97376151, "F": 18.9984032, "Cl": 34.96885271,
    "Br": 78.9183376, "I": 126.904468, "Si": 27.9769265327, "Se": 79.9165196,
    "B": 11.0093055, "Na": 22.98976928, "K": 38.96370668, "Li": 7.0160040,
    "Mg": 23.9850419, "Ca": 39.9625912, "Fe": 55.9349421, "Zn": 63.9291466,
    "Cu": 62.9295989, "Mn": 54.9380471, "Co": 58.9331976, "As": 74.9215942,
    "D": 2.0141017778,
}

ELECTRON = 0.00054858

# What each adduct ADDS to the neutral monoisotopic mass to give the measured
# m/z. All are singly charged, so m/z equals mass.
#
# The electron correction matters at the accuracy we need: a proton is a
# hydrogen ATOM minus its electron (1.007825 - 0.000549 = 1.007276). Ignoring
# that is a 1.5 ppm error at m/z 350 — comparable to the tolerance itself.
_H = MONOISOTOPIC["H"]
_H2O = 2 * _H + MONOISOTOPIC["O"]
ADDUCT_MASS = {
    "[M+H]+":        _H - ELECTRON,
    "[M+NH4]+":      MONOISOTOPIC["N"] + 4 * _H - ELECTRON,
    "[M-H2O+H]+":    _H - ELECTRON - _H2O,
    "[M-2H2O+H]+":   _H - ELECTRON - 2 * _H2O,
    "[M+Na]+":       MONOISOTOPIC["Na"] - ELECTRON,
    "[M+K]+":        MONOISOTOPIC["K"] - ELECTRON,
    "[M-H]-":        -_H + ELECTRON,
    "[M-H2O-H]-":    -_H + ELECTRON - _H2O,
    "[M+CH2O2-H]-":  MONOISOTOPIC["C"] + 2 * _H + 2 * MONOISOTOPIC["O"] - _H + ELECTRON,
    "[M+Cl]-":       MONOISOTOPIC["Cl"] + ELECTRON,
}

# Matches one element symbol plus an optional count: "C16" -> ("C", "16"),
# "S" -> ("S", ""). [A-Z][a-z]? is an uppercase letter optionally followed by a
# lowercase one, which is exactly how element symbols are written.
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float | None:
    """Monoisotopic mass of a molecular formula, or None if it cannot be read.

    Returns None for formulas containing an unknown element, rather than
    silently treating it as weightless — a wrong mass would quietly exclude the
    right candidate from every future search.
    """
    if not isinstance(formula, str) or not formula:
        return None
    # Charge markers and hydrates would corrupt a neutral mass, so reject them.
    if any(c in formula for c in "+-."):
        return None

    total = 0.0
    consumed = 0
    for sym, count in _FORMULA_TOKEN.findall(formula):
        if sym not in MONOISOTOPIC:
            return None
        total += MONOISOTOPIC[sym] * (int(count) if count else 1)
        consumed += len(sym) + len(count)

    # findall skips anything it cannot match, so compare the consumed length
    # against the input to catch stray characters instead of ignoring them.
    if consumed != len(formula):
        return None
    return total


def build_pool(coconut_csv: Path, train_parquet: Path, out: Path) -> pd.DataFrame:
    """Merge COCONUT and the training structures into one mass-indexed pool."""
    # --- COCONUT: the natural-product database ---------------------------
    # Only the four columns we need; the file is 577 MB and has 40 columns.
    coco = pd.read_csv(
        coconut_csv,
        usecols=["canonical_smiles", "standard_inchi_key", "molecular_formula",
                 "exact_molecular_weight"],
    )
    coco = coco.rename(columns={"canonical_smiles": "smiles"})
    coco["inchikey14"] = coco.standard_inchi_key.str.slice(0, 14)
    coco["source"] = "coconut"

    # --- training structures ---------------------------------------------
    # Already in the competition data, and they cover class 1. Included so one
    # pool serves every class.
    tr = pd.read_parquet(
        train_parquet, columns=["inchikey14", "normalized_smiles", "molecular_formula"]
    )
    tr = tr.rename(columns={"normalized_smiles": "smiles"})
    tr = tr.drop_duplicates(subset="inchikey14")
    tr["source"] = "train"
    tr["exact_molecular_weight"] = np.nan

    pool = pd.concat(
        [coco[["inchikey14", "smiles", "molecular_formula", "exact_molecular_weight",
               "source"]], tr],
        ignore_index=True,
    )

    # Drop rows with no usable key or structure before doing expensive work.
    pool = pool[pool.inchikey14.notna() & pool.smiles.notna()]

    # Prefer training entries on collision: their SMILES are RDKit-standardised
    # the same way the competition labels are. keep="last" works because the
    # training rows were concatenated after COCONUT's.
    pool = pool.drop_duplicates(subset="inchikey14", keep="last")

    # Mass from formula, which is exact and fast. Parsing a million SMILES with
    # RDKit would take half an hour and give the same answer.
    mass = pool.molecular_formula.map(formula_mass)
    pool["exact_mass"] = mass
    pool = pool[pool.exact_mass.notna()]

    # Sorting by mass is what lets a query use binary search instead of scanning
    # the whole pool. reset_index makes positions contiguous afterwards.
    pool = pool.sort_values("exact_mass").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    pool[["inchikey14", "smiles", "molecular_formula", "exact_mass", "source"]].to_parquet(
        out, index=False
    )
    return pool


class CandidatePool:
    """Mass-sorted structures, queried by precursor m/z and adduct."""

    def __init__(self, path: Path):
        self.df = pd.read_parquet(path)
        # Hot arrays pulled out of pandas once; indexing numpy is far cheaper.
        self.mass = self.df.exact_mass.to_numpy()
        self.smiles = self.df.smiles.to_numpy()
        self.key = self.df.inchikey14.to_numpy()
        assert np.all(np.diff(self.mass) >= 0), "pool must be sorted by mass"

    def __len__(self) -> int:
        return len(self.df)

    def neutral_mass(
        self, precursor_mz: float, adduct: str, offset_ppm: float = 0.0
    ) -> float | None:
        """Undo the adduct: measured ion m/z -> mass of the neutral molecule.

        `offset_ppm` removes a known systematic calibration bias. Measured from
        train.parquet's `precursor_error_ppm`, every source library reads HIGH
        against the mass implied by its own labelled structure:

            enveda-180 (timsTOF)  +0.98 ppm   <- the test instrument
            gnps       (Orbitrap) +1.55 ppm
            massbank   (QTOF)     +1.87 ppm
            riken      (QTOF)     +3.29 ppm

        This is not noise, it is calibration. Correcting it lets the search
        window be CENTRED on the truth rather than straddling it, so a narrower
        window keeps the same recall — and a narrower window means fewer
        candidates competing for 25 slots.
        """
        add = ADDUCT_MASS.get(adduct)
        if add is None:
            return None
        m = precursor_mz - add
        # The measurement is high by offset_ppm, so divide it back down.
        return m / (1.0 + offset_ppm * 1e-6)

    def query(
        self,
        precursor_mz: float,
        adduct: str,
        ppm: float = 10.0,
        offset_ppm: float = 0.0,
    ) -> np.ndarray:
        """Pool positions whose mass matches this measurement within `ppm`.

        Tolerance is in parts-per-million, not Daltons, because instrument mass
        error scales with mass: 5 ppm is 0.0017 Da at m/z 350 but 0.005 Da at
        m/z 1000.
        """
        m = self.neutral_mass(precursor_mz, adduct, offset_ppm)
        if m is None or m <= 0:
            return np.empty(0, dtype=np.int64)
        tol = m * ppm * 1e-6
        # Two binary searches on the sorted mass array bound the window.
        lo, hi = np.searchsorted(self.mass, [m - tol, m + tol])
        return np.arange(lo, hi, dtype=np.int64)


def _self_check() -> None:
    # Caffeine C8H10N4O2, monoisotopic 194.08038
    m = formula_mass("C8H10N4O2")
    assert abs(m - 194.080376) < 1e-4, m
    # Glucose C6H12O6, 180.06339
    assert abs(formula_mass("C6H12O6") - 180.063388) < 1e-4
    # Multi-letter element symbols must not be read as two elements.
    assert abs(formula_mass("CCl4") - (12.0 + 4 * 34.96885271)) < 1e-6
    # Unknown or malformed input must be rejected, never silently zero-weighted.
    assert formula_mass("C6H12Xx2") is None
    assert formula_mass("C6H12O6+") is None   # charged
    assert formula_mass("C6H12O6.H2O") is None  # hydrate
    assert formula_mass("") is None

    # An [M+H]+ ion of caffeine is measured at 195.0877.
    off = ADDUCT_MASS["[M+H]+"]
    assert abs((194.080376 + off) - 195.087652) < 1e-4, 194.080376 + off
    # [M-H]- of glucose is 179.0561.
    assert abs((180.063388 + ADDUCT_MASS["[M-H]-"]) - 179.056112) < 1e-4

    print("pool self-check passed")


if __name__ == "__main__":
    _self_check()
