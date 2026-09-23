"""Is our test-like chemistry actually in PubChem? Check before downloading it.

    uv run python scripts/pubchem_coverage.py -n 200

Samples molecules measured on timsTOF (the test instrument) from the spectral
library and asks PubChem, one at a time, whether a structure with the same
CONNECTIVITY exists -- the same notion of "same molecule" the metric uses
(InChIKey first block: atoms and bonds, stereochemistry ignored).

If coverage is high, adding PubChem to the candidate pool is worth the
multi-GB download. If it is low, that plan is wrong and this cost two minutes.
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastidentity/smiles/"
       "cids/TXT?identity_type=same_connectivity")


def in_pubchem(smiles: str) -> bool | None:
    """True/False, or None if the request itself failed (not counted)."""
    body = urllib.parse.urlencode({"smiles": smiles}).encode()
    try:
        with urllib.request.urlopen(URL, data=body, timeout=30) as r:
            return any(line.strip() not in ("", "0")
                       for line in r.read().decode().splitlines())
    except urllib.error.HTTPError as e:
        # PubChem answers 404 "No CID found" when the structure is absent.
        return False if e.code == 404 else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_parquet(ROOT / "data/index/meta.parquet",
                           columns=["inchikey14", "normalized_smiles", "instrument_type"])
    tims = (meta[meta.instrument_type.eq("timsTOF")]
            .drop_duplicates("inchikey14")
            .sample(args.n, random_state=args.seed))

    hits = misses = errors = 0
    for i, smi in enumerate(tims.normalized_smiles, 1):
        r = in_pubchem(smi)
        if r is None:
            errors += 1
        elif r:
            hits += 1
        else:
            misses += 1
        time.sleep(0.25)   # PubChem allows 5 requests/second; stay under it
        if i % 50 == 0:
            print(f"  {i}/{args.n}: {hits} found, {misses} absent, {errors} errors",
                  flush=True)

    n = hits + misses
    print(f"\ntimsTOF library molecules found in PubChem: {hits}/{n} = {hits / n:.1%}"
          f"  ({errors} request errors excluded)")


if __name__ == "__main__":
    main()
