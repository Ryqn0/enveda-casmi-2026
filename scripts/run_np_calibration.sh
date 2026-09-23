#!/usr/bin/env bash
# Dump the SAME natural-product holdout for three fpnet versions, so every
# submitted configuration can be re-scored offline and checked against its
# leaderboard score (docs/12).
set -euo pipefail
cd "$(dirname "$0")/.."

F=data/fpnet
SRC=gnps,massbank,mona,riken,msdial,pluskal_ms2,spectraverse,drug_plus,masaryk,enveda-np-examples
COMMON=(-n 400 --seed 21 --source "$SRC" --coconut-only --unseen-only
        --no-platt --frag --class1-frac 0.15
        --exclude-seen-by "$F/fpnet_v2_nofeat.pt"
        --exclude-seen-by "$F/fpnet.pt"
        --exclude-seen-by "$F/fpnet_v4c.pt")

for m in fpnet_v2_nofeat fpnet fpnet_v4c; do
  echo "=== $m"
  uv run python -u scripts/validate_fpnet.py --model "$F/$m.pt" "${COMMON[@]}" \
      --dump "/tmp/np_$m.parquet" > "/tmp/np_$m.log" 2>&1
  grep -E "holdout|molecules in|excluded|restricted" "/tmp/np_$m.log"
done
echo ALLDONE
