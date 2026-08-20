#!/usr/bin/env bash
set -u
cd /home/mila/t/tanga/CODE/irt-contamination-detection-2
export TQDM_DISABLE=1
CK=outputs/adapter_seed0_e6
# Priority order: epochs 1, 3, 6 first, then the rest.
for pair in "epoch1:176" "epoch3:528" "epoch6:1056" "epoch2:352" "epoch4:704" "epoch5:880"; do
  tag="${pair%%:*}"; step="${pair##*:}"
  echo "=== START ${tag} (checkpoint-${step}) $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/score.py --tag "$tag" --adapter "${CK}/checkpoint-${step}" --force
  echo "=== DONE ${tag} $(date +%H:%M:%S) ==="
done
echo "=== ALL SIX COMPLETE ==="
