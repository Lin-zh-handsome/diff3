#!/usr/bin/env bash
set -euo pipefail
cd /home/hanjinwei/p1/project/thesis-ch3-tm-dp
GPU="${1:-0}"
case "$GPU" in 0|1) ;; *) echo "Choose physical GPU 0 or 1" >&2; exit 2 ;; esac
PYTHON=/home/hanjinwei/miniconda3/envs/robodiff/bin/python

is_complete() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path
mode = sys.argv[1]
p = Path(f'data/outputs/tm_dp/{mode}/logs.json.txt')
if not p.exists():
    raise SystemExit(1)
lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
raise SystemExit(0 if lines and lines[-1].get('epoch') == 3049 else 1)
PY
}

# One training job at a time on the same physical GPU. Existing runs resume.
for mode in vanilla tsm me full; do
  if is_complete "$mode"; then
    continue
  fi
  bash run_tm_dp.sh "$mode" "$GPU" >> "data/outputs/tm_dp/${mode}_train.log" 2>&1
  is_complete "$mode"
done

# Evaluate only after all training jobs finish, on the same physical GPU.
for mode in vanilla tsm me full; do
  checkpoint=$("$PYTHON" - "$mode" <<'PY'
import glob, re, sys
mode = sys.argv[1]
files = glob.glob(f'data/outputs/tm_dp/{mode}/checkpoints/epoch=*test_mean_score=*.ckpt')
if not files:
    raise SystemExit(f'No best checkpoint for {mode}')
def score(path):
    match = re.search(r'test_mean_score=([0-9]+(?:\.[0-9]+)?)\.ckpt$', path)
    if match is None:
        raise ValueError(path)
    return float(match.group(1))
print(max(files, key=score))
PY
)
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" evaluate_tm_dp.py \
    --checkpoint "$checkpoint" --output-dir "data/outputs/tm_dp/eval_${mode}" \
    --device cuda:0 > "data/outputs/tm_dp/${mode}_eval.log" 2>&1
done
