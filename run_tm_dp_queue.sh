#!/usr/bin/env bash
set -euo pipefail
cd /home/hanjinwei/p1/project/thesis-ch3-tm-dp
mkdir -p data/outputs/tm_dp
GPU="${1:-0}"
case "$GPU" in 0|1) ;; *) echo "Choose physical GPU 0 or 1" >&2; exit 2 ;; esac
PYTHON=/home/hanjinwei/miniconda3/envs/robodiff/bin/python

is_complete() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
from pathlib import Path
import dill
import torch

mode = sys.argv[1]
run_dir = Path(f'data/outputs/tm_dp/{mode}')
log = run_dir / 'logs.json.txt'
if not log.exists():
    raise SystemExit(1)
last = None
with log.open() as stream:
    for line in stream:
        if line.strip():
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                # JsonLogger truncates an interrupted final line on resume.
                raise SystemExit(1 if not line.endswith('\n') else 2)
if not last or last.get('epoch') != 3049:
    raise SystemExit(1)
checkpoint = run_dir / 'checkpoints' / 'latest.ckpt'
if not checkpoint.is_file():
    print(f'{mode}: final log exists, but latest checkpoint is missing', file=sys.stderr)
    raise SystemExit(2)
try:
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill)
    if not payload.get('state_dicts', {}).get('model') or 'cfg' not in payload:
        raise ValueError('checkpoint lacks model state or config')
except Exception as exc:
    print(f'{mode}: latest checkpoint cannot be loaded: {exc}', file=sys.stderr)
    raise SystemExit(2)
PY
}

# One training job at a time on the same physical GPU. Existing runs resume.
for mode in vanilla tsm me full; do
  if is_complete "$mode"; then
    continue
  else
    status=$?
    if (( status != 1 )); then exit "$status"; fi
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
