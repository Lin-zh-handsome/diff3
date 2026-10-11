#!/usr/bin/env bash
set -euo pipefail
cd /home/hanjinwei/p1/project/thesis-ch3-tm-dp
VANILLA_PID="${1:?Pass the already running vanilla trainer PID}"
while kill -0 "$VANILLA_PID" 2>/dev/null; do sleep 120; done
/home/hanjinwei/miniconda3/envs/robodiff/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('data/outputs/tm_dp/vanilla/logs.json.txt')
lines = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
if not lines or lines[-1].get('epoch') != 3049:
    raise SystemExit('Vanilla training did not complete 3050 epochs; stopping queue')
PY
for mode in tsm me full; do
  bash run_tm_dp.sh "$mode" 0 > "data/outputs/tm_dp/${mode}_train.log" 2>&1
done
for mode in vanilla tsm me full; do
  checkpoint=$(/home/hanjinwei/miniconda3/envs/robodiff/bin/python - "$mode" <<'PY'
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
  /home/hanjinwei/miniconda3/envs/robodiff/bin/python evaluate_tm_dp.py \
    --checkpoint "$checkpoint" --output-dir "data/outputs/tm_dp/eval_${mode}" \
    --device cuda:0 > "data/outputs/tm_dp/${mode}_eval.log" 2>&1
done
