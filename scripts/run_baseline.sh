#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
if [ ! -f artifacts/environment.json ]; then
  echo '请先运行 python scripts/check_environment.py --simulator' >&2
  exit 1
fi
python -c "import json; d=json.load(open('artifacts/environment.json')); assert d['status']=='passed' and d.get('simulator',{}).get('real_reset_and_step'), 'Phase 1 尚未通过'"
python scripts/download_checkpoint.py
python scripts/download_vlm.py
export HF_HUB_OFFLINE=1
checkpoint_path=$(python -c "import json; print(json.load(open('artifacts/checkpoint.json'))['snapshot_path'])")
run_id=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "artifacts/baseline/$run_id"
export BASELINE_EVIDENCE_DIR="artifacts/baseline/$run_id"
python scripts/observed_eval.py \
  --policy.path="$checkpoint_path" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1 \
  --seed=42 \
  --output_dir="artifacts/baseline/$run_id" \
  2>&1 | tee "logs/baseline-$run_id.log"
