#!/bin/bash

# ================================================================
# Launch missing RUN2 jobs from already completed RUN1 outputs.
#
# Logic:
#  - Check /scratch/.../outputs/results for run1 and run2 result files
#  - If run1 exists and run2 is missing, submit a RUN2-only SLURM job
#  - RUN2 uses curated tactics JSON from the same results directory
#
# This script is intended for the legacy experiment set under:
#   scripts/configs/experiments
# ================================================================

set -euo pipefail

RESULTS_DIR="/scratch/mrahma45/jailbreaking_repos/outputs/results"
REPO_ROOT="/scratch/mrahma45/jailbreaking_repos/jailbreak-pkg"
EXPERIMENTS_DIR="$REPO_ROOT/scripts/configs/experiments"
EXPERIMENT_LIST="$EXPERIMENTS_DIR/experiment_list.txt"
PATCHED_CONFIG_DIR="$EXPERIMENTS_DIR/run2_from_results"

# Optional controls
DELAY_SECONDS="${DELAY_SECONDS:-2}"
MAX_SUBMITS="${MAX_SUBMITS:-0}" # 0 means no limit
DRY_RUN="${DRY_RUN:-0}"         # 1 => print actions only

echo "================================================"
echo "MISSING RUN2 LAUNCHER (from existing results)"
echo "================================================"
echo "Date: $(date)"
echo "Results dir: $RESULTS_DIR"
echo "Repo root:   $REPO_ROOT"
echo "Experiments: $EXPERIMENTS_DIR"
echo ""

if [ ! -f "$EXPERIMENT_LIST" ]; then
  echo "ERROR: Experiment list not found: $EXPERIMENT_LIST"
  exit 1
fi

mkdir -p "$PATCHED_CONFIG_DIR"
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/experiments

submitted=0
skipped=0
missing_inputs=0

while IFS= read -r exp_name || [[ -n "$exp_name" ]]; do
  [ -z "$exp_name" ] && continue

  run1_results="$RESULTS_DIR/${exp_name}_run1_results.json"
  run2_results="$RESULTS_DIR/${exp_name}_run2_results.json"
  curated_tactics="$RESULTS_DIR/${exp_name}_curated_tactics.json"
  run2_config="$EXPERIMENTS_DIR/${exp_name}_run2.yaml"
  patched_run2_config="$PATCHED_CONFIG_DIR/${exp_name}_run2_from_results.yaml"

  # Only consider experiments that completed run1 but not run2
  if [ ! -f "$run1_results" ]; then
    ((skipped+=1))
    continue
  fi

  if [ -f "$run2_results" ]; then
    ((skipped+=1))
    continue
  fi

  # Need both curated tactics and original run2 config
  if [ ! -f "$curated_tactics" ] || [ ! -f "$run2_config" ]; then
    echo "SKIP (missing curated/config): $exp_name"
    [ ! -f "$curated_tactics" ] && echo "  - missing: $curated_tactics"
    [ ! -f "$run2_config" ] && echo "  - missing: $run2_config"
    ((missing_inputs+=1))
    continue
  fi

  # Build patched run2 config pointing tactics_file to curated JSON in results dir
  python - "$run2_config" "$patched_run2_config" "$curated_tactics" <<'PY'
import sys
import yaml

src, dst, curated = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["agents"]["attacker"]["tactics_file"] = curated

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "QUEUE RUN2: $exp_name"
  echo "  patched config: $patched_run2_config"

  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi

  submit_output=$(sbatch \
    --job-name="${exp_name}_run2" \
    --partition=gpuq \
    --qos=gpu \
    --mem=100GB \
    --time=1-20:00 \
    --cpus-per-task=1 \
    --ntasks=1 \
    --gres=gpu:A100.80gb:1 \
    --output="/scratch/%u/jailbreaking_repos/outputs/experiments/${exp_name}_run2_%j.out" \
    --error="/scratch/%u/jailbreaking_repos/outputs/experiments/${exp_name}_run2_%j.err" \
    --mail-type=FAIL \
    --mail-user=mrahma45@gmu.edu \
    --wrap="cd /scratch/mrahma45/jailbreaking_repos && source venv_jailbreak/bin/activate && cd $REPO_ROOT && export HF_HOME=/scratch/mrahma45/hf_models TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub && python runs/pipeline/orchestrator.py --config_path $patched_run2_config" 2>&1)

  echo "  -> $submit_output"
  ((submitted+=1))

  if [ "$MAX_SUBMITS" -gt 0 ] && [ "$submitted" -ge "$MAX_SUBMITS" ]; then
    echo "Reached MAX_SUBMITS=$MAX_SUBMITS, stopping."
    break
  fi

  sleep "$DELAY_SECONDS"
done < "$EXPERIMENT_LIST"

echo ""
echo "================================================"
echo "DONE"
echo "Submitted RUN2 jobs : $submitted"
echo "Skipped             : $skipped"
echo "Missing inputs      : $missing_inputs"
echo "================================================"
