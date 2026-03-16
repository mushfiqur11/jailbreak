#!/bin/sh

#SBATCH --job-name=jb_smoke_v2

#SBATCH --partition=contrib-gpuq
#SBATCH --qos=ist_dept
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.out
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mrahma45@gmu.edu

#SBATCH --mem=100GB
#SBATCH --time=0-06:00

#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:A100.80gb:1

# ============================================
# SMOKE V2 3-STAGE SLURM SCRIPT (5 samples)
# Stages:
#   1) run1 with default tactics
#   2) independent tactic curation command
#   3) run2 with curated tactics
# ============================================

set -euo pipefail
set echo
umask 0022

module load gnu10

cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

pip install -r /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/setup/requirements.txt

mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/experiments
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/results

# CONFIG_NAME="exp_smoke_qwen06b_llama1b"
# CONFIG_NAME="exp_smoke_qwen06b_gemma1b"
CONFIG_NAME="exp_smoke_qwen06b_phi35"

echo "============================================"
echo "SMOKE V2 EXPERIMENT: ${CONFIG_NAME}"
echo "============================================"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo ""

export HF_HOME=/scratch/mrahma45/hf_models
export TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models
export HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub
# Optional override for orchestrator logging level (takes precedence over YAML logging.level)
# Uncomment for full raw-response debug logs:
export LOG_LEVEL=DEBUG
mkdir -p $HF_HOME

echo "LOG_LEVEL override: ${LOG_LEVEL:-<not set; using config logging.level>}"

cd /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg

RUN1_CONFIG_PATH="scripts/configs/experiments_v2/${CONFIG_NAME}_run1.yaml"
RUN2_CONFIG_PATH="scripts/configs/experiments_v2/${CONFIG_NAME}_run2.yaml"
CURATED_TACTICS_PATH="scripts/configs/experiments_v2/${CONFIG_NAME}_curated_tactics.json"

if [ ! -f "$RUN1_CONFIG_PATH" ]; then
    echo "ERROR: Run1 config file not found: $RUN1_CONFIG_PATH"
    exit 1
fi

if [ ! -f "$RUN2_CONFIG_PATH" ]; then
    echo "ERROR: Run2 config file not found: $RUN2_CONFIG_PATH"
    exit 1
fi

echo "Run1 config: $RUN1_CONFIG_PATH"
echo "Run2 config: $RUN2_CONFIG_PATH"
echo "Curated tactics output: $CURATED_TACTICS_PATH"
echo ""

echo "[Stage 1/3] Running run1 with default tactics..."
python runs/pipeline/orchestrator.py \
    --config_path "$RUN1_CONFIG_PATH"

echo "[Stage 2/3] Curating tactics from run1 successful jailbreaks..."
python scripts/curate_experiment_tactics.py \
    --config_path "$RUN1_CONFIG_PATH" \
    --output_path "$CURATED_TACTICS_PATH"

if [ ! -f "$CURATED_TACTICS_PATH" ]; then
    echo "ERROR: Curated tactics file missing after curation step: $CURATED_TACTICS_PATH"
    exit 1
fi

echo "Verified curated tactics exists for run2: $CURATED_TACTICS_PATH"

echo "[Stage 3/3] Running run2 with curated tactics..."
python runs/pipeline/orchestrator.py \
    --config_path "$RUN2_CONFIG_PATH"

echo ""
echo "Copying results to output directory..."
mv ${CONFIG_NAME}_*.json /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true
cp "$CURATED_TACTICS_PATH" /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true

echo ""
echo "============================================"
echo "SMOKE V2 EXPERIMENT COMPLETED at: $(date)"
echo "============================================"
