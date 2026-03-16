#!/bin/bash

#SBATCH --job-name=jb_exp_v2

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.out
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mrahma45@gmu.edu

#SBATCH --mem=100GB
#SBATCH --time=1-20:00

#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:A100.80gb:1

# ============================================
# V2 PARAMETRIZED 3-STAGE SLURM SCRIPT
# Usage: sbatch --export=CONFIG_NAME=exp_qwen8b_llama1b run_experiment_v2.slurm
# Stages:
#   1) run1 with default tactics
#   2) tactic curation + tactic generalization from run1 outputs
#   3) run2 with curated tactics (curated tactics are appended to defaults)
#
# IMPORTANT V2 CHANGE:
#   curated tactics JSON is stored in scripts/configs/experiments_v2/
#   (same directory family as configs), avoiding previous path mismatch.
# ============================================

# set -euo pipefail
# set -x
set echo
umask 0022

module load gnu10

cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

# pip install --upgrade -r /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/setup/requirements.txt

mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/experiments
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/results

echo "============================================"
echo "V2 EXPERIMENT: ${CONFIG_NAME}"
echo "============================================"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo ""

if [ -z "${CONFIG_NAME}" ]; then
    echo "ERROR: CONFIG_NAME environment variable not set!"
    echo "Usage: sbatch --export=CONFIG_NAME=exp_name run_experiment_v2.slurm"
    exit 1
fi

export HF_HOME=/scratch/mrahma45/hf_models
export TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models
export HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub
mkdir -p $HF_HOME

cd /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg

# Knowledge folder for tactics (absolute path)
KNOWLEDGE_FOLDER="/scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/jailbreak/agentic_module/agents/knowledge"

RUN1_CONFIG_PATH="scripts/configs/experiments_v2/${CONFIG_NAME}_run1.yaml"
RUN2_CONFIG_PATH="scripts/configs/experiments_v2/${CONFIG_NAME}_run2.yaml"
CURATED_TACTICS_PATH="${KNOWLEDGE_FOLDER}/${CONFIG_NAME}_curated_tactics.json"

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

echo "[Stage 2/3] Running tactic generalization from run1 successes..."
python scripts/generalize_experiment_tactics.py \
    --config_path "$RUN1_CONFIG_PATH"

if [ ! -f "$CURATED_TACTICS_PATH" ]; then
    echo "ERROR: Curated tactics file missing after curation step: $CURATED_TACTICS_PATH"
    exit 1
fi

echo "Verified curated tactics exists for run2: $CURATED_TACTICS_PATH"

echo "[Stage 3/3] Running run2 with curated tactics..."
python runs/pipeline/orchestrator.py \
    --config_path "$RUN2_CONFIG_PATH"

echo ""
echo "Moving results to output directory..."
mv ${CONFIG_NAME}_*.json /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true
cp "$CURATED_TACTICS_PATH" /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true

echo ""
echo "============================================"
echo "V2 EXPERIMENT COMPLETED at: $(date)"
echo "============================================"
