#!/bin/bash

#SBATCH --job-name=jb_exp_v3

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.out
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mrahma45@gmu.edu

#SBATCH --mem=100GB
#SBATCH --time=2-00:00

#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:A100.80gb:1

# ============================================
# V3 PARAMETRIZED 5-STAGE CHAINED SLURM SCRIPT
# Usage: sbatch --export=CONFIG_NAME=exp_v3_qwen06b_llama1b run_experiment_v3.slurm
# Stages:
#   1) run1 with default tactics
#   2) run2 with curated tactics from run1
#   3) run3 with curated tactics from run2
#   4) run4 with curated tactics from run3
#   5) run5 with curated tactics from run4
# ============================================

set echo
umask 0022

module load gnu10

cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/experiments
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/results

echo "============================================"
echo "V3 EXPERIMENT: ${CONFIG_NAME}"
echo "============================================"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo ""

if [ -z "${CONFIG_NAME}" ]; then
    echo "ERROR: CONFIG_NAME environment variable not set!"
    echo "Usage: sbatch --export=CONFIG_NAME=exp_name run_experiment_v3.slurm"
    exit 1
fi

export HF_HOME=/scratch/mrahma45/hf_models
export TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models
export HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub
mkdir -p $HF_HOME

cd /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg

KNOWLEDGE_FOLDER="/scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/jailbreak/agentic_module/agents/knowledge"

for RUN_IDX in 1 2 3 4 5; do
    RUN_CONFIG_PATH="scripts/configs/experiments_v3/${CONFIG_NAME}_run${RUN_IDX}.yaml"

    if [ ! -f "$RUN_CONFIG_PATH" ]; then
        echo "ERROR: Config file not found: $RUN_CONFIG_PATH"
        exit 1
    fi

    echo "[Stage ${RUN_IDX}/5] Running run${RUN_IDX}..."
    python runs/pipeline/orchestrator.py --config_path "$RUN_CONFIG_PATH"

    if [ "$RUN_IDX" -lt 5 ]; then
        NEXT_CURATED_TACTICS_PATH="${KNOWLEDGE_FOLDER}/${CONFIG_NAME}_run${RUN_IDX}_curated_tactics.json"
        echo "[Stage ${RUN_IDX}/5] Curating tactics for next run -> ${NEXT_CURATED_TACTICS_PATH}"
        python scripts/curate_experiment_tactics.py \
            --config_path "$RUN_CONFIG_PATH" \
            --output_path "$NEXT_CURATED_TACTICS_PATH"

        if [ ! -f "$NEXT_CURATED_TACTICS_PATH" ]; then
            echo "ERROR: Curated tactics file missing: $NEXT_CURATED_TACTICS_PATH"
            exit 1
        fi

        echo "[Stage ${RUN_IDX}/5] Running tactic generalization for run${RUN_IDX}"
        python scripts/generalize_experiment_tactics.py --config_path "$RUN_CONFIG_PATH"
    fi
done

echo ""
echo "============================================"
echo "V3 EXPERIMENT COMPLETED at: $(date)"
echo "============================================"
