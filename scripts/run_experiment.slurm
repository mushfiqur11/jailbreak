#!/bin/sh

#SBATCH --job-name=jb_exp

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.out
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/experiments/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mrahma45@gmu.edu

# Resource allocation - will vary by model size
# Default for most experiments (up to 8B models)
#SBATCH --mem=100GB
#SBATCH --time=0-18:00  # 18 hours per experiment

#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:A100.80gb:1

# ============================================
# PARAMETRIZED 3-STAGE SLURM SCRIPT
# Usage: sbatch --export=CONFIG_NAME=exp_qwen8b_llama1b run_experiment.slurm
# Stages:
#   1) run1 with default tactics
#   2) independent tactic curation command
#   3) run2 with curated tactics
# ============================================

## Load modules
set echo 
umask 0022 

module load gnu10

## Setup environment
cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

pip install -r /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/setup/requirements.txt

## Create output directories if they don't exist
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/experiments
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/results

## Print run info
echo "============================================"
echo "EXPERIMENT: ${CONFIG_NAME}"
echo "============================================"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo ""

## Validate CONFIG_NAME
if [ -z "${CONFIG_NAME}" ]; then
    echo "ERROR: CONFIG_NAME environment variable not set!"
    echo "Usage: sbatch --export=CONFIG_NAME=exp_name run_experiment.slurm"
    exit 1
fi

## Set HuggingFace cache directory
export HF_HOME=/scratch/mrahma45/hf_models
export TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models
export HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub

## Create cache directory if it doesn't exist
mkdir -p $HF_HOME

## Navigate to project directory
cd /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg

## Resolve config paths
RUN1_CONFIG_PATH="scripts/configs/experiments/${CONFIG_NAME}_run1.yaml"
RUN2_CONFIG_PATH="scripts/configs/experiments/${CONFIG_NAME}_run2.yaml"
CURATED_TACTICS_PATH="scripts/configs/experiments/curated_tactics/${CONFIG_NAME}_curated_tactics.json"

if [ ! -f "$RUN1_CONFIG_PATH" ]; then
    echo "ERROR: Run1 config file not found: $RUN1_CONFIG_PATH"
    exit 1
fi

if [ ! -f "$RUN2_CONFIG_PATH" ]; then
    echo "ERROR: Run2 config file not found: $RUN2_CONFIG_PATH"
    exit 1
fi

mkdir -p scripts/configs/experiments/curated_tactics

echo "Run1 config: $RUN1_CONFIG_PATH"
echo "Run2 config: $RUN2_CONFIG_PATH"
echo ""

## Stage 1: run1 (default tactics)
echo "[Stage 1/3] Running run1 with default tactics..."
python runs/pipeline/orchestrator.py \
    --config_path "$RUN1_CONFIG_PATH"

## Stage 2: independent tactic curation command
echo "[Stage 2/3] Curating tactics from run1 successful jailbreaks..."
python scripts/curate_experiment_tactics.py \
    --config_path "$RUN1_CONFIG_PATH" \
    --output_path "$CURATED_TACTICS_PATH"

## Stage 3: run2 (curated tactics)
echo "[Stage 3/3] Running run2 with curated tactics..."
python runs/pipeline/orchestrator.py \
    --config_path "$RUN2_CONFIG_PATH"

## Move output files to results directory
echo ""
echo "Moving results to output directory..."
mv ${CONFIG_NAME}_*.json /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true
mv scripts/configs/experiments/curated_tactics/${CONFIG_NAME}_curated_tactics.json /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETED at: $(date)"
echo "============================================"
echo ""
echo "Results location: /scratch/mrahma45/jailbreaking_repos/outputs/results/"
echo "  - ${CONFIG_NAME}_run1_results.json"
echo "  - ${CONFIG_NAME}_run1_results_stats.json"
echo "  - ${CONFIG_NAME}_run1_knowledge.json"
echo "  - ${CONFIG_NAME}_run1_successes.json"
echo "  - ${CONFIG_NAME}_run2_results.json"
echo "  - ${CONFIG_NAME}_run2_results_stats.json"
echo "  - ${CONFIG_NAME}_run2_knowledge.json"
echo "  - ${CONFIG_NAME}_run2_successes.json"
echo "  - ${CONFIG_NAME}_curated_tactics.json"
