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
# PARAMETRIZED SLURM SCRIPT
# Usage: sbatch --export=CONFIG_NAME=exp_qwen8b_llama1b run_experiment.slurm
# Or:    sbatch --job-name=exp_qwen8b_llama1b --export=CONFIG_NAME=exp_qwen8b_llama1b run_experiment.slurm
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

## Check if config file exists
CONFIG_PATH="scripts/configs/experiments/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Config file: $CONFIG_PATH"
echo ""

## Run the orchestrator pipeline
python runs/pipeline/orchestrator.py \
    --config_path "$CONFIG_PATH"

## Move output files to results directory
echo ""
echo "Moving results to output directory..."
mv ${CONFIG_NAME}_*.json /scratch/mrahma45/jailbreaking_repos/outputs/results/ 2>/dev/null || true

echo ""
echo "============================================"
echo "EXPERIMENT COMPLETED at: $(date)"
echo "============================================"
echo ""
echo "Results location: /scratch/mrahma45/jailbreaking_repos/outputs/results/"
echo "  - ${CONFIG_NAME}_results.json"
echo "  - ${CONFIG_NAME}_results_stats.json"
echo "  - ${CONFIG_NAME}_knowledge.json"
echo "  - ${CONFIG_NAME}_successes.json"
