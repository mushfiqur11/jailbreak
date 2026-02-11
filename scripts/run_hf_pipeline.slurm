#!/bin/sh

#SBATCH --job-name=jailbreak_hf_qwen_llama

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/hf_qwen_llama_%j.out
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/hf_qwen_llama_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mrahma45@gmu.edu

# Resource allocation for running multiple HF models
#SBATCH --mem=120GB  # Enough memory for Qwen3-8B + Llama-3.2-3B
#SBATCH --time=1-00:00  # 1 day (100 goals can take a long time with HF models)

#SBATCH --cpus-per-task=4  # More CPUs for HF tokenization/processing
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1  # 80GB GPU for running multiple 8B models

## Load modules
set echo 
umask 0022 

module load gnu10
module load cuda/12.1  # Required for PyTorch/HuggingFace

## Setup environment
cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

## Create output directory if it doesn't exist
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs

## Pre-download models (optional - helps with caching)
echo "Starting HuggingFace Jailbreak Pipeline..."
echo "Models: Qwen3-8B (attacker/reasoning/judge) + Llama-3.2-3B (target)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

## Set HuggingFace cache directory
export HF_HOME=/scratch/mrahma45/hf_models
export TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models
export HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub

## Create cache directory if it doesn't exist
mkdir -p $HF_HOME

## Navigate to project directory
cd /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg

## Run the orchestrator pipeline with HF config
python runs/pipeline/orchestrator.py \
    --config_path scripts/configs/hf_qwen_llama_config.yaml

echo "Pipeline completed at: $(date)"
