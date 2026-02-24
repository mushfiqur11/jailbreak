#!/bin/sh

#SBATCH --job-name=jb_short_test

#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --output=/scratch/%u/jailbreaking_repos/outputs/hf_short_test_%j.out
#SBATCH --error=/scratch/%u/jailbreaking_repos/outputs/hf_short_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mrahma45@gmu.edu

# Reduced resources for short test (5 samples only)
#SBATCH --mem=120GB
#SBATCH --time=0-04:00:00  # 4 hours (should be plenty for 5 samples)

#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1

## Load modules
set echo 
umask 0022 

module load gnu10
module load cuda/12.1

## Setup environment
cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

## Create output directory if it doesn't exist
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs

## Print run info
echo "============================================"
echo "SHORT TEST RUN - 5 Samples Only"
echo "Testing error handling & fallback tracking"
echo "============================================"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Models: Qwen3-8B (attacker/reasoning/judge) + Llama-3.2-3B (target)"
echo "Max Goals: 5"
echo ""

## Set HuggingFace cache directory
export HF_HOME=/scratch/mrahma45/hf_models
export TRANSFORMERS_CACHE=/scratch/mrahma45/hf_models
export HF_HUB_CACHE=/scratch/mrahma45/hf_models/hub

## Create cache directory if it doesn't exist
mkdir -p $HF_HOME

## Navigate to project directory
cd /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg

## Run the orchestrator pipeline with short test config
python runs/pipeline/orchestrator.py \
    --config_path scripts/configs/hf_short_test_config.yaml

echo ""
echo "============================================"
echo "SHORT TEST COMPLETED at: $(date)"
echo "============================================"
echo ""
echo "Check output files:"
echo "  - hf_short_test_results.json"
echo "  - hf_short_test_results_stats.json (for fallback counts)"
echo "  - hf_short_test_knowledge.json"
echo "  - hf_short_test_successes.json"
