#!/bin/bash

#SBATCH --job-name=valid_judge

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

set echo
umask 0022

module load gnu10

cd /scratch/mrahma45/jailbreaking_repos/
source venv_jailbreak/bin/activate

mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/experiments
mkdir -p /scratch/mrahma45/jailbreaking_repos/outputs/judge_validation

cd /scratch/mrahma45/jailbreaking_repos/runs/pipeline

python judge_single_pass.py \
    --dataset_path /scratch/mrahma45/jailbreaking_repos/jailbreak-pkg/data/new_data.json \
    --output_path /scratch/mrahma45/jailbreaking_repos/outputs/judge_validation/qwen3_32b_q4_single_pass.json \
    --model qwen \
    --model_id Qwen/Qwen3-32B \
    --hf_token_path /home/mrahma45/HUGGINGFACE_KEY \
    --hf_model_path /scratch/mrahma45/hf_models \
    --quantization 4bit