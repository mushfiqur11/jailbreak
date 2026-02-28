#!/bin/bash

# ============================================
# MASTER LAUNCHER SCRIPT
# Launches all 39 base experiment SLURM jobs.
# Each job executes 3 stages internally:
#   run1 (default tactics) -> tactic curation -> run2 (curated tactics)
# 13 targets × 3 attackers = 39 base experiments
# ============================================

echo "============================================"
echo "JAILBREAK EXPERIMENT LAUNCHER"
echo "============================================"
echo "Date: $(date)"
echo ""

# Navigate to scripts directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if experiment configs exist
EXPERIMENTS_DIR="configs/experiments"
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Experiment configs directory not found!"
    echo "Generating experiment configurations first..."
    python generate_experiment_configs.py
fi

# Check if experiment list file exists
EXPERIMENT_LIST="$EXPERIMENTS_DIR/experiment_list.txt"
if [ ! -f "$EXPERIMENT_LIST" ]; then
    echo "Experiment list file not found!"
    echo "Generating experiment configurations first..."
    python generate_experiment_configs.py
fi

# Count total experiments
TOTAL_EXPERIMENTS=$(wc -l < "$EXPERIMENT_LIST")
echo "Total experiments to launch: $TOTAL_EXPERIMENTS"
echo ""

# Delay between job submissions (to avoid overwhelming SLURM)
DELAY_SECONDS=2

# Counter for submitted jobs
SUBMITTED=0
FAILED=0

# Read experiment list and submit jobs
while IFS= read -r exp_name || [[ -n "$exp_name" ]]; do
    # Skip empty lines
    [ -z "$exp_name" ] && continue
    
    # Check if both stage configs exist
    RUN1_CONFIG_FILE="$EXPERIMENTS_DIR/${exp_name}_run1.yaml"
    RUN2_CONFIG_FILE="$EXPERIMENTS_DIR/${exp_name}_run2.yaml"
    if [ ! -f "$RUN1_CONFIG_FILE" ] || [ ! -f "$RUN2_CONFIG_FILE" ]; then
        echo "WARNING: Stage config file(s) not found for $exp_name, skipping..."
        ((FAILED++))
        continue
    fi
    
    # Submit the job
    echo "Submitting: $exp_name"
    JOB_ID=$(sbatch --job-name="$exp_name" --export=CONFIG_NAME="$exp_name" run_experiment.slurm 2>&1)
    
    if [[ $JOB_ID == *"Submitted batch job"* ]]; then
        ((SUBMITTED++))
        echo "  -> $JOB_ID"
    else
        echo "  -> FAILED: $JOB_ID"
        ((FAILED++))
    fi
    
    # Delay between submissions
    sleep $DELAY_SECONDS
    
done < "$EXPERIMENT_LIST"

echo ""
echo "============================================"
echo "LAUNCH COMPLETE"
echo "============================================"
echo "Submitted: $SUBMITTED / $TOTAL_EXPERIMENTS"
echo "Failed: $FAILED"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all jobs with: scancel -u \$USER"
echo ""
echo "Results will be saved to:"
echo "  /scratch/mrahma45/jailbreaking_repos/outputs/results/"
echo ""
