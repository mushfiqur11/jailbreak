#!/bin/bash

# ============================================
# V2 MASTER LAUNCHER
# Launches V2 3-stage experiments from experiments_v2 folder.
# Supports optional omission list for experiments to skip.
# ============================================

echo "============================================"
echo "JAILBREAK EXPERIMENT LAUNCHER V2"
echo "============================================"
echo "Date: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENTS_DIR="configs/experiments_v2"
EXPERIMENT_LIST="$EXPERIMENTS_DIR/experiment_list.txt"

if [ ! -d "$EXPERIMENTS_DIR" ] || [ ! -f "$EXPERIMENT_LIST" ]; then
    echo "V2 configs/list missing. Generating now..."
    python generate_experiment_configs_v2.py
fi

# Optional omit list path (one exp_name per line)
OMIT_LIST_FILE="${OMIT_LIST_FILE:-}"

should_skip() {
    local exp_name="$1"
    if [ -z "$OMIT_LIST_FILE" ] || [ ! -f "$OMIT_LIST_FILE" ]; then
        return 1
    fi
    grep -Fxq "$exp_name" "$OMIT_LIST_FILE"
}

TOTAL_EXPERIMENTS=$(wc -l < "$EXPERIMENT_LIST")
echo "Total experiments listed: $TOTAL_EXPERIMENTS"
if [ -n "$OMIT_LIST_FILE" ]; then
    echo "Using omit list: $OMIT_LIST_FILE"
fi
echo ""

DELAY_SECONDS=2
SUBMITTED=0
FAILED=0
SKIPPED=0

while IFS= read -r exp_name || [[ -n "$exp_name" ]]; do
    [ -z "$exp_name" ] && continue

    if should_skip "$exp_name"; then
        echo "Skipping (omit list): $exp_name"
        ((SKIPPED++))
        continue
    fi

    RUN1_CONFIG_FILE="$EXPERIMENTS_DIR/${exp_name}_run1.yaml"
    RUN2_CONFIG_FILE="$EXPERIMENTS_DIR/${exp_name}_run2.yaml"
    if [ ! -f "$RUN1_CONFIG_FILE" ] || [ ! -f "$RUN2_CONFIG_FILE" ]; then
        echo "WARNING: Missing run1/run2 config for $exp_name, skipping..."
        ((FAILED++))
        continue
    fi

    echo "Submitting V2: $exp_name"
    JOB_ID=$(sbatch --job-name="${exp_name}_v2" --export=CONFIG_NAME="$exp_name" run_experiment_v2.slurm 2>&1)

    if [[ $JOB_ID == *"Submitted batch job"* ]]; then
        ((SUBMITTED++))
        echo "  -> $JOB_ID"
    else
        echo "  -> FAILED: $JOB_ID"
        ((FAILED++))
    fi

    sleep $DELAY_SECONDS
done < "$EXPERIMENT_LIST"

echo ""
echo "============================================"
echo "V2 LAUNCH COMPLETE"
echo "============================================"
echo "Submitted: $SUBMITTED"
echo "Skipped (omit list): $SKIPPED"
echo "Failed: $FAILED"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
