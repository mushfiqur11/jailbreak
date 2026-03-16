#!/bin/bash

# ============================================
# V3 MASTER LAUNCHER
# Launches V3 5-stage chained experiments from experiments_v3 folder.
# Supports optional omission list for experiments to skip.
# ============================================

echo "============================================"
echo "JAILBREAK EXPERIMENT LAUNCHER V3"
echo "============================================"
echo "Date: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENTS_DIR="configs/experiments_v3"
EXPERIMENT_LIST="$EXPERIMENTS_DIR/experiment_list.txt"

if [ ! -d "$EXPERIMENTS_DIR" ] || [ ! -f "$EXPERIMENT_LIST" ]; then
    echo "V3 configs/list missing. Generating now..."
    python generate_experiment_configs_v3.py
fi

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

    missing=0
    for run_idx in 1 2 3 4 5; do
        RUN_CONFIG_FILE="$EXPERIMENTS_DIR/${exp_name}_run${run_idx}.yaml"
        if [ ! -f "$RUN_CONFIG_FILE" ]; then
            echo "WARNING: Missing config for $exp_name run${run_idx}, skipping..."
            missing=1
            break
        fi
    done
    if [ "$missing" -eq 1 ]; then
        ((FAILED++))
        continue
    fi

    echo "Submitting V3: $exp_name"
    JOB_ID=$(sbatch --job-name="${exp_name}_v3" --export=CONFIG_NAME="$exp_name" run_experiment_v3.slurm 2>&1)

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
echo "V3 LAUNCH COMPLETE"
echo "============================================"
echo "Submitted: $SUBMITTED"
echo "Skipped (omit list): $SKIPPED"
echo "Failed: $FAILED"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
