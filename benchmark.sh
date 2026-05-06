#!/bin/bash

ROOT="/dCLIMBA"
BASE_DIR="$ROOT/outputs/job_NAME"

# Configuration
VALIDATION_FLAG=""  # Set to "--validation" if needed
TEST_PERIOD="--test_period 1970,2014"  # Adjust as needed

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Running benchmarking for all runs"
echo "Base directory: $BASE_DIR"
echo "Test period: 2001-2014"
echo "=========================================="

total_runs=0
success=0
failed=0
failed_runs=()

# Iterate through model directories
for model_dir in "$BASE_DIR"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(basename "$model_dir")
    echo -e "\n${BLUE}Processing model: $model_name${NC}"
    echo "----------------------------------------"
    
    # Iterate through experiment directories
    for exp_dir in "$model_dir"/*; do
        if [ ! -d "$exp_dir" ]; then
            continue
        fi
        
        exp_name=$(basename "$exp_dir")
        
        # Iterate through run directories
        for run_dir in "$exp_dir"/*_*_*; do
            if [ ! -d "$run_dir" ]; then
                continue
            fi
            
            # Extract run_id
            full_name=$(basename "$run_dir")
            run_id=$(echo "$full_name" | cut -d'_' -f1)
            
            total_runs=$((total_runs + 1))
            
            echo ""
            echo -e "${YELLOW}[$total_runs] Processing:${NC}"
            echo "  Model: $model_name"
            echo "  Experiment: $exp_name"
            echo "  Run ID: $run_id"
            
            # Build command
            cmd="python $ROOT/benchmarking.py --run_id $run_id --base_dir $BASE_DIR"
            
            if [ -n "$VALIDATION_FLAG" ]; then
                cmd="$cmd $VALIDATION_FLAG"
            fi
            
            if [ -n "$TEST_PERIOD" ]; then
                cmd="$cmd $TEST_PERIOD"
            fi
            
            # Run benchmarking
            start_time=$(date +%s)
            
            if $cmd; then
                end_time=$(date +%s)
                duration=$((end_time - start_time))
                echo -e "  ${GREEN}✓ SUCCESS${NC} (${duration}s)"
                success=$((success + 1))
            else
                end_time=$(date +%s)
                duration=$((end_time - start_time))
                echo -e "  ${RED}✗ FAILED${NC} (${duration}s)"
                failed=$((failed + 1))
                failed_runs+=("$model_name/$exp_name/$run_id")
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "  Total runs: $total_runs"
echo -e "  ${GREEN}Successful: $success${NC}"
echo -e "  ${RED}Failed: $failed${NC}"

if [ $failed -gt 0 ]; then
    echo ""
    echo "Failed runs:"
    for run in "${failed_runs[@]}"; do
        echo "  - $run"
    done
fi

echo "=========================================="