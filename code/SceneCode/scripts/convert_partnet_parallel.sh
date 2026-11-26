#!/bin/bash
# Parallel PartNet-Mobility conversion using multiple independent workers.
#
# Each worker calls convert_partnet_mobility.py with a subset of asset IDs.
# Each worker manages its own CoACD server internally.
#
# Usage:
#   ./scripts/convert_partnet_parallel.sh <input_dir> <output_dir> [num_workers]
#
# Example:
#   ./scripts/convert_partnet_parallel.sh data/partnet-mobility-v0 output 10

set -e

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [num_workers]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [num_workers]}"
NUM_WORKERS="${3:-10}"

CPU_COUNT=$(nproc)
OMP_PER_WORKER=$((CPU_COUNT / NUM_WORKERS))
OMP_PER_WORKER=$((OMP_PER_WORKER > 0 ? OMP_PER_WORKER : 1))

echo "=== PartNet Parallel Conversion ==="
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Workers:     $NUM_WORKERS"
echo "CPU count:   $CPU_COUNT"
echo "OMP threads: $OMP_PER_WORKER per worker"
echo "===================================="

# Get list of asset IDs (directory names with mobility.urdf).
ASSET_IDS=($(find "$INPUT_DIR" -maxdepth 1 -mindepth 1 -type d \
    -exec test -f {}/mobility.urdf \; -print \
    | xargs -n1 basename | sort))
TOTAL_ASSETS=${#ASSET_IDS[@]}

if [ "$TOTAL_ASSETS" -eq 0 ]; then
    echo "Error: No assets found in $INPUT_DIR"
    exit 1
fi

echo "Found $TOTAL_ASSETS assets to process"

# Create output directory.
mkdir -p "$OUTPUT_DIR"

# Calculate assets per worker.
ASSETS_PER_WORKER=$(( (TOTAL_ASSETS + NUM_WORKERS - 1) / NUM_WORKERS ))

# Build comma-separated ID lists for each worker.
declare -a WORKER_IDS
for ((i=0; i<NUM_WORKERS; i++)); do
    START=$((i * ASSETS_PER_WORKER))
    END=$((START + ASSETS_PER_WORKER))
    if [ $END -gt $TOTAL_ASSETS ]; then
        END=$TOTAL_ASSETS
    fi
    if [ $START -ge $TOTAL_ASSETS ]; then
        continue
    fi

    # Build comma-separated list of IDs for this worker.
    IDS=""
    for ((j=START; j<END; j++)); do
        if [ -n "$IDS" ]; then
            IDS="$IDS,"
        fi
        IDS="$IDS${ASSET_IDS[$j]}"
    done
    WORKER_IDS[$i]="$IDS"

    COUNT=$((END - START))
    echo "Worker $i: $COUNT assets (indices $START-$((END-1)))"
done

# Cleanup function - kills all worker processes and their children.
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Kill all background jobs and their children.
    for PID in $(jobs -p 2>/dev/null); do
        pkill -P $PID 2>/dev/null || true
        kill $PID 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# Launch workers in parallel.
echo ""
echo "Launching $NUM_WORKERS workers..."
PIDS=()

for ((i=0; i<NUM_WORKERS; i++)); do
    IDS="${WORKER_IDS[$i]}"
    if [ -z "$IDS" ]; then
        continue
    fi

    LOG_FILE="$OUTPUT_DIR/worker_${i}.log"
    echo "Starting worker $i (log: $LOG_FILE)"

    # Each worker runs the conversion script with its assigned IDs.
    # Script creates its own CoACD server internally.
    OMP_NUM_THREADS=$OMP_PER_WORKER \
    OPENBLAS_NUM_THREADS=$OMP_PER_WORKER \
    MKL_NUM_THREADS=$OMP_PER_WORKER \
        python scripts/convert_partnet_mobility.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --ids "$IDS" \
        --skip-existing \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo "Workers launched. PIDs: ${PIDS[*]}"
echo ""
echo "Monitoring progress (Ctrl+C to stop)..."
echo "Logs in: $OUTPUT_DIR/worker_*.log"
echo ""

# Wait for all workers to complete.
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait $PID; then
        FAILED=$((FAILED + 1))
    fi
done

# Count results.
SUCCESS_COUNT=$(find "$OUTPUT_DIR" -name "mobility.sdf" | wc -l)
echo ""
echo "=== Conversion Complete ==="
echo "Successful: $SUCCESS_COUNT / $TOTAL_ASSETS"
echo "Workers failed: $FAILED"
echo "==========================="
