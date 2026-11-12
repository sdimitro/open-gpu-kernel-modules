#!/usr/bin/env bash
#
# A/B test of UVM TLB batch range merging on a single boot.
#
# Flips the runtime module parameter uvm_perf_tlb_batch_coalesce between
# interleaved runs (ABBA order, so slow drift such as thermals cancels out),
# snapshots the per-GPU TLB counters around each run, and writes one CSV row
# per run. Then summarizes with compare.py.
#
# Requires the patched nvidia-uvm (it has the parameter and the counters) and
# root (to write the parameter).
#
# Usage:
#   sudo ./ab_uvm_tlb.sh [-r rounds] [-g uvm_gpu_id] [-o out.csv] [-- bench args...]
#
#   -r  rounds; each round runs both settings once (default 8)
#   -g  UVM GPU id, i.e. /proc/driver/nvidia-uvm/gpus/<id> (default 1)
#   -o  output CSV (default uvm_tlb_ab.csv)
#   bench args are passed to bench/uvm_tlb_bench (default: --mode managed)
#
# Example:
#   sudo ./ab_uvm_tlb.sh -r 10 -- --mode system --mb 2048 --touch 0.5

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
# Overridable for testing the script without a GPU
BENCH=${BENCH:-$HERE/../bench/uvm_tlb_bench}
PARAM=${PARAM:-/sys/module/nvidia_uvm/parameters/uvm_perf_tlb_batch_coalesce}
UVM_PROC=${UVM_PROC:-/proc/driver/nvidia-uvm/gpus}

ROUNDS=8
UVM_GPU=1
OUT=uvm_tlb_ab.csv

while getopts "r:g:o:h" opt; do
    case "$opt" in
        r) ROUNDS=$OPTARG ;;
        g) UVM_GPU=$OPTARG ;;
        o) OUT=$OPTARG ;;
        *) sed -n '2,25p' "$0"; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift
BENCH_ARGS=("$@")
[[ ${#BENCH_ARGS[@]} -eq 0 ]] && BENCH_ARGS=(--mode managed)

INFO=$UVM_PROC/$UVM_GPU/info

if [[ ! -w $PARAM ]]; then
    echo "error: $PARAM is missing or not writable." >&2
    echo "Load the patched nvidia-uvm and run as root. On a stock driver, compare" >&2
    echo "builds with bpftrace/uvm_tlb_invalidates.bt instead (see README.md)." >&2
    exit 1
fi
if [[ ! -r $INFO ]] || ! grep -q '^tlb_batch_invalidate_all' "$INFO"; then
    echo "error: $INFO has no tlb_batch_* counters. Wrong -g id, or unpatched nvidia-uvm." >&2
    echo "Available: $(ls "$UVM_PROC" 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi
if [[ ! -x $BENCH ]]; then
    echo "error: $BENCH not found; run 'make uvm_tlb_bench' in bench/" >&2
    exit 1
fi

ORIG=$(cat "$PARAM")
trap 'echo "$ORIG" > "$PARAM"' EXIT

COUNTERS=(tlb_batch_targeted tlb_batch_targeted_ranges tlb_batch_invalidate_all tlb_batch_coalesced)

read_counters() {
    local name
    for name in "${COUNTERS[@]}"; do
        awk -v n="$name" '$1 == n { print $2 }' "$INFO"
    done
}

{
    printf 'round,coalesce,'
    "$BENCH" --header | tr -d '\n'
    printf ',%s' "${COUNTERS[@]}"
    printf '\n'
} > "$OUT"

# One unmeasured warm-up run
"$BENCH" "${BENCH_ARGS[@]}" > /dev/null

for ((round = 1; round <= ROUNDS; round++)); do
    if ((round % 2)); then order=(0 1); else order=(1 0); fi

    for setting in "${order[@]}"; do
        echo "$setting" > "$PARAM"

        mapfile -t before < <(read_counters)
        line=$("$BENCH" "${BENCH_ARGS[@]}" | tail -n 1)
        mapfile -t after < <(read_counters)

        row="$round,$setting,$line"
        for i in "${!COUNTERS[@]}"; do
            row+=",$((after[i] - before[i]))"
        done
        echo "$row" | tee -a "$OUT"
    done
done

echo
echo "Results: $OUT"
python3 "$HERE/compare.py" "$OUT" --group coalesce --baseline 0 \
    --metric tlb_batch_invalidate_all \
    --metric victim_concurrent_ms \
    --metric victim_slowdown_pct \
    --metric fault_kernel_ms \
    --metric cpu_evict_ms
