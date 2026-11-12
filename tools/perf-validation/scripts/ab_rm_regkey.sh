#!/usr/bin/env bash
#
# A/B test of an RM registry key by reloading the nvidia module between runs.
#
# Used for the two RM changes, whose old behavior is restored by a regkey:
#   RmGspRpcPollHealthCheckIntervalUs  0 = old (check every poll), 100 = new
#   RmApiLockTraceReaders              1 = old (trace readers),    0 = new
#
# For every run it reloads nvidia (+ nvidia_uvm) with
# NVreg_RegistryDwords="<extra>;<key>=<value>", does one unmeasured warm-up
# run of the load command, then runs it again under a bpftrace script and
# records the load command's CSV line plus selected bpftrace fields. Values
# are interleaved in ABBA order across rounds.
#
# WARNING: this unloads the NVIDIA kernel modules. Use a headless machine with
# no display server, no CUDA processes and nvidia-persistenced stopped.
# NVreg_RegistryDwords given here replaces the one from /etc/modprobe.d; pass
# any keys you rely on with -x.
#
# Usage:
#   sudo ./ab_rm_regkey.sh -k KEY -a OLD_VALUE -b NEW_VALUE [-r rounds]
#        [-t bpftrace_script] [-f field,field,...] [-x "Key1=1;Key2=2"]
#        [-o out.csv] -- LOAD_COMMAND [args...]
#
# The load command must print a CSV header when given -H and one CSV result
# line as its last line of output (bench/nvml_rpc_bench does both).
#
# Examples:
#   # GSP RPC poll health checks
#   sudo ./ab_rm_regkey.sh -k RmGspRpcPollHealthCheckIntervalUs -a 0 -b 100 \
#        -t ../bpftrace/gsp_rpc_poll.bt \
#        -f rpc_wait_ns.count,rpc_wait_ns.avg,health_checks_in_rpc_poll,rm_ioctl_ns.avg \
#        -- ../bench/nvml_rpc_bench -p 1 -t 10
#
#   # API lock trace ring, many concurrent readers
#   sudo ./ab_rm_regkey.sh -k RmApiLockTraceReaders -a 1 -b 0 \
#        -t ../bpftrace/rm_api_lock.bt \
#        -f acquire_wait_ns_read.count,acquire_wait_ns_read.avg,hold_ns_read.avg \
#        -- ../bench/nvml_rpc_bench -p 32 -t 10

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

# Overridable for testing the script without a GPU
MODPROBE=${MODPROBE:-modprobe}
LSMOD=${LSMOD:-lsmod}
BPFTRACE=${BPFTRACE:-bpftrace}
BT_ATTACH_WAIT=${BT_ATTACH_WAIT:-3}

KEY=
VAL_A=
VAL_B=
ROUNDS=6
BT=
FIELDS=
EXTRA=
OUT=rm_regkey_ab.csv

while getopts "k:a:b:r:t:f:x:o:h" opt; do
    case "$opt" in
        k) KEY=$OPTARG ;;
        a) VAL_A=$OPTARG ;;
        b) VAL_B=$OPTARG ;;
        r) ROUNDS=$OPTARG ;;
        t) BT=$OPTARG ;;
        f) FIELDS=$OPTARG ;;
        x) EXTRA=$OPTARG ;;
        o) OUT=$OPTARG ;;
        *) sed -n '2,45p' "$0"; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift
LOAD=("$@")

if [[ -z $KEY || -z $VAL_A || -z $VAL_B || ${#LOAD[@]} -eq 0 ]]; then
    sed -n '2,45p' "$0"
    exit 2
fi
if [[ $EUID -ne 0 && $MODPROBE == modprobe ]]; then
    echo "error: must run as root (reloads kernel modules)" >&2
    exit 1
fi
if [[ -n $BT ]] && ! command -v "$BPFTRACE" > /dev/null; then
    echo "error: bpftrace not found" >&2
    exit 1
fi
if pgrep -x nvidia-persistenced > /dev/null; then
    echo "error: stop nvidia-persistenced first (systemctl stop nvidia-persistenced)" >&2
    exit 1
fi

IFS=',' read -r -a FIELD_LIST <<< "$FIELDS"

reload_nvidia() {
    local dwords=$1 m
    for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem nvidia; do
        if $LSMOD | awk '{print $1}' | grep -qx "$m"; then
            $MODPROBE -r "$m"
        fi
    done
    $MODPROBE nvidia NVreg_RegistryDwords="$dwords"
    $MODPROBE nvidia_uvm
}

# Ask bpftrace to exit and print its END output. bpftrace installs its own
# SIGINT handler; fall back to SIGTERM in case SIGINT stays ignored (background
# jobs of a non-interactive shell start with SIGINT ignored).
stop_bpftrace() {
    local pid=$1 i
    kill -INT "$pid" 2> /dev/null || return 0
    for ((i = 0; i < 100; i++)); do
        kill -0 "$pid" 2> /dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2> /dev/null; then
        kill -TERM "$pid" 2> /dev/null || true
    fi
    wait "$pid" 2> /dev/null || true
}

run_one() {
    local value=$1 round=$2 dwords bt_out line pid
    dwords="$KEY=$value"
    [[ -n $EXTRA ]] && dwords="$EXTRA;$dwords"

    reload_nvidia "$dwords"

    # Warm-up: the first client initializes the GPU, which is slow
    "${LOAD[@]}" > /dev/null

    if [[ -n $BT ]]; then
        bt_out=$(mktemp)
        "$BPFTRACE" -q "$BT" > "$bt_out" 2>&1 &
        pid=$!
        sleep "$BT_ATTACH_WAIT"   # let probes attach
        line=$("${LOAD[@]}" | tail -n 1)
        stop_bpftrace "$pid"
        if [[ ${#FIELD_LIST[@]} -gt 0 ]]; then
            line+=",$(python3 "$HERE/bt_fields.py" "$bt_out" "${FIELD_LIST[@]}")"
        fi
        rm -f "$bt_out"
    else
        line=$("${LOAD[@]}" | tail -n 1)
    fi

    echo "$round,$value,$line" | tee -a "$OUT"
}

{
    printf 'round,%s,' "$KEY"
    "${LOAD[@]}" -H | tr -d '\n'
    if [[ ${#FIELD_LIST[@]} -gt 0 ]]; then
        printf ',%s' "$(python3 "$HERE/bt_fields.py" --header "${FIELD_LIST[@]}")"
    fi
    printf '\n'
} > "$OUT"

for ((round = 1; round <= ROUNDS; round++)); do
    if ((round % 2)); then order=("$VAL_A" "$VAL_B"); else order=("$VAL_B" "$VAL_A"); fi
    for value in "${order[@]}"; do
        run_one "$value" "$round"
    done
done

# Leave the driver loaded with the new default behavior
reload_nvidia "$EXTRA"

echo
echo "Results: $OUT"
metrics=(--metric calls_per_sec:higher --metric p50_us --metric p99_us)
for f in "${FIELD_LIST[@]}"; do
    metrics+=(--metric "${f//./_}")
done
python3 "$HERE/compare.py" "$OUT" --group "$KEY" --baseline "$VAL_A" "${metrics[@]}"
