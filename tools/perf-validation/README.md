# Validating the perf changes

This directory measures the three performance changes on this branch:

| # | Commit | What changed | Old behavior on the same build |
|---|--------|--------------|--------------------------------|
| 1 | `uvm: merge adjacent ranges in TLB invalidate batches` | Adjacent same-page-size ranges in a UVM TLB batch merge into one, instead of overflowing the 4-entry batch and invalidating the whole GPU VA space | `echo 0 > /sys/module/nvidia_uvm/parameters/uvm_perf_tlb_batch_coalesce` (runtime, no reload) |
| 2 | `rm/gsp: stop health-checking GSP on every RPC poll iteration` | GSP health check (2 PCIe register reads) runs at most every 100us while polling for an RPC reply, and not after the reply arrived; RM busy-wait loops now execute `cpu_relax()` | `NVreg_RegistryDwords="RmGspRpcPollHealthCheckIntervalUs=0"` (module reload). The `cpu_relax()` part has no switch; see below |
| 3 | `rm/rmapi: stop read-mode API lock users from writing the shared trace ring` | Read-mode RM API lock acquires no longer write the global lock trace ring | `NVreg_RegistryDwords="RmApiLockTraceReaders=1"` (module reload) |

Because every change can be switched back on the same build, A/B runs compare
exactly one variable. The bpftrace scripts only probe functions that exist in
the stock driver too, so the same scripts also compare the stock build against
this branch.

## What counts as proof

Each change is checked at three levels. A change is worth keeping when level 1
shows the mechanism works, and level 2 or 3 shows a statistically significant
improvement with no significant regression elsewhere.

1. **Mechanism.** The work the change removes is gone. This is a counter or a
   probe hit count, and it should move by a large factor. If it doesn't, the
   workload isn't exercising the path and levels 2 and 3 mean nothing.
2. **Operation.** The affected operation is faster: latency distributions
   from bpftrace.
3. **End to end.** A workload metric improves.

For levels 2 and 3, use at least 8 runs per variant, interleaved in ABBA order
(the scripts do this), and judge with `scripts/compare.py`. It reports the
median change with a 95% bootstrap confidence interval and a Mann-Whitney U
p-value, and only calls a result "better" or "WORSE" when the CI excludes 0
and p < 0.05.

## Setup

```sh
# Build and install this branch's modules (from the repository root)
make modules -j"$(nproc)"
sudo make modules_install -j"$(nproc)"   # or insmod the .ko files directly

# Benchmarks
cd tools/perf-validation/bench
make                     # uvm_tlb_bench needs nvcc; nvml_rpc_bench needs nvml.h + libnvidia-ml
```

Tools: `bpftrace` 0.16 or later, `perf` (for `perf c2c`), Python 3.

Reduce noise before measuring: an otherwise idle machine, the `performance`
CPU frequency governor, no other GPU clients, and benchmarks pinned to CPUs on
the GPU's NUMA node (`numactl --cpunodebind=N`).

### Correctness first

Run nvidia-uvm's built-in page tree test, which includes the new
`test_tlb_batch_coalesce()`. It runs in the kernel against fake GPUs and fake
TLB HALs, so it checks the exact invalidates a batch emits:

```sh
sudo modprobe -r nvidia_uvm
sudo modprobe nvidia_uvm uvm_enable_builtin_tests=1
./bench/uvm_page_tree_test          # prints "page tree test passed ..."
sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm   # back to normal
```

---

## 1. UVM TLB batch range merging

**Hypothesis.** When UVM maps or revokes 5 or more adjacent 4K or 64K pages in
a block, the stock driver invalidates every TLB entry of the GPU VA space.
Everything running in that process then refills its TLBs. Merging the ranges
turns that into one targeted invalidate.

**Mechanism (level 1).** Count whole-VA-space invalidates:

```sh
# Old behavior
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_tlb_batch_coalesce
sudo bpftrace bpftrace/uvm_tlb_invalidates.bt -c './bench/uvm_tlb_bench --mode managed'

# New behavior
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_tlb_batch_coalesce
sudo bpftrace bpftrace/uvm_tlb_invalidates.bt -c './bench/uvm_tlb_bench --mode managed'
```

`@invalidate_all` should drop by a large factor. The patched driver also keeps
cumulative per-GPU counters, readable at any time without tracing:

```sh
grep tlb_batch /proc/driver/nvidia-uvm/gpus/1/info
# tlb_batch_targeted / _targeted_ranges / _invalidate_all / _coalesced
```

If the old behavior shows few `invalidate_all`, the workload isn't hitting the
path on this GPU. Try `--touch 0.25` or `--touch 0.75`, or `--mode system` on
an HMM or ATS system, where GPU mappings always use 4K PTEs.

**Operation and end to end (levels 2 and 3).** `bench/uvm_tlb_bench` runs a
fault/evict loop on a buffer next to a "victim" kernel doing dependent random
reads over a large `cudaMalloc` buffer in the same process. The victim's run
time depends on GPU TLB hit rate, so it shows the collateral cost of
whole-VA-space invalidates. Run the full A/B, which flips the runtime
parameter between interleaved runs and records the counters for each run:

```sh
sudo scripts/ab_uvm_tlb.sh -r 10 -- --mode managed --mb 2048 --touch 0.5
sudo scripts/ab_uvm_tlb.sh -r 10 -o hmm.csv -- --mode system --mb 2048 --touch 0.5
```

Metrics, all medians:

| Column | Meaning | Expected direction |
|--------|---------|--------------------|
| `tlb_batch_invalidate_all` | whole-VA-space invalidates per run | large drop (level 1) |
| `victim_concurrent_ms`, `victim_slowdown_pct` | victim kernel time while faults are serviced, and its slowdown versus running alone | lower |
| `fault_kernel_ms` | time to fault the buffer back onto the GPU | lower or unchanged |
| `cpu_evict_ms` | CPU time pulling pages back, which revokes GPU mappings | unchanged or lower |

**Real applications.** Profile the application with
`nsys profile --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true`
once with the parameter at 0 and once at 1, and compare kernel durations in
migration-heavy phases. Read `/proc/driver/nvidia-uvm/gpus/<id>/info` before
and after each run to confirm the application hits the path.

**Overhead check.** Merging adds one comparison per queued range. For a
workload with no adjacent ranges, the counters show `tlb_batch_coalesced`
near 0 and run time should be unchanged.

---

## 2. GSP RPC poll loop

**Hypothesis.** Every synchronous GSP RPC (most RM controls and allocations on
GSP-based GPUs) ran the GSP health check on every poll iteration, including
the one that found the reply. That adds about two PCIe register reads (1-2us)
to every RPC. The poll loop also spun without a pause instruction.

**Mechanism (level 1).** `@health_checks_in_rpc_poll` divided by
`@rpc_wait_ns` count should go from at least 1 per RPC to about 0 for RPCs
shorter than 100us. `@crashcat_queue_reads` drops with it.

**Operation (level 2).** `@rpc_wait_ns` average and the `@rpc_wait_us`
histogram (1us buckets) should shift down by roughly the cost of the skipped
reads. `@rm_ioctl_ns` shows the end-to-end ioctl effect.

```sh
sudo bpftrace bpftrace/gsp_rpc_poll.bt -c './bench/nvml_rpc_bench -p 1 -t 10'
```

**End to end (level 3).** `nvml_rpc_bench` issues RM controls in a loop and
reports calls per second and per-call latency. A single process is latency
bound, which suits this change. The A/B script reloads the driver with the
regkey for each run, warms it up, and records the bench line plus the bpftrace
fields:

```sh
sudo scripts/ab_rm_regkey.sh -k RmGspRpcPollHealthCheckIntervalUs -a 0 -b 100 -r 8 \
     -t ../bpftrace/gsp_rpc_poll.bt \
     -f rpc_wait_ns.count,rpc_wait_ns.avg,health_checks_in_rpc_poll,rm_ioctl_ns.avg \
     -- ../bench/nvml_rpc_bench -p 1 -t 10
```

`ab_rm_regkey.sh` unloads the NVIDIA modules. Use a headless machine, stop
`nvidia-persistenced`, and pass any `NVreg_RegistryDwords` keys you normally
set with `-x "Key=val;..."`, because the script's value replaces the one from
`/etc/modprobe.d`.

**Separating the two parts.** The regkey only switches the health-check rate
limit. To isolate `cpu_relax()`, compare three configurations:

| Configuration | Health checks | `cpu_relax()` |
|---------------|---------------|---------------|
| stock 620.70.06 | every iteration | no |
| this branch, `RmGspRpcPollHealthCheckIntervalUs=0` | every iteration | yes |
| this branch, default | rate limited | yes |

The first pair isolates `cpu_relax()`, the second pair the rate limit.
`cpu_relax()` mostly helps the SMT sibling of a polling CPU. To measure that,
pin `nvml_rpc_bench` to one hardware thread and a CPU-bound loop to its
sibling (`lscpu -e` lists siblings by CORE), for example
`taskset -c <sibling> stress-ng --cpu 1 --metrics-brief -t 30`, and compare the
sibling's bogo-ops between the stock build and regkey=0.

**Risk check.** A dead GSP is now detected up to 100us later while polling,
against RPC timeouts of seconds. The poll loop still returns
`NV_ERR_RESET_REQUIRED` whenever draining the queue fails, exactly as before.

---

## 3. RM API lock trace ring

**Hypothesis.** Every read-mode RM API lock acquire and release wrote the same
global trace ring, so concurrent readers bounced its cache lines between CPUs.
That serializes the read side of an RW lock that exists to let them run in
parallel.

**Mechanism (level 1).** Look for false sharing on `g_RmApiLock` with
`perf c2c`. It reports cache lines with HITM (loads that hit a line modified in
another core's cache) and resolves them to symbols:

```sh
sudo perf c2c record -a -- ./bench/nvml_rpc_bench -p 32 -g all -t 10
sudo perf c2c report --stdio | less      # search for g_RmApiLock
```

With `RmApiLockTraceReaders=1`, lines in `g_RmApiLock` (which contains
`traceInfo`) should rank high in HITM. With the default they should drop out
or fall far down the list.

**Confirm the load uses read mode.** `rm_api_lock.bt` splits acquire wait and
hold times by requested mode. `@acquire_wait_ns_read` should dominate:

```sh
sudo bpftrace bpftrace/rm_api_lock.bt -c './bench/nvml_rpc_bench -p 32 -g all -t 10'
```

**End to end (levels 2 and 3).** Throughput of concurrent read-locked controls,
swept across process counts:

```sh
for p in 1 4 16 64; do
  sudo scripts/ab_rm_regkey.sh -k RmApiLockTraceReaders -a 1 -b 0 -r 8 -o api_lock_p$p.csv \
       -t ../bpftrace/rm_api_lock.bt \
       -f acquire_wait_ns_read.count,acquire_wait_ns_read.avg,hold_ns_read.avg \
       -- ../bench/nvml_rpc_bench -p "$p" -g all -t 10
done
```

Expect no change at `-p 1` and a growing gap as concurrency rises. Controls to
the same GPU also serialize on that GPU's lock, so the effect is largest with
`-g all` on multi-GPU machines. On a single GPU, "no significant change" is a
legitimate result.

---

## Files

| Path | Purpose |
|------|---------|
| `bpftrace/uvm_tlb_invalidates.bt` | Counts UVM TLB batches, targeted and whole-VA-space invalidates |
| `bpftrace/gsp_rpc_poll.bt` | GSP RPC wait time, health checks per RPC, RM ioctl latency |
| `bpftrace/rm_api_lock.bt` | RM API lock acquire wait and hold time by mode |
| `bench/uvm_tlb_bench.cu` | Fault/evict loop next to a TLB-sensitive victim kernel |
| `bench/nvml_rpc_bench.c` | Multi-process RM control load generator (NVML) |
| `bench/uvm_page_tree_test.c` | Runs nvidia-uvm's built-in page tree test |
| `scripts/ab_uvm_tlb.sh` | Interleaved A/B for change 1 using the runtime parameter |
| `scripts/ab_rm_regkey.sh` | Interleaved A/B for changes 2 and 3 using module reloads |
| `scripts/compare.py` | Median change, bootstrap CI, Mann-Whitney U |
| `scripts/bt_fields.py` | Extracts bpftrace END output into CSV fields |

## Status of these tools

Everything here was compiled or exercised without NVIDIA hardware:

- The driver changes build cleanly against Linux 6.8 headers.
- The changed `uvm_tlb_batch.c` passed a userspace harness of the in-kernel
  test cases plus 200,000 randomized batches. Every requested page stayed
  covered by an invalidate of the same page size, and no batch exceeded 4
  entries.
- The bpftrace scripts pass `bpftrace -d`.
- `nvml_rpc_bench.c` and `uvm_page_tree_test.c` compile with `-Wall -Wextra
  -Werror`. `uvm_tlb_bench.cu` compiles under clang's CUDA mode (host and
  sm_80 device), but has not been built with nvcc.
- The A/B scripts were dry-run end to end against stub drivers.
- `compare.py` matches SciPy's Mann-Whitney U p-values.

None of it has run on a GPU yet. The first real run should start with the
level 1 checks, which confirm each benchmark actually exercises its code path.
