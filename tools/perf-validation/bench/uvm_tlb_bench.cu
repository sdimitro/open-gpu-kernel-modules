// uvm_tlb_bench: measure what UVM fault servicing costs other GPU work through
// GPU TLB invalidates.
//
// Two things run in one process, so they share one GPU VA space:
//
//   faulter  Repeatedly migrates a buffer to the GPU by writing the first
//            --touch fraction of every 2MB block from a kernel, then pulls it
//            back with CPU writes. With --touch 0.5 each block is serviced as
//            a run of adjacent 64K (managed) or 4K (system/HMM) pages, and the
//            evictions revoke those runs again. That is the pattern where the
//            stock driver queues one TLB batch entry per page and falls back
//            to invalidating the whole VA space.
//
//   victim   A kernel doing dependent random reads over a large cudaMalloc
//            buffer on its own stream. Its run time depends on the GPU TLB hit
//            rate, so whole-VA-space invalidates triggered by the faulter slow
//            it down.
//
// The benchmark times the victim alone, then the victim while the faulter
// runs, and the faulter's own GPU fault-in time.
//
// Build:  make uvm_tlb_bench          (needs nvcc)
// Usage:  uvm_tlb_bench [options]
//   --mode managed|system  managed = cudaMallocManaged (default),
//                          system  = malloc (needs HMM or ATS)
//   --mb N                 faulted buffer size in MB (default 1024)
//   --touch F              fraction of each 2MB block to touch, 0 < F <= 1
//                          (default 0.5)
//   --iters N              fault/evict iterations (default 20)
//   --victim-mb N          victim buffer size in MB (default 4096)
//   --victim-reads N       dependent reads per victim thread (default 4096)
//   --victim-alone N       victim launches measured alone (default 20)
//   --header               print the CSV header and exit
//
// Output: one CSV line
//   mode,mb,touch,iters,victim_mb,victim_alone_ms,victim_concurrent_ms,
//   victim_slowdown_pct,victim_concurrent_launches,fault_kernel_ms,cpu_evict_ms
// All *_ms values are medians.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(call)                                                                 \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,    \
                    cudaGetErrorString(err_));                                      \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

static const size_t kBlockSize = 2ull << 20;   // UVM VA block size
static const size_t kHostPage = 4096;

struct Options
{
    std::string mode = "managed";
    size_t mb = 1024;
    double touch = 0.5;
    int iters = 20;
    size_t victim_mb = 4096;
    unsigned victim_reads = 4096;
    int victim_alone = 20;
};

// Write one word per 4K page in the first touch_bytes of every 2MB block.
__global__ void fault_in(unsigned *buf, size_t num_blocks, size_t pages_per_block)
{
    size_t total = num_blocks * pages_per_block;

    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        size_t block = i / pages_per_block;
        size_t page = i % pages_per_block;
        size_t word = (block * kBlockSize + page * kHostPage) / sizeof(unsigned);

        buf[word] += 1;
    }
}

// Dependent random reads: each load's address depends on the previous value,
// so the kernel is latency bound and sensitive to TLB misses.
__global__ void victim(const unsigned long long *buf, size_t n, unsigned reads,
                       unsigned long long seed, unsigned long long *sink)
{
    unsigned long long x = seed ^ ((blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x) *
                                   0x9E3779B97F4A7C15ull);
    unsigned long long acc = 0;

    for (unsigned i = 0; i < reads; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        // Random over a multi-GB buffer, so L1 hits are negligible and the
        // load cost is dominated by L2/DRAM latency and TLB misses.
        unsigned long long v = buf[(x ^ acc) % n];
        acc += v;
    }

    if (acc == 0x5eed5eed5eed5eedull)
        *sink = acc;
}

__global__ void fill(unsigned long long *buf, size_t n)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        buf[i] = i * 0x9E3779B97F4A7C15ull;
}

static double median(std::vector<double> v)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--mode managed|system] [--mb N] [--touch F] [--iters N]\n"
            "          [--victim-mb N] [--victim-reads N] [--victim-alone N] [--header]\n",
            argv0);
}

static bool parse(int argc, char **argv, Options &o)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](void) -> const char * { return (i + 1 < argc) ? argv[++i] : nullptr; };
        const char *v = nullptr;

        if (a == "--header") {
            printf("mode,mb,touch,iters,victim_mb,victim_alone_ms,victim_concurrent_ms,"
                   "victim_slowdown_pct,victim_concurrent_launches,fault_kernel_ms,cpu_evict_ms\n");
            exit(0);
        }
        if (!(v = next()))
            return false;

        if (a == "--mode")              o.mode = v;
        else if (a == "--mb")           o.mb = strtoull(v, nullptr, 0);
        else if (a == "--touch")        o.touch = atof(v);
        else if (a == "--iters")        o.iters = atoi(v);
        else if (a == "--victim-mb")    o.victim_mb = strtoull(v, nullptr, 0);
        else if (a == "--victim-reads") o.victim_reads = (unsigned)strtoul(v, nullptr, 0);
        else if (a == "--victim-alone") o.victim_alone = atoi(v);
        else                            return false;
    }

    return (o.mode == "managed" || o.mode == "system") && o.mb >= 2 && o.touch > 0.0 &&
           o.touch <= 1.0 && o.iters > 0 && o.victim_mb > 0 && o.victim_alone > 0;
}

int main(int argc, char **argv)
{
    Options o;
    int dev = 0;
    cudaDeviceProp prop;

    if (!parse(argc, argv, o)) {
        usage(argv[0]);
        return 2;
    }

    CHECK(cudaSetDevice(dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));

    int concurrent_managed = 0;
    int pageable_access = 0;
    CHECK(cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, dev));
    CHECK(cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, dev));

    if (o.mode == "managed" && !concurrent_managed) {
        fprintf(stderr, "GPU does not support concurrent managed access (on-demand faulting)\n");
        return 1;
    }
    if (o.mode == "system" && !pageable_access) {
        fprintf(stderr, "GPU cannot access pageable memory (needs HMM or ATS)\n");
        return 1;
    }

    // Faulted buffer, rounded to whole 2MB blocks
    size_t num_blocks = (o.mb << 20) / kBlockSize;
    size_t bytes = num_blocks * kBlockSize;
    size_t pages_per_block = std::max<size_t>(1, (size_t)(o.touch * (kBlockSize / kHostPage)));
    unsigned *buf = nullptr;

    if (o.mode == "managed") {
        CHECK(cudaMallocManaged(&buf, bytes));
    }
    else {
        if (posix_memalign((void **)&buf, kBlockSize, bytes) != 0) {
            fprintf(stderr, "posix_memalign(%zu) failed\n", bytes);
            return 1;
        }
    }
    memset(buf, 0, bytes);

    // Victim buffer, resident on the GPU
    size_t victim_n = (o.victim_mb << 20) / sizeof(unsigned long long);
    unsigned long long *vbuf = nullptr;
    unsigned long long *sink = nullptr;
    CHECK(cudaMalloc(&vbuf, victim_n * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&sink, sizeof(unsigned long long)));
    fill<<<prop.multiProcessorCount * 4, 256>>>(vbuf, victim_n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaStream_t fs, vs;
    CHECK(cudaStreamCreateWithFlags(&fs, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&vs, cudaStreamNonBlocking));

    // One block per SM leaves room for the faulter's kernel to run alongside
    dim3 vgrid(prop.multiProcessorCount), vblock(256);
    dim3 fgrid(prop.multiProcessorCount * 2), fblock(256);

    cudaEvent_t v0, v1, f0, f1;
    CHECK(cudaEventCreate(&v0));
    CHECK(cudaEventCreate(&v1));
    CHECK(cudaEventCreate(&f0));
    CHECK(cudaEventCreate(&f1));

    auto run_victim = [&](unsigned long long seed) -> double {
        CHECK(cudaEventRecord(v0, vs));
        victim<<<vgrid, vblock, 0, vs>>>(vbuf, victim_n, o.victim_reads, seed, sink);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(v1, vs));
        CHECK(cudaEventSynchronize(v1));
        return elapsed_ms(v0, v1);
    };

    // Warm up: populate the faulted buffer's GPU page tables and the victim's
    // TLB working set once.
    fault_in<<<fgrid, fblock, 0, fs>>>(buf, num_blocks, pages_per_block);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(fs));
    run_victim(1);

    // 1) Victim alone
    std::vector<double> victim_alone;
    for (int i = 0; i < o.victim_alone; ++i)
        victim_alone.push_back(run_victim(100 + i));

    // 2) Victim while the faulter runs
    std::atomic<bool> faulter_done(false);
    std::vector<double> fault_kernel_ms, cpu_evict_ms;

    std::thread faulter([&]() {
        for (int it = 0; it < o.iters; ++it) {
            // Pull the touched pages back to the CPU. This evicts them from
            // the GPU and revokes the GPU mappings.
            auto c0 = std::chrono::steady_clock::now();
            for (size_t b = 0; b < num_blocks; ++b) {
                volatile unsigned *base = buf + (b * kBlockSize) / sizeof(unsigned);
                for (size_t p = 0; p < pages_per_block; ++p)
                    base[(p * kHostPage) / sizeof(unsigned)] += 1;
            }
            auto c1 = std::chrono::steady_clock::now();
            cpu_evict_ms.push_back(std::chrono::duration<double, std::milli>(c1 - c0).count());

            // Fault them back in on the GPU
            CHECK(cudaEventRecord(f0, fs));
            fault_in<<<fgrid, fblock, 0, fs>>>(buf, num_blocks, pages_per_block);
            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(f1, fs));
            CHECK(cudaEventSynchronize(f1));
            fault_kernel_ms.push_back(elapsed_ms(f0, f1));
        }
        faulter_done.store(true);
    });

    std::vector<double> victim_concurrent;
    unsigned long long seed = 1000;
    while (!faulter_done.load())
        victim_concurrent.push_back(run_victim(seed++));

    faulter.join();
    CHECK(cudaDeviceSynchronize());

    double alone = median(victim_alone);
    double conc = median(victim_concurrent);

    printf("%s,%zu,%.3f,%d,%zu,%.4f,%.4f,%.2f,%zu,%.4f,%.4f\n",
           o.mode.c_str(),
           num_blocks * 2,
           o.touch,
           o.iters,
           o.victim_mb,
           alone,
           conc,
           alone > 0.0 ? (conc / alone - 1.0) * 100.0 : 0.0,
           victim_concurrent.size(),
           median(fault_kernel_ms),
           median(cpu_evict_ms));

    CHECK(cudaFree(vbuf));
    CHECK(cudaFree(sink));
    if (o.mode == "managed")
        CHECK(cudaFree(buf));
    else
        free(buf);

    return 0;
}
