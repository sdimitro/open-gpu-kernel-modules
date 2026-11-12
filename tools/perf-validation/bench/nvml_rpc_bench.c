/*
 * nvml_rpc_bench: generate RM control traffic from N processes and report
 * throughput and per-call latency.
 *
 * Every NVML query below is an RM control ioctl. On GSP-based GPUs most of
 * them are forwarded to GSP as synchronous RPCs, and many take the RM API
 * lock in read mode, which is what the "rm/gsp" and "rm/rmapi" changes
 * target. Run it under bpftrace/gsp_rpc_poll.bt or bpftrace/rm_api_lock.bt to
 * confirm what the load actually exercises on your system (RPCs per call,
 * read vs write lock mode) before comparing builds.
 *
 * Build:  make nvml_rpc_bench      (needs nvml.h and libnvidia-ml.so)
 * Usage:  nvml_rpc_bench [-p procs] [-t seconds] [-g gpu] [-q query]
 *           -p  number of worker processes (default 1)
 *           -t  run time in seconds (default 10)
 *           -g  GPU index, or "all" to spread workers round robin across
 *               every GPU (default 0)
 *           -q  power | clocks | temp | util | mix (default mix)
 *           -H  print the CSV header and exit
 *
 * Output: one CSV line
 *   query,procs,seconds,calls,calls_per_sec,p50_us,p99_us,max_p99_us
 * p50_us/p99_us are medians across processes of each process's own
 * percentile; max_p99_us is the worst process's p99.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <nvml.h>

#define MAX_SAMPLES (1u << 20)

typedef struct
{
    uint64_t calls;
    uint64_t errors;
    double   p50_us;
    double   p99_us;
} worker_result_t;

static double now_us(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x > y) - (x < y);
}

static double percentile(double *v, size_t n, double p)
{
    size_t idx;

    if (n == 0)
        return 0.0;

    idx = (size_t)(p * (double)(n - 1) + 0.5);
    return v[idx];
}

// One NVML query. Returns the NVML status of the last call made.
static nvmlReturn_t do_query(nvmlDevice_t dev, const char *query, uint64_t iter)
{
    unsigned int value;
    nvmlUtilization_t util;

    if (strcmp(query, "mix") == 0) {
        static const char *mix[] = { "power", "clocks", "temp", "util" };
        query = mix[iter % 4];
    }

    if (strcmp(query, "power") == 0)
        return nvmlDeviceGetPowerUsage(dev, &value);
    if (strcmp(query, "clocks") == 0)
        return nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &value);
    if (strcmp(query, "temp") == 0)
        return nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &value);
    if (strcmp(query, "util") == 0)
        return nvmlDeviceGetUtilizationRates(dev, &util);

    return NVML_ERROR_INVALID_ARGUMENT;
}

static int worker(int gpu, int worker_index, const char *query, double seconds, worker_result_t *out)
{
    nvmlDevice_t dev;
    nvmlReturn_t ret;
    double *samples;
    size_t nsamples = 0;
    double start;
    double deadline;
    uint64_t iter = 0;

    memset(out, 0, sizeof(*out));

    samples = malloc(sizeof(*samples) * MAX_SAMPLES);
    if (!samples)
        return -1;

    ret = nvmlInit_v2();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "nvmlInit_v2: %s\n", nvmlErrorString(ret));
        free(samples);
        return -1;
    }

    // gpu < 0: spread workers across all GPUs. Read-locked RM controls to
    // the same GPU still serialize on that GPU's lock, so cross-GPU spread is
    // what exposes contention on global RM state such as the API lock.
    if (gpu < 0) {
        unsigned int count = 0;

        ret = nvmlDeviceGetCount_v2(&count);
        if (ret != NVML_SUCCESS || count == 0) {
            fprintf(stderr, "nvmlDeviceGetCount_v2: %s\n", nvmlErrorString(ret));
            nvmlShutdown();
            free(samples);
            return -1;
        }
        gpu = worker_index % (int)count;
    }

    ret = nvmlDeviceGetHandleByIndex_v2((unsigned int)gpu, &dev);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "nvmlDeviceGetHandleByIndex_v2(%d): %s\n", gpu, nvmlErrorString(ret));
        nvmlShutdown();
        free(samples);
        return -1;
    }

    // Fail early on an unsupported query instead of measuring error paths
    ret = do_query(dev, query, 0);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "query '%s' failed: %s\n", query, nvmlErrorString(ret));
        nvmlShutdown();
        free(samples);
        return -1;
    }

    start = now_us();
    deadline = start + seconds * 1e6;

    for (;;) {
        double t0 = now_us();
        double t1;

        if (t0 >= deadline)
            break;

        ret = do_query(dev, query, iter++);
        t1 = now_us();

        if (ret != NVML_SUCCESS)
            out->errors++;

        out->calls++;
        if (nsamples < MAX_SAMPLES)
            samples[nsamples++] = t1 - t0;
    }

    qsort(samples, nsamples, sizeof(*samples), cmp_double);
    out->p50_us = percentile(samples, nsamples, 0.50);
    out->p99_us = percentile(samples, nsamples, 0.99);

    nvmlShutdown();
    free(samples);
    return 0;
}

static void usage(const char *argv0)
{
    fprintf(stderr, "usage: %s [-p procs] [-t seconds] [-g gpu|all] [-q power|clocks|temp|util|mix] [-H]\n", argv0);
}

int main(int argc, char **argv)
{
    int procs = 1;
    double seconds = 10.0;
    int gpu = 0;
    const char *query = "mix";
    worker_result_t *results;
    double *p50s;
    double *p99s;
    uint64_t total_calls = 0;
    uint64_t total_errors = 0;
    double max_p99 = 0.0;
    int failed = 0;
    int opt;
    int i;

    while ((opt = getopt(argc, argv, "p:t:g:q:H")) != -1) {
        switch (opt) {
            case 'p': procs = atoi(optarg); break;
            case 't': seconds = atof(optarg); break;
            case 'g': gpu = (strcmp(optarg, "all") == 0) ? -1 : atoi(optarg); break;
            case 'q': query = optarg; break;
            case 'H':
                printf("query,procs,seconds,calls,calls_per_sec,p50_us,p99_us,max_p99_us\n");
                return 0;
            default:
                usage(argv[0]);
                return 2;
        }
    }

    if (procs < 1 || seconds <= 0.0) {
        usage(argv[0]);
        return 2;
    }

    results = calloc((size_t)procs, sizeof(*results));
    p50s = calloc((size_t)procs, sizeof(*p50s));
    p99s = calloc((size_t)procs, sizeof(*p99s));
    if (!results || !p50s || !p99s)
        return 1;

    // Each worker writes its result struct to its own pipe
    {
        int (*fds)[2] = calloc((size_t)procs, sizeof(*fds));
        pid_t *pids = calloc((size_t)procs, sizeof(*pids));

        if (!fds || !pids)
            return 1;

        for (i = 0; i < procs; i++) {
            if (pipe(fds[i]) != 0) {
                perror("pipe");
                return 1;
            }

            pids[i] = fork();
            if (pids[i] < 0) {
                perror("fork");
                return 1;
            }

            if (pids[i] == 0) {
                worker_result_t r;
                int rc;

                close(fds[i][0]);
                rc = worker(gpu, i, query, seconds, &r);
                if (write(fds[i][1], &r, sizeof(r)) != (ssize_t)sizeof(r))
                    rc = -1;
                close(fds[i][1]);
                _exit(rc == 0 ? 0 : 1);
            }

            close(fds[i][1]);
        }

        for (i = 0; i < procs; i++) {
            int status;

            if (read(fds[i][0], &results[i], sizeof(results[i])) != (ssize_t)sizeof(results[i]))
                failed++;
            close(fds[i][0]);

            if (waitpid(pids[i], &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
                failed++;
        }

        free(fds);
        free(pids);
    }

    if (failed) {
        fprintf(stderr, "%d worker(s) failed\n", failed);
        return 1;
    }

    for (i = 0; i < procs; i++) {
        total_calls += results[i].calls;
        total_errors += results[i].errors;
        p50s[i] = results[i].p50_us;
        p99s[i] = results[i].p99_us;
        if (results[i].p99_us > max_p99)
            max_p99 = results[i].p99_us;
    }

    qsort(p50s, (size_t)procs, sizeof(*p50s), cmp_double);
    qsort(p99s, (size_t)procs, sizeof(*p99s), cmp_double);

    if (total_errors)
        fprintf(stderr, "warning: %llu NVML calls returned errors\n", (unsigned long long)total_errors);

    printf("%s,%d,%.1f,%llu,%.1f,%.2f,%.2f,%.2f\n",
           query,
           procs,
           seconds,
           (unsigned long long)total_calls,
           (double)total_calls / seconds,
           percentile(p50s, (size_t)procs, 0.5),
           percentile(p99s, (size_t)procs, 0.5),
           max_p99);

    free(results);
    free(p50s);
    free(p99s);
    return 0;
}
