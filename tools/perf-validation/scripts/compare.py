#!/usr/bin/env python3
"""Compare A/B benchmark results with robust statistics.

Reads a CSV where one column names the variant of each run (for example
coalesce=0/1 or the regkey value) and other columns hold metrics. For each
metric it reports, per variant, the run count, median, mean and spread, then
the change of the treatment median relative to the baseline median with:

  - a 95% bootstrap confidence interval for that relative change
  - a two-sided Mann-Whitney U test p-value (normal approximation with tie
    correction), which does not assume normally distributed run times

A change is "significant" here when the CI excludes 0 and p < 0.05. Use at
least 8 runs per variant, interleaved (the ab_*.sh scripts do this), so slow
drift (thermals, background load) does not land on one variant.

Usage:
  compare.py RESULTS.csv --group COL --baseline VALUE \\
             --metric NAME[:lower|higher] [--metric ...]

  :lower  (default) means smaller is better, :higher means larger is better;
  it only affects the "better/worse" label.

Example:
  compare.py uvm_tlb_ab.csv --group coalesce --baseline 0 \\
      --metric victim_concurrent_ms --metric fault_kernel_ms \\
      --metric tlb_batch_invalidate_all
"""

import argparse
import csv
import math
import random
import statistics
import sys


def mann_whitney_p(a, b):
    """Two-sided Mann-Whitney U p-value, normal approximation, tie corrected."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")

    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(combined)
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = rank
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1

    r1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma_sq = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma_sq <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(sigma_sq)
    z = max(z, 0.0)
    return math.erfc(z / math.sqrt(2.0))


def bootstrap_ci(base, treat, iters=10000, seed=12345):
    """95% CI of (median(treat) / median(base) - 1)."""
    rng = random.Random(seed)
    ratios = []
    for _ in range(iters):
        mb = statistics.median(rng.choices(base, k=len(base)))
        mt = statistics.median(rng.choices(treat, k=len(treat)))
        if mb != 0:
            ratios.append(mt / mb - 1.0)
    if not ratios:
        return float("nan"), float("nan")
    ratios.sort()
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]


def fmt(v):
    if v != v:  # NaN
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--group", required=True, help="column that names the variant")
    ap.add_argument("--baseline", required=True, help="variant value treated as the baseline")
    ap.add_argument("--metric", action="append", required=True, help="NAME[:lower|higher]")
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    if args.group not in rows[0]:
        print(f"column {args.group!r} not found; have: {', '.join(rows[0])}", file=sys.stderr)
        return 1

    variants = sorted({r[args.group] for r in rows})
    if args.baseline not in variants:
        print(f"baseline {args.baseline!r} not among variants {variants}", file=sys.stderr)
        return 1
    treatments = [v for v in variants if v != args.baseline]

    for spec in args.metric:
        name, _, direction = spec.partition(":")
        direction = direction or "lower"
        if name not in rows[0]:
            print(f"\n{name}: column not found", file=sys.stderr)
            continue

        data = {}
        for v in variants:
            vals = []
            for r in rows:
                if r[args.group] == v and r[name] not in ("", None):
                    try:
                        vals.append(float(r[name]))
                    except ValueError:
                        pass
            data[v] = vals

        print(f"\n== {name} ({direction} is better)")
        print(f"   {'variant':<12} {'n':>3} {'median':>12} {'mean':>12} {'stdev':>12} {'min':>12} {'max':>12}")
        for v in variants:
            vals = data[v]
            if not vals:
                print(f"   {v:<12} {0:>3}")
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"   {v:<12} {len(vals):>3} {fmt(statistics.median(vals)):>12} {fmt(statistics.mean(vals)):>12} "
                  f"{fmt(sd):>12} {fmt(min(vals)):>12} {fmt(max(vals)):>12}")

        base = data[args.baseline]
        for t in treatments:
            treat = data[t]
            if len(base) < 2 or len(treat) < 2:
                print(f"   {t} vs {args.baseline}: need at least 2 runs per variant")
                continue
            mb, mt = statistics.median(base), statistics.median(treat)
            change = (mt / mb - 1.0) if mb else float("nan")
            lo, hi = bootstrap_ci(base, treat)
            p = mann_whitney_p(base, treat)
            significant = (lo > 0 or hi < 0) and p < 0.05
            if change != change or change == 0 or not significant:
                verdict = "no significant change"
            else:
                better = (change < 0) if direction == "lower" else (change > 0)
                verdict = "better" if better else "WORSE"
            print(f"   {t} vs {args.baseline}: median change {change * 100:+.2f}% "
                  f"(95% CI {lo * 100:+.2f}% .. {hi * 100:+.2f}%), Mann-Whitney p={p:.4f} -> {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
