#!/usr/bin/env python3
"""Extract numbers from bpftrace END output as CSV fields.

bpftrace prints scalar maps as
    @name: 1234
and stats() maps as
    @name: count 12, average 3456, total 41472

For each requested field this prints one comma-separated value:
    name          scalar value (or the stats count)
    name.count    stats count
    name.avg      stats average
    name.total    stats total
Missing maps (bpftrace omits maps that were never written) print 0.

Usage: bt_fields.py OUTPUT_FILE FIELD [FIELD...]
       bt_fields.py --header FIELD [FIELD...]
"""

import re
import sys

SCALAR = re.compile(r"^@(\w+): (-?\d+)\s*$")
STATS = re.compile(r"^@(\w+): count (\d+), average (-?\d+), total (-?\d+)\s*$")


def parse(path):
    values = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STATS.match(line)
            if m:
                name, count, avg, total = m.groups()
                values[name] = count
                values[name + ".count"] = count
                values[name + ".avg"] = avg
                values[name + ".total"] = total
                continue
            m = SCALAR.match(line)
            if m:
                values[m.group(1)] = m.group(2)
    return values


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    if argv[1] == "--header":
        print(",".join(f.replace(".", "_") for f in argv[2:]))
        return 0

    values = parse(argv[1])
    print(",".join(values.get(f, "0") for f in argv[2:]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
