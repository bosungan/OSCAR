import csv, sys
from collections import defaultdict
agg = defaultdict(list); unit = {}
r = csv.reader(sys.stdin)
hdr = next(r)
mi = hdr.index("Metric Name"); vi = hdr.index("Metric Value"); ui = hdr.index("Metric Unit")
for row in r:
    if len(row) <= vi:
        continue
    name = row[mi]; val = row[vi].replace(",", "")
    if not name:
        continue
    try:
        agg[name].append(float(val)); unit[name] = row[ui]
    except ValueError:
        pass
for k in sorted(agg):
    v = agg[k]
    if not v:
        continue
    print("  %-60s %14.4f  %-6s (n=%d)" % (k, sum(v)/len(v), unit.get(k, ""), len(v)))
