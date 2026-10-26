"""Can sector similarity be derived from the data instead of hand-set?

Two candidate signals:
  1. Buyer overlap: sectors are similar if the same acquirers buy in both (cosine on acquirer counts).
  2. Financial profile: sectors are similar if their deals look alike (size, margin, growth, multiples).
Standard library only. Usage: python3 -I sector_similarity.py <csv>
"""
import csv
import math
import statistics as st
import sys
from collections import Counter, defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
sectors = sorted({r["sector"] for r in rows})
short = {s: s[:10] for s in sectors}


def show(title, sim):
    print(title)
    print(" " * 22 + " ".join(f"{short[s]:>10}" for s in sectors))
    for a in sectors:
        print(f"{a:>22}" + " ".join(f"{sim[a][b]:>10.2f}" for b in sectors))
    print()


def overlap(subset):
    vec = defaultdict(Counter)
    for r in subset:
        vec[r["sector"]][r["acquirer"]] += 1
    norm = {s: math.sqrt(sum(v * v for v in vec[s].values())) or 1 for s in sectors}
    return {a: {b: sum(vec[a][k] * vec[b][k] for k in vec[a]) / (norm[a] * norm[b]) for b in sectors} for a in sectors}


show("Buyer overlap (all acquirers)", overlap(rows))
show("Buyer overlap (sponsors only)", overlap([r for r in rows if r["acquirer_type"] == "Financial Sponsor"]))
show("Buyer overlap (strategics only)", overlap([r for r in rows if r["acquirer_type"] == "Strategic"]))

feats = ["deal_size_mm", "ebitda_margin_pct", "revenue_growth_pct", "ev_ebitda_multiple", "ev_revenue_multiple"]
med = {s: {} for s in sectors}
for s in sectors:
    sub = [r for r in rows if r["sector"] == s]
    for f in feats:
        vals = [float(r[f]) for r in sub]
        med[s][f] = math.log(st.median(vals)) if f == "deal_size_mm" else st.median(vals)
print("Sector medians (log EV, margin, growth, EV/EBITDA, EV/Rev):")
for s in sectors:
    print(f"{s:>22}", " ".join(f"{med[s][f]:>7.2f}" for f in feats))
print()
mu = {f: st.mean(med[s][f] for s in sectors) for f in feats}
sd = {f: st.stdev(med[s][f] for s in sectors) for f in feats}
z = {s: [(med[s][f] - mu[f]) / sd[f] for f in feats] for s in sectors}
dist = {a: {b: math.dist(z[a], z[b]) for b in sectors} for a in sectors}
scale = st.median(dist[a][b] for a in sectors for b in sectors if a != b)
show("Financial-profile similarity exp(-d / median d)", {a: {b: math.exp(-dist[a][b] / scale) for b in sectors} for a in sectors})
