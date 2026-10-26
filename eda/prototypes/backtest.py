"""Backtest: can the scoring parameters be calibrated on the data instead of hand-set?

Each historical deal becomes a pseudo-target. Using only deals announced *before* it, rank all
acquirers and check whether the real acquirer lands in the top 10 (hit@10) and at what rank (MRR).
Compares baselines against the relevance model, sweeping one parameter at a time from a base config.
Standard library only. Usage: python3 -I backtest.py <csv>
"""
import csv
import math
import statistics as st
import sys
from collections import Counter, defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    r["ev"] = float(r["deal_size_mm"])
    r["year"] = int(r["deal_year"]) + (int(r["deal_quarter"][1]) - 1) / 4
    r["margin"] = float(r["ebitda_margin_pct"])
sectors = sorted({r["sector"] for r in rows})

# --- sector similarity candidates (computed on all data; leakage is ~1 deal per sector, noted) ---
vec = defaultdict(Counter)
for r in rows:
    vec[r["sector"]][r["acquirer"]] += 1
norm = {s: math.sqrt(sum(v * v for v in vec[s].values())) for s in sectors}
overlap = {a: {b: sum(vec[a][k] * vec[b][k] for k in vec[a]) / (norm[a] * norm[b]) for b in sectors} for a in sectors}

feats = ["deal_size_mm", "ebitda_margin_pct", "revenue_growth_pct", "ev_ebitda_multiple", "ev_revenue_multiple"]
med = {s: [st.median(float(r[f]) for r in rows if r["sector"] == s) for f in feats] for s in sectors}
for s in sectors:
    med[s][0] = math.log(med[s][0])
mu = [st.mean(med[s][i] for s in sectors) for i in range(len(feats))]
sd = [st.stdev(med[s][i] for s in sectors) for i in range(len(feats))]
z = {s: [(med[s][i] - mu[i]) / sd[i] for i in range(len(feats))] for s in sectors}
dist = {a: {b: math.dist(z[a], z[b]) for b in sectors} for a in sectors}
scale = st.median(dist[a][b] for a in sectors for b in sectors if a != b)
profile = {a: {b: math.exp(-dist[a][b] / scale) for b in sectors} for a in sectors}

SECTOR_SIMS = {
    "exact": {a: {b: 1.0 if a == b else 0.05 for b in sectors} for a in sectors},
    "overlap": overlap,
    "profile": profile,
    "blend": {a: {b: (overlap[a][b] + profile[a][b]) / 2 for b in sectors} for a in sectors},
}
OUTCOME_W = {"Closed": 1.0, "Pending": 0.8, "Withdrawn": 0.5, "Terminated": 0.5, "Rumored": 0.3}

BASE = dict(sector="blend", sigma=math.log(2), half_life=4.0, k=5, profile_mult=0.85)


def score_model(target, past, p):
    sim = SECTOR_SIMS[p["sector"]][target["sector"]]
    per_acq = defaultdict(list)
    for d in past:
        rel = sim[d["sector"]]
        if p["sigma"]:
            rel *= math.exp(-(math.log(d["ev"] / target["ev"]) ** 2) / (2 * p["sigma"] ** 2))
        if p["half_life"]:
            rel *= 0.5 ** ((target["year"] - d["year"]) / p["half_life"])
        rel *= OUTCOME_W[d["outcome"]]
        m = p["profile_mult"]
        rel *= 1.0 if d["target_ownership_pre"] == target["target_ownership_pre"] else m
        rel *= 1.0 if (d["geography"] in ("National", "Multi-Regional")) == (target["geography"] in ("National", "Multi-Regional")) else m
        per_acq[d["acquirer"]].append(rel)
    return {a: sum(sorted(v, reverse=True)[: p["k"]]) for a, v in per_acq.items()}


def popularity(target, past, _):
    return Counter(d["acquirer"] for d in past)


def sector_popularity(target, past, _):
    c = Counter(d["acquirer"] for d in past if d["sector"] == target["sector"])
    for d in past:  # break ties / fill with overall popularity, tiny weight
        c[d["acquirer"]] += 0.001
    return c


def evaluate(scorer, p, targets):
    hits, rr, n, rand = 0, 0.0, 0, 0.0
    for t in targets:
        past = [d for d in rows if d["year"] < t["year"]]
        known = {d["acquirer"] for d in past}
        if t["acquirer"] not in known:  # first-time acquirer: no method can find it; count as miss
            n += 1
            rand += 10 / max(len(known), 1)
            continue
        scores = scorer(t, past, p)
        ranked = sorted(known, key=lambda a: (-scores.get(a, 0.0), a))
        rank = ranked.index(t["acquirer"]) + 1
        hits += rank <= 10
        rr += 1 / rank
        n += 1
        rand += 10 / len(known)
    return hits / n, rr / n, rand / n, n


def report(label, targets):
    print(f"\n=== {label} ===")
    hit, mrr, rand, n = evaluate(popularity, None, targets)
    print(f"{'random (expected)':<34} hit@10 {rand:5.1%}            n={n}")
    print(f"{'popularity (deal count)':<34} hit@10 {hit:5.1%}  MRR {mrr:.3f}")
    hit, mrr, *_ = evaluate(sector_popularity, None, targets)
    print(f"{'same-sector deal count':<34} hit@10 {hit:5.1%}  MRR {mrr:.3f}")
    hit, mrr, *_ = evaluate(score_model, BASE, targets)
    print(f"{'model, base config':<34} hit@10 {hit:5.1%}  MRR {mrr:.3f}")
    sweeps = {
        "sector": ["exact", "overlap", "profile", "blend"],
        "sigma": [math.log(1.5), math.log(2), math.log(3), None],
        "half_life": [2.0, 4.0, 8.0, None],
        "k": [1, 3, 5, 10, 1000],
        "profile_mult": [0.5, 0.85, 1.0],
    }
    for key, values in sweeps.items():
        for v in values:
            p = dict(BASE, **{key: v})
            hit, mrr, *_ = evaluate(score_model, p, targets)
            shown = f"{math.exp(v):.1f}x band" if key == "sigma" and v else v
            print(f"  {key + ' = ' + str(shown):<32} hit@10 {hit:5.1%}  MRR {mrr:.3f}")


all_targets = [t for t in rows if t["year"] >= 2018]
report("All deals 2018+ as targets", all_targets)
report("Targets like the profile: sector sim (blend) >= 0.45 to Healthcare Services, EV $75-500M, 2018+",
       [t for t in all_targets if SECTOR_SIMS["blend"]["Healthcare Services"][t["sector"]] >= 0.45 and 75 <= t["ev"] <= 500])
