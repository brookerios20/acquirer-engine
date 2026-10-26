"""Compare ranking options on the same leak-free held-out deals.

Source of the backtest table in docs/decisions.md (decision 4) and the conviction hit
rates (decision 7). Sector similarity is computed from pre-2022 deals only. Test set:
2022-2024 deals; also a subset of test deals that look like the target profile (sector
similarity >= 0.6 to Healthcare Services, EV $75-500M).
Options: deal count, same-sector count, similarity-weighted deal count (no fitting),
and the 6-feature conditional logit trained on 2018-2021.
Standard library only. Usage: python3 -I ranking_compare.py <csv>
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
sectors = sorted({r["sector"] for r in rows})


def similarity(deals):
    vec = defaultdict(Counter)
    for r in deals:
        if r["acquirer_type"] == "Financial Sponsor":
            vec[r["sector"]][r["acquirer"]] += 1
    norm = {s: math.sqrt(sum(v * v for v in vec[s].values())) or 1 for s in sectors}
    return {a: {b: 1.0 if a == b else sum(vec[a][k] * vec[b][k] for k in vec[a]) / (norm[a] * norm[b]) for b in sectors} for a in sectors}


sim = similarity([r for r in rows if r["year"] < 2022])


def features(t, deals):
    log_ev = [math.log(d["ev"]) for d in deals]
    return [
        math.log1p(len(deals)),
        math.log1p(sum(d["sector"] == t["sector"] for d in deals)),
        st.mean(sim[t["sector"]][d["sector"]] for d in deals),
        -abs(math.log(t["ev"]) - st.median(log_ev)),
        t["year"] - max(d["year"] for d in deals),
        st.mean(d["target_ownership_pre"] in ("Private", "PE-Backed") for d in deals),
    ]


def choice_sets(targets):
    out = []
    for t in targets:
        past = defaultdict(list)
        for d in rows:
            if d["year"] < t["year"]:
                past[d["acquirer"]].append(d)
        if t["acquirer"] not in past:
            out.append(None)
            continue
        names = sorted(past)
        out.append((t, names, past, names.index(t["acquirer"])))
    return out


def fit(train, iters=400, lr=0.5, l2=0.01):
    X_all = [[features(t, past[a]) for a in names] for t, names, past, _ in (s for s in train if s)]
    flat = [x for X in X_all for x in X]
    mu = [st.mean(c) for c in zip(*flat)]
    sd = [st.stdev(c) or 1 for c in zip(*flat)]
    z = lambda x: [(v - m) / s for v, m, s in zip(x, mu, sd)]
    data = [([z(x) for x in X], s[3]) for X, s in zip(X_all, (s for s in train if s))]
    w = [0.0] * 6
    for _ in range(iters):
        grad = [-l2 * wi for wi in w]
        for X, y in data:
            u = [sum(wi * xi for wi, xi in zip(w, x)) for x in X]
            m = max(u)
            e = [math.exp(v - m) for v in u]
            tot = sum(e)
            for j in range(6):
                grad[j] += X[y][j] - sum(ei * x[j] for ei, x in zip(e, X)) / tot
        w = [wi + lr * g / len(data) for wi, g in zip(w, grad)]
    return lambda t, deals: sum(wi * xi for wi, xi in zip(w, z(features(t, deals))))


def evaluate(sets, score):
    hits = rr = 0.0
    n = 0
    for s in sets:
        n += 1
        if s is None:
            continue
        t, names, past, y = s
        u = [score(t, past[a]) + 1e-9 * i for i, a in enumerate(reversed(names))]  # tiny tie-break
        u = list(reversed(u))
        rank = 1 + sum(v > u[y] for v in u)
        hits += rank <= 10
        rr += 1 / rank
    return hits / n, rr / n, n


options = {
    "deal count": lambda t, ds: len(ds),
    "same-sector count": lambda t, ds: sum(d["sector"] == t["sector"] for d in ds) + 0.001 * len(ds),
    "similarity-weighted count": lambda t, ds: sum(sim[t["sector"]][d["sector"]] for d in ds),
    "logit, 6 features": fit(choice_sets([t for t in rows if 2018 <= t["year"] < 2022])),
}

test = choice_sets([t for t in rows if t["year"] >= 2022])
profile = [s for s, t in zip(test, [t for t in rows if t["year"] >= 2022])
           if sim["Healthcare Services"][t["sector"]] >= 0.6 and 75 <= t["ev"] <= 500]
for label, sets in (("All 2022-2024 deals", test), ("Profile-like 2022-2024 deals", profile)):
    print(f"\n{label}")
    for name, fn in options.items():
        hit, mrr, n = evaluate(sets, fn)
        print(f"  {name:<27} hit@10 {hit:5.1%}  MRR {mrr:.3f}  n={n}")

# Target top 10 under each option, using all deals and full-data similarity
sim_full = similarity(rows)
hist = defaultdict(list)
for d in rows:
    hist[d["acquirer"]].append(d)
target = {"sector": "Healthcare Services", "ev": 200.0, "year": 2025.0}
print("\nTarget top 10 (all deals):")
for name in ("deal count", "similarity-weighted count"):
    fn = (lambda t, ds: len(ds)) if name == "deal count" else (lambda t, ds: sum(sim_full[t["sector"]][d["sector"]] for d in ds))
    top = sorted(hist, key=lambda a: (-fn(target, hist[a]), a))[:10]
    print(f"  {name}: " + ", ".join(top))
other = {"sector": "Dental", "ev": 200.0, "year": 2025.0}
top = sorted(hist, key=lambda a: (-len(hist[a]), a))[:10]
print(f"  deal count, Dental target: " + ", ".join(top) + "  (same list for any target)")
top = sorted(hist, key=lambda a: (-sum(sim_full['Dental'][d['sector']] for d in hist[a]), a))[:10]
print(f"  similarity-weighted, Dental target: " + ", ".join(top))

# Conviction check on the similarity-weighted ranking: among each test deal's top 10, hit rate by level
def conviction(t, deals):
    near = [s for s, _ in sorted(sim[t["sector"]].items(), key=lambda kv: -kv[1]) if s != t["sector"]][:2]
    direct = [d for d in deals if d["outcome"] == "Closed" and d["sector"] in {t["sector"], *near}
              and 0.5 * t["ev"] <= d["ev"] <= 2 * t["ev"] and d["year"] >= t["year"] - 5]
    exact = sum(d["sector"] == t["sector"] for d in deals)
    level = 2 if len(direct) >= 3 else 1 if direct or exact >= 2 else 0
    if t["ev"] > 2 * max(d["ev"] for d in deals):
        level = max(0, level - 1)
    return ["Low", "Medium", "High"][level]


preds, hits = Counter(), Counter()
score = options["similarity-weighted count"]
for s in test:
    if s is None:
        continue
    t, names, past, y = s
    top = sorted(range(len(names)), key=lambda i: (-score(t, past[names[i]]), names[i]))[:10]
    for i in top:
        lvl = conviction(t, past[names[i]])
        preds[lvl] += 1
        hits[lvl] += i == y
print("\nConviction hit rate, similarity-weighted top 10 per test deal:")
for k in ("High", "Medium", "Low"):
    print(f"  {k:<6} {hits[k]}/{preds[k]} = {hits[k] / preds[k]:.1%}")
