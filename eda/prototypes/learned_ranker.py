"""Learn the ranking weights from the deals themselves (conditional logit), instead of hand-setting them.

Each historical deal is a "choice": out of all acquirers active before it, one bought it.
Features describe the (target, acquirer) pair using only deals before the target. Weights are fit
by maximizing the likelihood that the real acquirer is chosen. Train on 2018-2021, test on 2022-2024.
Standard library only. Usage: python3 -I learned_ranker.py <csv>
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

# Data-derived sector similarity: buyer overlap among sponsors (strategics are single-sector).
vec = defaultdict(Counter)
for r in rows:
    if r["acquirer_type"] == "Financial Sponsor":
        vec[r["sector"]][r["acquirer"]] += 1
norm = {s: math.sqrt(sum(v * v for v in vec[s].values())) for s in sectors}
sim = {a: {b: 1.0 if a == b else sum(vec[a][k] * vec[b][k] for k in vec[a]) / (norm[a] * norm[b]) for b in sectors} for a in sectors}

FEATURES = ["log_activity", "same_sector", "sector_fit", "size_fit", "years_since_last", "private_share"]


def features(t, deals):
    n = len(deals)
    log_ev = [math.log(d["ev"]) for d in deals]
    return [
        math.log1p(n),                                                      # how active overall
        math.log1p(sum(d["sector"] == t["sector"] for d in deals)),         # deals in the exact sector
        st.mean(sim[t["sector"]][d["sector"]] for d in deals),              # average sector closeness
        -abs(math.log(t["ev"]) - st.median(log_ev)),                        # size distance from typical deal
        t["year"] - max(d["year"] for d in deals),                          # staleness
        st.mean(d["target_ownership_pre"] in ("Private", "PE-Backed") for d in deals),
    ]


def choice_sets(targets):
    sets = []
    for t in targets:
        past = defaultdict(list)
        for d in rows:
            if d["year"] < t["year"]:
                past[d["acquirer"]].append(d)
        if t["acquirer"] not in past:
            sets.append(None)  # first-time acquirer: unrankable, counts as a miss
            continue
        names = sorted(past)
        sets.append((names, [features(t, past[a]) for a in names], names.index(t["acquirer"])))
    return sets


def standardize(train):
    allx = [x for s in train if s for x in s[1]]
    mu = [st.mean(c) for c in zip(*allx)]
    sd = [st.stdev(c) or 1 for c in zip(*allx)]
    return lambda x: [(v - m) / s for v, m, s in zip(x, mu, sd)]


def fit(train, z, iters=400, lr=0.5, l2=0.01):
    w = [0.0] * len(FEATURES)
    data = [([z(x) for x in s[1]], s[2]) for s in train if s]
    for _ in range(iters):
        grad = [-l2 * wi for wi in w]
        for X, y in data:
            u = [sum(wi * xi for wi, xi in zip(w, x)) for x in X]
            m = max(u)
            e = [math.exp(v - m) for v in u]
            tot = sum(e)
            for j in range(len(w)):
                grad[j] += X[y][j] - sum(ei * x[j] for ei, x in zip(e, X)) / tot
        w = [wi + lr * g / len(data) for wi, g in zip(w, grad)]
    return w


def evaluate(sets, scorer):
    hits = rr = 0.0
    for s in sets:
        if s is None:
            continue
        names, X, y = s
        u = scorer(X)
        rank = 1 + sum(v > u[y] for v in u)
        hits += rank <= 10
        rr += 1 / rank
    return hits / len(sets), rr / len(sets), len(sets)


train_t = [t for t in rows if 2018 <= t["year"] < 2022]
test_t = [t for t in rows if t["year"] >= 2022]
train, test = choice_sets(train_t), choice_sets(test_t)
z = standardize(train)
w = fit(train, z)

print("Learned weights (standardized features; bigger magnitude = matters more):")
for f, wi in zip(FEATURES, w):
    print(f"  {f:<18} {wi:+.2f}")
learned = lambda X: [sum(wi * xi for wi, xi in zip(w, z(x))) for x in X]
popular = lambda X: [x[0] for x in X]
for label, sets in [("train 2018-2021", train), ("test 2022-2024", test)]:
    print(f"\n{label}")
    for name, fn in [("popularity", popular), ("learned", learned)]:
        hit, mrr, n = evaluate(sets, fn)
        print(f"  {name:<11} hit@10 {hit:5.1%}  MRR {mrr:.3f}  n={n}")

# Apply to the target profile, using all deals as history
target = {"sector": "Healthcare Services", "ev": 200.0, "year": 2025.0}
hist = defaultdict(list)
for d in rows:
    hist[d["acquirer"]].append(d)
scored = sorted(((sum(wi * xi for wi, xi in zip(w, z(features(target, ds)))), a) for a, ds in hist.items()), reverse=True)
print("\nTop 15 for Healthcare Services ~$200M (learned weights):")
for s, a in scored[:15]:
    ds = hist[a]
    print(f"  {s:+.2f}  {a:<26} {ds[0]['acquirer_type']:<17} deals={len(ds):<3} HS={sum(d['sector'] == 'Healthcare Services' for d in ds)}")

# --- Conviction lift: probability vs. an average acquirer, overall and within buyer type ---
u = {a: s for s, a in scored}
mx = max(u.values())
p = {a: math.exp(v - mx) for a, v in u.items()}
tot = sum(p.values())
p = {a: v / tot for a, v in p.items()}
types = {a: ds[0]["acquirer_type"] for a, ds in hist.items()}
print(f"\nLift for the target profile ({len(p)} acquirers; average p = {1 / len(p):.2%})")
for t in ["Financial Sponsor", "Strategic"]:
    group = sorted((a for a in p if types[a] == t), key=lambda a: -p[a])
    share = sum(p[a] for a in group)
    print(f"\n{t}: {len(group)} acquirers, {share:.0%} of total probability")
    print(f"  {'acquirer':<26} {'p':>6} {'lift vs all':>12} {'lift in type':>13}")
    for a in group[:8]:
        print(f"  {a:<26} {p[a]:6.2%} {p[a] * len(p):11.1f}x {p[a] / share * len(group):12.1f}x")
