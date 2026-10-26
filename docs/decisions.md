# Design Decisions

Each entry records what was decided, why, and what it costs. Entries follow the order of the pipeline. "The EDA" refers to [`eda/FINDINGS.md`](../eda/FINDINGS.md), which lists each data finding and the notebook check behind it.

## 1. Layered architecture with constructor injection

### Decision

| Folder | Responsibility |
|---|---|
| `controllers/` | CLI entry point and composition root: parses arguments, builds every object, calls the service, hands results to a view |
| `services/` | Business logic: scoring, evidence building, one-pager generation, validation, and one orchestrating service |
| `models/` | Plain dataclasses shared by all layers. No I/O |
| `repositories/` | `TransactionRepository` interface and `CsvTransactionRepository` |
| `clients/` | `LlmClient` interface, `AnthropicClient` (live), `RecordingClient` (live, saves responses), and `RecordedClient` (replays them) |
| `prompts/` | Prompt templates, kept apart from the clients so the clients stay provider-only |
| `views/` | Write `AcquirerOnePager` objects and the run log to output files |

The full folder structure is in the [appendix](#appendix-folder-structure).

Dependency rules:

1. Constructor injection everywhere. No class builds its own dependencies. The controller wires everything.
2. A `Protocol` for each dependency that does I/O, calls an outside system, or has two implementations. Today: `TransactionRepository`, `LlmClient`, and `AcquirerRankingModel` (decision 4). Later candidates: a Databricks table, MCP tools, a CRM.
3. Deterministic services are passed as concrete classes and tested with real instances on fixture data. `unittest.mock.create_autospec` covers the rare case where one needs isolating.

The diagram is in [architecture.md](architecture.md).

### Rationale

Constructor injection makes every class testable without patching, and the controller is the one place that shows how the system is assembled. The data source and the LLM are slow, external, or nondeterministic, so tests replace them with fakes through their `Protocol`s. The deterministic services give the same answer every time, so tests run the real code. In Python a concrete class can still be injected, autospecced, or swapped later.

### Tradeoff

A service without a `Protocol` can't take a hand-written fake without a type checker objecting. Adding one later is one `Protocol` and one type annotation, with no caller changes. One set of models is shared across layers, which keeps the prototype small. A layer that later needs its own shape would need its own model and some refactoring.

## 2. The target profile is a CLI input, with stated readings of each trait

### Decision

`TargetProfile` comes from CLI flags whose defaults are the brief's target, so `acquirers` with no flags reproduces it:

| Flag | Default | Reading |
|---|---|---|
| `--sector` | Healthcare Services | Must match a sector in the CSV |
| `--ev` | 200 | Enterprise value in $M |
| `--ownership` | Private | "Private" means `Private` only. A sponsor-owned target would be described as PE-backed, and that changes who is selling |
| `--margin` | Top quartile of the sector | "Strong EBITDA margins" means at or above the sector's 75th percentile (16.8% for Healthcare Services) |
| `--region` | None | "Regional" means not `National` or `Multi-Regional`. No specific region is assumed unless one is passed |

`TargetProfileService` checks each flag against the values in the data and resolves the margin default. The floor is rounded to the CSV's one decimal, so a deal at exactly 16.8% counts as strong.

"Mid-market" is a label carried by the size. Sector drives the ranking (decision 4), and sector and EV define direct precedents for conviction (decision 7). Ownership, margin, and region affect neither, because the EDA found no acquirer-level signal for them (decision 7). They appear in the prompt's target profile and in each acquirer's evidence (decision 6), where the acquirer's history with similar targets is shown next to the peer baseline.

### Rationale

The brief's profile is vague in three places. Writing each reading down lets a banker correct it with one flag. Flags with defaults also cover the stretch goal of arbitrary targets at almost no cost.

### Tradeoff

Each reading is a judgment a banker might make differently, especially "private." Acquirers aren't filtered by geography, because the EDA found no regional concentration among strategics (p = 0.54).

## 3. Data rules live in the repository

### Decision

`CsvTransactionRepository` applies the rules found in the [EDA](../eda/FINDINGS.md#data-quality-issues-ranked-by-impact), so services never handle raw quirks:

- Use `ev_ebitda_multiple` and `ebitda_margin_pct` as reported and drop `target_ebitda_mm`. EV, revenue, and EV/Revenue reconcile in every row; the EBITDA fields disagree by 18–28% in a typical row.
- Ignore `sub_sector`, which equals `sector` in every row.
- Accept `Rumored` as a fifth outcome (10 rows). It counts as buyer interest and is excluded from comps.
- Keep names as reported (for example, `Cerner (Oracle)`).

Comps use closed deals only and always show their sample size. The closest set (closed Healthcare Services deals at $100–400M) has 10 deals, median EV/EBITDA 12.1x.

### Rationale

Keeping the rules in the repository puts them in one place, and tests can check them against fixture rows.

### Tradeoff

The rules are specific to this file. Another data source gets its own repository implementation.

## 4. Acquirers are ranked by similarity-weighted deal count

### Decision

`CandidateScoringService` ranks acquirers through an `AcquirerRankingModel` interface:

```python
class AcquirerRankingModel(Protocol):
    def fit(self, transactions: list[Transaction]) -> None: ...
    def score(self, target: TargetProfile, candidates: list[str]) -> list[RankingScore]: ...
# RankingScore: acquirer, score, contributions (score per sector)
```

- `SimilarityWeightedCount` (default). An acquirer's score is the sum, over all its past deals, of how similar each deal's sector is to the target's. A Healthcare Services deal counts 1.0, a Physician Groups deal 0.83, a Health IT deal 0.44. `fit` stores the deals and computes sector similarity.
- `DealCountBaseline`, which ranks by number of deals. It returns the same list for every target, so it serves only as the bar to beat in the backtest.

Ties are broken by name. Each one-pager shows the score split by sector, for example "Nordic Capital: 4 Healthcare Services deals × 1.0, 10 Physician Groups × 0.83, 3 Behavioral Health × 0.77, ...".

#### Sector similarity

`SectorSimilarityService` represents each sector by how often each sponsor bought there and takes the cosine between sectors, computed from the deals it is given. Strategics are excluded because 90 of 92 bought in one sector only. On the full CSV, closest to Healthcare Services: Physician Groups 0.83, Behavioral Health 0.77, Medical Devices 0.76, Dental 0.72.

#### Validation

`BacktestService` (the `backtest` command) holds out every 2022–2024 deal. For each one, it fits the model on all deals announced in an earlier quarter, ranks the acquirers active by then, and records where the real buyer ranked. A buyer with no earlier deal counts as a miss. Because `fit()` computes sector similarity, similarity is refit from those earlier deals at every held-out deal, so no test-period deal after the one being ranked can leak in. "Profile-like" deals are those in sectors with similarity ≥ 0.6 to Healthcare Services (computed from pre-2022 deals) at $75–500M.

| Method | All: Hit@10 | All: MRR | Profile-like: Hit@10 | Profile-like: MRR |
|---|---|---|---|---|
| Deal count | 40.1% | 0.119 | 52.3% | 0.149 |
| Similarity-weighted count | 38.0% | 0.132 | 50.0% | 0.183 |

n = 142 (all) and 44 (profile-like). In deals, deal count puts the real buyer in the top 10 for 57 and 23, and the similarity-weighted count for 54 and 22. A replacement model has to beat the current one on this test.

#### Note: prototype numbers

`eda/prototypes/ranking_compare.py` ran the comparison before the command existed. It fixed sector similarity at pre-2022 deals for every held-out deal, which is also leak-free, and it tested two methods the engine doesn't implement. Its deal-count row matches the command exactly; its similarity-weighted row differs by 2 deals on all deals and 1 on profile-like deals.

| Method (prototype) | All: Hit@10 | All: MRR | Profile-like: Hit@10 | Profile-like: MRR |
|---|---|---|---|---|
| Deal count | 40.1% | 0.119 | 52.3% | 0.149 |
| Same-sector count | 38.7% | 0.143 | 47.7% | 0.188 |
| Similarity-weighted count | 39.4% | 0.132 | 52.3% | 0.173 |
| Conditional logit, 6 features | 39.4% | 0.171 | 43.2% | 0.188 |

### Rationale

No method beats deal count at putting the real buyer in the top 10, and the gaps are a few deals out of 142 or 44, within noise. Fifteen sponsors did 58% of the deals, so every method returns mostly the same large sponsors and differs mainly in order. Given that, the choice rests on simplicity, explanation, and responding to the target. The similarity-weighted count is 3 deals behind deal count on all deals and 1 behind on profile-like deals. It ranks the real buyer higher (MRR 0.132 vs. 0.119 on all deals and 0.183 vs. 0.149 on profile-like deals), changes with the target's sector, has no learned weights, and explains itself in one sentence. Plain deal count returns an identical list for every target, which reads as a hardcoded list.

### Alternatives considered

- *Conditional logit* (six features fit by maximum likelihood): the best MRR on all deals in the prototype, but the worst hit@10 on profile-like deals, more code to build and defend, and its earlier advantage (44.4% hit@10) came from sector similarity leaking test-period deals into training. It is the next model to try, as another `AcquirerRankingModel`, adopted only if it beats this one in the backtest, ideally on real deal data.
- *Hand-weighted scoring* (sector table, size kernel, recency half-life): 37.0% hit@10 against deal count's 40.7% across all 2018+ deals (`eda/prototypes/backtest.py`), and removing its size or recency factors improved it.

### Tradeoff

- On hit@10 it trails plain deal count by 3 of 142 held-out deals.
- The score ignores deal size and recency. In the backtests neither helped rank buyers; size enters through conviction (decision 7) and the size-stretch risk (decision 10).
- Similarity is computed from sponsors and applied to strategics too, so a Behavioral Health operator scores as close to the target. The highest-scoring strategics here have no Healthcare Services deals; decision 5's eligibility rule keeps them off the list.
- The score rewards activity, so the most active sponsors rank highest, as they do in the data. The README states the sponsor skew.
- Acquirers with no history can't be ranked.

## 5. Ranking decides the buyer mix by default, with an opt-in strategic floor

### Decision

By default the shortlist is the top 10 by score. `--min-strategics N` guarantees at least N strategic buyers:

1. Take the top 10 by score.
2. If it has fewer than N strategics, replace the lowest-ranked sponsors with the highest-scoring eligible strategics not on the list.
3. Order by score. Reserved entries have `reserved_slot = True`, and the one-pager says so.
4. If fewer than N strategics are eligible, include all of them and warn (for example, "requested 4 strategics, 3 eligible"). Eligible means at least one deal in the target's sector.

Reserved entries keep the conviction their evidence earns. The flag is passed to `CandidateScoringService` separately from `TargetProfile`, because it controls how the list is built.

### Rationale

Strategics have 1–6 deals each and will rarely make the top 10 on evidence. Banker buyer lists usually include both types, so a banker can ask for strategics without the system hiding that they scored lower. A floor is simpler and more predictable than a fixed ratio. The eligibility rule matters here: the highest-scoring strategics are Behavioral Health operators (Lyra Health, Universal Health Services) and Home Health operators (CenterWell Home Health, VNS Health) with no Healthcare Services deals.

### Tradeoff

The default list for this target is all sponsors, which may look unbalanced to a banker. The right N is a banker's call; the default is 0.

## 6. The evidence pack holds every fact the LLM may use

### Decision

`EvidenceBuilderService` builds one `EvidencePack` per acquirer. Every item has an ID (`F1`, `F2`, ...) or a transaction ID:

- Identity: acquirer type, total deals, closed deals, years active.
- Sector mix: deals per sector with each sector's similarity weight (decision 4).
- Size: smallest, median, and largest deal, and where the target's EV falls in that range.
- Recent activity: deals in the last 2 years of data (2023–2024) and the date of the last deal.
- Profile history: the share of the acquirer's deals with private, top-quartile-margin, and regional targets (decision 2).
- Direct precedents: the count and list (decision 7).
- Precedent table: up to 5 deals. Direct precedents come first; remaining rows are the acquirer's other deals ordered by sector similarity, then by distance from the target's EV on a log scale. Each row shows sector, EV, deal type, outcome, EV/EBITDA, and EV/Revenue as reported, and is labeled direct or not.
- Valuation: the market comps (decision 3) and the acquirer's own median EV/EBITDA and EV/Revenue on closed deals, each with its sample size.
- Outcomes: withdrawn and terminated counts.
- Peer baselines: the typical value of each item above for acquirers of the same type. "Typical" is the median across every acquirer of that type in the data, the acquirer itself included.

Summary statistics such as "3 direct precedents, median 12.5x" are computed over the direct precedents only.

### Rationale

The overview needs size, priorities, and recent activity; the brief's example citation needs a defined deal set with a count and a median; and the valuation section needs a number that differs by acquirer, or all 10 show the same market comps. "Strategic priorities" are read from sector mix and size range. Filling the precedent table past the direct precedents means an acquirer with none, such as Warburg Pincus, still shows its closest deals, labeled as such.

### Tradeoff

Without rationale tags (decision 8), priorities are thinner than a banker's sense of them. Acquirer-level multiples don't differ detectably between sponsors (p = 0.40), so the baselines present a gap as a fact about past deals. Widening thin comp sets is not built: it never triggers for this target (n = 10).

## 7. Conviction is computed in code from direct precedents

### Decision

Conviction answers: how much evidence is there that this acquirer wants a deal like this one? It is absolute and separate from rank. `EvidenceBuilderService` computes it:

- Direct precedent: a closed deal in the target's sector or its 2 nearest sectors (decision 4), at 0.5–2x the target's EV, in the last 5 years of data.
- High: 3 or more direct precedents.
- Medium: 1–2 direct precedents, or 2 or more deals of any outcome in the target's exact sector, at any size.
- Low: none of the above.
- Downgrade: a high-severity data risk (decision 10) lowers the level by one.

The LLM writes the 1–2 sentence reason from the same counts. It can add qualitative risks but cannot change the level. The ranking score is shown separately, next to conviction.

For this target, the top 10 come out 4 High, 5 Medium, 1 Low. Warburg Pincus ranks 6th but has no direct precedents.

#### Signals dropped

These signals showed no detectable difference between acquirers in the EDA:

| Signal | Finding | Shown instead in |
|---|---|---|
| Deal-type fit | Labels look randomly assigned (58 PE mergers of equals or SPACs, 69 strategic LBOs, MBOs or platform investments) | Precedent table |
| Profile fit (margin, region, private) | p = 0.16–0.54 | Evidence pack, next to peer baselines (decision 6) |
| Close rate | p = 0.59; strategics have too few deals | One-pager facts |
| Multiples paid vs. comp median | p = 0.40 | Valuation section |

### Rationale

A banker can check a count in seconds ("3 closed deals in Healthcare Services or adjacent sectors at $100–400M since 2020"). Counting direct precedents separates one perfect-match deal from many loose ones, which an MD would ask about. Code-computed levels are reproducible and testable; LLM-assigned levels tend to cluster at High/Medium and can drift between runs. The downgrade comes from a computed risk, so it stays reproducible.

### Tradeoff

- Levels don't predict who buys. In the backtest, among each held-out deal's top 10 predictions, the share that were the real buyer was 5.2% for High, 4.5% for Medium, and 4.0% for Low (7 of 134, 36 of 808, and 13 of 328; `eda/prototypes/ranking_compare.py`). The order is right, but the gaps rest on a handful of deals. Conviction measures the amount of evidence, and the README explains that it is only weakly related to purchase probability.
- The cut points (3 and 1) are conventions, kept in one place for a banker to calibrate.
- The downgrade never fires in the default top 10; every acquirer there has done a deal above $1B. It applies to 2 of the 13 eligible strategics (Trinity Health, HCA Healthcare) if `--min-strategics` reserves them.
- Qualitative risks written by the LLM (financing, antitrust) don't lower the level. Next step: let the LLM lower it one notch for a stated risk, logged and shown.

## 8. Prompt structure

### Decision

`prompts/` holds versioned templates filled with `string.Template`:

- `system.md`: the role (an associate writing for an MD who is deciding whether to call this buyer); hard rules (use only the facts given, cite a fact or transaction ID for every claim, no figures outside the facts, say "the data doesn't show" when a fact is missing); a short list of banned phrases ("leading," "well-positioned," "track record of strategic acquisitions").
- `lens_sponsor.md` or `lens_strategic.md`, chosen by code from the acquirer type:
  - *Sponsor:* breadth across adjacent sectors, where the target's size sits in their range, recent pace, and (labeled as outside the data) exit and holding-period angles.
  - *Strategic:* depth in the target's sector, size relative to their largest deal, and (labeled as outside the data) synergy, antitrust, and integration angles.
- `acquirer.md`: the target profile, market comps, the evidence pack as JSON with an ID on every fact (`F1`, `F2`, ...), peer baselines, the one-line summary of the other shortlisted acquirers (decision 9) with an instruction to say what distinguishes this buyer, and the output schema.

`acquirer.md` also states each section's word range. The ranges come from the validator's config, so the prompt and the check can't disagree; stating them cut first-try length failures from 3 of 10 to 0 of 10 in the live runs.

The prompt leaves out `deal_type`, `financing_type`, and `strategic_rationale_tags`; they still appear as reported in the precedent tables code renders. Peer baselines give the typical value of a trait across acquirers of the same type (for example, the share of high-margin targets).

### Rationale

Generic output is the main failure the brief names. The lens gives sponsors and strategics different arguments, the peer line pushes each one-pager to say what is specific to its buyer, and the fact IDs let the validator (decision 11) check every claim. `deal_type` and `financing_type` often contradict the acquirer type, and the tags show no acquirer-level pattern (EDA issues #2–3), so reasoning from them would produce confident claims built on noise. Sponsors are statistically similar on most traits, so without baselines the model would present shared traits as one firm's strategy.

### Tradeoff

The fit thesis has less material and relies on sector, size, recency, and the specific deals cited. The lenses avoid deal type and financing even though bankers think in those terms; a cleaner data source could put them back.

## 9. One LLM call per acquirer, run concurrently

### Decision

Each one-pager gets its own LLM call, run concurrently. Each prompt holds that acquirer's evidence pack plus one line per other shortlisted acquirer (name, type, relevant deal count).

### Rationale

One acquirer per prompt gives more specific output, a failed call can be retried alone, concurrency fits the 60-second budget, and each call has its own recorded response for tests.

### Tradeoff

Contrasts between acquirers are limited to what the summary line supports. A single batched call would allow direct comparison, but its long output is slower, degrades toward the end, and one failure loses all 10.

## 10. `AcquirerOnePager` is the output model

### Decision

```python
@dataclass
class AcquirerOnePager:
    acquirer: str
    acquirer_type: str
    rank: int
    reserved_slot: bool                 # included by --min-strategics (decision 5)
    ranking: RankingScore               # score and per-sector contributions (decision 4)
    headline: str
    overview: str
    fit_thesis: str
    precedents: list[PrecedentRow]      # transaction, is_direct, sector weight (decision 6)
    precedent_commentary: str
    valuation: ValuationComps
    valuation_commentary: str
    risks: list[RiskFlag]
    conviction: ConvictionLevel
    conviction_reason: str
    cited_ids: dict[str, list[str]]     # section -> fact and transaction IDs
    review_flags: list[str]             # validation failures (decision 11)
```

Code fills the factual fields (`acquirer`, `rank`, `ranking`, `precedents`, `valuation`, `conviction`, and data risks). The LLM returns JSON for the prose fields, `cited_ids`, and its share of `risks`.

The LLM's JSON is a Pydantic model, `OnePagerDraft`, kept in `models/` next to the dataclasses. Citations are one list per section, because structured output needs a fixed schema. It has a `data_risk_notes` list (one sentence per computed risk, matched by exact title and stored as `RiskFlag.explanation`) and a `qualitative_risks` list (becomes `RiskFlag`s with `source = llm`).

Each `RiskFlag` has a `source`, a `severity`, and the transaction IDs behind it. Data risks are computed by code, with thresholds in one config file; only those that trip are passed to the LLM to explain:

| Data risk | Trips when | Severity |
|---|---|---|
| Size stretch | Target EV is above the acquirer's largest deal (26 strategics have never done a $200M deal) | High if above 2x their largest deal |
| Thin evidence | Fewer than 2 direct precedents (decision 7) | Normal |
| Recent pace | 5 or more deals in the last 2 years of data (integration bandwidth). Sponsors did 1–7 in 2023–2024, median 3; only Nordic Capital (6) and Vista Equity (7) trip it | Normal |
| Competitive process | At least half the comp set drew 3 or more bidders. For this target 5 of 10 did, so it trips. The comp set depends only on the target, so this is a target-level risk: it trips for all acquirers or none, and `summary.md` shows it once | Normal |

LLM risks (`source = llm`) cover what the CSV can't measure: antitrust, integration complexity, financing capacity. Withdrawn history and premium paid appear in the one-pager facts, because neither differs between acquirers in the EDA.

### Rationale

Sizes, multiples, and precedents go to an MD, so they never pass through the model. Computed risks are grounded wherever the data supports one, and `source` shows which ones the data backs. They don't guarantee two risks per one-pager: for this target most acquirers trip only competitive process, so the rule of at least 2 risks (decision 11) relies on the LLM's risks plus the validator. A typed object lets the validator check concrete rules and keeps views simple.

### Tradeoff

The thresholds are conventions, listed in the README for a banker to adjust. Antitrust is left to the LLM because almost every strategic is single-sector, so a computed concentration flag would trip for all of them.

## 11. Grounding validation, one retry, then flag for review

### Decision

`OutputValidatorService` checks each LLM response:

1. It matches the output schema.
2. Every cited ID exists in that acquirer's evidence pack.
3. Every number in the prose (`13.2x`, `$150M`, `42%`, years, counts) matches a value in the evidence pack or target profile at the precision displayed.
4. There are at least 2 risks, no banned phrases, and each section is within its length range.

How the rules are applied:

- "In the evidence pack" means any number that appears in the data blocks of the prompt: target profile, definitions, market comps, facts, precedent table, data risks, and the other-buyer lines. Numbers that appear only in the instructions don't count. Numbers are compared after removing commas and trailing zeros, so `2.90x` matches `2.9x` but `12x` doesn't match `12.1x`.
- IDs written inline in the prose ("(MA-2021-0310)") are checked as well as the `citations` lists, and each section must cite at least one ID.
- Every computed data risk needs a note with its exact title.
- Banned phrases match whole words, so "leading" is caught and "misleading" isn't.

On failure, `AcquirerOnePagerService` retries once in the same conversation: the first response goes back as the assistant turn, followed by the specific violations ("valuation_commentary uses 14.1; that value isn't in the facts"). An API error, refusal, or truncated response counts as a failed attempt and is retried the same way. If it fails again:

- The code-filled fields stay. The prose fields are kept if the response parsed, or left empty if it didn't.
- `review_flags` lists the violations. The Markdown file opens with a "Needs review" line, `summary.md` marks the row, and the JSON includes the list.
- The CLI prints a warning naming the acquirer, and the run continues.

### Rationale

The one-pagers are drafts for VP review, and the tool should say where to look. A banker keeps the parts that are guaranteed correct even when the prose fails, and the other 9 one-pagers are unaffected.

### Tradeoff

Regex grounding misses numbers written as words ("six deals") and can't judge whether reasoning is sound, only whether its numbers are real. Because the other-buyer lines are grounded too, a number that belongs to another acquirer passes if the model attaches it to this one. A model-graded review could be added on top of these checks later.

## 12. Markdown and JSON output

### Decision

Each run writes `summary.md` (ranked table with conviction and key stats), one Markdown file per acquirer, `one_pagers.json`, and `run_log.json` (decision 13), into `runs/<timestamp>/`. A recorded run is copied to `sample_output/`.

### Rationale

Markdown renders on GitHub, diffs cleanly, and can be committed as sample output. JSON serves tests and future consumers such as a CRM write-back.

### Tradeoff

Bankers work in PowerPoint and PDF. An HTML or PDF view can be added in `views/` without changing the services.

## 13. Record and replay, run log, and non-determinism

### Decision

- `--record` wraps `AnthropicClient` in `RecordingClient`, which saves each response keyed by a hash of model, prompt version, and the full prompt. `--replay` uses `RecordedClient`, which needs no API key and fails with a clear message when no recording matches.
- The key also covers the output schema, so a schema change can't replay an old response. Each recording is one JSON file in `recordings/` that holds the prompt as well as the response, so a reviewer can read exactly what the model was given. Failed calls aren't recorded.
- A replay that finds no matching recording stops the run with exit code 1, because a partial replay wouldn't reproduce the recorded output. Changing any input (`--ev 250`, a prompt edit, `LLM_MODEL`) causes a miss.
- `RunLogView` writes `run_log.json`: the CSV's SHA-256, configuration, prompt version, model, whether the LLM step ran live, recorded, or replayed, per-call latency and tokens, validation results, and any rows skipped at load. This is the audit log.
- `sample_output/` was written by `acquirers --record --out sample_output`. Replaying it with no API key reproduces every file except `run_log.json` (which has timestamps), and a test checks this.
- The README links a short recorded demo (a replay run, then a live run), as the brief asks for tools that need a paid key. Replay and the committed sample output let anyone check the same run without a key.

How each part handles non-determinism:

| Part | Behavior |
|---|---|
| Ranking, numbers, tables, conviction, data risks | Deterministic. Same CSV and config give the same output |
| Prose | Varies between live runs. The Messages API rejects `temperature`, `top_p`, and `top_k` with a 400, so there is no sampling control. Prose can't change a number (decision 11) or a conviction level (decision 7) |
| Replay | Reproduces a recorded run exactly. The committed sample output comes from one |

### Rationale

Live runs need a paid key, so replay lets anyone run the tool offline and reproduce the sample output.

### Tradeoff

Recordings go stale when a prompt changes and must be re-recorded with a key. Prompt changes are compared by re-recording and diffing the reports. Re-recording adds files and never deletes them, so old recordings stay in `recordings/` until someone clears the folder before recording. The replay test then fails until `sample_output/` is re-recorded too. Replay tests the pipeline around the LLM; the prose of a new live run can differ from the recording.

## 14. Runtime, model, and how it runs

### Decision

- Python with pandas. pandas loads the CSV and computes sector similarity (a sector-by-sponsor crosstab) and the margin quartile; numpy comes with pandas and isn't used directly. Per-acquirer evidence is plain Python over the `Transaction` dataclasses, because each acquirer has at most 25 deals and a list comprehension reads more directly than a DataFrame round trip. Other dependencies: `pydantic` (output schema), `anthropic`, `python-dotenv`.
- API keys come from a `.env` file. `ANTHROPIC_API_KEY` is read from `.env` (see `.env.example`); an optional `LLM_MODEL` overrides the model. Without `--replay`, a missing key fails before any analysis with "Set ANTHROPIC_API_KEY in .env, or run with --replay."
- Model: `claude-sonnet-5-5`, as the balance of quality and latency for 10 parallel calls.
- Structured output: `client.beta.messages.parse()` with the Pydantic output schema, so the SDK validates the JSON. The response's stop reason is checked first; a refusal or a `max_tokens` cut-off is a failed call, handled by decision 11.
- Refusal fallback: each call sends `fallbacks="default"` with the `server-side-fallback-2026-07-01` beta. If a safety classifier declines, the API re-runs the request on Anthropic's recommended fallback model within the same call, and the run log records which model served it. This is why the call uses the `beta` namespace.
- Thinking effort: `medium` (in `LlmConfig`). Sonnet 5.5 thinks by default and effort is the main latency lever; at `medium` a call took 9–16 s in the first live run.
- Concurrency and limits: a thread pool of 10 workers. Each HTTP attempt times out at 30 s and the SDK retries once (rate limits, server errors, timeouts), so timeout × (retries + 1) = 60 s per call. Two live runs of the default target took 28.9 s (4 validation retries) and 21.7 s (none).
- How it runs: `pyproject.toml` defines an `acquirers` command. With `uv`, `uv run acquirers` installs and runs in one step; without it, `pip install -e .` then `acquirers`.

### Rationale

pandas makes the data rules and evidence building short and readable. The SDK's schema parsing removes a hand-written JSON parser. A `.env` file keeps keys out of the repository and matches the brief's environment-variable rule.

### Tradeoff

Sonnet-class output may be less polished than Opus-class; the model is one setting to change. The SDK counts a timed-out call as retryable, so a tight timeout can make one slow call fail; decision 11 makes that failure visible. The 60-second budget applies per call. An acquirer that needs a validation retry makes two calls in sequence, so a run where both are slow can pass 60 s. Lower effort trades prose quality for speed.

## 15. Edge cases and tests

### Decision

| Case | Behavior |
|---|---|
| CSV missing, or a required column absent | Fail at load, naming the file or column |
| Bad row (non-numeric EV, unknown outcome) | Skip it, count it, list it in the run log |
| Target sector not in the data, or an unknown region | Fail, listing the known values |
| Fewer than 10 rankable acquirers | Produce fewer, with a note in `summary.md` |
| Fewer eligible strategics than `--min-strategics` | Include all and warn (decision 5) |
| Missing API key | Fail before analysis (decision 14) |
| An LLM call fails twice | "Needs review"; the run continues (decision 11) |
| No recording matches in replay | Fail with a clear message (decision 13) |

Tests (`pytest`):

- Unit, on fixture CSVs (repository) and small hand-built deal lists (`tests/factories.py`): repository rules, sector similarity, ranking and tie-breaks, `--min-strategics`, conviction thresholds and the downgrade, risk thresholds, precedent selection. The evidence-builder tests fix sector similarity through `create_autospec` so each test controls which sectors count as nearest.
- Unit, documented numbers: every figure this file and `eda/FINDINGS.md` quote about the real CSV (similarities, comps, the 4/5/1 conviction mix, the two downgrades) is asserted in `tests/unit/test_documented_numbers.py`, so the code and the documents can't drift apart.
- Unit, validator: a planted fake number, a bad citation, and a banned phrase are each caught, along with rounded figures, missing risk notes, too few risks, and length.
- Unit, one-pager service and views, with a fake `LlmClient` (`tests/llm_fakes.py`): one call per acquirer in rank order, the retry carries the violations, two failures flag the one-pager and keep its computed fields, and a failed acquirer doesn't stop the others.
- Replay, end to end (`tests/replay/`): the full CLI runs offline from committed recordings, writes 10 one-pagers that pass schema and grounding checks, and reproduces `sample_output/` file for file. The same file checks that a changed target misses with a clear message and that a missing key fails before any analysis.
- Unit, backtest (`test_backtest_service.py`): hit@k and MRR on a hand-built deal list, a first-time buyer counts as a miss, each fit sees only earlier quarters, and the profile-like filter. The real-CSV table is pinned in `test_documented_numbers.py`.
- Unit, record and replay clients: a recorded response replays exactly; a changed prompt, prompt version, or model misses; failed calls aren't saved.

No test calls the live API or reads `.env`.

### Rationale

Each failure either stops early with a message that says what to fix, or is contained to one row or one one-pager and reported. Fixture CSVs make the expected answer obvious, and replay tests the whole pipeline without a key.

### Tradeoff

The tests check that outputs are correct and grounded. Prose quality is judged by reading recorded runs side by side when a prompt changes.

## Appendix: Folder structure

```
acquirer-engine/
├── README.md
├── pyproject.toml                      # defines the `acquirers` command (decision 14)
├── .env.example                        # ANTHROPIC_API_KEY, optional LLM_MODEL
├── .gitignore
├── data/
│   └── ma_transactions_500.csv
├── docs/
│   ├── architecture.md
│   └── decisions.md
├── eda/
│   ├── FINDINGS.md
│   ├── eda.ipynb
│   ├── requirements.txt
│   └── prototypes/                     # analysis scripts behind decisions 4 and 7
├── src/acquirer_engine/
│   ├── config.py                       # risk thresholds, conviction cut points, timeouts
│   ├── errors.py                       # DataError and InputError: stop with a message (decision 15)
│   ├── formatting.py                   # number formats shared by facts, views, and the validator
│   ├── controllers/
│   │   └── cli_controller.py           # flags, composition root, report and backtest commands
│   ├── services/
│   │   ├── acquirer_report_service.py  # orchestrates one run
│   │   ├── target_profile_service.py   # checks CLI inputs, resolves the margin floor (decision 2)
│   │   ├── candidate_scoring_service.py
│   │   ├── acquirer_ranking_model.py   # AcquirerRankingModel protocol
│   │   ├── similarity_weighted_count.py
│   │   ├── deal_count_baseline.py
│   │   ├── sector_similarity_service.py
│   │   ├── evidence_builder_service.py # evidence pack, conviction, data risks
│   │   ├── acquirer_one_pager_service.py
│   │   ├── output_validator_service.py
│   │   └── backtest_service.py
│   ├── models/                         # Transaction, TargetProfile, EvidencePack, AcquirerOnePager, ...
│   ├── repositories/
│   │   ├── transaction_repository.py   # TransactionRepository protocol
│   │   └── csv_transaction_repository.py
│   ├── clients/
│   │   ├── llm_client.py               # LlmClient protocol
│   │   ├── anthropic_client.py
│   │   ├── recording_key.py            # hash of model, prompt version, prompt, schema
│   │   ├── recording_client.py
│   │   └── recorded_client.py
│   ├── prompts/
│   │   ├── system.md
│   │   ├── lens_sponsor.md
│   │   ├── lens_strategic.md
│   │   └── acquirer.md
│   └── views/
│       ├── markdown_view.py
│       ├── json_view.py
│       └── run_log_view.py
├── recordings/                         # saved LLM responses for --replay (decision 13)
├── sample_output/                      # a recorded run, committed (decision 12)
├── runs/                               # one folder per run, gitignored
└── tests/
    ├── factories.py                    # make_transaction, make_target
    ├── llm_fakes.py                    # FakeLlmClient, small evidence fixture, a valid draft
    ├── fixtures/                       # small CSVs with obvious answers
    ├── unit/
    └── replay/
```
