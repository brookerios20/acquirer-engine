# M&A Acquirer Identification Engine

Given a target company, this tool ranks the 10 most likely acquirers from a CSV of 500 healthcare M&A transactions and drafts a one-page rationale for each. Code computes the ranking, every figure, the precedent tables, the valuation comps, the conviction level, and the data-backed risks. An LLM writes the prose from those facts, and a validator checks that every number and citation in the prose comes from them.

A recorded run for the brief's target is in [`sample_output/`](sample_output/summary.md).

## How to Run

The tool needs Python 3.11 or later. With [uv](https://docs.astral.sh/uv/), one command installs the dependencies and runs it.

Replay the recorded run. This needs no API key:

```bash
uv run acquirers --replay
```

Run it live against the Anthropic API:

```bash
cp .env.example .env
```

Then set `ANTHROPIC_API_KEY` in `.env` and run:

```bash
uv run acquirers
```

Without uv, install the package with pip and use the same commands without `uv run`:

```bash
pip install -e .
```

Each run writes to `runs/<timestamp>/`:

- `summary.md`, the ranked table with conviction, comps, and target-level risks
- one Markdown one-pager per acquirer, for example `01-nordic-capital.md`
- `one_pagers.json`, the same content for other programs
- `run_log.json`, the audit log described under [Handling Output Non-Determinism](#handling-output-non-determinism)

A live run of the default target took 14 to 27 seconds in testing. It makes 10 calls, plus 1 for each validation retry, and uses about 63K input and 15K output tokens.

### Other targets

With no flags, the tool runs the brief's target. Each flag overrides one trait:

| Flag | Default | Meaning |
|---|---|---|
| `--sector` | Healthcare Services | Target sector, as named in the CSV |
| `--ev` | 200 | Enterprise value in $M |
| `--ownership` | Private | Ownership before the deal |
| `--margin` | Sector's top quartile (16.8% for Healthcare Services) | EBITDA margin that counts as strong |
| `--region` | Any regional | A specific region, such as Midwest |
| `--min-strategics` | 0 | Reserve at least this many slots for strategic buyers |
| `--csv` | `data/ma_transactions_500.csv` | Transaction data |
| `--out` | `runs/<timestamp>` | Output folder |
| `--record` / `--replay` | Live | Save responses for replay, or replay saved ones |

For example, `uv run acquirers --sector Dental --ev 150 --min-strategics 3`. An unknown sector, region, or ownership value fails with a list of the values in the data. Replay only works for the recorded target, so changing a target flag needs an API key.

### Backtest

```bash
uv run acquirers backtest
```

This ranks the real buyer of each 2022–2024 deal using only deals announced before it, and reports hit@10 and mean reciprocal rank for the ranking model and a deal-count baseline. It needs no API key.

### Tests

```bash
uv run pytest
```

The 90 tests run offline. They cover the data rules, ranking, conviction, risks, the validator, the retry logic, the views, the backtest, and an end-to-end replay that must reproduce every file in `sample_output/` except `run_log.json`, which records timing. One test file checks every number quoted in [`docs/decisions.md`](docs/decisions.md) against the CSV.

## Configuration: Models and API Keys

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | For live runs | Read from `.env`. Without it, a live run stops before any analysis and asks for the key or `--replay` |
| `LLM_MODEL` | No | Overrides the default model, `claude-sonnet-5-5` |

`.env` is gitignored, and no key appears in the code or the recordings.

The other settings are in [`src/acquirer_engine/config.py`](src/acquirer_engine/config.py):

- `LlmConfig`: thinking effort (`medium`), a 30-second timeout per HTTP attempt with one SDK retry, and 10 parallel calls.
- `ValidationConfig`: the banned phrases, the minimum of 2 risks, and the word range for each section.
- `AnalysisConfig`: the thresholds below. Each is a convention I chose from the data, and a banker should calibrate them.

| Setting | Value |
|---|---|
| Size band for comps and direct precedents | 0.5–2x target EV ($100–400M) |
| Direct precedent window | Last 5 years of data (2020–2024) |
| Nearest sectors that count as direct | 2 (Physician Groups, Behavioral Health) |
| Conviction | High: 3+ direct precedents. Medium: 1–2, or 2+ deals of any kind in the target's sector. Low: otherwise |
| Size stretch risk | Target EV above the buyer's largest deal; high severity above 2x, which lowers conviction one level |
| Thin evidence risk | Fewer than 2 direct precedents |
| Recent pace risk | 5 or more deals in the last 2 years of data |
| Competitive process risk | Half or more of the comp set drew 3 or more bidders |

## Recorded Demo

Demo video: [Loom recording](https://www.loom.com/share/4b680ad56b4246f899b1da76612f262c).

The demo shows a replay run and then a live run. The committed [`sample_output/`](sample_output/summary.md) and `recordings/` let anyone reproduce the recorded run without a key.

## Architecture Decisions

Each decision, with its reasoning and tradeoffs, is in [`docs/decisions.md`](docs/decisions.md). The layer diagram is in [`docs/architecture.md`](docs/architecture.md).

The run is a fixed sequence of steps, and only the last step calls the LLM:

1. `CsvTransactionRepository` loads the CSV and applies the data rules.
2. `TargetProfileService` checks the flags against the data and resolves the margin floor.
3. `CandidateScoringService` scores every acquirer and selects 10, honoring `--min-strategics`.
4. `EvidenceBuilderService` builds an evidence pack per acquirer: numbered facts, a precedent table, valuation comps, peer baselines, conviction, and data risks.
5. `AcquirerOnePagerService` makes one LLM call per acquirer, in parallel. `OutputValidatorService` checks each response, and a failed response is retried once with the specific problems listed. A response that fails twice keeps its computed fields and is marked "Needs review", and the other one-pagers are unaffected.
6. The views write Markdown, JSON, and the run log.

The code is layered into controllers, services, repositories, clients, models, prompts, and views, with constructor injection throughout. The CLI controller is the only module that constructs the services, repository, and clients. The data source, the LLM client, and the ranking model sit behind interfaces, so a Databricks table, another model provider, or a learned ranking model can be added as a new class without changes to the services.

### Ranking

An acquirer's score is the sum, over its past deals, of how similar each deal's sector is to the target's. Sector similarity comes from the data, as the cosine between sectors' sponsor deal counts; Physician Groups is 0.83 to Healthcare Services and Health IT is 0.44. In the backtest, no method I tried beat plain deal count at putting the real buyer in the top 10, because 15 sponsors did 58% of the deals. The similarity-weighted count is within 3 of 142 deals of it on that measure, ranks the real buyer higher on average, changes with the target, and explains itself in one line on each one-pager.

### Conviction

Conviction measures how much evidence there is that a buyer wants a deal like this one, and code computes it from direct precedents: closed deals in the target's sector or its 2 nearest sectors, at 0.5–2x the target's EV, in the last 5 years of data. The LLM explains the level and cannot change it. For the brief's target, the top 10 come out 4 High, 5 Medium, and 1 Low.

### LLM Prompt Structure

The prompt templates are in [`src/acquirer_engine/prompts/`](src/acquirer_engine/prompts/):

- `system.md` sets the role (an associate writing for an MD who is deciding whether to call this buyer) and the hard rules: use only the facts given, cite a fact or transaction ID for each claim, write numbers exactly as given, say "the data doesn't show" when a fact is missing, and label general-knowledge points as outside the data. It also lists banned phrases such as "well-positioned."
- `lens_sponsor.md` or `lens_strategic.md` is chosen by acquirer type. Sponsors are framed on sector breadth, size range, and pace; strategics on depth in the sector and size against their largest deal.
- `acquirer.md` holds the target profile, definitions, market comps, the evidence pack as JSON with an ID on every fact (`F1`, `F2`, ...), the precedent table, the computed risks, one line on each other shortlisted buyer, the sections to write, and each section's word limit.

Peer baselines in the facts show the typical value of each trait for acquirers of the same type. Sponsors look statistically alike on most traits in this data, and the baselines keep the model from presenting a shared trait as one firm's strategy. The model returns JSON through structured output (`client.beta.messages.parse` with a Pydantic schema), so the SDK validates the shape.

The validator then checks that every cited ID exists in that acquirer's evidence, that every number in the prose appears in the data the prompt gave the model at the precision shown, that there are at least 2 risks and a note on each computed risk, that no banned phrase appears, and that each section is within its word range.

### How the CSV Is Used

The data's quirks and how each field is treated are in [`eda/FINDINGS.md`](eda/FINDINGS.md). The notebook that produced them is [`eda/eda.ipynb`](eda/eda.ipynb).

The repository applies these rules:

- EV, revenue, and EV/Revenue reconcile in every row. The EBITDA fields disagree with each other by 18–28% in a typical row, so the tool uses `ev_ebitda_multiple` and `ebitda_margin_pct` as stated and drops `target_ebitda_mm`.
- `sub_sector` equals `sector` in every row and is ignored.
- `Rumored` (10 rows) is accepted as a fifth outcome. It counts as buyer interest and is excluded from comps.
- A row with a non-numeric or out-of-range value, or an unknown outcome or acquirer type, is skipped and listed in the run log.

The brief warns against filtering narrowly. The tool uses the full dataset in three ways. The ranking counts every past deal, weighted by sector similarity. Direct precedents include the 2 nearest sectors. The precedent table fills past the direct precedents with the buyer's closest other deals, labeled as such. The market comps stay narrow (closed Healthcare Services deals at $100–400M, n = 10, median 12.1x EV/EBITDA) and always show their sample size.

`deal_type`, `financing_type`, and `strategic_rationale_tags` stay out of the prompt. Their labels often contradict the acquirer type (58 PE deals labeled merger of equals or SPAC), and the EDA found no acquirer-level pattern in them. The precedent tables still show deal type as reported.

### What I Would Improve With More Time

- Test a conditional logit ranking model as another `AcquirerRankingModel`, and adopt it only if it beats the current model in the backtest, ideally on real deal data.
- Let the LLM lower conviction by one level for a stated qualitative risk, such as financing, with the change logged and shown.
- Add a model-graded review on top of the deterministic checks, to judge whether the reasoning holds as well as whether its numbers are real.
- Add a PDF or PowerPoint view, since bankers work in those formats.
- Add a feedback loop where a banker marks a buyer as irrelevant and the ranking adjusts.
- Widen thin comp sets automatically for targets where the closest set has few deals.
- Measure outcomes with deal teams: how often recommended buyers reach an IOI, and how much analyst time a buyer list takes with and without the tool.

## Assumptions

About the target:

- "Private" means `Private` ownership only. A sponsor-owned company would be described as PE-backed.
- "Strong EBITDA margins" means at or above the sector's 75th percentile, 16.8% for Healthcare Services.
- "Regional" means any region except `National` and `Multi-Regional`. No specific region is assumed.
- "Mid-market" is carried by the $200M size.
- Sector and size drive the ranking and conviction. Ownership, margin, and region appear in each one-pager's evidence next to the peer baseline, because the EDA found no acquirer-level preference for them.

About the data:

- `deal_size_mm` is enterprise value in $M, in one currency.
- The data ends in 2024, so "the last 5 years" is 2020–2024 and "recent" is 2023–2024.
- The data appears to be synthetic, and acquirer names are kept as reported, for example `Cerner (Oracle)`.

I don't have a banking background. The thresholds, the reading of each trait, and the default buyer mix are my judgment from the data and the brief, and each is one setting or flag for a banker to change.

## Known Limitations and Failure Modes

- The default list for this target is all financial sponsors, because sponsors did most of the deals and strategics have 1–6 deals each. `--min-strategics` reserves slots for strategics that have at least one deal in the target's sector.
- The ranking rewards activity, so the most active sponsors rank highest.
- Conviction is weakly related to who actually buys. In the backtest, the real buyer was 5.2% of High predictions, 4.5% of Medium, and 4.0% of Low.
- The validator matches numbers with a regular expression, so it misses numbers written as words. It also accepts a number that belongs to another shortlisted buyer, since those appear in the prompt.
- A run usually finishes in 14 to 27 seconds, inside the brief's 60-second target. Each LLM call is capped at 60 seconds: a 30-second timeout and one SDK retry. An acquirer that needs a validation retry makes two calls in sequence, so a slow run can go over 60 seconds.
- A buyer with no deals in the data can't be ranked.

How each failure is handled:

| Case | Behavior |
|---|---|
| CSV missing, or a required column absent | Stops at load, naming the file or column |
| Bad row | Skipped and listed in the run log |
| Unknown sector, region, or ownership | Stops, listing the known values |
| Fewer than 10 rankable acquirers | Produces fewer, with a note in `summary.md` |
| Fewer eligible strategics than `--min-strategics` | Includes all of them and warns |
| Missing API key | Stops before any analysis |
| A model refusal | The API re-runs the request on Anthropic's fallback model; the run log records which model answered |
| An LLM response fails twice | The one-pager is marked "Needs review" and the run continues |
| No recording matches in replay | Stops with a message to re-record |

## Handling Output Non-Determinism

The ranking, every figure and table, the conviction levels, and the data risks are deterministic: the same CSV and settings give the same output.

The prose varies between live runs. The Messages API rejects `temperature`, `top_p`, and `top_k` for this model, so there is no sampling control. The validator bounds that variation, because prose can't introduce a number outside the facts or change a conviction level.

`--record` saves each LLM response in `recordings/`, keyed by a hash of the model, prompt version, full prompt, and output schema. `--replay` serves those responses back with no API key, which reproduces every output file except the run log; a test checks this against `sample_output/`. Any change to the prompt, data, target flags, or model gives a new key, so replay fails clearly until the run is re-recorded.

Each run writes `run_log.json` with the CSV's SHA-256, the configuration, the prompt version, the model that served each call, per-call latency and tokens, the validation results, and any skipped rows.
