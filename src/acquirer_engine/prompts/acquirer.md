Draft the one-pager for **$acquirer** as a buyer of the target below.

## Target
$target_profile

## Definitions
$definitions

## Market comps
$market_comps

## Facts about $acquirer
Every fact you may use. Cite by ID.
```json
$facts_json
```

## Precedent table
Code shows these deals in the one-pager. Direct precedents come first.
$precedent_table

## Data risks computed by code
$data_risks

## Other buyers on the shortlist
Say what sets $acquirer apart from these.
$other_buyers

## How to frame this buyer
$lens

## Sections to write
- `headline`: one line, under 20 words, naming the main reason to call them.
- `overview`: 2-4 sentences. Who they are in this data: deal count, size range, where they buy, recent activity.
- `fit_thesis`: 3-5 sentences. Why this target fits this buyer specifically, citing the deals and counts that support it.
- `precedent_commentary`: 2-4 sentences on the precedent table: which deals matter most and why.
- `valuation_commentary`: 2-4 sentences comparing the market comps with what this buyer has paid. Report the medians and sample sizes as given; don't compute a gap.
- `data_risk_notes`: one note per data risk above, using its exact title. Empty if there are none.
- `qualitative_risks`: risks the data can't measure (financing capacity, antitrust, integration, competition from other buyers). Write enough that data risks plus qualitative risks total at least $min_risks.
- `conviction_reason`: 1-2 sentences explaining the computed conviction level ($conviction) from the counts behind it.
- `citations`: the IDs each section relies on.

Word limits, enforced by a checker: $length_limits.
