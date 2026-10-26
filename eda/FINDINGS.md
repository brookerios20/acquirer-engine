# EDA findings: `ma_transactions_500.csv`

Source: `eda/eda.ipynb`, run on `data/ma_transactions_500.csv`. [obs] = printed by the analysis; [inf] = inference from that output. Section numbers (§) refer to the notebook's headings. The ranking prototypes behind the design's backtest results are in `eda/prototypes/`.

Scope: this file reports what the data shows and how each field should be treated as data. It makes no design decisions. Design choices that rely on these findings (scoring, comps rules, thresholds, prompting) belong in the design documents and `docs/decisions.md`.

_Revised 2026-10-25:_
- _Corrected two counts that were misread from tables._
- _Added §7 acquirer-level and target-slice checks._
- _Took design rules out of the script and this file._
- _Reworded the non-significant results as "no detectable difference."_

## What the data is
500 healthcare M&A transactions from 2015 to 2024, one row per deal, keyed by `transaction_id`. The IDs are unique and contiguous, and each ID's year matches `deal_year` [obs].

Acquirers: 107 distinct.
- 15 PE sponsors account for 290 deals, 14–25 each, spread across 6–10 sectors [obs].
- 92 strategics account for 210 deals, median 2 each. 90 of the 92 buy in a single sector [obs].

The file has no duplicate rows, no whitespace or case variants, and missing values in only one column (`days_to_close`) [obs]. The issues are in the meaning of the values, as described in the table below. The data appears to be synthetic [inf].

## Distributions
All values below are [obs].
- Sector: Behavioral Health 70, Home Health/Hospice 62, Physician Groups 60, Health IT 50, Dental 46, Healthcare Services 46, Revenue Cycle 43, Health Insurance 43, Medical Devices 42, Pharma/Biotech 38.
- Acquirer type: Financial Sponsor 290, Strategic 210.
- Outcome: Closed 406, Withdrawn 49, Pending 21, Terminated 14, Rumored 10.
- Target ownership before the deal: Private 208, PE-Backed 141, Public 76, Non-Profit 75.
- Deal size (EV, $M): min 6.1, median 153.6, max 14,353.7.
- Deals at $100–400M: 154 across all sectors, and 12 in Healthcare Services from 11 acquirers.
- Sponsors by deal count: Blackstone 25, Nordic Capital 24, Bain Capital 24, Vista Equity 24, Welsh Carson 23, Warburg Pincus 23, New Mountain Capital 21, TPG Capital 18, Advent International 18, KKR 17, Veritas Capital 16, Hellman & Friedman 15, GTCR 14, General Atlantic 14, Francisco Partners 14.
- Strategic buyers in Healthcare Services are mostly health systems (Atrium Health 3, UPMC 3, Steward Health Care 3, CHRISTUS Health 2, Ardent Health Services 2, Ascension Health 2, Advocate Health 2), plus UnitedHealth Group 2.

## Data-quality issues, ranked by impact
| # | Issue | Evidence | Data treatment | Why |
|---|---|---|---|---|
| 1 | EBITDA-side fields don't reconcile. EV, revenue and EV/Revenue agree (100% within 5%). EBITDA, margin and EV/EBITDA disagree under every identity. | Median gaps 18–28%; only 10–15% of rows within 5% [obs] | Flag. Take `ev_ebitda_multiple` and `ebitda_margin_pct` as stated and never recompute them. Treat `target_ebitda_mm` as unreliable. | No two of the three EBITDA-side fields imply the third, so no derived value is trustworthy. The stated multiple is the CSV's own valuation field. |
| 2 | Structure labels conflict with buyer type. PE in Merger of Equals or SPAC deals; PE paying in stock; strategics doing LBOs or MBOs; carve-outs from private or non-profit targets; $13B non-profit pharma targets | PE in Merger of Equals or SPAC: 58 rows. PE paying in Stock or Cash + Stock: 82. Strategics in LBO, MBO or Platform Investment: 69. Carve-outs from Private or Non-Profit targets: 12. Non-profit pharma targets: e.g. MA-2023-0500 ($13.3B) [obs]. No detectable per-sponsor difference in `deal_type` (p=0.19) or `financing_type` (p=0.62) (§7C) [obs] | Flag. Usable as reported values for a deal. Unreliable as a signal of an acquirer's behavior. | The labels look randomly assigned [inf]. |
| 3 | Rationale tags. Fixed 12-term vocabulary; "Geographic Expansion" is on 90% of deals; "Platform Build" appears only on Bolt-on and Platform deals; "Carve-out Complexity" only on Carve-outs | §6 [obs]. No detectable per-sponsor difference for the 6 tags tested (p=0.30–0.94, §7C) [obs] | Flag, same as #2. | Some tags restate `deal_type`; the rest show no detectable acquirer pattern [inf]. |
| 4 | `outcome = Rumored` isn't listed in the brief's field table. | 10 rows [obs] | Clean: map to a non-closed status. | Only `Closed` represents a completed transaction price. |
| 5 | `sub_sector` duplicates `sector`. | 100% equal [obs] | Exclude. Sub-specialty appears in target names (e.g. "Ambulatory Care", "Medical Group") [obs]. | It carries no extra signal. |
| 6 | Thin target slice. Healthcare Services has 46 deals, 12 of them at $100–400M (inclusive), and 10 of those 12 closed (§5, §7A) [obs]. | | Report sample sizes with any statistic computed on this slice. | The brief warns against filtering narrowly. Statistics on n≈10 are fragile. |
| 7 | Heavy-tailed size. Up to $14.4B, with 64 IQR outliers above $1,085M: Health Insurance 21, Pharma 17, Medical Devices 15, Healthcare Services 9, Home Health 2 [obs] | §3 | Keep. Compare sizes on a log scale or in bands. | The values are plausible for large deals in these sectors. |
| 8 | Minor items. One target is named "Integrated Integrated Behavioral"; two acquirer names have parentheticals ("Aspen Dental (ADMI)", "Cerner (Oracle)"); `days_to_close` is blank exactly where a deal didn't close; `pd.cut` size bands are (100, 400] while slice filters are [100, 400], and one deal is exactly $100M | [obs] | Leave as is. Use one inclusive size-band definition consistently. | None affects the data's meaning. |

## Acquirer-level signal (§7)
### Sponsors show no detectable difference on almost every trait
Each trait was tested against 2000 shuffles of acquirer labels [obs].
- Only bolt-on share came in under 0.05 (p=0.016), and about 1 of 19 tests is expected to do that by chance. Deal size is weak (p=0.09).
- No detectable difference: sector mix (p=0.43), geography (p=0.21), ownership (p=0.16), number of bidders (p=0.67), not-closed rate (p=0.59), EV/EBITDA paid (p=0.40), leverage (p=0.50), top-quartile-margin share (p=0.54), synergy % (p=0.82).
- These results do not show that sponsors behave the same way [inf]. With 15 firms at 14–25 deals each, the sample is too small to detect differences on these traits, so per-sponsor averages should not be read as characteristics of a firm.
- What clearly differs among sponsors is how many deals they did (14–25) and which specific deals they did [obs].

### The strategics' multiple premium is mostly sector mix
Overall, closed strategic deals have a 17.0x median vs. 15.1x for sponsors [obs]. Within sectors the gap largely disappears:
- Deal-weighted mean within-sector gap: +0.06x. Strategics are higher in 5 of 9 sectors, and in only 2 of 9 for $100–400M deals (median −0.5x) [obs].
- Physician Groups is 100% PE (60 deals) and low-multiple, which pulls the PE median down [obs].
- Healthcare Services alone shows a gap: 13.9x vs. 12.2x (18 vs. 19 closed deals) [obs].

### Other patterns
- No detectable regional concentration for strategics: 6% of the 47 strategics with ≥2 deals in a named region did them all in one region, no detectable difference from shuffled geography (p=0.54) [obs].
- Strategics are single-sector (90 of 92), the clearest structural pattern in the data [obs].
- Largest deal below $200M: 26 of 92 strategics, 0 of 15 sponsors [obs].

### Closed $100–400M deals since 2020 (§7G)
- Healthcare Services: 7 acquirers have ≥1, and the maximum is 2 (Blackstone) [obs].
- Any sector: 41 acquirers have ≥1 [obs].
- Per sponsor: mean 0.40 in Healthcare Services and 3.73 in any sector. Per strategic (of the 26 with any): 0.08 and 1.15 [obs].

## What the data can and can't support
- Distinguishes acquirers:
  - acquirer type
  - number of deals
  - the specific deals done, with their stated sector, size, year, outcome and multiples
  - a strategic's sector
  - deal size (weakly, among sponsors)
- No detectable acquirer-level signal at this sample size: `deal_type`, `financing_type`, rationale tags, geography, target ownership, `num_bidders`, outcome rate, multiples paid, synergy %.
- Unreliable as values: `target_ebitda_mm` (issue #1).

## Data interpretation assumptions
- `deal_size_mm` is enterprise value in USD millions. All amounts are in one currency and unit (there are no mixed-unit markers in the file).
- Where stated and derived EBITDA-side values disagree, the stated multiples and margin are taken as the CSV's values (issue #1).
- `acquirer_type` is fixed per acquirer (verified: no acquirer has two types) [obs].
- Data ends in 2024. 2024 has 23 deals (7/6/7/3 by quarter) vs. 49 in 2023 [obs]. A partial Q4 is possible but not shown [inf].

## Reference values for the stated target (facts only)
- Healthcare Services EBITDA margin: median 15.5%, 75th percentile 16.8% (pandas' default linear quantile) [obs].
- Closed Healthcare Services deals at $100–400M: n=10. EV/EBITDA median 12.1x (IQR 10.3–14.4x); EV/Revenue median 2.90x (IQR 2.62–3.22x) [obs].

## Open questions for the design
The data raises the following questions, which will be addressed in the design documents:
1. Which EBITDA field is the source of truth? (Evidence: issue #1. No pair of fields reconciles.)
2. Should `deal_type`, `financing_type` or tags inform anything beyond display? (Evidence: issues #2–3, §7C.)
3. How should non-closed deals, including "Rumored," count? (Evidence: issue #4. 94 non-closed deals.)
4. What threshold defines "strong margins"? (Evidence: sector margin percentiles, §5.)
5. How should breadth (PE, many sectors) be weighed against focus (strategic, one sector)? (Evidence: §7 single-sector strategics; sponsor differences mostly undetectable.)
6. Does "private" mean Private only, or Private + PE-Backed? (Evidence: ownership counts above; no detectable per-sponsor ownership preference, p=0.16.)
7. With little detectable per-sponsor signal, what makes each acquirer's rationale specific? (Evidence: "What the data can and can't support.")
