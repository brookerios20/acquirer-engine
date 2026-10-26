import re
from pathlib import Path

from .. import formatting as fmt
from ..models import AcquirerOnePager, AcquirerReport, RiskSource


def one_pager_filename(one_pager: AcquirerOnePager) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", one_pager.acquirer.lower()).strip("-")
    return f"{one_pager.rank:02d}-{slug}.md"


class MarkdownView:
    """Writes `summary.md` and one Markdown file per acquirer (decision 12)."""

    def write(self, report: AcquirerReport, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [out_dir / "summary.md"]
        paths[0].write_text(self._summary(report), encoding="utf-8")
        for one_pager in report.one_pagers:
            path = out_dir / one_pager_filename(one_pager)
            path.write_text(self._one_pager(one_pager, report), encoding="utf-8")
            paths.append(path)
        return paths

    # --- summary.md ---------------------------------------------------------------------

    def _summary(self, report: AcquirerReport) -> str:
        target = report.target
        evidence = report.evidence
        market = evidence.market_comps
        lines = [
            "# Likely acquirers",
            "",
            f"**Target:** {target.sector}, {fmt.money(target.ev_mm)} EV, {target.ownership.lower()}, "
            f"EBITDA margin at or above {target.min_margin_pct:.1f}%, "
            + (f"{target.region}." if target.region else "regional."),
            "",
            "Drafts for VP review. Rankings, figures, conviction, and data risks are computed from the "
            "transaction data; the prose is LLM-written and checked against those figures.",
            "",
        ]
        if report.min_strategics:
            lines += [f"Strategic slots reserved: at least {report.min_strategics}.", ""]
        for warning in report.warnings:
            lines += [f"> **Note:** {warning}", ""]

        lines += [
            "| Rank | Acquirer | Type | Score | Conviction | Direct precedents | Deals | Review |",
            "|---|---|---|---|---|---|---|---|",
        ]
        packs = {p.acquirer: p for p in evidence.packs}
        for one_pager in report.one_pagers:
            pack = packs[one_pager.acquirer]
            name = f"[{one_pager.acquirer}]({one_pager_filename(one_pager)})"
            if one_pager.reserved_slot:
                name += " (reserved)"
            lines.append(
                f"| {one_pager.rank} | {name} | {one_pager.acquirer_type} | {one_pager.ranking.score:.1f} | "
                f"{one_pager.conviction.level.value} | {one_pager.conviction.direct_precedents} | "
                f"{pack.stats.total_deals} | {'Needs review' if one_pager.needs_review else 'OK'} |"
            )

        lines += ["", "## Market comps", ""]
        if market.median_ev_ebitda is not None and market.median_ev_revenue is not None:
            lines.append(
                f"{market.count} closed {market.sector} deals at {fmt.money(market.min_ev_mm)}-"
                f"{fmt.money(market.max_ev_mm)}: median EV/EBITDA {fmt.ev_ebitda(market.median_ev_ebitda)}, "
                f"median EV/Revenue {fmt.ev_revenue(market.median_ev_revenue)}."
            )
        else:
            lines.append(f"No closed {market.sector} deals at {fmt.money(market.min_ev_mm)}-{fmt.money(market.max_ev_mm)}.")
        lines += ["", f"Deals: {', '.join(market.transaction_ids) or 'none'}", ""]

        if evidence.target_risks:
            lines += ["## Risks for every buyer", ""]
            lines += [f"- **{r.title}:** {r.detail}." for r in evidence.target_risks]
            lines.append("")

        first, last = evidence.direct_years
        lines += [
            "## Definitions",
            "",
            f"- **Direct precedent:** a closed deal in {' / '.join(evidence.direct_sectors)}, at "
            f"{fmt.money(market.min_ev_mm)}-{fmt.money(market.max_ev_mm)} EV, in {first}-{last}.",
            "- **Score:** each past deal counts by how similar its sector is to the target's (1.0 for the "
            "same sector). It measures activity in related sectors, not purchase probability.",
            "- **Conviction:** how much evidence there is that the buyer wants a deal like this one, "
            "from direct-precedent counts. It is computed in code, separately from the score.",
            "",
            f"_Model: {report.model}. Prompt version: {report.prompt_version}. "
            f"Data: {report.csv_source} (SHA-256 {report.csv_sha256[:12]})._",
            "",
        ]
        return "\n".join(lines)

    # --- One file per acquirer ----------------------------------------------------------

    def _one_pager(self, one_pager: AcquirerOnePager, report: AcquirerReport) -> str:
        valuation = one_pager.valuation
        market = valuation.market
        lines = [f"# {one_pager.rank}. {one_pager.acquirer}", ""]
        if one_pager.needs_review:
            lines += [
                "> **Needs review.** The draft failed these checks after one retry. "
                "Figures, tables, and conviction below are computed and unaffected.",
                "",
            ]
            lines += [f"> - {flag}" for flag in one_pager.review_flags]
            lines.append("")

        subtitle = f"{one_pager.acquirer_type} · Score {one_pager.ranking.score:.1f} · Conviction: **{one_pager.conviction.level.value}**"
        if one_pager.reserved_slot:
            subtitle += " · Strategic slot reserved by `--min-strategics`"
        lines += [subtitle, ""]
        if one_pager.headline:
            lines += [f"**{one_pager.headline}**", ""]

        lines += ["## Acquirer overview", "", one_pager.overview or "_Not drafted._", ""]
        lines += ["## Strategic fit thesis", "", one_pager.fit_thesis or "_Not drafted._", ""]

        lines += ["## Precedent activity", ""]
        if one_pager.precedent_commentary:
            lines += [one_pager.precedent_commentary, ""]
        lines += [
            "| ID | Target | Year | Sector | EV | Deal type | Outcome | EV/EBITDA | EV/Revenue | Direct |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in one_pager.precedents:
            t = row.transaction
            lines.append(
                f"| {t.transaction_id} | {t.target_company} | {t.deal_year} | {t.sector} | {fmt.money(t.ev_mm)} | "
                f"{t.deal_type} | {t.outcome} | {fmt.ev_ebitda(t.ev_ebitda)} | {fmt.ev_revenue(t.ev_revenue)} | "
                f"{'yes' if row.is_direct else 'no'} |"
            )
        lines += ["", "Deal type is shown as reported; the EDA found it unreliable as a signal of buyer behavior.", ""]

        lines += ["## Valuation context", ""]
        if one_pager.valuation_commentary:
            lines += [one_pager.valuation_commentary, ""]
        lines += [
            "| Set | n | Median EV/EBITDA | Median EV/Revenue |",
            "|---|---|---|---|",
            f"| Market: closed {market.sector}, {fmt.money(market.min_ev_mm)}-{fmt.money(market.max_ev_mm)} | "
            f"{market.count} | {_multiple(market.median_ev_ebitda, fmt.ev_ebitda)} | "
            f"{_multiple(market.median_ev_revenue, fmt.ev_revenue)} |",
            f"| {one_pager.acquirer}: closed deals | {valuation.acquirer_closed_count} | "
            f"{_multiple(valuation.acquirer_median_ev_ebitda, fmt.ev_ebitda)} | "
            f"{_multiple(valuation.acquirer_median_ev_revenue, fmt.ev_revenue)} |",
            f"| {one_pager.acquirer}: direct precedents | {valuation.direct_count} | "
            f"{_multiple(valuation.direct_median_ev_ebitda, fmt.ev_ebitda)} | n/a |",
            "",
        ]

        lines += ["## Risk flags", ""]
        for risk in one_pager.risks:
            source = "computed from data" if risk.source == RiskSource.DATA else "LLM-written"
            text = f"- **{risk.title}** ({source}, {risk.severity.value} severity): {risk.detail.rstrip('.')}."
            if risk.explanation:
                text += f" {risk.explanation}"
            if risk.transaction_ids:
                text += f" [{', '.join(risk.transaction_ids)}]"
            lines.append(text)
        lines.append("")

        conviction = one_pager.conviction
        lines += [
            "## Conviction",
            "",
            f"**{conviction.level.value}.** {one_pager.conviction_reason}".rstrip(),
            "",
            f"Basis: {conviction.direct_precedents} direct precedents, "
            f"{conviction.exact_sector_deals} deals in {report.target.sector}"
            + (f"; lowered one level by the {conviction.downgraded_by.lower()} risk." if conviction.downgraded_by else "."),
            "",
            "## Score breakdown",
            "",
        ]
        lines += [
            f"- {c.deal_count} {c.sector} {'deal' if c.deal_count == 1 else 'deals'} × {fmt.weight(c.weight)}"
            for c in one_pager.ranking.contributions
        ]
        lines += ["", f"Total: {one_pager.ranking.score:.1f}", ""]

        if one_pager.cited_ids:
            lines += ["## Sources cited", ""]
            lines += [f"- {section}: {', '.join(ids)}" for section, ids in one_pager.cited_ids.items() if ids]
            lines.append("")
        return "\n".join(lines)


def _multiple(value: float | None, formatter) -> str:
    return formatter(value) if value is not None else "n/a"
