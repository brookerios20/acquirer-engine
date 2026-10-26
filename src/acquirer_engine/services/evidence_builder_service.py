import math
import statistics
from collections import Counter
from dataclasses import dataclass

from .. import formatting as fmt
from ..config import AnalysisConfig
from ..models import (
    AcquirerStats,
    Conviction,
    ConvictionLevel,
    Evidence,
    EvidencePack,
    Fact,
    MarketComps,
    PeerBaseline,
    PrecedentRow,
    RiskFlag,
    RiskSeverity,
    RiskSource,
    SectorSimilarity,
    Shortlist,
    ShortlistEntry,
    TargetProfile,
    Transaction,
    ValuationComps,
    group_by_acquirer,
)
from .sector_similarity_service import SectorSimilarityService


@dataclass(frozen=True)
class _Context:
    """Values every per-acquirer step needs, computed once per run."""

    target: TargetProfile
    similarity: SectorSimilarity
    direct_sectors: list[str]
    direct_years: tuple[int, int]
    recent_years: tuple[int, int]
    min_ev_mm: float           # size band, inclusive
    max_ev_mm: float


class EvidenceBuilderService:
    """Builds one `EvidencePack` per shortlisted acquirer: the facts the LLM may use,
    the precedent table, valuation comps, conviction, and data risks (decisions 6, 7, 10).
    """

    def __init__(self, similarity_service: SectorSimilarityService, config: AnalysisConfig):
        self._similarity_service = similarity_service
        self._config = config

    def build(
        self, transactions: list[Transaction], target: TargetProfile, shortlist: Shortlist
    ) -> Evidence:
        config = self._config
        similarity = self._similarity_service.compute(transactions)
        data_end_year = max(t.deal_year for t in transactions)
        context = _Context(
            target=target,
            similarity=similarity,
            direct_sectors=[target.sector, *similarity.nearest(target.sector, config.nearest_sectors)],
            direct_years=(data_end_year - config.direct_precedent_years + 1, data_end_year),
            recent_years=(data_end_year - config.recent_years + 1, data_end_year),
            min_ev_mm=target.ev_mm * config.size_band_low,
            max_ev_mm=target.ev_mm * config.size_band_high,
        )

        deals_by_acquirer = group_by_acquirer(transactions)
        # Stats for every acquirer, not just the shortlist, so peer baselines cover all peers
        all_stats = {
            name: self._acquirer_stats(name, deals, context)
            for name, deals in deals_by_acquirer.items()
        }
        baselines = self._peer_baselines(list(all_stats.values()))

        comp_deals = [
            t for t in transactions
            if t.is_closed and t.sector == target.sector and self._in_size_band(t, context)
        ]
        market = self._market_comps(comp_deals, context)
        target_risks = self._target_risks(comp_deals)

        packs = [
            self._pack(
                entry,
                all_stats[entry.acquirer],
                deals_by_acquirer[entry.acquirer],
                baselines[entry.acquirer_type],
                market,
                target_risks,
                context,
                shortlist_size=len(shortlist.entries),
            )
            for entry in shortlist.entries
        ]
        return Evidence(
            direct_sectors=context.direct_sectors,
            direct_years=context.direct_years,
            recent_years=context.recent_years,
            market_comps=market,
            target_risks=target_risks,
            packs=packs,
        )

    # --- Per-acquirer stats -------------------------------------------------------------

    def _acquirer_stats(self, name: str, deals: list[Transaction], context: _Context) -> AcquirerStats:
        target = context.target
        by_size = sorted(deals, key=lambda t: (t.ev_mm, t.transaction_id))
        by_date = sorted(deals, key=lambda t: (t.deal_year, t.deal_quarter, t.transaction_id))
        outcomes = Counter(t.outcome for t in deals)

        return AcquirerStats(
            acquirer=name,
            acquirer_type=deals[0].acquirer_type,
            total_deals=len(deals),
            closed_deals=sum(t.is_closed for t in deals),
            first_year=by_date[0].deal_year,
            last_year=by_date[-1].deal_year,
            sector_counts=dict(Counter(t.sector for t in deals)),
            exact_sector_deals=sum(t.sector == target.sector for t in deals),
            min_ev=by_size[0],
            median_ev_mm=statistics.median(t.ev_mm for t in deals),
            max_ev=by_size[-1],
            share_below_target_ev=sum(t.ev_mm < target.ev_mm for t in deals) / len(deals),
            recent_deals=[t for t in by_date if t.deal_year >= context.recent_years[0]],
            last_deal=by_date[-1],
            private_deals=sum(t.target_ownership == target.ownership for t in deals),
            high_margin_deals=sum(t.ebitda_margin_pct >= target.min_margin_pct for t in deals),
            regional_deals=sum(self._is_regional(t, target) for t in deals),
            direct_precedents=[t for t in by_date if self._is_direct_precedent(t, context)],
            withdrawn=outcomes["Withdrawn"],
            terminated=outcomes["Terminated"],
        )

    def _is_direct_precedent(self, t: Transaction, context: _Context) -> bool:
        """Closed, in the target's sector or its nearest sectors, at 0.5-2x the target's EV,
        in the last 5 years of data (decision 7)."""
        return (
            t.is_closed
            and t.sector in context.direct_sectors
            and self._in_size_band(t, context)
            and t.deal_year >= context.direct_years[0]
        )

    def _in_size_band(self, t: Transaction, context: _Context) -> bool:
        return context.min_ev_mm <= t.ev_mm <= context.max_ev_mm

    def _is_regional(self, t: Transaction, target: TargetProfile) -> bool:
        if target.region is not None:
            return t.geography == target.region
        return t.geography not in self._config.non_regional_geographies

    def _peer_baselines(self, all_stats: list[AcquirerStats]) -> dict[str, PeerBaseline]:
        baselines = {}
        for acquirer_type in sorted({s.acquirer_type for s in all_stats}):
            peers = [s for s in all_stats if s.acquirer_type == acquirer_type]
            median = statistics.median
            baselines[acquirer_type] = PeerBaseline(
                acquirer_type=acquirer_type,
                peer_count=len(peers),
                total_deals=median(s.total_deals for s in peers),
                recent_deals=median(len(s.recent_deals) for s in peers),
                median_ev_mm=median(s.median_ev_mm for s in peers),
                max_ev_mm=median(s.max_ev.ev_mm for s in peers),
                exact_sector_deals=median(s.exact_sector_deals for s in peers),
                direct_precedents=median(len(s.direct_precedents) for s in peers),
                private_share=median(s.private_share for s in peers),
                high_margin_share=median(s.high_margin_share for s in peers),
                regional_share=median(s.regional_share for s in peers),
            )
        return baselines

    # --- Market comps and target-level risk ---------------------------------------------

    def _market_comps(self, comp_deals: list[Transaction], context: _Context) -> MarketComps:
        return MarketComps(
            sector=context.target.sector,
            min_ev_mm=context.min_ev_mm,
            max_ev_mm=context.max_ev_mm,
            transaction_ids=sorted(t.transaction_id for t in comp_deals),
            median_ev_ebitda=_median_or_none([t.ev_ebitda for t in comp_deals]),
            median_ev_revenue=_median_or_none([t.ev_revenue for t in comp_deals]),
        )

    def _target_risks(self, comp_deals: list[Transaction]) -> list[RiskFlag]:
        """Competitive process depends only on the target, so it trips for all acquirers or none."""
        threshold = self._config.contested_min_bidders
        contested = [t for t in comp_deals if t.num_bidders >= threshold]
        if not comp_deals or len(contested) / len(comp_deals) < self._config.contested_min_share:
            return []
        return [
            RiskFlag(
                title="Competitive process",
                detail=(
                    f"{len(contested)} of {len(comp_deals)} comparable closed deals drew "
                    f"{threshold} or more bidders"
                ),
                source=RiskSource.DATA,
                severity=RiskSeverity.NORMAL,
                transaction_ids=sorted(t.transaction_id for t in contested),
            )
        ]

    # --- One acquirer's pack ------------------------------------------------------------

    def _pack(
        self,
        entry: ShortlistEntry,
        stats: AcquirerStats,
        deals: list[Transaction],
        baseline: PeerBaseline,
        market: MarketComps,
        target_risks: list[RiskFlag],
        context: _Context,
        shortlist_size: int,
    ) -> EvidencePack:
        acquirer_risks = self._acquirer_risks(stats, context)
        conviction = self._conviction(stats, acquirer_risks)

        closed = [t for t in deals if t.is_closed]
        valuation = ValuationComps(
            market=market,
            acquirer_closed_count=len(closed),
            acquirer_median_ev_ebitda=_median_or_none([t.ev_ebitda for t in closed]),
            acquirer_median_ev_revenue=_median_or_none([t.ev_revenue for t in closed]),
            direct_count=len(stats.direct_precedents),
            direct_median_ev_ebitda=_median_or_none([t.ev_ebitda for t in stats.direct_precedents]),
        )
        risks = acquirer_risks + target_risks
        precedents = self._precedent_table(stats, deals, context)
        facts = self._facts(entry, stats, baseline, valuation, conviction, risks, context, shortlist_size)

        return EvidencePack(
            entry=entry,
            stats=stats,
            peer_baseline=baseline,
            precedents=precedents,
            valuation=valuation,
            conviction=conviction,
            data_risks=risks,
            facts=facts,
        )

    def _acquirer_risks(self, stats: AcquirerStats, context: _Context) -> list[RiskFlag]:
        config = self._config
        target = context.target
        risks = []

        largest = stats.max_ev
        if target.ev_mm > largest.ev_mm:
            is_high = target.ev_mm > config.size_stretch_high_multiple * largest.ev_mm
            detail = (
                f"Target EV {fmt.money(target.ev_mm)} is above their largest deal in the data "
                f"({fmt.money(largest.ev_mm)}, {largest.transaction_id})"
            )
            if is_high:
                detail += f", more than {config.size_stretch_high_multiple:g}x"
            risks.append(RiskFlag(
                title="Size stretch",
                detail=detail,
                source=RiskSource.DATA,
                severity=RiskSeverity.HIGH if is_high else RiskSeverity.NORMAL,
                transaction_ids=[largest.transaction_id],
            ))

        direct_count = len(stats.direct_precedents)
        if direct_count < config.thin_evidence_below_direct:
            risks.append(RiskFlag(
                title="Thin evidence",
                detail=(
                    f"{direct_count} direct precedent(s); fewer than "
                    f"{config.thin_evidence_below_direct} closely comparable closed deals"
                ),
                source=RiskSource.DATA,
                severity=RiskSeverity.NORMAL,
                transaction_ids=[t.transaction_id for t in stats.direct_precedents],
            ))

        recent = stats.recent_deals
        if len(recent) >= config.recent_pace_min_deals:
            first, last = context.recent_years
            risks.append(RiskFlag(
                title="Recent pace",
                detail=f"{len(recent)} deals in {first}-{last}; integration bandwidth may be stretched",
                source=RiskSource.DATA,
                severity=RiskSeverity.NORMAL,
                transaction_ids=[t.transaction_id for t in recent],
            ))
        return risks

    def _conviction(self, stats: AcquirerStats, acquirer_risks: list[RiskFlag]) -> Conviction:
        config = self._config
        direct_count = len(stats.direct_precedents)

        if direct_count >= config.high_min_direct:
            level = ConvictionLevel.HIGH
        elif (
            direct_count >= config.medium_min_direct
            or stats.exact_sector_deals >= config.medium_min_exact_sector_deals
        ):
            level = ConvictionLevel.MEDIUM
        else:
            level = ConvictionLevel.LOW

        # A high-severity data risk lowers the level by one
        high_risk = next((r for r in acquirer_risks if r.severity == RiskSeverity.HIGH), None)
        if high_risk is not None:
            level = {
                ConvictionLevel.HIGH: ConvictionLevel.MEDIUM,
                ConvictionLevel.MEDIUM: ConvictionLevel.LOW,
                ConvictionLevel.LOW: ConvictionLevel.LOW,
            }[level]

        return Conviction(
            level=level,
            direct_precedents=direct_count,
            exact_sector_deals=stats.exact_sector_deals,
            downgraded_by=high_risk.title if high_risk else None,
        )

    def _precedent_table(
        self, stats: AcquirerStats, deals: list[Transaction], context: _Context
    ) -> list[PrecedentRow]:
        """Direct precedents first, then the closest other deals: by sector similarity,
        then by distance from the target's EV on a log scale (decision 6)."""
        target = context.target

        def closeness(t: Transaction):
            return (
                -context.similarity.weight(target.sector, t.sector),
                abs(math.log(t.ev_mm / target.ev_mm)),
                t.transaction_id,
            )

        direct_ids = {t.transaction_id for t in stats.direct_precedents}
        direct = sorted(stats.direct_precedents, key=closeness)
        others = sorted((t for t in deals if t.transaction_id not in direct_ids), key=closeness)
        return [
            PrecedentRow(t, t.transaction_id in direct_ids, context.similarity.weight(target.sector, t.sector))
            for t in (direct + others)[: self._config.precedent_table_size]
        ]

    def _facts(
        self,
        entry: ShortlistEntry,
        stats: AcquirerStats,
        baseline: PeerBaseline,
        valuation: ValuationComps,
        conviction: Conviction,
        risks: list[RiskFlag],
        context: _Context,
        shortlist_size: int,
    ) -> list[Fact]:
        """Every fact the LLM may cite, numbered F1, F2, ... in a fixed order."""
        target = context.target
        facts: list[Fact] = []

        def add(label: str, value: str) -> None:
            facts.append(Fact(f"F{len(facts) + 1}", label, value))

        # Identity and ranking
        add("Acquirer type", stats.acquirer_type)
        rank_text = f"{entry.ranking.score:.1f}, rank {entry.rank} of {shortlist_size}"
        if entry.reserved_slot:
            rank_text += " (strategic slot reserved by --min-strategics)"
        add("Ranking score", rank_text)
        add(
            "Deals in data",
            f"{stats.total_deals} total, {stats.closed_deals} closed, {stats.first_year}-{stats.last_year}",
        )

        # Sector mix, in the order of the score's explanation
        for contribution in entry.ranking.contributions:
            add(
                f"Deals in {contribution.sector}",
                f"{contribution.deal_count} (similarity to {target.sector}: "
                f"{fmt.weight(context.similarity.weight(target.sector, contribution.sector))})",
            )

        # Size
        add(
            "Deal size range",
            f"smallest {fmt.money(stats.min_ev.ev_mm)} ({stats.min_ev.transaction_id}), "
            f"median {fmt.money(stats.median_ev_mm)}, "
            f"largest {fmt.money(stats.max_ev.ev_mm)} ({stats.max_ev.transaction_id})",
        )
        add(
            "Target EV against their deals",
            f"{fmt.money(target.ev_mm)} is larger than {fmt.percent(stats.share_below_target_ev)} of their deals",
        )

        # Recent activity
        first, last = context.recent_years
        last_deal = stats.last_deal
        add(
            "Recent activity",
            f"{len(stats.recent_deals)} deals in {first}-{last}; last deal {last_deal.deal_year} "
            f"Q{last_deal.deal_quarter} ({last_deal.transaction_id})",
        )

        # Profile history (decision 2)
        regional_label = (
            f"Targets in {target.region}" if target.region
            else "Regional targets (not " + " or ".join(self._config.non_regional_geographies) + ")"
        )
        for label, count, share in (
            (f"{target.ownership} targets", stats.private_deals, stats.private_share),
            (f"Targets with EBITDA margin at or above {target.min_margin_pct:.1f}%",
             stats.high_margin_deals, stats.high_margin_share),
            (regional_label, stats.regional_deals, stats.regional_share),
        ):
            add(label, f"{count} of {stats.total_deals} deals ({fmt.percent(share)})")

        # Direct precedents and sector depth (decision 7)
        direct_ids = ", ".join(t.transaction_id for t in stats.direct_precedents) or "none"
        years_first, years_last = context.direct_years
        add(
            f"Direct precedents (closed, {' / '.join(context.direct_sectors)}, "
            f"{fmt.money(context.min_ev_mm)}-{fmt.money(context.max_ev_mm)}, {years_first}-{years_last})",
            f"{len(stats.direct_precedents)} ({direct_ids})",
        )
        if valuation.direct_median_ev_ebitda is not None:
            add("Direct precedents, median EV/EBITDA", fmt.ev_ebitda(valuation.direct_median_ev_ebitda))
        add(f"Deals in {target.sector}, any outcome or size", str(stats.exact_sector_deals))

        # Valuation (decision 3)
        market = valuation.market
        market_text = f"{market.count} deals"
        if market.median_ev_ebitda is not None and market.median_ev_revenue is not None:
            market_text += (
                f", median EV/EBITDA {fmt.ev_ebitda(market.median_ev_ebitda)}, "
                f"median EV/Revenue {fmt.ev_revenue(market.median_ev_revenue)}"
            )
        add(
            f"Market comps (closed {market.sector}, {fmt.money(market.min_ev_mm)}-{fmt.money(market.max_ev_mm)})",
            market_text,
        )
        if valuation.acquirer_median_ev_ebitda is not None and valuation.acquirer_median_ev_revenue is not None:
            add(
                "Their closed deals, median multiples",
                f"EV/EBITDA {fmt.ev_ebitda(valuation.acquirer_median_ev_ebitda)}, "
                f"EV/Revenue {fmt.ev_revenue(valuation.acquirer_median_ev_revenue)} "
                f"(n={valuation.acquirer_closed_count})",
            )

        # Outcomes
        add("Withdrawn or terminated deals", f"{stats.withdrawn} withdrawn, {stats.terminated} terminated")

        # Peer baselines (decision 6): what is typical for this acquirer type
        peers = f"Peer median ({baseline.acquirer_type}, {baseline.peer_count} acquirers)"
        add(f"{peers}: total deals", _count(baseline.total_deals))
        add(f"{peers}: deals in {first}-{last}", _count(baseline.recent_deals))
        add(f"{peers}: median deal size", fmt.money(baseline.median_ev_mm))
        add(f"{peers}: largest deal", fmt.money(baseline.max_ev_mm))
        add(f"{peers}: deals in {target.sector}", _count(baseline.exact_sector_deals))
        add(f"{peers}: direct precedents", _count(baseline.direct_precedents))
        add(f"{peers}: share of {target.ownership.lower()} targets", fmt.percent(baseline.private_share))
        add(f"{peers}: share of targets with margin at or above {target.min_margin_pct:.1f}%",
            fmt.percent(baseline.high_margin_share))
        add(f"{peers}: share of {'targets in ' + target.region if target.region else 'regional targets'}",
            fmt.percent(baseline.regional_share))

        # Conviction and data risks, computed in code
        conviction_text = (
            f"{conviction.level.value}: {conviction.direct_precedents} direct precedents, "
            f"{conviction.exact_sector_deals} deals in {target.sector}"
        )
        if conviction.downgraded_by:
            conviction_text += f"; lowered one level by the {conviction.downgraded_by.lower()} risk"
        add("Conviction (computed)", conviction_text)
        for risk in risks:
            add(f"Data risk, {risk.severity.value} severity: {risk.title}", risk.detail)

        return facts


def _median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _count(value: float) -> str:
    """Peer medians of counts can land on a half: show 2 or 2.5, never 2.0."""
    return f"{value:g}"
