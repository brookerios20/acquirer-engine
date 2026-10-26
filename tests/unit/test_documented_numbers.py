"""The numbers quoted in docs/decisions.md and eda/FINDINGS.md, checked against the real CSV.

If one of these fails, either the code or the document is wrong.
"""
from collections import Counter

import pytest

from acquirer_engine.config import AnalysisConfig
from acquirer_engine.models import STRATEGIC, ConvictionLevel, RankingScore, Shortlist, ShortlistEntry
from acquirer_engine.repositories import CsvTransactionRepository
from acquirer_engine.services import (
    BacktestService,
    CandidateScoringService,
    DealCountBaseline,
    EvidenceBuilderService,
    SectorSimilarityService,
    SimilarityWeightedCount,
    TargetProfileService,
)
from tests.factories import REAL_CSV

CONFIG = AnalysisConfig()


@pytest.fixture(scope="module")
def transactions():
    loaded = CsvTransactionRepository(REAL_CSV).load()
    assert loaded.skipped_rows == []
    return loaded.transactions


@pytest.fixture(scope="module")
def target(transactions):
    # The brief's target with the CLI defaults (decision 2)
    return TargetProfileService(CONFIG).build(
        transactions, "Healthcare Services", 200.0, "Private", None, None
    )


def run(transactions, target, min_strategics=0):
    similarity_service = SectorSimilarityService()
    scoring = CandidateScoringService(SimilarityWeightedCount(similarity_service), CONFIG)
    shortlist = scoring.shortlist(transactions, target, min_strategics)
    evidence = EvidenceBuilderService(similarity_service, CONFIG).build(transactions, target, shortlist)
    return shortlist, evidence


def test_loads_all_500_rows(transactions):
    assert len(transactions) == 500
    assert Counter(t.outcome for t in transactions)["Rumored"] == 10


def test_margin_floor_is_16_8(target):
    assert target.min_margin_pct == 16.8


def test_sector_similarity_to_healthcare_services(transactions):
    similarity = SectorSimilarityService().compute(transactions)
    closest = {s: round(similarity.weight("Healthcare Services", s), 2) for s in similarity.sectors}

    assert closest["Physician Groups"] == 0.83
    assert closest["Behavioral Health"] == 0.77
    assert closest["Medical Devices"] == 0.76
    assert closest["Dental"] == 0.72
    assert closest["Health IT"] == 0.44


def test_market_comps(transactions, target):
    _, evidence = run(transactions, target)

    assert evidence.market_comps.count == 10
    assert round(evidence.market_comps.median_ev_ebitda, 1) == 12.1
    assert round(evidence.market_comps.median_ev_revenue, 2) == 2.90


def test_default_top_10_conviction_mix(transactions, target):
    shortlist, evidence = run(transactions, target)
    levels = Counter(p.conviction.level for p in evidence.packs)

    assert len(shortlist.entries) == 10
    assert all(e.acquirer_type != STRATEGIC for e in shortlist.entries)
    assert levels == {ConvictionLevel.HIGH: 4, ConvictionLevel.MEDIUM: 5, ConvictionLevel.LOW: 1}
    assert not any(p.conviction.downgraded_by for p in evidence.packs)

    warburg = next(p for p in evidence.packs if p.acquirer == "Warburg Pincus")
    assert warburg.entry.rank == 6
    assert warburg.conviction.direct_precedents == 0


def test_backtest_table():
    similarity_service = SectorSimilarityService()
    service = BacktestService(
        CsvTransactionRepository(REAL_CSV),
        {"count": DealCountBaseline(), "weighted": SimilarityWeightedCount(similarity_service)},
        similarity_service,
        CONFIG,
    )
    rows = {(r.model, r.subset): r for r in service.run("Healthcare Services", 200.0).rows}

    def summary(model, subset):
        row = rows[(model, subset)]
        return row.deal_count, round(row.hit_at_k * 100, 1), round(row.mrr, 3)

    assert summary("count", "All") == (142, 40.1, 0.119)
    assert summary("count", "Profile-like") == (44, 52.3, 0.149)
    assert summary("weighted", "All") == (142, 38.0, 0.132)
    assert summary("weighted", "Profile-like") == (44, 50.0, 0.183)


def test_min_strategics_reserves_slots(transactions, target):
    shortlist, _ = run(transactions, target, min_strategics=4)

    strategics = [e for e in shortlist.entries if e.acquirer_type == STRATEGIC]
    assert len(shortlist.entries) == 10
    assert len(strategics) == 4
    assert all(e.reserved_slot for e in strategics)


def test_downgrade_applies_to_two_of_thirteen_eligible_strategics(transactions, target):
    eligible = sorted({
        t.acquirer for t in transactions
        if t.acquirer_type == STRATEGIC and t.sector == "Healthcare Services"
    })
    shortlist = Shortlist([
        ShortlistEntry(rank, name, STRATEGIC, RankingScore(name, 0.0, []), True)
        for rank, name in enumerate(eligible, start=1)
    ])

    evidence = EvidenceBuilderService(SectorSimilarityService(), CONFIG).build(transactions, target, shortlist)
    downgraded = sorted(p.acquirer for p in evidence.packs if p.conviction.downgraded_by)

    assert len(eligible) == 13
    assert downgraded == ["HCA Healthcare", "Trinity Health"]
