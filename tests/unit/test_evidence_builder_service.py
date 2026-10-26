from unittest.mock import create_autospec

import pytest

from acquirer_engine.config import AnalysisConfig
from acquirer_engine.models import (
    STRATEGIC,
    ConvictionLevel,
    RankingScore,
    RiskSeverity,
    SectorSimilarity,
    Shortlist,
    ShortlistEntry,
)
from acquirer_engine.services import EvidenceBuilderService, SectorSimilarityService
from tests.factories import make_target, make_transaction

HCS, PG, BH, DENTAL = "Healthcare Services", "Physician Groups", "Behavioral Health", "Dental"

# Fixed similarity so each test controls which sectors are "nearest" (PG and BH)
SIMILARITY = SectorSimilarity({
    HCS: {PG: 0.9, BH: 0.8, DENTAL: 0.1},
    PG: {HCS: 0.9, BH: 0.5, DENTAL: 0.2},
    BH: {HCS: 0.8, PG: 0.5, DENTAL: 0.3},
    DENTAL: {HCS: 0.1, PG: 0.2, BH: 0.3},
})
TARGET = make_target(sector=HCS, ev_mm=200.0)


def deal(acquirer="A", **overrides):
    return make_transaction(acquirer=acquirer, **overrides)


def build(transactions, *acquirers, config=AnalysisConfig(), target=TARGET):
    """Run the builder with sector similarity isolated (decision 1 allows autospec here)."""
    similarity_service = create_autospec(SectorSimilarityService, instance=True)
    similarity_service.compute.return_value = SIMILARITY
    types = {t.acquirer: t.acquirer_type for t in transactions}
    shortlist = Shortlist([
        ShortlistEntry(rank, name, types[name], RankingScore(name, 1.0, []), False)
        for rank, name in enumerate(acquirers or ["A"], start=1)
    ])
    return EvidenceBuilderService(similarity_service, config).build(transactions, target, shortlist)


def pack(transactions, acquirer="A"):
    return next(p for p in build(transactions, acquirer).packs if p.acquirer == acquirer)


def ids(transactions):
    return [t.transaction_id for t in transactions]


# --- Direct precedents and conviction (decision 7) ------------------------------------


def test_direct_precedent_rules():
    direct = [
        deal(sector=HCS, ev_mm=200.0),
        deal(sector=PG, ev_mm=100.0, deal_year=2020),     # lower size and year bounds are inclusive
        deal(sector=BH, ev_mm=400.0),                      # upper size bound is inclusive
    ]
    not_direct = [
        deal(sector=DENTAL, ev_mm=200.0),                  # not a near sector
        deal(sector=HCS, ev_mm=99.9),                      # below 0.5x
        deal(sector=HCS, ev_mm=400.1),                     # above 2x
        deal(sector=HCS, ev_mm=200.0, deal_year=2019),     # older than 5 years of data
        deal(sector=HCS, ev_mm=200.0, outcome="Withdrawn"),  # not closed
    ]

    result = pack(direct + not_direct)

    assert sorted(ids(result.stats.direct_precedents)) == sorted(ids(direct))
    assert result.conviction.level == ConvictionLevel.HIGH
    assert result.conviction.direct_precedents == 3


@pytest.mark.parametrize(
    "transactions, expected",
    [
        ([deal(sector=HCS), deal(sector=DENTAL, ev_mm=500.0)], ConvictionLevel.MEDIUM),  # 1 direct
        # 0 direct, but 2 deals of any outcome in the exact sector
        ([deal(sector=HCS, deal_year=2016, ev_mm=300.0), deal(sector=HCS, outcome="Withdrawn")],
         ConvictionLevel.MEDIUM),
        ([deal(sector=HCS, deal_year=2016, ev_mm=300.0), deal(sector=DENTAL)], ConvictionLevel.LOW),
    ],
)
def test_conviction_levels(transactions, expected):
    assert pack(transactions).conviction.level == expected


def test_high_severity_size_stretch_lowers_conviction_one_level():
    # Medium on 2 exact-sector deals, but the target is more than 2x their largest deal
    result = pack([deal(sector=HCS, ev_mm=50.0), deal(sector=HCS, ev_mm=40.0)])

    stretch = next(r for r in result.data_risks if r.title == "Size stretch")
    assert stretch.severity == RiskSeverity.HIGH
    assert result.conviction.level == ConvictionLevel.LOW
    assert result.conviction.downgraded_by == "Size stretch"


@pytest.mark.parametrize("largest", [100.0, 150.0])
def test_size_stretch_at_or_below_2x_is_normal_and_does_not_downgrade(largest):
    result = pack([deal(sector=HCS, ev_mm=largest), deal(sector=HCS, ev_mm=largest)])

    stretch = next(r for r in result.data_risks if r.title == "Size stretch")
    assert stretch.severity == RiskSeverity.NORMAL
    assert result.conviction.downgraded_by is None


def test_no_size_stretch_when_they_have_done_a_larger_deal():
    result = pack([deal(sector=HCS, ev_mm=250.0)])

    assert "Size stretch" not in [r.title for r in result.data_risks]


# --- Data risks (decision 10) ----------------------------------------------------------


def test_thin_evidence_trips_below_two_direct_precedents():
    one = pack([deal(sector=HCS)])
    two = pack([deal(sector=HCS), deal(sector=PG)])

    assert "Thin evidence" in [r.title for r in one.data_risks]
    assert "Thin evidence" not in [r.title for r in two.data_risks]


def test_recent_pace_trips_at_five_deals_in_last_two_years():
    four_recent = [deal(sector=DENTAL) for _ in range(4)]

    busy = pack(four_recent + [deal(sector=DENTAL, deal_year=2023)])
    steady = pack(four_recent + [deal(sector=DENTAL, deal_year=2022)])

    pace = next(r for r in busy.data_risks if r.title == "Recent pace")
    assert len(pace.transaction_ids) == 5
    assert "Recent pace" not in [r.title for r in steady.data_risks]


def test_competitive_process_is_a_target_level_risk():
    comps = [deal("Other", sector=HCS, num_bidders=n) for n in (3, 4, 1, 1)]  # 2 of 4 contested
    transactions = comps + [deal("A", sector=DENTAL), deal("B", sector=DENTAL)]

    evidence = build(transactions, "A", "B")

    assert [r.title for r in evidence.target_risks] == ["Competitive process"]
    assert evidence.target_risks[0].detail.startswith("2 of 4")
    for result in evidence.packs:
        assert "Competitive process" in [r.title for r in result.data_risks]


def test_competitive_process_does_not_trip_below_half():
    comps = [deal("Other", sector=HCS, num_bidders=n) for n in (3, 1, 1)]

    assert build(comps + [deal("A", sector=DENTAL)], "A").target_risks == []


# --- Valuation and precedent table (decisions 3 and 6) ---------------------------------


def test_market_comps_use_closed_target_sector_deals_in_size_band():
    in_set = [
        deal("X", sector=HCS, ev_mm=100.0, ev_ebitda=10.0, ev_revenue=2.0),
        deal("Y", sector=HCS, ev_mm=300.0, ev_ebitda=12.0, ev_revenue=3.0),
        deal("Z", sector=HCS, ev_mm=400.0, ev_ebitda=20.0, ev_revenue=4.0, deal_year=2015),  # any year
    ]
    out_of_set = [
        deal("X", sector=HCS, ev_mm=500.0, ev_ebitda=30.0),
        deal("Y", sector=HCS, ev_mm=200.0, ev_ebitda=30.0, outcome="Rumored"),
        deal("Z", sector=PG, ev_mm=200.0, ev_ebitda=30.0),
    ]

    market = build(in_set + out_of_set, "X").market_comps

    assert market.transaction_ids == sorted(ids(in_set))
    assert market.median_ev_ebitda == 12.0
    assert market.median_ev_revenue == 3.0


def test_acquirer_valuation_uses_their_closed_deals():
    result = pack([
        deal(sector=HCS, ev_ebitda=10.0),
        deal(sector=DENTAL, ev_mm=900.0, ev_ebitda=14.0),
        deal(sector=HCS, ev_ebitda=30.0, outcome="Terminated"),
    ])

    assert result.valuation.acquirer_closed_count == 2
    assert result.valuation.acquirer_median_ev_ebitda == 12.0
    assert result.valuation.direct_median_ev_ebitda == 10.0


def test_precedent_table_puts_direct_first_then_closest():
    direct = deal(sector=HCS, ev_mm=200.0)
    withdrawn_hcs = deal(sector=HCS, ev_mm=200.0, outcome="Withdrawn")
    pg_near_size = deal(sector=PG, ev_mm=300.0, deal_year=2018)
    pg_far_size = deal(sector=PG, ev_mm=1000.0, deal_year=2018)
    bh = deal(sector=BH, ev_mm=200.0, deal_year=2018)
    dental = deal(sector=DENTAL, ev_mm=200.0, deal_year=2018)

    result = pack([dental, bh, pg_far_size, pg_near_size, withdrawn_hcs, direct])

    assert [r.transaction for r in result.precedents] == [direct, withdrawn_hcs, pg_near_size, pg_far_size, bh]
    assert [r.is_direct for r in result.precedents] == [True, False, False, False, False]


# --- Profile history and peer baselines (decisions 2 and 6) ----------------------------


def test_profile_history_counts():
    result = pack([
        deal(sector=DENTAL, target_ownership="Private", ebitda_margin_pct=20.0, geography="Midwest"),
        deal(sector=DENTAL, target_ownership="PE-Backed", ebitda_margin_pct=16.8, geography="National"),
        deal(sector=DENTAL, target_ownership="Public", ebitda_margin_pct=10.0, geography="Multi-Regional"),
        deal(sector=DENTAL, target_ownership="Private", ebitda_margin_pct=12.0, geography="Southeast"),
    ])

    assert result.stats.private_deals == 2
    assert result.stats.high_margin_deals == 2          # 16.8 is at the floor, so it counts
    assert result.stats.regional_deals == 2
    assert result.stats.private_share == 0.5


def test_peer_baseline_is_median_across_same_type():
    transactions = (
        [deal("A", sector=DENTAL)]
        + [deal("B", sector=DENTAL) for _ in range(3)]
        + [deal("C", sector=DENTAL) for _ in range(5)]
        + [deal("Op", sector=DENTAL, acquirer_type=STRATEGIC) for _ in range(9)]
    )

    evidence = build(transactions, "A", "Op")
    sponsor, strategic = (p.peer_baseline for p in evidence.packs)

    assert (sponsor.peer_count, sponsor.total_deals) == (3, 3)
    assert (strategic.peer_count, strategic.total_deals) == (1, 9)


# --- Facts -----------------------------------------------------------------------------


def test_facts_are_numbered_and_cite_direct_precedents():
    direct = deal(sector=HCS, ev_mm=200.0)
    result = pack([direct, deal(sector=DENTAL, ev_mm=900.0)])

    assert [f.id for f in result.facts] == [f"F{i}" for i in range(1, len(result.facts) + 1)]
    direct_fact = next(f for f in result.facts if f.label.startswith("Direct precedents (closed"))
    assert direct_fact.value == f"1 ({direct.transaction_id})"
    assert "Healthcare Services / Physician Groups / Behavioral Health" in direct_fact.label
    assert {"F1", direct.transaction_id} <= result.citable_ids
