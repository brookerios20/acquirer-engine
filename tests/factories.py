"""Builders for test data. Each returns a valid object with plain defaults."""
import itertools
from pathlib import Path

from acquirer_engine.models import SPONSOR, TargetProfile, Transaction

REAL_CSV = Path(__file__).parent.parent / "data" / "ma_transactions_500.csv"

_ids = itertools.count(1)


def make_transaction(**overrides) -> Transaction:
    """A valid closed deal with plain defaults; tests override only what they check."""
    year = overrides.get("deal_year", 2024)
    fields = dict(
        transaction_id=f"MA-{year}-{next(_ids):04d}",
        target_company="Test Target",
        acquirer="Sponsor A",
        acquirer_type=SPONSOR,
        sector="Healthcare Services",
        deal_year=year,
        deal_quarter=2,
        deal_type="Platform Investment",
        geography="Midwest",
        financing_type="All Cash",
        ev_mm=200.0,
        revenue_mm=70.0,
        ebitda_margin_pct=15.0,
        revenue_growth_pct=10.0,
        ev_ebitda=12.0,
        ev_revenue=2.9,
        synergy_pct=10.0,
        outcome="Closed",
        rationale_tags=("Geographic Expansion",),
        num_bidders=2,
        days_to_close=120,
        target_ownership="Private",
    )
    fields.update(overrides)
    return Transaction(**fields)


def make_target(**overrides) -> TargetProfile:
    fields = dict(
        sector="Healthcare Services", ev_mm=200.0, ownership="Private", min_margin_pct=16.8, region=None
    )
    fields.update(overrides)
    return TargetProfile(**fields)
