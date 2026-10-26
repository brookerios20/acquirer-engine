import pytest

from acquirer_engine.config import AnalysisConfig
from acquirer_engine.errors import InputError
from acquirer_engine.services import TargetProfileService
from tests.factories import make_transaction

# Healthcare Services margins 10, 12, 14, 16, 18: 75th percentile (linear) = 16.0
TRANSACTIONS = [
    make_transaction(sector="Healthcare Services", ebitda_margin_pct=m, geography=g)
    for m, g in ((10, "Midwest"), (12, "Southeast"), (14, "National"), (16, "Midwest"), (18, "Multi-Regional"))
] + [make_transaction(sector="Dental", ebitda_margin_pct=40.0, target_ownership="PE-Backed")]


def build(**overrides):
    args = dict(sector="Healthcare Services", ev_mm=200.0, ownership="Private", min_margin_pct=None, region=None)
    args.update(overrides)
    return TargetProfileService(AnalysisConfig()).build(TRANSACTIONS, **args)


def test_default_margin_floor_is_sector_top_quartile():
    assert build().min_margin_pct == 16.0


def test_explicit_margin_is_kept():
    assert build(min_margin_pct=20.0).min_margin_pct == 20.0


def test_unknown_sector_lists_known_values():
    with pytest.raises(InputError, match="Known sectors: Dental, Healthcare Services"):
        build(sector="Veterinary")


def test_region_must_be_a_named_region():
    assert build(region="Midwest").region == "Midwest"
    with pytest.raises(InputError, match="Known regions: Midwest, Southeast"):
        build(region="National")


def test_unknown_ownership_fails():
    with pytest.raises(InputError, match="PE-Backed, Private"):
        build(ownership="Family-Owned")


def test_ev_must_be_positive():
    with pytest.raises(InputError):
        build(ev_mm=0)
