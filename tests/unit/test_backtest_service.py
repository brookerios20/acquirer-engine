import pytest

from acquirer_engine.config import AnalysisConfig
from acquirer_engine.errors import InputError
from acquirer_engine.models import LoadedTransactions
from acquirer_engine.services import BacktestService, DealCountBaseline, SectorSimilarityService
from tests.factories import make_transaction


class FixedRepository:
    def __init__(self, transactions):
        self._transactions = transactions

    def load(self):
        return LoadedTransactions(self._transactions, [], "fixture", "sha")


class SpyModel(DealCountBaseline):
    """Deal count, recording what each fit() was given."""

    def __init__(self):
        super().__init__()
        self.fits = []

    def fit(self, transactions):
        self.fits.append(list(transactions))
        super().fit(transactions)


def deal(acquirer, year, quarter=2, sector="Healthcare Services", ev_mm=200.0):
    return make_transaction(acquirer=acquirer, deal_year=year, deal_quarter=quarter, sector=sector, ev_mm=ev_mm)


# Before 2022: A has 3 deals, B 2, C 1. Held out: A, then C, then a first-time buyer.
TRAIN = [deal("A", 2019), deal("A", 2020), deal("A", 2021), deal("B", 2020), deal("B", 2021), deal("C", 2021)]
TEST = [deal("A", 2022, 1), deal("C", 2022, 2), deal("Newcomer", 2023)]


def run(transactions, models, config=AnalysisConfig(backtest_k=1)):
    service = BacktestService(FixedRepository(transactions), models, SectorSimilarityService(), config)
    return service.run("Healthcare Services", 200.0)


def test_hit_rate_and_mrr():
    result = run(TRAIN + TEST, {"count": DealCountBaseline()})
    all_deals = next(r for r in result.rows if r.subset == "All")

    # A ranks 1st; C ranks 3rd (A has 4 deals by then, B 2, C 1); Newcomer can't be ranked
    assert all_deals.deal_count == 3
    assert all_deals.hit_at_k == pytest.approx(1 / 3)
    assert all_deals.mrr == pytest.approx((1 + 1 / 3) / 3)
    assert result.test_years == (2022, 2023)


def test_each_fit_sees_only_earlier_quarters():
    same_quarter = [deal("B", 2022, 1)]  # announced alongside A's held-out deal
    spy = SpyModel()

    run(TRAIN + TEST + same_quarter, {"spy": spy})

    a_fit = spy.fits[0]  # held-out deals are processed in date order; A's 2022 Q1 deal is first
    assert all((t.deal_year, t.deal_quarter) < (2022, 1) for t in a_fit)
    assert len(a_fit) == len(TRAIN)


def test_profile_like_subset_uses_sector_similarity_and_size():
    off_profile = [deal("A", 2023, sector="Dental", ev_mm=200.0), deal("A", 2023, ev_mm=1000.0)]

    result = run(TRAIN + TEST + off_profile, {"count": DealCountBaseline()})
    sizes = {r.subset: r.deal_count for r in result.rows}

    # Dental shares no sponsor with Healthcare Services before 2022; $1B is outside $75-500M
    assert sizes == {"All": 5, "Profile-like": 3}


def test_unknown_sector_fails():
    service = BacktestService(FixedRepository(TRAIN), {}, SectorSimilarityService(), AnalysisConfig())

    with pytest.raises(InputError, match="Known sectors"):
        service.run("Veterinary", 200.0)
