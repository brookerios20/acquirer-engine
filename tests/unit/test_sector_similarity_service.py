import math

import pytest

from acquirer_engine.models import STRATEGIC
from acquirer_engine.services import SectorSimilarityService
from tests.factories import make_transaction


def deals(acquirer, *sectors, acquirer_type="Financial Sponsor"):
    return [make_transaction(acquirer=acquirer, sector=s, acquirer_type=acquirer_type) for s in sectors]


def test_cosine_of_sponsor_deal_counts():
    # S1 = (A: 2, B: 1), S2 = (A: 1, B: 1), S3 = (C: 1)
    transactions = deals("A", "S1", "S1", "S2") + deals("B", "S1", "S2") + deals("C", "S3")

    similarity = SectorSimilarityService().compute(transactions)

    assert similarity.weight("S1", "S2") == pytest.approx(3 / (math.sqrt(5) * math.sqrt(2)))
    assert similarity.weight("S2", "S1") == similarity.weight("S1", "S2")
    assert similarity.weight("S1", "S3") == 0.0
    assert similarity.weight("S1", "S1") == 1.0


def test_strategics_are_ignored():
    sponsors = deals("A", "S1", "S2") + deals("B", "S3")
    strategic = deals("Op Co", "S1", "S3", acquirer_type=STRATEGIC)

    with_strategic = SectorSimilarityService().compute(sponsors + strategic)
    without = SectorSimilarityService().compute(sponsors)

    assert with_strategic.weights == without.weights
    assert with_strategic.weight("S1", "S3") == 0.0


def test_sector_without_sponsor_deals_scores_zero():
    transactions = deals("A", "S1", "S2") + deals("Op Co", "S4", acquirer_type=STRATEGIC)

    similarity = SectorSimilarityService().compute(transactions)

    assert similarity.weight("S1", "S4") == 0.0
    assert similarity.weight("S4", "S4") == 1.0


def test_nearest_breaks_ties_by_name():
    # S2 and S3 are equally close to S1; S4 is unrelated
    transactions = deals("A", "S1", "S2", "S3") + deals("B", "S4")

    similarity = SectorSimilarityService().compute(transactions)

    assert similarity.nearest("S1", 2) == ["S2", "S3"]
    assert similarity.nearest("S1", 3) == ["S2", "S3", "S4"]
