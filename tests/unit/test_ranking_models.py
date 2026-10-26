import pytest

from acquirer_engine.services import DealCountBaseline, SectorSimilarityService, SimilarityWeightedCount
from tests.factories import make_target, make_transaction


def deals(acquirer, *sectors):
    return [make_transaction(acquirer=acquirer, sector=s) for s in sectors]


# A: 2 target-sector deals. B: 3 deals in a loosely related sector. C: 4 unrelated deals.
# "Near" shares one sponsor (Bridge) with the target sector; "Far" shares none.
TRANSACTIONS = (
    deals("A", "Target", "Target")
    + deals("B", "Near", "Near", "Near")
    + deals("C", "Far", "Far", "Far", "Far")
    + deals("Bridge", "Target", "Near")
)


def test_similarity_weighted_count_sums_sector_weights():
    model = SimilarityWeightedCount(SectorSimilarityService())
    model.fit(TRANSACTIONS)
    target = make_target(sector="Target")

    scores = {s.acquirer: s for s in model.score(target, ["A", "B", "C"])}
    near_weight = SectorSimilarityService().compute(TRANSACTIONS).weight("Target", "Near")

    assert scores["A"].score == pytest.approx(2.0)
    assert scores["B"].score == pytest.approx(3 * near_weight)
    assert scores["C"].score == 0.0
    contribution = scores["B"].contributions[0]
    assert (contribution.sector, contribution.deal_count) == ("Near", 3)
    assert contribution.weight == pytest.approx(near_weight)


def test_similarity_weighted_count_changes_with_target():
    model = SimilarityWeightedCount(SectorSimilarityService())
    model.fit(TRANSACTIONS)

    far_scores = {s.acquirer: s.score for s in model.score(make_target(sector="Far"), ["A", "C"])}

    assert far_scores["C"] == 4.0
    assert far_scores["A"] == 0.0


def test_similarity_weighted_count_requires_fit():
    with pytest.raises(RuntimeError):
        SimilarityWeightedCount(SectorSimilarityService()).score(make_target(), ["A"])


def test_deal_count_baseline_ignores_target():
    model = DealCountBaseline()
    model.fit(TRANSACTIONS)

    for sector in ("Target", "Far"):
        scores = {s.acquirer: s.score for s in model.score(make_target(sector=sector), ["A", "B", "C"])}
        assert scores == {"A": 2.0, "B": 3.0, "C": 4.0}
