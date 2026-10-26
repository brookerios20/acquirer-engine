from collections import Counter

from ..models import (
    RankingScore,
    SectorContribution,
    SectorSimilarity,
    TargetProfile,
    Transaction,
    group_by_acquirer,
)
from .sector_similarity_service import SectorSimilarityService


class SimilarityWeightedCount:
    """Default ranking model (decision 4).

    Score = sum over the acquirer's past deals of how similar each deal's sector is to the
    target's. A deal in the target's sector counts 1.0; a deal in a sector with similarity
    0.83 counts 0.83.
    """

    def __init__(self, similarity_service: SectorSimilarityService):
        self._similarity_service = similarity_service
        self._similarity: SectorSimilarity | None = None
        self._deals_by_acquirer: dict[str, list[Transaction]] = {}

    def fit(self, transactions: list[Transaction]) -> None:
        self._similarity = self._similarity_service.compute(transactions)
        self._deals_by_acquirer = group_by_acquirer(transactions)

    def score(self, target: TargetProfile, candidates: list[str]) -> list[RankingScore]:
        if self._similarity is None:
            raise RuntimeError("Call fit() before score()")

        scores = []
        for acquirer in candidates:
            sector_counts = Counter(t.sector for t in self._deals_by_acquirer.get(acquirer, []))
            contributions = [
                SectorContribution(sector, count, self._similarity.weight(target.sector, sector))
                for sector, count in sector_counts.items()
            ]
            contributions.sort(key=lambda c: (-c.points, c.sector))
            scores.append(RankingScore(acquirer, sum(c.points for c in contributions), contributions))
        return scores
