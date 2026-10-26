from collections import Counter

from ..models import RankingScore, SectorContribution, TargetProfile, Transaction, group_by_acquirer


class DealCountBaseline:
    """Ranks by number of deals, ignoring the target (decision 4).

    It returns the same order for every target, so it serves only as the bar to beat in
    the backtest.
    """

    def __init__(self) -> None:
        self._deals_by_acquirer: dict[str, list[Transaction]] = {}

    def fit(self, transactions: list[Transaction]) -> None:
        self._deals_by_acquirer = group_by_acquirer(transactions)

    def score(self, target: TargetProfile, candidates: list[str]) -> list[RankingScore]:
        scores = []
        for acquirer in candidates:
            sector_counts = Counter(t.sector for t in self._deals_by_acquirer.get(acquirer, []))
            contributions = [SectorContribution(s, n, 1.0) for s, n in sector_counts.items()]
            contributions.sort(key=lambda c: (-c.points, c.sector))
            scores.append(RankingScore(acquirer, float(sum(sector_counts.values())), contributions))
        return scores
