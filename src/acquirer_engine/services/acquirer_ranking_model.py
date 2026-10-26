from typing import Protocol

from ..models import RankingScore, TargetProfile, Transaction


class AcquirerRankingModel(Protocol):
    """Scores how likely each candidate is to buy the target (decision 4).

    A replacement model must beat the current one in the `backtest` command.
    """

    def fit(self, transactions: list[Transaction]) -> None: ...

    def score(self, target: TargetProfile, candidates: list[str]) -> list[RankingScore]: ...
