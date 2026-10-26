from ..config import AnalysisConfig
from ..errors import InputError
from ..models import BacktestResult, BacktestRow, TargetProfile, Transaction
from ..repositories import TransactionRepository
from .acquirer_ranking_model import AcquirerRankingModel
from .sector_similarity_service import SectorSimilarityService


class BacktestService:
    """Time-split check of ranking models (decision 4).

    For each held-out deal, the model is fit only on deals announced before it, scores
    every acquirer active before it, and we record where the real buyer ranked. A buyer
    with no earlier deal can't be ranked and counts as a miss.
    """

    def __init__(
        self,
        repository: TransactionRepository,
        models: dict[str, AcquirerRankingModel],
        similarity_service: SectorSimilarityService,
        config: AnalysisConfig,
    ):
        self._repository = repository
        self._models = models
        self._similarity_service = similarity_service
        self._config = config

    def run(self, sector: str, ev_mm: float) -> BacktestResult:
        """`sector` and `ev_mm` define the "profile-like" subset; ranking ignores them."""
        config = self._config
        transactions = self._repository.load().transactions
        sectors = sorted({t.sector for t in transactions})
        if sector not in sectors:
            raise InputError(f"Unknown sector '{sector}'. Known sectors: {', '.join(sectors)}")
        target = TargetProfile(sector, ev_mm, "", 0.0, None)
        split_year = config.backtest_test_from_year
        held_out = sorted(
            (t for t in transactions if t.deal_year >= split_year),
            key=lambda t: (t.deal_year, t.deal_quarter, t.transaction_id),
        )

        # "Profile-like" uses only pre-split deals, so the test set itself can't leak in
        similarity = self._similarity_service.compute([t for t in transactions if t.deal_year < split_year])
        min_ev, max_ev = target.ev_mm * config.profile_ev_low, target.ev_mm * config.profile_ev_high
        profile_like = [
            t for t in held_out
            if similarity.weight(target.sector, t.sector) >= config.profile_min_similarity
            and min_ev <= t.ev_mm <= max_ev
        ]

        rows = []
        for name, model in self._models.items():
            ranks = {t.transaction_id: self._rank_of_buyer(model, t, transactions) for t in held_out}
            for subset_name, subset in (("All", held_out), ("Profile-like", profile_like)):
                rows.append(self._metrics(name, subset_name, [ranks[t.transaction_id] for t in subset]))

        last_year = max(t.deal_year for t in transactions)
        profile = (
            f"sector similarity >= {config.profile_min_similarity:g} to {target.sector} "
            f"(from pre-{split_year} deals), EV ${min_ev:,.0f}-{max_ev:,.0f}M"
        )
        return BacktestResult(config.backtest_k, (split_year, last_year), profile, rows)

    def _rank_of_buyer(
        self, model: AcquirerRankingModel, deal: Transaction, transactions: list[Transaction]
    ) -> int | None:
        """1-based rank of the real buyer among acquirers active before the deal, or None."""
        past = [t for t in transactions if _period(t) < _period(deal)]
        candidates = sorted({t.acquirer for t in past})
        if deal.acquirer not in candidates:
            return None

        model.fit(past)
        # The models use sector and size; the other traits don't affect ranking
        target = TargetProfile(deal.sector, deal.ev_mm, deal.target_ownership, 0.0, None)
        ranked = sorted(model.score(target, candidates), key=lambda s: (-s.score, s.acquirer))
        return 1 + [s.acquirer for s in ranked].index(deal.acquirer)

    def _metrics(self, model: str, subset: str, ranks: list[int | None]) -> BacktestRow:
        k = self._config.backtest_k
        count = len(ranks)
        hits = sum(1 for r in ranks if r is not None and r <= k)
        reciprocal = sum(1 / r for r in ranks if r is not None)
        return BacktestRow(
            model=model,
            subset=subset,
            deal_count=count,
            hit_at_k=hits / count if count else 0.0,
            mrr=reciprocal / count if count else 0.0,
        )


def _period(t: Transaction) -> tuple[int, int]:
    """Deals in the same quarter don't see each other."""
    return (t.deal_year, t.deal_quarter)
