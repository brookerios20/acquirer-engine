from ..config import AnalysisConfig
from ..errors import InputError
from ..models import (
    STRATEGIC,
    RankingScore,
    Shortlist,
    ShortlistEntry,
    TargetProfile,
    Transaction,
)
from .acquirer_ranking_model import AcquirerRankingModel


class CandidateScoringService:
    """Ranks every acquirer in the data and selects the shortlist (decisions 4 and 5)."""

    def __init__(self, ranking_model: AcquirerRankingModel, config: AnalysisConfig):
        self._ranking_model = ranking_model
        self._config = config

    def shortlist(
        self, transactions: list[Transaction], target: TargetProfile, min_strategics: int = 0
    ) -> Shortlist:
        size = self._config.shortlist_size
        if not 0 <= min_strategics <= size:
            raise InputError(f"--min-strategics must be between 0 and {size}, got {min_strategics}")

        acquirer_types = {t.acquirer: t.acquirer_type for t in transactions}
        candidates = sorted(acquirer_types)

        self._ranking_model.fit(transactions)
        ranked = sorted(
            self._ranking_model.score(target, candidates),
            key=lambda s: (-s.score, s.acquirer),  # ties broken by name
        )

        selected = ranked[:size]
        reserved: set[str] = set()
        warnings: list[str] = []

        if len(selected) < size:
            warnings.append(f"Only {len(selected)} rankable acquirers in the data; showing all of them.")

        def is_strategic(score: RankingScore) -> bool:
            return acquirer_types[score.acquirer] == STRATEGIC

        strategics_selected = sum(is_strategic(s) for s in selected)
        if strategics_selected < min_strategics:
            # Eligible = a strategic with at least one deal in the target's sector (decision 5)
            in_sector = {t.acquirer for t in transactions if t.sector == target.sector}
            selected_names = {s.acquirer for s in selected}
            eligible = [
                s for s in ranked
                if is_strategic(s) and s.acquirer in in_sector and s.acquirer not in selected_names
            ]
            to_add = eligible[: min_strategics - strategics_selected]
            available = strategics_selected + len(eligible)
            if available < min_strategics:
                warnings.append(f"Requested {min_strategics} strategics, {available} eligible.")

            # Free one slot per added strategic by dropping the lowest-ranked sponsors.
            # The list is full here: if it were short, every acquirer would already be on it.
            sponsors_low_to_high = [s for s in reversed(selected) if not is_strategic(s)]
            dropped = {s.acquirer for s in sponsors_low_to_high[: len(to_add)]}

            selected = [s for s in selected if s.acquirer not in dropped] + to_add
            selected.sort(key=lambda s: (-s.score, s.acquirer))
            reserved = {s.acquirer for s in to_add}

        entries = [
            ShortlistEntry(
                rank=rank,
                acquirer=score.acquirer,
                acquirer_type=acquirer_types[score.acquirer],
                ranking=score,
                reserved_slot=score.acquirer in reserved,
            )
            for rank, score in enumerate(selected, start=1)
        ]
        return Shortlist(entries, warnings)
