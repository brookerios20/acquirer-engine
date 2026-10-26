from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestRow:
    """One model's score on one set of held-out deals (decision 4)."""

    model: str
    subset: str
    deal_count: int
    hit_at_k: float          # share of deals whose real buyer ranked in the top k
    mrr: float               # mean of 1 / rank of the real buyer; 0 when it can't be ranked


@dataclass(frozen=True)
class BacktestResult:
    k: int
    test_years: tuple[int, int]
    profile_description: str
    rows: list[BacktestRow]
