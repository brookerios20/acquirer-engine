from collections import defaultdict
from dataclasses import dataclass

SPONSOR = "Financial Sponsor"
STRATEGIC = "Strategic"

CLOSED = "Closed"
# Rumored isn't in the brief's field table but appears in 10 rows (decision 3)
OUTCOMES = ("Closed", "Withdrawn", "Pending", "Terminated", "Rumored")


@dataclass(frozen=True)
class Transaction:
    """One CSV row after the repository's data rules. `target_ebitda_mm` and
    `sub_sector` are dropped (decision 3)."""

    transaction_id: str
    target_company: str
    acquirer: str
    acquirer_type: str
    sector: str
    deal_year: int
    deal_quarter: int
    deal_type: str
    geography: str
    financing_type: str
    ev_mm: float
    revenue_mm: float
    ebitda_margin_pct: float
    revenue_growth_pct: float
    ev_ebitda: float
    ev_revenue: float
    synergy_pct: float
    outcome: str
    rationale_tags: tuple[str, ...]
    num_bidders: int
    days_to_close: int | None
    target_ownership: str

    @property
    def is_closed(self) -> bool:
        return self.outcome == CLOSED


@dataclass(frozen=True)
class SkippedRow:
    """A CSV row the repository couldn't use, listed in the run log."""

    line_number: int
    transaction_id: str
    reason: str


@dataclass(frozen=True)
class LoadedTransactions:
    transactions: list[Transaction]
    skipped_rows: list[SkippedRow]
    source: str
    sha256: str


def group_by_acquirer(transactions: list[Transaction]) -> dict[str, list[Transaction]]:
    groups: dict[str, list[Transaction]] = defaultdict(list)
    for transaction in transactions:
        groups[transaction.acquirer].append(transaction)
    return dict(groups)
