from dataclasses import dataclass
from enum import Enum

from .ranking import ShortlistEntry
from .transaction import Transaction


class ConvictionLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class RiskSource(str, Enum):
    DATA = "data"      # computed by code from the CSV
    LLM = "llm"        # written by the model; the CSV can't measure it


class RiskSeverity(str, Enum):
    HIGH = "high"
    NORMAL = "normal"


@dataclass(frozen=True)
class RiskFlag:
    title: str
    detail: str
    source: RiskSource
    severity: RiskSeverity
    transaction_ids: list[str]
    explanation: str = ""              # LLM's note on why it matters for this deal


@dataclass(frozen=True)
class Fact:
    """One item the LLM may cite, by `id` (F1, F2, ...). `value` is display text."""

    id: str
    label: str
    value: str


@dataclass(frozen=True)
class AcquirerStats:
    """Per-acquirer figures behind the evidence pack and the peer baselines (decision 6)."""

    acquirer: str
    acquirer_type: str
    total_deals: int
    closed_deals: int
    first_year: int
    last_year: int
    sector_counts: dict[str, int]
    exact_sector_deals: int            # any outcome, any size, in the target's sector
    min_ev: Transaction
    median_ev_mm: float
    max_ev: Transaction
    share_below_target_ev: float       # share of their deals smaller than the target
    recent_deals: list[Transaction]
    last_deal: Transaction
    private_deals: int                 # target ownership matches the profile
    high_margin_deals: int             # margin at or above the profile's floor
    regional_deals: int
    direct_precedents: list[Transaction]
    withdrawn: int
    terminated: int

    @property
    def private_share(self) -> float:
        return self.private_deals / self.total_deals

    @property
    def high_margin_share(self) -> float:
        return self.high_margin_deals / self.total_deals

    @property
    def regional_share(self) -> float:
        return self.regional_deals / self.total_deals


@dataclass(frozen=True)
class PeerBaseline:
    """Median of each stat across acquirers of the same type."""

    acquirer_type: str
    peer_count: int
    total_deals: float
    recent_deals: float
    median_ev_mm: float
    max_ev_mm: float
    exact_sector_deals: float
    direct_precedents: float
    private_share: float
    high_margin_share: float
    regional_share: float


@dataclass(frozen=True)
class MarketComps:
    """Closed deals in the target's sector inside the size band (decision 3)."""

    sector: str
    min_ev_mm: float
    max_ev_mm: float
    transaction_ids: list[str]
    median_ev_ebitda: float | None
    median_ev_revenue: float | None

    @property
    def count(self) -> int:
        return len(self.transaction_ids)


@dataclass(frozen=True)
class ValuationComps:
    market: MarketComps
    acquirer_closed_count: int
    acquirer_median_ev_ebitda: float | None
    acquirer_median_ev_revenue: float | None
    direct_count: int
    direct_median_ev_ebitda: float | None


@dataclass(frozen=True)
class PrecedentRow:
    transaction: Transaction
    is_direct: bool
    sector_weight: float


@dataclass(frozen=True)
class Conviction:
    """Computed in code (decision 7). The LLM writes the reason but can't change the level."""

    level: ConvictionLevel
    direct_precedents: int
    exact_sector_deals: int
    downgraded_by: str | None          # title of the high-severity risk, if any


@dataclass(frozen=True)
class EvidencePack:
    entry: ShortlistEntry
    stats: AcquirerStats
    peer_baseline: PeerBaseline
    precedents: list[PrecedentRow]
    valuation: ValuationComps
    conviction: Conviction
    data_risks: list[RiskFlag]
    facts: list[Fact]

    @property
    def acquirer(self) -> str:
        return self.entry.acquirer

    @property
    def citable_ids(self) -> set[str]:
        """Fact IDs plus every transaction ID the pack mentions."""
        ids = {fact.id for fact in self.facts}
        ids.update(row.transaction.transaction_id for row in self.precedents)
        ids.update(t.transaction_id for t in self.stats.direct_precedents)
        ids.update(self.valuation.market.transaction_ids)
        for risk in self.data_risks:
            ids.update(risk.transaction_ids)
        ids.update(t.transaction_id for t in (self.stats.min_ev, self.stats.max_ev, self.stats.last_deal))
        return ids


@dataclass(frozen=True)
class Evidence:
    """Everything `EvidenceBuilderService` produces for one run."""

    direct_sectors: list[str]          # target sector plus its nearest sectors
    direct_years: tuple[int, int]
    recent_years: tuple[int, int]
    market_comps: MarketComps
    target_risks: list[RiskFlag]       # trip for every acquirer or none; summary shows them once
    packs: list[EvidencePack]
