"""Plain dataclasses shared by all layers. No I/O."""
from .backtest import BacktestResult, BacktestRow
from .evidence import (
    AcquirerStats,
    Conviction,
    ConvictionLevel,
    Evidence,
    EvidencePack,
    Fact,
    MarketComps,
    PeerBaseline,
    PrecedentRow,
    RiskFlag,
    RiskSeverity,
    RiskSource,
    ValuationComps,
)
from .one_pager import (
    AcquirerOnePager,
    AcquirerReport,
    CallRecord,
    Citations,
    DataRiskNote,
    OnePagerDraft,
    QualitativeRisk,
)
from .ranking import RankingScore, SectorContribution, Shortlist, ShortlistEntry
from .sector_similarity import SectorSimilarity
from .target_profile import ReportRequest, TargetProfile
from .transaction import (
    CLOSED,
    OUTCOMES,
    SPONSOR,
    STRATEGIC,
    LoadedTransactions,
    SkippedRow,
    Transaction,
    group_by_acquirer,
)
