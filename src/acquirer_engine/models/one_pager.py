from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .evidence import Conviction, Evidence, PrecedentRow, RiskFlag, ValuationComps
from .ranking import RankingScore
from .target_profile import TargetProfile
from .transaction import SkippedRow

# --- What the LLM returns (decision 10). Pydantic, so the SDK validates the JSON. -------


class Citations(BaseModel):
    """Fact IDs (F1, F2, ...) and transaction IDs each section relies on."""

    overview: list[str]
    fit_thesis: list[str]
    precedent_commentary: list[str]
    valuation_commentary: list[str]
    conviction_reason: list[str]


class DataRiskNote(BaseModel):
    title: str = Field(description="Exact title of a data risk from the facts")
    note: str = Field(description="One sentence on why it matters for this deal")


class QualitativeRisk(BaseModel):
    title: str
    detail: str = Field(description="1-2 sentences; label anything not in the facts as outside the data")
    cited_ids: list[str]


class OnePagerDraft(BaseModel):
    headline: str
    overview: str
    fit_thesis: str
    precedent_commentary: str
    valuation_commentary: str
    data_risk_notes: list[DataRiskNote]
    qualitative_risks: list[QualitativeRisk]
    conviction_reason: str
    citations: Citations


# --- The finished one-pager -------------------------------------------------------------


@dataclass
class AcquirerOnePager:
    """Code fills the factual fields; the LLM fills the prose (decision 10)."""

    acquirer: str
    acquirer_type: str
    rank: int
    reserved_slot: bool
    ranking: RankingScore
    precedents: list[PrecedentRow]
    valuation: ValuationComps
    conviction: Conviction
    risks: list[RiskFlag]
    headline: str = ""
    overview: str = ""
    fit_thesis: str = ""
    precedent_commentary: str = ""
    valuation_commentary: str = ""
    conviction_reason: str = ""
    cited_ids: dict[str, list[str]] = field(default_factory=dict)
    review_flags: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_flags)


@dataclass(frozen=True)
class CallRecord:
    """One LLM attempt, for the run log (decision 13)."""

    acquirer: str
    attempt: int
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    violations: list[str]
    error: str | None = None


@dataclass
class AcquirerReport:
    """Everything one run produces; the views turn it into files."""

    target: TargetProfile
    min_strategics: int
    evidence: Evidence
    one_pagers: list[AcquirerOnePager]
    warnings: list[str]
    call_records: list[CallRecord]
    csv_source: str
    csv_sha256: str
    skipped_rows: list[SkippedRow]
    prompt_version: str
    model: str
