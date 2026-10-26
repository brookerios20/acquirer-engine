from dataclasses import dataclass


@dataclass(frozen=True)
class TargetProfile:
    """The company being sold, with each vague trait resolved to a stated reading (decision 2)."""

    sector: str
    ev_mm: float
    ownership: str
    min_margin_pct: float      # "strong margins": at or above this EBITDA margin
    region: str | None         # None = "regional" means any region except National or Multi-Regional


@dataclass(frozen=True)
class ReportRequest:
    """The CLI inputs for one run, before they are checked against the data."""

    sector: str
    ev_mm: float
    ownership: str
    min_margin_pct: float | None   # None = the sector's top quartile
    region: str | None
    min_strategics: int
