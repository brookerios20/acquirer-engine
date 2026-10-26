from dataclasses import dataclass, field


@dataclass(frozen=True)
class SectorContribution:
    """One line of a score's explanation: "10 Physician Groups deals x 0.83"."""

    sector: str
    deal_count: int
    weight: float

    @property
    def points(self) -> float:
        return self.deal_count * self.weight


@dataclass(frozen=True)
class RankingScore:
    acquirer: str
    score: float
    contributions: list[SectorContribution]


@dataclass(frozen=True)
class ShortlistEntry:
    rank: int
    acquirer: str
    acquirer_type: str
    ranking: RankingScore
    reserved_slot: bool        # included by --min-strategics (decision 5)


@dataclass(frozen=True)
class Shortlist:
    entries: list[ShortlistEntry]
    warnings: list[str] = field(default_factory=list)
