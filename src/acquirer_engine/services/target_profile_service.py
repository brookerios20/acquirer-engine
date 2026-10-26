import pandas as pd

from ..config import AnalysisConfig
from ..errors import InputError
from ..models import TargetProfile, Transaction


class TargetProfileService:
    """Turns CLI inputs into a `TargetProfile` (decision 2).

    Checks each input against the values in the data and resolves the default margin
    floor to the sector's top quartile.
    """

    def __init__(self, config: AnalysisConfig):
        self._config = config

    def build(
        self,
        transactions: list[Transaction],
        sector: str,
        ev_mm: float,
        ownership: str,
        min_margin_pct: float | None,
        region: str | None,
    ) -> TargetProfile:
        sectors = sorted({t.sector for t in transactions})
        if sector not in sectors:
            raise InputError(f"Unknown sector '{sector}'. Known sectors: {', '.join(sectors)}")

        if ev_mm <= 0:
            raise InputError(f"EV must be positive, got {ev_mm}")

        ownerships = sorted({t.target_ownership for t in transactions})
        if ownership not in ownerships:
            raise InputError(f"Unknown ownership '{ownership}'. Known values: {', '.join(ownerships)}")

        if region is not None:
            regions = sorted(
                {t.geography for t in transactions} - set(self._config.non_regional_geographies)
            )
            if region not in regions:
                raise InputError(f"Unknown region '{region}'. Known regions: {', '.join(regions)}")

        if min_margin_pct is None:
            min_margin_pct = self.default_margin_floor(transactions, sector)

        return TargetProfile(sector, ev_mm, ownership, min_margin_pct, region)

    def default_margin_floor(self, transactions: list[Transaction], sector: str) -> float:
        """The sector's EBITDA margin at `margin_quantile` (pandas' linear quantile),
        rounded to the CSV's one-decimal precision so a deal at exactly the floor counts."""
        margins = pd.Series([t.ebitda_margin_pct for t in transactions if t.sector == sector])
        return round(float(margins.quantile(self._config.margin_quantile)), 1)
