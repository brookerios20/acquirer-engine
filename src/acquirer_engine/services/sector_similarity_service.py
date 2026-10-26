import math

import pandas as pd

from ..models import SPONSOR, SectorSimilarity, Transaction


class SectorSimilarityService:
    """Sector closeness from sponsor overlap (decision 4).

    Each sector is a vector of how many deals each sponsor did there; similarity is the
    cosine between two sectors' vectors. Strategics are left out because 90 of 92 buy in
    one sector only, so they carry no information about which sectors go together.
    """

    def compute(self, transactions: list[Transaction]) -> SectorSimilarity:
        sectors = sorted({t.sector for t in transactions})
        sponsor_deals = [t for t in transactions if t.acquirer_type == SPONSOR]
        if not sponsor_deals:
            return SectorSimilarity({a: {b: 0.0 for b in sectors if b != a} for a in sectors})

        # Rows are sectors, columns are sponsors, cells are deal counts
        counts = pd.crosstab(
            pd.Series([t.sector for t in sponsor_deals], name="sector"),
            pd.Series([t.acquirer for t in sponsor_deals], name="acquirer"),
        ).reindex(sectors, fill_value=0)

        dot_products = counts @ counts.T
        norms = {s: math.sqrt(dot_products.loc[s, s]) for s in sectors}

        def cosine(a: str, b: str) -> float:
            # A sector no sponsor bought in has norm 0; its similarity to everything is 0
            if norms[a] == 0 or norms[b] == 0:
                return 0.0
            return float(dot_products.loc[a, b]) / (norms[a] * norms[b])

        return SectorSimilarity({a: {b: cosine(a, b) for b in sectors if b != a} for a in sectors})
