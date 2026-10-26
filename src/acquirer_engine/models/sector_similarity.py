from dataclasses import dataclass


@dataclass(frozen=True)
class SectorSimilarity:
    """Closeness between sectors, 0 to 1 (decision 4). A sector is always 1.0 to itself."""

    weights: dict[str, dict[str, float]]

    @property
    def sectors(self) -> list[str]:
        return sorted(self.weights)

    def weight(self, sector_a: str, sector_b: str) -> float:
        if sector_a == sector_b:
            return 1.0
        return self.weights.get(sector_a, {}).get(sector_b, 0.0)

    def nearest(self, sector: str, count: int) -> list[str]:
        """The `count` most similar other sectors, ties broken by name."""
        others = [s for s in self.sectors if s != sector]
        others.sort(key=lambda s: (-self.weight(sector, s), s))
        return others[:count]
