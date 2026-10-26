"""Thresholds and cut points, kept in one place for a banker to adjust.

Each value is a convention, not a fitted parameter. The README lists them.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    # Shortlist (decisions 4 and 5)
    shortlist_size: int = 10

    # Target profile (decision 2)
    margin_quantile: float = 0.75          # "strong margins" = sector's top quartile
    non_regional_geographies: tuple[str, ...] = ("National", "Multi-Regional")

    # Direct precedents and comps (decisions 3 and 7)
    size_band_low: float = 0.5             # x target EV, inclusive
    size_band_high: float = 2.0            # x target EV, inclusive
    direct_precedent_years: int = 5        # last 5 years of data
    nearest_sectors: int = 2               # target sector plus its 2 nearest

    # Conviction cut points (decision 7)
    high_min_direct: int = 3
    medium_min_direct: int = 1
    medium_min_exact_sector_deals: int = 2

    # Evidence pack (decision 6)
    recent_years: int = 2                  # "recent activity" = last 2 years of data
    precedent_table_size: int = 5

    # Data risks (decision 10)
    thin_evidence_below_direct: int = 2
    recent_pace_min_deals: int = 5         # deals in the recent window
    size_stretch_high_multiple: float = 2.0
    contested_min_bidders: int = 3
    contested_min_share: float = 0.5       # share of comp deals with 3+ bidders

    # Backtest (decision 4)
    backtest_k: int = 10
    backtest_test_from_year: int = 2022    # deals from this year on are held out
    profile_min_similarity: float = 0.6    # "profile-like" held-out deals
    profile_ev_low: float = 0.375          # x target EV: $75M for a $200M target
    profile_ev_high: float = 2.5           # x target EV: $500M for a $200M target


@dataclass(frozen=True)
class LlmConfig:
    """LLM call settings (decision 14). `LLM_MODEL` in .env overrides the model."""

    model: str = "claude-sonnet-5-5"
    effort: str = "medium"                 # thinking depth; the main latency lever
    max_tokens: int = 16000
    # Measured 9-16 s per call at medium effort. timeout x (retries + 1) = 60 s (decision 14)
    timeout_s: float = 30.0                # per HTTP attempt
    sdk_max_retries: int = 1               # SDK retries for rate limits, server errors, timeouts
    max_workers: int = 10                  # one thread per acquirer (decision 9)


@dataclass(frozen=True)
class ValidationConfig:
    """Rules the validator applies to each LLM response (decision 11)."""

    min_risks: int = 2
    banned_phrases: tuple[str, ...] = (
        "leading",
        "well-positioned",
        "well positioned",
        "track record of strategic acquisitions",
        "best-in-class",
        "synergistic",
    )
    # Word-count ranges per prose section
    section_words: tuple[tuple[str, int, int], ...] = (
        ("headline", 4, 25),
        ("overview", 30, 130),
        ("fit_thesis", 40, 170),
        ("precedent_commentary", 25, 130),
        ("valuation_commentary", 25, 120),
        ("conviction_reason", 12, 70),
    )
