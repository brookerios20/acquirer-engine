"""Number formats shared by the evidence pack, the views, and the validator.

The validator matches numbers at the precision shown here (decision 11), so every
displayed figure goes through these functions.
"""


def money(value_mm: float) -> str:
    return f"${value_mm:,.1f}M"


def ev_ebitda(value: float) -> str:
    return f"{value:.1f}x"


def ev_revenue(value: float) -> str:
    return f"{value:.2f}x"


def percent(share: float) -> str:
    return f"{share * 100:.0f}%"


def weight(value: float) -> str:
    return f"{value:.2f}"
