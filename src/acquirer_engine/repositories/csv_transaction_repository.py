import hashlib
from pathlib import Path

import pandas as pd

from ..errors import DataError
from ..models import OUTCOMES, SPONSOR, STRATEGIC, LoadedTransactions, SkippedRow, Transaction

# Columns the engine reads. `sub_sector` (equals `sector` in every row) and
# `target_ebitda_mm` (doesn't reconcile with margin or EV/EBITDA) are ignored (decision 3).
REQUIRED_COLUMNS = (
    "transaction_id", "target_company", "acquirer", "acquirer_type", "sector",
    "deal_year", "deal_quarter", "deal_type", "geography", "financing_type",
    "deal_size_mm", "target_revenue_mm", "ebitda_margin_pct", "revenue_growth_pct",
    "ev_ebitda_multiple", "ev_revenue_multiple", "synergy_pct_of_deal", "outcome",
    "strategic_rationale_tags", "num_bidders", "days_to_close", "target_ownership_pre",
)
ACQUIRER_TYPES = (SPONSOR, STRATEGIC)


class CsvTransactionRepository:
    """Loads the deal CSV and applies the data rules found in the EDA (decision 3).

    A missing file or column stops the run. A bad row is skipped and reported.
    """

    def __init__(self, csv_path: Path):
        self._csv_path = csv_path

    def load(self) -> LoadedTransactions:
        if not self._csv_path.is_file():
            raise DataError(f"CSV not found: {self._csv_path}")

        raw_bytes = self._csv_path.read_bytes()
        # Read everything as text so each row can be checked and skipped on its own
        frame = pd.read_csv(self._csv_path, dtype=str, keep_default_na=False)

        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise DataError(f"{self._csv_path} is missing required column(s): {', '.join(missing)}")

        transactions: list[Transaction] = []
        skipped: list[SkippedRow] = []
        for index, row in enumerate(frame.to_dict("records")):
            line_number = index + 2  # line 1 is the header
            try:
                transactions.append(self._parse_row(row))
            except ValueError as error:
                skipped.append(SkippedRow(line_number, row["transaction_id"], str(error)))

        return LoadedTransactions(
            transactions=transactions,
            skipped_rows=skipped,
            source=str(self._csv_path),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    def _parse_row(self, row: dict[str, str]) -> Transaction:
        outcome = row["outcome"].strip()
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome '{outcome}'")
        acquirer_type = row["acquirer_type"].strip()
        if acquirer_type not in ACQUIRER_TYPES:
            raise ValueError(f"unknown acquirer_type '{acquirer_type}'")

        days_to_close = row["days_to_close"].strip()
        tags = row["strategic_rationale_tags"].strip()

        return Transaction(
            transaction_id=row["transaction_id"].strip(),
            target_company=row["target_company"].strip(),
            acquirer=row["acquirer"].strip(),  # names kept as reported, e.g. "Cerner (Oracle)"
            acquirer_type=acquirer_type,
            sector=row["sector"].strip(),
            deal_year=_to_int(row, "deal_year"),
            deal_quarter=_to_quarter(row["deal_quarter"]),
            deal_type=row["deal_type"].strip(),
            geography=row["geography"].strip(),
            financing_type=row["financing_type"].strip(),
            ev_mm=_to_positive_float(row, "deal_size_mm"),
            revenue_mm=_to_float(row, "target_revenue_mm"),
            ebitda_margin_pct=_to_float(row, "ebitda_margin_pct"),
            revenue_growth_pct=_to_float(row, "revenue_growth_pct"),
            ev_ebitda=_to_float(row, "ev_ebitda_multiple"),
            ev_revenue=_to_float(row, "ev_revenue_multiple"),
            synergy_pct=_to_float(row, "synergy_pct_of_deal"),
            outcome=outcome,
            rationale_tags=tuple(tags.split("|")) if tags else (),
            num_bidders=_to_int(row, "num_bidders"),
            # Blank exactly where a deal didn't close
            days_to_close=int(float(days_to_close)) if days_to_close else None,
            target_ownership=row["target_ownership_pre"].strip(),
        )


def _to_float(row: dict[str, str], column: str) -> float:
    text = row[column].strip()
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"non-numeric {column} '{text}'") from None


def _to_positive_float(row: dict[str, str], column: str) -> float:
    value = _to_float(row, column)
    if value <= 0:
        raise ValueError(f"{column} must be positive, got {value}")
    return value


def _to_int(row: dict[str, str], column: str) -> int:
    value = _to_float(row, column)
    if not value.is_integer():
        raise ValueError(f"non-integer {column} '{row[column]}'")
    return int(value)


def _to_quarter(text: str) -> int:
    text = text.strip()
    if text not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"unknown deal_quarter '{text}'")
    return int(text[1])
