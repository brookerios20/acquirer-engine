import hashlib

import pytest

from acquirer_engine.errors import DataError
from acquirer_engine.repositories import CsvTransactionRepository


def test_loads_rows_and_applies_data_rules(fixtures_dir):
    loaded = CsvTransactionRepository(fixtures_dir / "valid.csv").load()

    assert [t.transaction_id for t in loaded.transactions] == ["MA-2020-0001", "MA-2021-0002", "MA-2022-0003"]
    assert loaded.skipped_rows == []

    phreesia, cerner, blackstone = loaded.transactions
    assert phreesia.deal_quarter == 4
    assert phreesia.rationale_tags == ("Technology Capability", "Geographic Expansion", "Vertical Integration")
    assert phreesia.ev_ebitda == 18.3          # stated multiple used as reported
    assert cerner.acquirer == "Cerner (Oracle)"  # name kept as reported
    assert cerner.outcome == "Rumored"          # accepted as a fifth outcome
    assert not cerner.is_closed
    assert cerner.days_to_close is None         # blank where the deal didn't close
    assert blackstone.rationale_tags == ()
    assert blackstone.ev_revenue == 1.38


def test_records_source_and_hash(fixtures_dir):
    path = fixtures_dir / "valid.csv"
    loaded = CsvTransactionRepository(path).load()

    assert loaded.source == str(path)
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_skips_bad_rows_and_says_why(fixtures_dir):
    loaded = CsvTransactionRepository(fixtures_dir / "bad_rows.csv").load()

    assert [t.transaction_id for t in loaded.transactions] == ["MA-2020-0001"]
    assert [(s.line_number, s.transaction_id) for s in loaded.skipped_rows] == [
        (3, "MA-2021-0002"),
        (4, "MA-2022-0003"),
    ]
    assert "deal_size_mm" in loaded.skipped_rows[0].reason
    assert "Abandoned" in loaded.skipped_rows[1].reason


def test_missing_column_fails_naming_it(fixtures_dir):
    with pytest.raises(DataError, match="deal_size_mm"):
        CsvTransactionRepository(fixtures_dir / "missing_columns.csv").load()


def test_missing_file_fails_naming_it(tmp_path):
    with pytest.raises(DataError, match="nope.csv"):
        CsvTransactionRepository(tmp_path / "nope.csv").load()
