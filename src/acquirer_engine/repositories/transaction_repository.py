from typing import Protocol

from ..models import LoadedTransactions


class TransactionRepository(Protocol):
    """Source of cleaned transactions. Today a CSV; later a Databricks table."""

    def load(self) -> LoadedTransactions: ...
