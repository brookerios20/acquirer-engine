"""Errors that stop a run with a message telling the user what to fix (decision 15)."""


class DataError(Exception):
    """The data source is missing or unusable."""


class InputError(ValueError):
    """A CLI input doesn't match the data, such as an unknown sector."""
