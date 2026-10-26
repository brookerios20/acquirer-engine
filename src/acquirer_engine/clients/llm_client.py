from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True)
class LlmMessage:
    role: str          # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class LlmRequest:
    """Provider-neutral request. `prompt_version` keys recordings (decision 13)."""

    system: str
    messages: list[LlmMessage]
    output_schema: type[BaseModel]
    prompt_version: str


@dataclass(frozen=True)
class LlmResponse:
    parsed: BaseModel
    raw_json: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class LlmCallError(Exception):
    """The call failed after the client's own retries: API error, refusal, truncation,
    or output that didn't match the schema. The caller decides what to do next.

    `retryable` is False when sending the same request again can't succeed, such as a
    bad API key or an empty credit balance (decision 11).
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class LlmClient(Protocol):
    @property
    def model(self) -> str: ...

    def generate(self, request: LlmRequest) -> LlmResponse: ...
