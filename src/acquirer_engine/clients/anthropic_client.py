import time

import anthropic
import pydantic

from ..config import LlmConfig
from .llm_client import LlmCallError, LlmRequest, LlmResponse

# Server-side refusal fallback: if a safety classifier declines, the API re-runs the
# request on Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Statuses the SDK treats as transient. Any other 4xx means the request or the account
# is wrong (bad key, no credits, invalid request), so a retry would fail the same way.
TRANSIENT_STATUSES = {408, 409, 429}


class AnthropicClient:
    """Calls the Messages API with structured output (decision 14).

    The SDK validates the JSON against the request's Pydantic schema. Rate limits and
    server errors are retried by the SDK (`sdk_max_retries`); anything still failing
    becomes an `LlmCallError`.
    """

    def __init__(self, api_key: str, config: LlmConfig):
        self._config = config
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=config.timeout_s,
            max_retries=config.sdk_max_retries,
        )

    @property
    def model(self) -> str:
        return self._config.model

    def generate(self, request: LlmRequest) -> LlmResponse:
        started = time.perf_counter()
        try:
            response = self._client.beta.messages.parse(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                system=request.system,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                output_format=request.output_schema,
                # No temperature/top_p/top_k: the API rejects them (decision 13)
                output_config={"effort": self._config.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except pydantic.ValidationError as error:
            raise LlmCallError(f"response didn't match the output schema: {error}") from error
        except anthropic.APIStatusError as error:
            retryable = error.status_code in TRANSIENT_STATUSES or error.status_code >= 500
            raise LlmCallError(
                f"API error {error.status_code}: {_api_error_message(error)}", retryable=retryable
            ) from error
        except anthropic.APIConnectionError as error:
            # Includes timeouts (APITimeoutError is a subclass)
            raise LlmCallError(f"connection error: {error}") from error
        latency = time.perf_counter() - started

        # Check why the model stopped before reading the content
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise LlmCallError(f"model declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LlmCallError("response was cut off at max_tokens")
        if response.parsed_output is None:
            raise LlmCallError(f"no structured output (stop reason: {response.stop_reason})")

        raw_json = next((b.text for b in response.content if b.type == "text"), "")
        return LlmResponse(
            parsed=response.parsed_output,
            raw_json=raw_json,
            model=response.model,  # the fallback model, if one served the request
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=round(latency, 2),
        )


def _api_error_message(error: anthropic.APIStatusError) -> str:
    """The API's own message ("Your credit balance is too low..."), without the SDK's
    "Error code: 400 - {...}" wrapper around the raw response body."""
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message = body["error"].get("message")
        if message:
            return message
    return error.message
