import anthropic
import httpx2  # the SDK's HTTP library
import pytest

from acquirer_engine.clients import LlmCallError, LlmMessage, LlmRequest
from acquirer_engine.clients.anthropic_client import AnthropicClient
from acquirer_engine.config import LlmConfig
from acquirer_engine.models import OnePagerDraft

REQUEST = LlmRequest("system", [LlmMessage("user", "hi")], OnePagerDraft, "v-test")


def client_raising(status_code: int, message: str) -> AnthropicClient:
    """An AnthropicClient whose API call fails with the given status and error message."""
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    response = httpx2.Response(status_code, json=body, request=httpx2.Request("POST", "https://api.test"))
    error = anthropic.APIStatusError(f"Error code: {status_code} - {body}", response=response, body=body)

    def parse(**kwargs):
        raise error

    client = AnthropicClient("test-key", LlmConfig())
    client._client.beta.messages.parse = parse
    return client


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_request_and_account_errors_are_not_retryable(status_code):
    with pytest.raises(LlmCallError) as caught:
        client_raising(status_code, "Your credit balance is too low.").generate(REQUEST)

    assert not caught.value.retryable
    assert str(caught.value) == f"API error {status_code}: Your credit balance is too low."


@pytest.mark.parametrize("status_code", [408, 429, 500, 529])
def test_transient_errors_are_retryable(status_code):
    with pytest.raises(LlmCallError) as caught:
        client_raising(status_code, "Overloaded.").generate(REQUEST)

    assert caught.value.retryable
