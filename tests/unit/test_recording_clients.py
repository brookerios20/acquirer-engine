import json

import pytest

from acquirer_engine.clients import (
    LlmCallError,
    LlmMessage,
    LlmRequest,
    RecordedClient,
    RecordingClient,
    RecordingNotFoundError,
)
from acquirer_engine.models import OnePagerDraft
from tests.llm_fakes import FakeLlmClient, small_evidence, valid_draft

PACK = small_evidence().packs[0]
DRAFT = valid_draft(PACK)


def request(text="Draft the one-pager", version="v1"):
    return LlmRequest("system prompt", [LlmMessage("user", text)], OnePagerDraft, version)


def test_record_then_replay_returns_the_same_response(tmp_path):
    live = FakeLlmClient(lambda req, n: DRAFT)

    recorded = RecordingClient(live, tmp_path).generate(request())
    replayed = RecordedClient(tmp_path, live.model).generate(request())

    assert replayed.parsed == DRAFT
    assert replayed.raw_json == recorded.raw_json
    assert (replayed.model, replayed.input_tokens, replayed.latency_s) == ("fake-model", 100, 0.01)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["request"]["messages"] == [{"role": "user", "content": "Draft the one-pager"}]


@pytest.mark.parametrize(
    "changed",
    [
        {"text": "A different prompt"},
        {"version": "v2"},
    ],
)
def test_any_change_to_the_request_misses(tmp_path, changed):
    RecordingClient(FakeLlmClient(lambda req, n: DRAFT), tmp_path).generate(request())

    with pytest.raises(RecordingNotFoundError, match="Re-record with --record"):
        RecordedClient(tmp_path, "fake-model").generate(request(**changed))


def test_a_different_model_misses(tmp_path):
    RecordingClient(FakeLlmClient(lambda req, n: DRAFT), tmp_path).generate(request())

    with pytest.raises(RecordingNotFoundError):
        RecordedClient(tmp_path, "another-model").generate(request())


def test_failed_calls_are_not_recorded(tmp_path):
    def fail(req, n):
        raise LlmCallError("API error 500")

    with pytest.raises(LlmCallError):
        RecordingClient(FakeLlmClient(fail), tmp_path).generate(request())

    assert list(tmp_path.glob("*.json")) == []
