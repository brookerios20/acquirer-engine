import json
from pathlib import Path

from .llm_client import LlmRequest, LlmResponse
from .recording_key import recording_key


class RecordingNotFoundError(Exception):
    """Replay found no recording for a request. Unlike `LlmCallError`, this stops the
    run: a partial replay wouldn't reproduce the recorded output."""


class RecordedClient:
    """Replays responses saved by `RecordingClient`. Needs no API key (decision 13)."""

    def __init__(self, recordings_dir: Path, model: str):
        self._recordings_dir = recordings_dir
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def generate(self, request: LlmRequest) -> LlmResponse:
        key = recording_key(self._model, request)
        path = self._recordings_dir / f"{key}.json"
        if not path.is_file():
            raise RecordingNotFoundError(
                f"No recording for this request in {self._recordings_dir} (key {key}). "
                "The prompt, data, settings, or model differ from the recorded run. "
                "Re-record with --record and an API key."
            )

        saved = json.loads(path.read_text(encoding="utf-8"))["response"]
        return LlmResponse(
            parsed=request.output_schema.model_validate_json(saved["raw_json"]),
            raw_json=saved["raw_json"],
            model=saved["model"],
            input_tokens=saved["input_tokens"],
            output_tokens=saved["output_tokens"],
            latency_s=saved["latency_s"],
        )
