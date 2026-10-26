import json
from pathlib import Path

from .llm_client import LlmClient, LlmRequest, LlmResponse
from .recording_key import recording_key


class RecordingClient:
    """Wraps a live client and saves each successful response for replay (decision 13).

    One JSON file per call, named by the recording key. The file also holds the prompt,
    so a reviewer can see exactly what the model was given.
    """

    def __init__(self, inner: LlmClient, recordings_dir: Path):
        self._inner = inner
        self._recordings_dir = recordings_dir

    @property
    def model(self) -> str:
        return self._inner.model

    def generate(self, request: LlmRequest) -> LlmResponse:
        response = self._inner.generate(request)  # failures raise and aren't recorded

        key = recording_key(self.model, request)
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "requested_model": self.model,
            "prompt_version": request.prompt_version,
            "request": {
                "system": request.system,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            },
            "response": {
                "raw_json": response.raw_json,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_s": response.latency_s,
            },
        }
        path = self._recordings_dir / f"{key}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return response
