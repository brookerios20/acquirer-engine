import hashlib
import json

from .llm_client import LlmRequest


def recording_key(model: str, request: LlmRequest) -> str:
    """Hash of everything that determines a response: model, prompt version, the full
    prompt, and the output schema (decision 13). Any change gives a new key."""
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": request.prompt_version,
            "system": request.system,
            "messages": [[m.role, m.content] for m in request.messages],
            "schema": request.output_schema.model_json_schema(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
