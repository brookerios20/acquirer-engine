"""Prompt templates (decision 8), filled with `string.Template`.

Bump PROMPT_VERSION whenever a template changes: recordings are keyed by it (decision 13).
"""
from pathlib import Path
from string import Template

PROMPT_VERSION = "v2"  # v2: data-risk notes don't restate the risk's figures

_PROMPT_DIR = Path(__file__).parent


def load_template(name: str) -> Template:
    return Template((_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8"))
