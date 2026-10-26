"""End to end, offline: the full pipeline replays the committed recordings (decision 15).

No test calls the live API. If a prompt or the data changes, these fail until the
recordings and sample_output/ are re-recorded with `acquirers --record --out sample_output`.
"""
import json
from pathlib import Path

import pytest

from acquirer_engine.controllers.cli_controller import main

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLE_OUTPUT = REPO_ROOT / "sample_output"


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Tests never read the developer's .env, so no key or model override leaks in."""
    monkeypatch.setattr("acquirer_engine.controllers.cli_controller.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("LLM_MODEL", raising=False)


@pytest.fixture
def replay_out(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # default CSV and recordings paths are relative
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "replay"
    assert main(["--replay", "--out", str(out)]) == 0
    return out


def test_replay_writes_ten_grounded_one_pagers(replay_out):
    payload = json.loads((replay_out / "one_pagers.json").read_text())
    run_log = json.loads((replay_out / "run_log.json").read_text())

    assert len(payload["one_pagers"]) == 10
    assert not any(p["needs_review"] for p in payload["one_pagers"])
    assert all(flags == [] for flags in run_log["validation"].values())  # schema, IDs, numbers all pass
    assert run_log["llm_mode"] == "replay"
    assert len(list(replay_out.glob("[0-9][0-9]-*.md"))) == 10


def test_replay_reproduces_the_committed_sample_output(replay_out):
    expected = sorted(p.name for p in SAMPLE_OUTPUT.iterdir() if p.name != "run_log.json")
    actual = sorted(p.name for p in replay_out.iterdir() if p.name != "run_log.json")

    assert actual == expected
    for name in expected:
        assert (replay_out / name).read_text() == (SAMPLE_OUTPUT / name).read_text(), name


def test_replay_with_a_different_target_fails_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)

    assert main(["--replay", "--ev", "250", "--out", str(tmp_path)]) == 1
    assert "No recording for this request" in capsys.readouterr().err


def test_missing_key_fails_before_analysis(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main(["--out", str(tmp_path / "out")]) == 2
    assert "Set ANTHROPIC_API_KEY in .env, or run with --replay." in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
