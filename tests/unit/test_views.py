import json

from acquirer_engine.config import AnalysisConfig, LlmConfig, ValidationConfig
from acquirer_engine.models import AcquirerReport
from acquirer_engine.services import AcquirerOnePagerService, OutputValidatorService
from acquirer_engine.views import JsonView, MarkdownView, RunLogView
from tests.llm_fakes import TARGET, FakeLlmClient, acquirer_of, small_evidence, valid_draft

EVIDENCE = small_evidence()
PACKS = {p.acquirer: p for p in EVIDENCE.packs}


def make_report() -> AcquirerReport:
    def respond(request, call_number):
        draft = valid_draft(PACKS[acquirer_of(request)])
        if acquirer_of(request) == "Health System":
            return draft.model_copy(update={"overview": "Too short."})  # fails both attempts
        return draft

    service = AcquirerOnePagerService(
        FakeLlmClient(respond), OutputValidatorService(ValidationConfig()),
        AnalysisConfig(), LlmConfig(), ValidationConfig(),
    )
    one_pagers, records = service.generate_all(EVIDENCE, TARGET)
    return AcquirerReport(
        target=TARGET, min_strategics=0, evidence=EVIDENCE, one_pagers=one_pagers, warnings=[],
        call_records=records, csv_source="fixture.csv", csv_sha256="abc123" * 10, skipped_rows=[],
        prompt_version="v1", model="fake-model",
    )


def test_markdown_view_writes_summary_and_one_file_per_acquirer(tmp_path):
    paths = MarkdownView().write(make_report(), tmp_path)

    assert [p.name for p in paths] == ["summary.md", "01-sponsor-b.md", "02-sponsor-a.md", "03-health-system.md"]
    summary = (tmp_path / "summary.md").read_text()
    assert "| 3 | [Health System](03-health-system.md) | Strategic |" in summary
    assert summary.count("Needs review") == 1
    flagged = (tmp_path / "03-health-system.md").read_text()
    assert flagged.startswith("# 3. Health System\n\n> **Needs review.**")
    assert "Platform Investment" in flagged  # deal type shown as reported in the table
    assert "Needs review" not in (tmp_path / "01-sponsor-b.md").read_text()


def test_json_and_run_log_views(tmp_path):
    report = make_report()

    one_pagers = json.loads(JsonView().write(report, tmp_path).read_text())
    run_log = json.loads(
        RunLogView().write(report, tmp_path, AnalysisConfig(), LlmConfig(), ValidationConfig(), 1.5, "live").read_text()
    )

    assert [p["acquirer"] for p in one_pagers["one_pagers"]] == ["Sponsor B", "Sponsor A", "Health System"]
    assert one_pagers["one_pagers"][2]["needs_review"] is True
    assert one_pagers["one_pagers"][0]["conviction"]["level"] == "High"
    assert run_log["data"]["sha256"] == "abc123" * 10
    assert len(run_log["calls"]) == 4  # one each, plus a retry for Health System
    assert run_log["validation"]["Health System"] != []
