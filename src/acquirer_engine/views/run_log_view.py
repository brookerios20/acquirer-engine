import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import AnalysisConfig, LlmConfig, ValidationConfig
from ..models import AcquirerReport


class RunLogView:
    """Writes `run_log.json`, the audit log of what the run saw and produced (decision 13)."""

    def write(
        self,
        report: AcquirerReport,
        out_dir: Path,
        analysis_config: AnalysisConfig,
        llm_config: LlmConfig,
        validation_config: ValidationConfig,
        elapsed_s: float,
        llm_mode: str,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "run_log.json"
        payload = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed_s, 1),
            "data": {
                "source": report.csv_source,
                "sha256": report.csv_sha256,
                "skipped_rows": [dataclasses.asdict(s) for s in report.skipped_rows],
            },
            "target": dataclasses.asdict(report.target),
            "min_strategics": report.min_strategics,
            "config": {
                "analysis": dataclasses.asdict(analysis_config),
                "llm": dataclasses.asdict(llm_config),
                "validation": dataclasses.asdict(validation_config),
            },
            "llm_mode": llm_mode,           # live, record, or replay
            "prompt_version": report.prompt_version,
            "model": report.model,
            "warnings": report.warnings,
            "calls": [dataclasses.asdict(c) for c in report.call_records],
            "validation": {p.acquirer: p.review_flags for p in report.one_pagers},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
