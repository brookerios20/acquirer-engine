import dataclasses
import json
from pathlib import Path

from ..models import AcquirerReport


class JsonView:
    """Writes `one_pagers.json` for tests and future consumers such as a CRM write-back."""

    def write(self, report: AcquirerReport, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "one_pagers.json"
        payload = {
            "target": dataclasses.asdict(report.target),
            "min_strategics": report.min_strategics,
            "market_comps": dataclasses.asdict(report.evidence.market_comps),
            "target_risks": [dataclasses.asdict(r) for r in report.evidence.target_risks],
            "warnings": report.warnings,
            "one_pagers": [
                {**dataclasses.asdict(p), "needs_review": p.needs_review} for p in report.one_pagers
            ],
        }
        # Enums here subclass str, so json writes their values
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
