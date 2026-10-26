from ..errors import DataError
from ..models import AcquirerReport, ReportRequest
from ..prompts import PROMPT_VERSION
from ..repositories import TransactionRepository
from .acquirer_one_pager_service import AcquirerOnePagerService
from .candidate_scoring_service import CandidateScoringService
from .evidence_builder_service import EvidenceBuilderService
from .target_profile_service import TargetProfileService


class AcquirerReportService:
    """Runs one report: load, resolve the target, shortlist, build evidence, draft one-pagers.

    Deterministic steps run in code; only the last step calls the LLM.
    """

    def __init__(
        self,
        repository: TransactionRepository,
        target_profile_service: TargetProfileService,
        scoring_service: CandidateScoringService,
        evidence_builder: EvidenceBuilderService,
        one_pager_service: AcquirerOnePagerService,
    ):
        self._repository = repository
        self._target_profile_service = target_profile_service
        self._scoring_service = scoring_service
        self._evidence_builder = evidence_builder
        self._one_pager_service = one_pager_service

    def run(self, request: ReportRequest) -> AcquirerReport:
        loaded = self._repository.load()
        if not loaded.transactions:
            raise DataError(f"No usable rows in {loaded.source}")
        transactions = loaded.transactions

        target = self._target_profile_service.build(
            transactions, request.sector, request.ev_mm, request.ownership,
            request.min_margin_pct, request.region,
        )
        shortlist = self._scoring_service.shortlist(transactions, target, request.min_strategics)
        evidence = self._evidence_builder.build(transactions, target, shortlist)
        one_pagers, call_records = self._one_pager_service.generate_all(evidence, target)

        return AcquirerReport(
            target=target,
            min_strategics=request.min_strategics,
            evidence=evidence,
            one_pagers=one_pagers,
            warnings=shortlist.warnings,
            call_records=call_records,
            csv_source=loaded.source,
            csv_sha256=loaded.sha256,
            skipped_rows=loaded.skipped_rows,
            prompt_version=PROMPT_VERSION,
            model=self._one_pager_service.model,
        )
