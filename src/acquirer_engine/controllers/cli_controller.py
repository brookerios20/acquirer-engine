"""CLI entry point and composition root (decision 1).

This is the only module that names concrete classes. It builds every object, runs the
report, and hands the result to the views.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ..clients import (
    AnthropicClient,
    LlmClient,
    RecordedClient,
    RecordingClient,
    RecordingNotFoundError,
)
from ..config import AnalysisConfig, LlmConfig, ValidationConfig
from ..errors import DataError, InputError
from ..models import ReportRequest
from ..repositories import CsvTransactionRepository
from ..services import (
    AcquirerOnePagerService,
    AcquirerReportService,
    BacktestService,
    CandidateScoringService,
    DealCountBaseline,
    EvidenceBuilderService,
    OutputValidatorService,
    SectorSimilarityService,
    SimilarityWeightedCount,
    TargetProfileService,
)
from ..views import JsonView, MarkdownView, RunLogView

DEFAULT_CSV = Path("data/ma_transactions_500.csv")
DEFAULT_RECORDINGS = Path("recordings")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="acquirers",
        description="Rank the 10 most likely acquirers for a target and draft a one-pager for each.",
    )
    parser.add_argument("command", nargs="?", default="report", choices=["report", "backtest"],
                        help="report (default): shortlist and one-pagers. backtest: compare ranking models "
                             "on held-out deals; no API key needed")
    # Defaults reproduce the brief's target (decision 2)
    parser.add_argument("--sector", default="Healthcare Services", help="Target sector, as named in the CSV")
    parser.add_argument("--ev", type=float, default=200.0, help="Target enterprise value in $M")
    parser.add_argument("--ownership", default="Private", help="Target ownership before the deal")
    parser.add_argument("--margin", type=float, default=None,
                        help="Strong-margin floor in %% (default: the sector's top quartile)")
    parser.add_argument("--region", default=None, help="A specific region (default: any regional target)")
    parser.add_argument("--min-strategics", type=int, default=0,
                        help="Reserve at least this many slots for strategic buyers (default 0)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Transaction CSV")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default runs/<timestamp>)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="Call the API and save each response for replay")
    mode.add_argument("--replay", action="store_true", help="Use saved responses; no API key needed")
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS,
                        help=f"Folder of saved responses (default {DEFAULT_RECORDINGS})")
    return parser.parse_args(argv)


def build_llm_client(args: argparse.Namespace, llm_config: LlmConfig) -> LlmClient | None:
    """Live, recording, or replay client (decision 13). None means the API key is missing."""
    if args.replay:
        return RecordedClient(args.recordings, llm_config.model)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    live = AnthropicClient(api_key, llm_config)
    return RecordingClient(live, args.recordings) if args.record else live


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "backtest":
        return run_backtest(args)
    return run_report(args)


def run_backtest(args: argparse.Namespace) -> int:
    """Deterministic: no LLM, so no API key (decision 4)."""
    analysis_config = AnalysisConfig()
    similarity_service = SectorSimilarityService()
    backtest_service = BacktestService(
        repository=CsvTransactionRepository(args.csv),
        models={
            "Deal count (baseline)": DealCountBaseline(),
            "Similarity-weighted count": SimilarityWeightedCount(similarity_service),
        },
        similarity_service=similarity_service,
        config=analysis_config,
    )
    try:
        result = backtest_service.run(args.sector, args.ev)
    except (DataError, InputError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    first, last = result.test_years
    print(f"Held-out deals: {first}-{last}. Each model is fit only on deals announced before each one.")
    print(f"Profile-like: {result.profile_description}.\n")
    print(f"{'Model':<28} {'Deals':<14} {'n':>4} {f'Hit@{result.k}':>8} {'MRR':>7}")
    for row in result.rows:
        print(f"{row.model:<28} {row.subset:<14} {row.deal_count:>4} {row.hit_at_k:>8.1%} {row.mrr:>7.3f}")
    return 0


def run_report(args: argparse.Namespace) -> int:
    started = time.perf_counter()

    # Keys come from .env (decision 14); fail before any analysis if missing
    load_dotenv(Path.cwd() / ".env")
    analysis_config = AnalysisConfig()
    llm_config = LlmConfig(model=os.environ.get("LLM_MODEL", "").strip() or LlmConfig.model)
    validation_config = ValidationConfig()

    llm_client = build_llm_client(args, llm_config)
    if llm_client is None:
        print("Set ANTHROPIC_API_KEY in .env, or run with --replay.", file=sys.stderr)
        return 2

    similarity_service = SectorSimilarityService()
    report_service = AcquirerReportService(
        repository=CsvTransactionRepository(args.csv),
        target_profile_service=TargetProfileService(analysis_config),
        scoring_service=CandidateScoringService(SimilarityWeightedCount(similarity_service), analysis_config),
        evidence_builder=EvidenceBuilderService(similarity_service, analysis_config),
        one_pager_service=AcquirerOnePagerService(
            llm_client=llm_client,
            validator=OutputValidatorService(validation_config),
            analysis_config=analysis_config,
            llm_config=llm_config,
            validation_config=validation_config,
        ),
    )
    request = ReportRequest(
        sector=args.sector,
        ev_mm=args.ev,
        ownership=args.ownership,
        min_margin_pct=args.margin,
        region=args.region,
        min_strategics=args.min_strategics,
    )

    try:
        report = report_service.run(request)
    except (DataError, InputError, RecordingNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    out_dir = args.out or Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    MarkdownView().write(report, out_dir)
    JsonView().write(report, out_dir)
    elapsed = time.perf_counter() - started
    llm_mode = "replay" if args.replay else "record" if args.record else "live"
    RunLogView().write(report, out_dir, analysis_config, llm_config, validation_config, elapsed, llm_mode)

    for warning in report.warnings:
        print(f"Note: {warning}")
    for one_pager in report.one_pagers:
        status = "NEEDS REVIEW" if one_pager.needs_review else "ok"
        print(f"{one_pager.rank:>2}. {one_pager.acquirer:<28} {one_pager.conviction.level.value:<7} {status}")
        if one_pager.needs_review:
            print(f"    Warning: {len(one_pager.review_flags)} check(s) failed; see the one-pager.", file=sys.stderr)
    print(f"\nWrote {out_dir}/summary.md and {len(report.one_pagers)} one-pagers in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
