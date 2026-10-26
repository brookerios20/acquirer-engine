import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import formatting as fmt
from ..clients import LlmCallError, LlmClient, LlmMessage, LlmRequest
from ..config import AnalysisConfig, LlmConfig, ValidationConfig
from ..models import (
    STRATEGIC,
    AcquirerOnePager,
    CallRecord,
    Evidence,
    EvidencePack,
    OnePagerDraft,
    RiskFlag,
    RiskSeverity,
    RiskSource,
    TargetProfile,
)
from ..prompts import PROMPT_VERSION, load_template
from .output_validator_service import ID_PATTERN, OutputValidatorService

MAX_ATTEMPTS = 2  # one try plus one retry with the violations (decision 11)


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    grounding_text: str    # the data blocks only; numbers here count as grounded


class AcquirerOnePagerService:
    """Drafts each one-pager with one LLM call per acquirer, run concurrently (decision 9).

    Each draft is validated; on failure the call is retried once with the specific
    violations. If it still fails, the one-pager keeps its code-filled fields and is
    flagged for review (decision 11). One acquirer's failure never stops the run.
    """

    def __init__(
        self,
        llm_client: LlmClient,
        validator: OutputValidatorService,
        analysis_config: AnalysisConfig,
        llm_config: LlmConfig,
        validation_config: ValidationConfig,
    ):
        self._llm_client = llm_client
        self._validator = validator
        self._analysis_config = analysis_config
        self._llm_config = llm_config
        self._validation_config = validation_config
        self._system_template = load_template("system")
        self._acquirer_template = load_template("acquirer")
        self._lenses = {
            "sponsor": load_template("lens_sponsor").template,
            "strategic": load_template("lens_strategic").template,
        }

    @property
    def model(self) -> str:
        return self._llm_client.model

    def generate_all(
        self, evidence: Evidence, target: TargetProfile
    ) -> tuple[list[AcquirerOnePager], list[CallRecord]]:
        with ThreadPoolExecutor(max_workers=self._llm_config.max_workers) as pool:
            results = list(pool.map(lambda pack: self._generate_one(pack, evidence, target), evidence.packs))

        one_pagers = [one_pager for one_pager, _ in results]
        records = [record for _, pack_records in results for record in pack_records]
        return one_pagers, records

    def _generate_one(
        self, pack: EvidencePack, evidence: Evidence, target: TargetProfile
    ) -> tuple[AcquirerOnePager, list[CallRecord]]:
        prompt = self.build_prompt(pack, evidence, target)
        messages = [LlmMessage("user", prompt.user)]
        records: list[CallRecord] = []
        draft: OnePagerDraft | None = None
        violations: list[str] = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            request = LlmRequest(prompt.system, messages, OnePagerDraft, PROMPT_VERSION)
            try:
                response = self._llm_client.generate(request)
            except LlmCallError as error:
                records.append(CallRecord(pack.acquirer, attempt, self.model, 0.0, 0, 0, [], str(error)))
                # Keep an earlier draft's violations: they still describe the prose we have
                violations = [*violations, f"LLM call failed: {error}"]
                if not error.retryable:
                    break  # e.g. bad key or no credits: the same request would fail again
                continue  # retry the same request

            draft = response.parsed
            violations = self._validator.validate(draft, pack, prompt.grounding_text)
            records.append(CallRecord(
                pack.acquirer, attempt, response.model, response.latency_s,
                response.input_tokens, response.output_tokens, violations,
            ))
            if not violations:
                break
            # Retry in the same conversation, sending back exactly what was wrong
            messages = [
                *messages,
                LlmMessage("assistant", response.raw_json),
                LlmMessage("user", _retry_message(violations)),
            ]

        return self._assemble(pack, draft, violations), records

    # --- Prompt (decision 8) ------------------------------------------------------------

    def build_prompt(self, pack: EvidencePack, evidence: Evidence, target: TargetProfile) -> Prompt:
        config = self._analysis_config
        market = evidence.market_comps
        direct_first, direct_last = evidence.direct_years

        target_profile = "\n".join([
            f"- Sector: {target.sector}",
            f"- Enterprise value: {fmt.money(target.ev_mm)} (mid-market)",
            f"- Ownership before the deal: {target.ownership}",
            f"- Strong EBITDA margin: at or above {target.min_margin_pct:.1f}%",
            "- Region: " + (target.region or "regional (not National or Multi-Regional); no specific region given"),
        ])
        definitions = "\n".join([
            f"- Direct precedent: a closed deal in {' / '.join(evidence.direct_sectors)}, at "
            f"{fmt.money(market.min_ev_mm)}-{fmt.money(market.max_ev_mm)} EV, in {direct_first}-{direct_last}.",
            f"- Similarity weight: how close a sector is to {target.sector}, from 0 to 1, based on "
            "overlap in the sponsors that buy in both.",
            "- Ranking score: the sum of similarity weights over all of the buyer's deals.",
            f"- Conviction: High = {config.high_min_direct} or more direct precedents. Medium = "
            f"{config.medium_min_direct} or more, or {config.medium_min_exact_sector_deals} or more deals "
            f"of any outcome in {target.sector}. Low otherwise. A high-severity data risk lowers it one level.",
        ])
        market_comps = (
            f"{market.count} closed {market.sector} deals at {fmt.money(market.min_ev_mm)}-"
            f"{fmt.money(market.max_ev_mm)}"
        )
        if market.median_ev_ebitda is not None and market.median_ev_revenue is not None:
            market_comps += (
                f": median EV/EBITDA {fmt.ev_ebitda(market.median_ev_ebitda)}, "
                f"median EV/Revenue {fmt.ev_revenue(market.median_ev_revenue)}"
            )
        market_comps += f". IDs: {', '.join(market.transaction_ids) or 'none'}."

        facts_json = json.dumps(
            [{"id": f.id, "fact": f.label, "value": f.value} for f in pack.facts], indent=1
        )
        precedent_table = _precedent_table(pack)
        data_risks = "\n".join(
            f"- {r.title} ({r.severity.value} severity): {r.detail}. IDs: {', '.join(r.transaction_ids) or 'none'}"
            for r in pack.data_risks
        ) or "None."
        other_buyers = "\n".join(
            f"- {p.acquirer} ({p.entry.acquirer_type}, rank {p.entry.rank}): "
            f"{p.conviction.direct_precedents} direct precedents, "
            f"{p.stats.exact_sector_deals} deals in {target.sector}, {p.stats.total_deals} deals in total"
            for p in evidence.packs if p.acquirer != pack.acquirer
        ) or "None."

        lens = self._lenses["strategic" if pack.entry.acquirer_type == STRATEGIC else "sponsor"]
        user = self._acquirer_template.substitute(
            acquirer=pack.acquirer,
            target_profile=target_profile,
            definitions=definitions,
            market_comps=market_comps,
            facts_json=facts_json,
            precedent_table=precedent_table,
            data_risks=data_risks,
            other_buyers=other_buyers,
            lens=lens,
            min_risks=self._validation_config.min_risks,
            conviction=pack.conviction.level.value,
            length_limits="; ".join(
                f"{section} {low}-{high}" for section, low, high in self._validation_config.section_words
            ),
        )
        system = self._system_template.substitute(
            banned_phrases=", ".join(f'"{p}"' for p in self._validation_config.banned_phrases)
        )
        grounding_text = "\n".join([
            target_profile, definitions, market_comps, facts_json, precedent_table, data_risks, other_buyers,
        ])
        return Prompt(system, user, grounding_text)

    # --- Assembly (decision 10) ---------------------------------------------------------

    def _assemble(
        self, pack: EvidencePack, draft: OnePagerDraft | None, violations: list[str]
    ) -> AcquirerOnePager:
        one_pager = AcquirerOnePager(
            acquirer=pack.acquirer,
            acquirer_type=pack.entry.acquirer_type,
            rank=pack.entry.rank,
            reserved_slot=pack.entry.reserved_slot,
            ranking=pack.entry.ranking,
            precedents=pack.precedents,
            valuation=pack.valuation,
            conviction=pack.conviction,
            risks=list(pack.data_risks),
            review_flags=violations,
        )
        if draft is None:
            return one_pager  # prose left empty; the code-filled fields still stand

        notes = {n.title: n.note for n in draft.data_risk_notes}
        one_pager.risks = [
            dataclasses.replace(risk, explanation=notes.get(risk.title, "")) for risk in pack.data_risks
        ] + [
            RiskFlag(
                title=risk.title,
                detail=risk.detail,
                source=RiskSource.LLM,
                severity=RiskSeverity.NORMAL,
                transaction_ids=[i for i in risk.cited_ids if ID_PATTERN.fullmatch(i) and i.startswith("MA-")],
            )
            for risk in draft.qualitative_risks
        ]
        one_pager.headline = draft.headline
        one_pager.overview = draft.overview
        one_pager.fit_thesis = draft.fit_thesis
        one_pager.precedent_commentary = draft.precedent_commentary
        one_pager.valuation_commentary = draft.valuation_commentary
        one_pager.conviction_reason = draft.conviction_reason
        one_pager.cited_ids = draft.citations.model_dump()
        return one_pager


def _precedent_table(pack: EvidencePack) -> str:
    """Markdown table for the prompt. Deal type and financing are left out (decision 8)."""
    lines = [
        "| ID | Year | Sector | EV | Outcome | EV/EBITDA | EV/Revenue | Direct |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in pack.precedents:
        t = row.transaction
        lines.append(
            f"| {t.transaction_id} | {t.deal_year} | {t.sector} | {fmt.money(t.ev_mm)} | {t.outcome} | "
            f"{fmt.ev_ebitda(t.ev_ebitda)} | {fmt.ev_revenue(t.ev_revenue)} | {'yes' if row.is_direct else 'no'} |"
        )
    return "\n".join(lines)


def _retry_message(violations: list[str]) -> str:
    bullet_list = "\n".join(f"- {v}" for v in violations)
    return (
        "Your response broke these rules:\n"
        f"{bullet_list}\n"
        "Return the full corrected response. Use only figures that appear in the facts, exactly as written."
    )
