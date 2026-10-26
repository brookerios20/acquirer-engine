"""Test doubles for the LLM layer and a small evidence fixture to run them against."""
import threading
from typing import Callable
from unittest.mock import create_autospec

from acquirer_engine.clients import LlmCallError, LlmRequest, LlmResponse
from acquirer_engine.config import AnalysisConfig
from acquirer_engine.models import (
    STRATEGIC,
    Citations,
    DataRiskNote,
    Evidence,
    EvidencePack,
    OnePagerDraft,
    QualitativeRisk,
    RankingScore,
    SectorSimilarity,
    Shortlist,
    ShortlistEntry,
)
from acquirer_engine.services import EvidenceBuilderService, SectorSimilarityService
from tests.factories import make_target, make_transaction

HCS, PG, BH, DENTAL = "Healthcare Services", "Physician Groups", "Behavioral Health", "Dental"
SIMILARITY = SectorSimilarity({
    HCS: {PG: 0.9, BH: 0.8, DENTAL: 0.1},
    PG: {HCS: 0.9, BH: 0.5, DENTAL: 0.2},
    BH: {HCS: 0.8, PG: 0.5, DENTAL: 0.3},
    DENTAL: {HCS: 0.1, PG: 0.2, BH: 0.3},
})
TARGET = make_target(sector=HCS, ev_mm=200.0)


def build_evidence(transactions, acquirers, target=TARGET) -> Evidence:
    """Evidence for the named acquirers, in rank order, with sector similarity fixed."""
    similarity_service = create_autospec(SectorSimilarityService, instance=True)
    similarity_service.compute.return_value = SIMILARITY
    types = {t.acquirer: t.acquirer_type for t in transactions}
    shortlist = Shortlist([
        ShortlistEntry(rank, name, types[name], RankingScore(name, 1.0, []), False)
        for rank, name in enumerate(acquirers, start=1)
    ])
    return EvidenceBuilderService(similarity_service, AnalysisConfig()).build(transactions, target, shortlist)


def small_evidence() -> Evidence:
    """Two sponsors and a strategic. Sponsor A trips thin evidence; the strategic trips size stretch."""
    transactions = (
        [make_transaction(acquirer="Sponsor A", sector=HCS, ev_mm=250.0)]
        + [make_transaction(acquirer="Sponsor B", sector=s, ev_mm=250.0) for s in (HCS, PG, BH)]
        + [make_transaction(acquirer="Health System", acquirer_type=STRATEGIC, sector=HCS, ev_mm=60.0,
                            deal_type="Platform Investment", financing_type="Leveraged")]
    )
    return build_evidence(transactions, ["Sponsor B", "Sponsor A", "Health System"])


SENTENCE = "This buyer appears in the data with deals in related sectors and recent activity (F1)."


def valid_draft(pack: EvidencePack) -> OnePagerDraft:
    """A draft that passes every validator rule: no numbers, real IDs, enough words."""
    return OnePagerDraft(
        headline="Repeat buyer in adjacent care delivery sectors worth an early call",
        overview=f"{SENTENCE} {SENTENCE} {SENTENCE}",
        fit_thesis=f"{SENTENCE} {SENTENCE} {SENTENCE} {SENTENCE}",
        precedent_commentary=f"{SENTENCE} {SENTENCE}",
        valuation_commentary=f"{SENTENCE} {SENTENCE}",
        data_risk_notes=[DataRiskNote(title=r.title, note="This matters for the process.") for r in pack.data_risks],
        qualitative_risks=[
            QualitativeRisk(title="Financing capacity", detail="Outside the data: leverage markets may tighten.",
                            cited_ids=["F1"]),
            QualitativeRisk(title="Integration", detail="Outside the data: integration may take management time.",
                            cited_ids=["F1"]),
        ],
        conviction_reason="The computed level follows from the direct precedent count and sector depth (F1).",
        citations=Citations(
            overview=["F1"], fit_thesis=["F1"], precedent_commentary=["F1"],
            valuation_commentary=["F1"], conviction_reason=["F1"],
        ),
    )


class FakeLlmClient:
    """Answers each request with `respond(request, call_number)`, which returns a draft or
    raises `LlmCallError`. Records every request; safe to call from worker threads."""

    model = "fake-model"

    def __init__(self, respond: Callable[[LlmRequest, int], OnePagerDraft]):
        self._respond = respond
        self._lock = threading.Lock()
        self.requests: list[LlmRequest] = []

    def generate(self, request: LlmRequest) -> LlmResponse:
        with self._lock:
            self.requests.append(request)
            call_number = sum(r.messages[0] == request.messages[0] for r in self.requests)
        draft = self._respond(request, call_number)
        return LlmResponse(
            parsed=draft, raw_json=draft.model_dump_json(), model=self.model,
            input_tokens=100, output_tokens=50, latency_s=0.01,
        )


def acquirer_of(request: LlmRequest) -> str:
    """The acquirer a prompt is about, from the first line of the user message."""
    first_line = request.messages[0].content.splitlines()[0]
    return first_line.split("**")[1]


__all__ = ["FakeLlmClient", "LlmCallError", "acquirer_of", "build_evidence", "small_evidence", "valid_draft"]
