import json

from acquirer_engine.clients import LlmCallError
from acquirer_engine.config import AnalysisConfig, LlmConfig, ValidationConfig
from acquirer_engine.models import RiskSource
from acquirer_engine.services import AcquirerOnePagerService, OutputValidatorService
from tests.llm_fakes import TARGET, FakeLlmClient, acquirer_of, small_evidence, valid_draft

EVIDENCE = small_evidence()
PACKS = {p.acquirer: p for p in EVIDENCE.packs}


def service(client):
    return AcquirerOnePagerService(
        llm_client=client,
        validator=OutputValidatorService(ValidationConfig()),
        analysis_config=AnalysisConfig(),
        llm_config=LlmConfig(max_workers=3),
        validation_config=ValidationConfig(),
    )


def always_valid(request, call_number):
    return valid_draft(PACKS[acquirer_of(request)])


def test_one_call_per_acquirer_in_rank_order():
    client = FakeLlmClient(always_valid)

    one_pagers, records = service(client).generate_all(EVIDENCE, TARGET)

    assert [p.acquirer for p in one_pagers] == ["Sponsor B", "Sponsor A", "Health System"]
    assert len(client.requests) == 3
    assert all(not p.needs_review for p in one_pagers)
    assert [r.attempt for r in records] == [1, 1, 1]


def test_assembles_code_fields_and_llm_prose():
    one_pagers, _ = service(FakeLlmClient(always_valid)).generate_all(EVIDENCE, TARGET)
    sponsor_a = next(p for p in one_pagers if p.acquirer == "Sponsor A")
    pack = PACKS["Sponsor A"]

    assert sponsor_a.conviction == pack.conviction
    assert sponsor_a.precedents == pack.precedents
    assert sponsor_a.overview.startswith("This buyer")
    data_risks = [r for r in sponsor_a.risks if r.source == RiskSource.DATA]
    llm_risks = [r for r in sponsor_a.risks if r.source == RiskSource.LLM]
    assert [r.title for r in data_risks] == ["Thin evidence"]
    assert data_risks[0].explanation == "This matters for the process."
    assert [r.title for r in llm_risks] == ["Financing capacity", "Integration"]
    assert sponsor_a.cited_ids["overview"] == ["F1"]


def test_retries_once_with_the_violations():
    def banned_then_valid(request, call_number):
        draft = valid_draft(PACKS[acquirer_of(request)])
        if call_number == 1:
            return draft.model_copy(update={"headline": "A leading buyer of care delivery businesses"})
        return draft

    client = FakeLlmClient(banned_then_valid)
    one_pagers, records = service(client).generate_all(EVIDENCE, TARGET)

    assert all(not p.needs_review for p in one_pagers)
    assert len(client.requests) == 6
    retry = next(r for r in client.requests if len(r.messages) == 3)
    assert retry.messages[1].role == "assistant"
    assert "headline uses the banned phrase 'leading'." in retry.messages[2].content
    assert sorted(r.attempt for r in records) == [1, 1, 1, 2, 2, 2]


def test_flags_for_review_after_two_failures_and_keeps_the_prose():
    def always_bad(request, call_number):
        return valid_draft(PACKS[acquirer_of(request)]).model_copy(update={"overview": "Too short."})

    one_pagers, _ = service(FakeLlmClient(always_bad)).generate_all(EVIDENCE, TARGET)

    for one_pager in one_pagers:
        assert one_pager.review_flags == ["overview has 2 words; keep it between 30 and 130."]
        assert one_pager.overview == "Too short."
        assert one_pager.conviction == PACKS[one_pager.acquirer].conviction


def test_failed_calls_leave_prose_empty_and_the_run_continues():
    def fail_for_sponsor_a(request, call_number):
        if acquirer_of(request) == "Sponsor A":
            raise LlmCallError("model declined the request (category: None)")
        return valid_draft(PACKS[acquirer_of(request)])

    one_pagers, records = service(FakeLlmClient(fail_for_sponsor_a)).generate_all(EVIDENCE, TARGET)
    sponsor_a = next(p for p in one_pagers if p.acquirer == "Sponsor A")

    assert sponsor_a.overview == ""
    assert sponsor_a.review_flags == ["LLM call failed: model declined the request (category: None)"] * 2
    assert sponsor_a.risks == PACKS["Sponsor A"].data_risks
    assert [p.needs_review for p in one_pagers] == [False, True, False]
    assert [r.error is not None for r in records if r.acquirer == "Sponsor A"] == [True, True]


def test_prompt_holds_the_evidence_and_leaves_out_deal_type():
    prompt = service(FakeLlmClient(always_valid)).build_prompt(PACKS["Health System"], EVIDENCE, TARGET)

    facts = json.loads(prompt.user.split("```json\n")[1].split("\n```")[0])
    assert facts[0] == {"id": "F1", "fact": "Acquirer type", "value": "Strategic"}
    assert "Platform Investment" not in prompt.user      # deal_type stays out (decision 8)
    assert "Leveraged" not in prompt.user                # financing_type too
    assert "strategic (an operating company)" in prompt.user   # strategic lens
    assert "- Sponsor B (Financial Sponsor, rank 1)" in prompt.user
    assert "- Health System" not in prompt.user          # not listed among the other buyers
    assert "$" not in prompt.system.replace("$1,304.4M", "").replace("$1.3B", "")  # template fully filled
