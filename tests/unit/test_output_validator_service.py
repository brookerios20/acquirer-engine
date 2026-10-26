import pytest

from acquirer_engine.config import ValidationConfig
from acquirer_engine.models import QualitativeRisk
from acquirer_engine.services import OutputValidatorService
from tests.llm_fakes import small_evidence, valid_draft

EVIDENCE = small_evidence()
PACK = next(p for p in EVIDENCE.packs if p.acquirer == "Sponsor A")  # trips thin evidence
GROUNDING = "Market comps: 10 deals, median EV/EBITDA 12.1x, largest $1,304.4M, 38% private, 2020-2024"


def validate(draft, grounding=GROUNDING):
    return OutputValidatorService(ValidationConfig()).validate(draft, PACK, grounding)


def with_text(section, text):
    return valid_draft(PACK).model_copy(update={section: text})


def test_valid_draft_passes():
    assert validate(valid_draft(PACK)) == []


def test_planted_fake_number_is_caught():
    draft = with_text("fit_thesis", valid_draft(PACK).fit_thesis + " They paid 14.1x for comparable assets.")

    assert validate(draft) == ["fit_thesis uses 14.1; that value isn't in the facts."]


@pytest.mark.parametrize("text", ["12.1x", "$1,304.4M", "38%", "from 2020 to 2024"])
def test_numbers_from_the_facts_pass_at_displayed_precision(text):
    draft = with_text("valuation_commentary", valid_draft(PACK).valuation_commentary + f" Figure: {text}.")

    assert validate(draft) == []


@pytest.mark.parametrize("text, number", [("12x", "12"), ("$1.3B", "1.3")])
def test_rounded_or_rescaled_numbers_fail(text, number):
    draft = with_text("valuation_commentary", valid_draft(PACK).valuation_commentary + f" About {text}.")

    assert validate(draft) == [f"valuation_commentary uses {number}; that value isn't in the facts."]


def test_ids_are_not_read_as_numbers():
    deal_id = PACK.stats.direct_precedents[0].transaction_id
    draft = with_text("precedent_commentary", valid_draft(PACK).precedent_commentary + f" See ({deal_id}) and (F2).")

    assert validate(draft) == []


def test_bad_citations_are_caught():
    draft = valid_draft(PACK)
    draft.citations.overview = ["F999"]
    draft = draft.model_copy(update={"fit_thesis": draft.fit_thesis + " Compare (MA-1999-0001)."})

    violations = validate(draft)

    assert "overview cites F999; that ID isn't in the facts." in violations
    assert "fit_thesis cites MA-1999-0001; that ID isn't in the facts." in violations


def test_empty_citation_list_is_caught():
    draft = valid_draft(PACK)
    draft.citations.valuation_commentary = []

    assert validate(draft) == ["citations.valuation_commentary is empty; cite the facts it relies on."]


def test_banned_phrase_is_caught_as_a_whole_word():
    banned = with_text("headline", "A leading consolidator of care delivery businesses worth calling")
    allowed = with_text("headline", "Not a misleading signal: an active consolidator worth calling")

    assert validate(banned) == ["headline uses the banned phrase 'leading'."]
    assert validate(allowed) == []


def test_too_few_risks_is_caught():
    draft = valid_draft(PACK)
    draft.qualitative_risks = []  # Sponsor A has 1 data risk, so 1 in total

    assert validate(draft) == ["Only 1 risk(s) in total; add qualitative risks to reach 2."]


def test_every_data_risk_needs_a_note_with_its_exact_title():
    draft = valid_draft(PACK)
    draft.data_risk_notes[0].title = "Thin data"

    assert validate(draft) == [
        "data_risk_notes is missing a note for 'Thin evidence'.",
        "data_risk_notes has 'Thin data', which isn't one of the data risks.",
    ]


def test_section_length_is_checked():
    assert validate(with_text("overview", "Too short.")) == ["overview has 2 words; keep it between 30 and 130."]


def test_numbers_in_qualitative_risks_are_checked():
    draft = valid_draft(PACK)
    draft.qualitative_risks.append(QualitativeRisk(title="Leverage", detail="Debt near 6.5x.", cited_ids=["F1"]))

    assert validate(draft) == ["risk 'Leverage' uses 6.5; that value isn't in the facts."]
