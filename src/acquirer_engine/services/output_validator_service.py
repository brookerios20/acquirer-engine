import re
from decimal import Decimal, InvalidOperation

from ..config import ValidationConfig
from ..models import EvidencePack, OnePagerDraft

ID_PATTERN = re.compile(r"\b(?:F\d+|MA-\d{4}-\d{4})\b")
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


class OutputValidatorService:
    """Checks one LLM draft against its evidence pack (decision 11).

    Returns a list of violations, each phrased so it can be sent back to the model as-is.
    An empty list means the draft passed.
    """

    def __init__(self, config: ValidationConfig):
        self._config = config
        self._banned = [
            (phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
            for phrase in config.banned_phrases
        ]

    def validate(self, draft: OnePagerDraft, pack: EvidencePack, grounding_text: str) -> list[str]:
        violations: list[str] = []
        prose = self._prose_sections(draft)

        # 1. Schema: the SDK already parsed the JSON; check required text isn't blank
        for section, text in prose.items():
            if not text.strip():
                violations.append(f"{section} is empty.")

        # 2. Every cited ID, in `citations` or inline in the prose, is in the evidence pack
        citable = pack.citable_ids
        cited = {section: list(ids) for section, ids in draft.citations.model_dump().items()}
        for risk in draft.qualitative_risks:
            cited[f"risk '{risk.title}'"] = risk.cited_ids
        for section, text in prose.items():
            cited.setdefault(section, []).extend(ID_PATTERN.findall(text))
        for section, ids in cited.items():
            for unknown in sorted(set(ids) - citable):
                violations.append(f"{section} cites {unknown}; that ID isn't in the facts.")
        for section, ids in draft.citations.model_dump().items():
            if not ids:
                violations.append(f"citations.{section} is empty; cite the facts it relies on.")

        # 3. Every number in the prose appears in the data given to the model
        allowed = _numbers_in(grounding_text)
        for section, text in prose.items():
            for number in _number_strings(ID_PATTERN.sub(" ", text)):
                if _normalize(number) not in allowed:
                    violations.append(f"{section} uses {number}; that value isn't in the facts.")

        # 4. Risk count, data-risk notes, banned phrases, section lengths
        risk_count = len(pack.data_risks) + len(draft.qualitative_risks)
        if risk_count < self._config.min_risks:
            violations.append(
                f"Only {risk_count} risk(s) in total; add qualitative risks to reach {self._config.min_risks}."
            )
        data_titles = {r.title for r in pack.data_risks}
        noted_titles = {n.title for n in draft.data_risk_notes}
        for title in sorted(data_titles - noted_titles):
            violations.append(f"data_risk_notes is missing a note for '{title}'.")
        for title in sorted(noted_titles - data_titles):
            violations.append(f"data_risk_notes has '{title}', which isn't one of the data risks.")

        for section, text in prose.items():
            for phrase, pattern in self._banned:
                if pattern.search(text):
                    violations.append(f"{section} uses the banned phrase '{phrase}'.")

        for section, low, high in self._config.section_words:
            words = len(getattr(draft, section).split())
            if not low <= words <= high:
                violations.append(f"{section} has {words} words; keep it between {low} and {high}.")

        return violations

    def _prose_sections(self, draft: OnePagerDraft) -> dict[str, str]:
        sections = {
            "headline": draft.headline,
            "overview": draft.overview,
            "fit_thesis": draft.fit_thesis,
            "precedent_commentary": draft.precedent_commentary,
            "valuation_commentary": draft.valuation_commentary,
            "conviction_reason": draft.conviction_reason,
        }
        for note in draft.data_risk_notes:
            sections[f"data risk note '{note.title}'"] = note.note
        for risk in draft.qualitative_risks:
            sections[f"risk '{risk.title}'"] = f"{risk.title}. {risk.detail}"
        return sections


def _number_strings(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def _normalize(number: str) -> str:
    """'1,304.4' -> '1304.4'; '2.90' -> '2.9'. Precision beyond what's shown still differs:
    '12.1' and '12' are different values."""
    try:
        value = Decimal(number.replace(",", "")).normalize()
    except InvalidOperation:
        return number
    # normalize() turns 100 into 1E+2; format back to plain digits
    return format(value, "f")


def _numbers_in(text: str) -> set[str]:
    return {_normalize(n) for n in _number_strings(ID_PATTERN.sub(" ", text))}
