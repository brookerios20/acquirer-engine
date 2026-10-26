import pytest

from acquirer_engine.config import AnalysisConfig
from acquirer_engine.errors import InputError
from acquirer_engine.models import STRATEGIC
from acquirer_engine.services import CandidateScoringService, DealCountBaseline
from tests.factories import make_target, make_transaction

TARGET = make_target(sector="Healthcare Services")


def deals(acquirer, count, sector="Healthcare Services", acquirer_type="Financial Sponsor"):
    return [make_transaction(acquirer=acquirer, sector=sector, acquirer_type=acquirer_type) for _ in range(count)]


def service(shortlist_size=3):
    # Deal count keeps the expected order obvious; ranking models are tested separately
    return CandidateScoringService(DealCountBaseline(), AnalysisConfig(shortlist_size=shortlist_size))


SPONSORS = deals("Sponsor 5", 5) + deals("Sponsor 4", 4) + deals("Sponsor 3", 3)
STRATEGICS = (
    deals("Strategic 2", 2, acquirer_type=STRATEGIC)
    + deals("Strategic 1", 1, acquirer_type=STRATEGIC)
    # Scores higher than either strategic above, but has no deal in the target's sector
    + deals("Strategic Elsewhere", 3, sector="Dental", acquirer_type=STRATEGIC)
)


def test_takes_top_n_by_score():
    shortlist = service().shortlist(SPONSORS + STRATEGICS, TARGET)

    assert [e.acquirer for e in shortlist.entries] == ["Sponsor 5", "Sponsor 4", "Sponsor 3"]
    assert [e.rank for e in shortlist.entries] == [1, 2, 3]
    assert not any(e.reserved_slot for e in shortlist.entries)
    assert shortlist.warnings == []


def test_ties_broken_by_name():
    transactions = deals("Bravo", 2) + deals("Alpha", 2) + deals("Charlie", 2)

    shortlist = service(shortlist_size=2).shortlist(transactions, TARGET)

    assert [e.acquirer for e in shortlist.entries] == ["Alpha", "Bravo"]


def test_fewer_rankable_acquirers_than_n_warns():
    shortlist = service(shortlist_size=5).shortlist(deals("Only", 1), TARGET)

    assert [e.acquirer for e in shortlist.entries] == ["Only"]
    assert "Only 1 rankable" in shortlist.warnings[0]


def test_min_strategics_replaces_lowest_sponsors_with_eligible_strategics():
    shortlist = service().shortlist(SPONSORS + STRATEGICS, TARGET, min_strategics=2)

    # Strategic Elsewhere outscores both but has no Healthcare Services deal, so it is skipped
    assert [e.acquirer for e in shortlist.entries] == ["Sponsor 5", "Strategic 2", "Strategic 1"]
    assert [e.reserved_slot for e in shortlist.entries] == [False, True, True]
    assert [e.rank for e in shortlist.entries] == [1, 2, 3]
    assert shortlist.warnings == []


def test_min_strategics_warns_when_too_few_are_eligible():
    shortlist = service().shortlist(SPONSORS + STRATEGICS, TARGET, min_strategics=3)

    names = [e.acquirer for e in shortlist.entries]
    assert names == ["Sponsor 5", "Strategic 2", "Strategic 1"]
    assert "Strategic Elsewhere" not in names
    assert shortlist.warnings == ["Requested 3 strategics, 2 eligible."]


def test_strategics_already_on_the_list_count_toward_the_floor():
    transactions = SPONSORS + deals("Big Strategic", 10, acquirer_type=STRATEGIC) + STRATEGICS

    shortlist = service().shortlist(transactions, TARGET, min_strategics=1)

    assert [e.acquirer for e in shortlist.entries] == ["Big Strategic", "Sponsor 5", "Sponsor 4"]
    assert not any(e.reserved_slot for e in shortlist.entries)


@pytest.mark.parametrize("value", [-1, 4])
def test_min_strategics_out_of_range_fails(value):
    with pytest.raises(InputError):
        service().shortlist(SPONSORS, TARGET, min_strategics=value)
