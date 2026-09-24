"""Sentence-level grounding: answer sentences whose content isn't in the
passages are removed, even when every quote verified."""

from pathlib import Path

from agents.mock import MockAgent
from contracts import AskRequest, Compartment, Label, Passage, Principal, Tier
from harness import grounding
from harness.runner import run_query

PASSAGE = Passage(
    doc_id="emergency_rulebook",
    page=54,
    title="Emergency Rulebook",
    text=(
        "Emergency equipment to be carried in the vehicle: 1 roll of gunny / hessian cloth "
        "(about 10 mts. long), Breathing Apparatus (With spare filled cylinder and Canister gas "
        "masks), fire proximity suit, two DCP fire extinguishers of 10 kg capacity. Only "
        "reliable trained staff or qualified contractors may carry out maintenance work on a pipeline."
    ),
    label=Label(tier=Tier.INTERNAL, compartments=frozenset([Compartment.HSE])),
)

GROUNDED = "The vehicle must carry breathing apparatus with a spare filled cylinder and canister gas masks."
INVENTED = "The inspection found minor corrosion on the spark arrestors and the flame detectors."
INVENTED_NUMBER = "Two DCP fire extinguishers of 25 kg capacity must be carried."


def test_grounded_sentence_passes():
    result = grounding.check(GROUNDED, [PASSAGE])
    assert result.checked == 1
    assert result.unsupported == []


def test_invented_finding_flagged():
    result = grounding.check(f"{GROUNDED} {INVENTED}", [PASSAGE])
    assert result.checked == 2
    assert result.unsupported == [INVENTED]


def test_number_not_in_source_flagged():
    result = grounding.check(INVENTED_NUMBER, [PASSAGE])
    assert result.unsupported == [INVENTED_NUMBER]


def test_short_fragments_and_headings_not_scored():
    result = grounding.check("## Emergency Kit Findings Overview\nStay calm.", [PASSAGE])
    assert result.checked == 0


def test_remove_keeps_markdown_structure():
    answer = (
        "## Emergency kit\n"
        f"- {GROUNDED}\n"
        f"- {INVENTED}\n"
        "## Findings\n"
        f"{INVENTED}"
    )
    out = grounding.remove_sentences(answer, [INVENTED])
    assert out == f"## Emergency kit\n- {GROUNDED}"


def _principal():
    return Principal(
        person_id="safety-01",
        name="Head of Safety",
        job_title="Safety",
        grade="F",
        compartments=frozenset([Compartment.HSE]),
    )


def test_runner_removes_invented_sentence_and_says_so(tmp_path: Path):
    agent = MockAgent(custom_answer=f"{GROUNDED} {INVENTED}")
    response = run_query(
        request=AskRequest(question="what emergency equipment must the vehicle carry?", top_k=1),
        corpus=[PASSAGE],
        principal=_principal(),
        agent=agent,
        db_path=str(tmp_path / "g.db"),
    )
    assert response.status == "answered"
    assert GROUNDED in response.answer
    assert "spark arrestors" not in response.answer
    assert "1 statement was removed" in response.answer


def test_runner_retries_then_abstains_when_nothing_is_grounded(tmp_path: Path):
    agent = MockAgent(custom_answer=INVENTED)
    response = run_query(
        request=AskRequest(question="what emergency equipment must the vehicle carry?", top_k=1),
        corpus=[PASSAGE],
        principal=_principal(),
        agent=agent,
        db_path=str(tmp_path / "g.db"),
    )
    assert response.status == "abstained"
    assert agent.calls == 3


SAFETY = Passage(
    doc_id="fire_sop",
    page=3,
    title="Fire SOP",
    text=(
        "Do not use water on oil fires. Stop the engine. Move the vehicle to an open area. "
        "Call the fire brigade. Water must be used to cool the storage tank during a fire."
    ),
    label=Label(tier=Tier.INTERNAL, compartments=frozenset([Compartment.HSE])),
)


def test_derived_small_count_is_not_treated_as_invented():
    s = "The SOP lists 3 immediate steps: stop the engine, move the vehicle, call the fire brigade."
    assert grounding.check(s, [SAFETY]).unsupported == []


def test_dropped_negation_flagged():
    s2 = "Water should be used on oil fires to control them."
    assert grounding.check(s2, [SAFETY]).unsupported == [s2]
    assert grounding.check("Never use water on oil fires, per the SOP.", [SAFETY]).unsupported == []


def test_added_negation_flagged():
    s = "Water must not be used to cool the storage tank during a fire."
    assert grounding.check(s, [SAFETY]).unsupported == [s]


def test_prohibition_restating_a_negation_is_supported():
    s = "Using water on oil fires is prohibited by the fire SOP."
    assert grounding.check(s, [SAFETY]).unsupported == []
