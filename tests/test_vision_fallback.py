"""granite3.2-vision answers most question-style prompts with the bare word
"unanswerable"; analyze_image must never pass that through to the user."""

from agents.real import RealLlmAgent, _is_vision_refusal


def test_refusal_detection():
    assert _is_vision_refusal("unanswerable")
    assert _is_vision_refusal("  Unanswerable. ")
    assert _is_vision_refusal("")
    assert not _is_vision_refusal("The image shows a notebook page with handwritten notes.")


def _agent_with_replies(replies):
    agent = RealLlmAgent()
    prompts = []

    def fake_call(prompt, _image):
        prompts.append(prompt)
        return replies.pop(0)

    agent._vision_call = fake_call
    return agent, prompts


def test_focused_answer_used_when_model_answers():
    agent, prompts = _agent_with_replies(["A notebook page about reinforcement learning."])
    assert agent.analyze_image("analyze this pic", "b64") == "A notebook page about reinforcement learning."
    assert prompts == ["Describe this image in detail, focusing on: analyze this pic"]


def test_falls_back_to_general_description():
    agent, prompts = _agent_with_replies(["unanswerable", "A handwritten notebook page with equations."])
    out = agent.analyze_image("What does this handwritten note say?", "b64")
    assert "A handwritten notebook page with equations." in out
    assert "could not answer the question directly" in out
    assert prompts[1] == "Describe this image in detail."


def test_never_returns_bare_unanswerable():
    agent, _ = _agent_with_replies(["unanswerable", "unanswerable"])
    out = agent.analyze_image("read this", "b64")
    assert out.lower() != "unanswerable"
    assert "could not interpret this image" in out
