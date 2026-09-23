"""Live token streaming: the answer text shown while a model is still writing
its JSON must be exactly the decoded prefix of the final answer -- never raw
JSON, never a half-decoded escape sequence."""

import json

from agents.real import _stream_emitter, partial_json_string_field


def test_field_not_started_returns_none():
    assert partial_json_string_field('{"ans', "answer") is None
    assert partial_json_string_field('{"citations": []', "answer") is None


def test_partial_prefix_decoded():
    assert partial_json_string_field('{"answer": "## Head', "answer") == "## Head"
    assert partial_json_string_field('{"answer": "a\\nb', "answer") == "a\nb"


def test_incomplete_escapes_are_held_back():
    assert partial_json_string_field('{"answer": "x\\', "answer") == "x"
    assert partial_json_string_field('{"answer": "x\\u00', "answer") == "x"
    assert partial_json_string_field('{"answer": "x\\ud83d', "answer") == "x"


def test_stops_at_closing_quote():
    raw = '{"answer": "done \\"quoted\\" text", "citations": [{"quote": "other"}]}'
    assert partial_json_string_field(raw, "answer") == 'done "quoted" text'


def test_streamed_deltas_reassemble_exact_final_answer():
    final = {"answer": "## Findings\n- Pump **P-101** at 12.5 bar\n- \"Quoted\" note — ok \U0001F600", "citations": []}
    raw = json.dumps(final)
    events = []
    on_token = _stream_emitter(events.append, json_field="answer")
    for i in range(0, len(raw), 3):  # feed in small, escape-splitting chunks
        on_token(raw[i:i + 3])
    assert "".join(e["text"] for e in events) == final["answer"]
    assert all(e["stage"] == "stream" and e["status"] == "delta" for e in events)


def test_plain_text_stream_passes_through():
    events = []
    on_token = _stream_emitter(events.append)
    for piece in ["```python\n", "print(1)\n", "```"]:
        on_token(piece)
    assert "".join(e["text"] for e in events) == "```python\nprint(1)\n```"


def test_no_emit_means_no_streaming():
    assert _stream_emitter(None) is None
