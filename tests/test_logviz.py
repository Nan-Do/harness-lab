"""Tests for the log viewer's pretty-printing.

The events that carry the conversation with the model (``llm_request``,
``llm_continuation``) nest a whole chat transcript inside one JSON record, and
tool call arguments are nested once more as a JSON *string*. Rendering those
verbatim is what the viewer exists to avoid, so the guarantees worth pinning
down are that message bodies come out as text, that nested arguments are
decoded, and that neither of those breaks ``--compact``'s one line per event.
"""

import json

import pytest

import logviz


@pytest.fixture(autouse=True)
def plain_output(monkeypatch):
    """Render without colors, block clipping or compaction unless asked."""
    monkeypatch.setattr(logviz, "_NO_COLOR", True)
    monkeypatch.setattr(logviz, "_MAX_LINES", 0)
    monkeypatch.setattr(logviz, "_COMPACT", False)


FILE = 'import sys\n\n\ndef main():\n    print("hi")\n'

MESSAGES = [
    {"role": "system", "content": "You are harness-lab."},
    {"role": "user", "content": "Fix main.py"},
    {
        "role": "assistant",
        "content": "Reading it.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "main.py", "start": 1}),
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": FILE},
]


def render(event, **flags):
    full = flags.pop("full", False)
    width = flags.pop("width", 180)
    return logviz.render_event(event, width, full, flags.pop("compact", False))


def request(**extra):
    return {
        "ts": "2026-07-31T16:36:40.300000",
        "event": "llm_request",
        "session": "20260731-163639-a1b2c3",
        "model": "qwen2.5-coder-7b",
        "messages": MESSAGES,
        **extra,
    }


def test_message_bodies_are_rendered_as_text_not_json():
    out = render(request(), full=True)

    assert "You are harness-lab." in out
    assert 'def main():\n' in out
    # the escaped forms the raw record contains must not survive
    assert "\\n" not in out
    assert '{"role":' not in out


def test_tool_call_arguments_are_decoded_from_their_json_string():
    out = render(request(), full=True)

    assert "read_file(path=\"main.py\", start=1)" in out
    assert "id=call_1" in out
    assert '{"path": "main.py"' not in out


def test_transcript_is_summarized_until_full_is_asked_for():
    digest = "4 messages (system=1 user=1 assistant=1 tool=1"

    brief = render(request())
    assert digest in brief
    assert "You are harness-lab." not in brief

    assert digest in render(request(), full=True)


def test_continuation_renders_its_transcript_too():
    event = {
        "ts": "2026-07-31T16:36:42.100000",
        "event": "llm_continuation",
        "round": 1,
        "reason": "finish_reason=length; requesting continuation",
        "messages": MESSAGES,
    }

    out = render(event, full=True)

    assert "LLM ..." in out
    assert "requesting continuation" in out
    assert "You are harness-lab." in out


def test_reasoning_is_shown_apart_from_what_the_model_said():
    """A thinking model's log is mostly reasoning; labelling it as the answer
    would misreport what the model actually committed to."""
    event = {
        "ts": "2026-07-31T16:36:41.500000",
        "event": "llm_response",
        "round": 1,
        "finish_reason": "stop",
        "reasoning": "main.py is short, so one read is enough.",
        "content": "It prints hi.",
    }

    out = render(event, full=True)

    assert "thinks:" in out
    assert "main.py is short, so one read is enough." in out
    assert "says:" in out
    assert "It prints hi." in out


def test_unparseable_arguments_are_shown_raw():
    """A call cut off by the token limit is exactly this shape; hiding it
    would hide the reason the run went wrong."""
    truncated = {"role": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "write_file", "arguments": '{"path": "a.py'}}
    ]}

    out = "\n".join(logviz.fmt_message(1, truncated, 180, True))

    assert '{"path": "a.py' in out


def test_multiline_tool_arguments_become_a_block():
    event = {
        "ts": "2026-07-31T16:36:44.200000",
        "event": "tool_call",
        "step": 1,
        "name": "write_file",
        "args": {"path": "main.py", "content": FILE},
    }

    out = render(event, full=True)

    assert 'path: "main.py"' in out
    assert "| import sys" in out
    assert "\\n" not in out


def test_short_arguments_stay_on_one_line():
    event = {
        "ts": "2026-07-31T16:36:44.200000",
        "event": "tool_call",
        "step": 1,
        "name": "read_file",
        "args": {"path": "main.py", "start": 1},
    }

    assert 'read_file(path="main.py", start=1)' in render(event)


def test_max_lines_clips_long_blocks(monkeypatch):
    monkeypatch.setattr(logviz, "_MAX_LINES", 2)
    event = {
        "ts": "2026-07-31T16:36:44.400000",
        "event": "tool_result",
        "step": 1,
        "name": "read_file",
        "result": "one\ntwo\nthree\nfour",
    }

    out = render(event)

    assert "one" in out and "two" in out
    assert "three" not in out
    assert "[+2 more lines]" in out
    # --full overrides the clip
    assert "three" in render(event, full=True)


def test_compact_keeps_one_line_per_event(monkeypatch):
    monkeypatch.setattr(logviz, "_COMPACT", True)

    events = [
        request(),
        {
            "ts": "2026-07-31T16:36:44.200000",
            "event": "tool_call",
            "step": 1,
            "name": "write_file",
            "args": {"path": "main.py", "content": FILE},
        },
        {
            "ts": "2026-07-31T16:36:42.000000",
            "event": "llm_response",
            "round": 1,
            "content": "Here is the plan:\n1. read main.py",
        },
    ]

    for event in events:
        assert "\n" not in render(event, compact=True), event["event"]


def test_unknown_events_still_pretty_print_their_payload():
    event = {
        "ts": "2026-07-31T16:36:45.000000",
        "event": "some_new_event",
        "count": 3,
        "patch": "line one\nline two",
    }

    out = render(event)

    assert "count=3" in out
    assert "| line one" in out
    assert "| line two" in out
    assert "\\n" not in out
