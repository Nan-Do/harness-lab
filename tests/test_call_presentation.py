"""Tests for showing a tool call without dumping its payload.

A front-end has to answer two questions about a call it is about to run: what
is it doing (short enough for a prompt line) and what exactly is it about to
write (long, and worth reading as code somewhere else). Splitting those two
apart is what keeps an approval dialog readable, so the split is pinned here --
along with reading the body straight out of a call that is still streaming.
"""

from tool_support import (
    describe_call,
    size_note,
    split_args,
    streaming_body,
)

FILE = 'def add(a, b):\n    return a + b\n'


def test_short_arguments_stay_inline():
    inline, bodies = split_args({"path": "calc.py", "start": 1, "end": 20})

    assert bodies == []
    assert dict(inline) == {"path": "calc.py", "start": "1", "end": "20"}


def test_multiline_and_long_values_become_bodies():
    long_line = "x" * 200
    inline, bodies = split_args(
        {"path": "calc.py", "content": FILE, "command": long_line}
    )

    assert dict(inline) == {"path": "calc.py"}
    assert dict(bodies) == {"content": FILE, "command": long_line}


def test_describe_call_sizes_the_body_instead_of_quoting_it():
    described = describe_call("write_file", {"path": "calc.py", "content": FILE})

    assert "path=calc.py" in described
    assert "content: 2 lines" in described
    assert "def add" not in described


def test_size_note_counts_a_final_line_without_a_newline():
    assert size_note("a\nb\n") == "2 lines, 4 B"
    assert size_note("a\nb") == "2 lines, 3 B"
    assert size_note("") == "0 lines, 0 B"


def test_streaming_body_decodes_a_half_written_argument():
    partial = '{"path": "calc.py", "content": "def add(a, b):\\n    return'

    key, body = streaming_body(partial)

    assert key == "content"
    assert body == "def add(a, b):\n    return"


def test_streaming_body_waits_for_a_split_escape_sequence():
    """A chunk that ends mid-escape must not leak the backslash to the screen."""
    key, body = streaming_body('{"content": "line one\\')

    assert key == "content"
    assert body == "line one"


def test_streaming_body_stops_at_the_closing_quote():
    finished = '{"path": "calc.py", "content": "done\\n", "mode": "w"}'

    assert streaming_body(finished) == ("content", "done\n")


def test_streaming_body_follows_the_argument_being_written_now():
    """patch_file fills in old_text and then new_text; the live one wins."""
    raw = '{"path": "calc.py", "old_text": "return a", "new_text": "return a + b'

    assert streaming_body(raw) == ("new_text", "return a + b")


def test_streaming_body_is_empty_before_a_body_starts():
    assert streaming_body('{"path": "ca') == ("", "")
    assert streaming_body("") == ("", "")


def test_streaming_body_grows_monotonically():
    """The printed text is a diff against what was shown, so it may only grow."""
    raw = '{"content": "a\\tb\\u00e9c"}'
    seen = ""
    for size in range(1, len(raw) + 1):
        _, body = streaming_body(raw[:size])
        assert body[: len(seen)] == seen, (seen, body)
        seen = body
    assert seen == "a\tbéc"
