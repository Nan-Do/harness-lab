"""Tests for the plain-text transcript.

The mode exists to show the whole interaction, which makes showing something
*twice* the failure worth guarding against: a streamed answer is already on
screen by the time the agent reports it, and so is the body of a streamed tool
call by the time it is announced (and approved).
"""

import io
from types import SimpleNamespace

from rich.console import Console

from plain import PlainView

FILE = "def add(a, b):\n    return a + b\n"


def view(show_reasoning: bool = True) -> PlainView:
    console = Console(
        file=io.StringIO(),
        width=100,
        soft_wrap=True,
        highlight=False,
        no_color=True,
    )
    agent = SimpleNamespace(logger=SimpleNamespace(log=lambda *a, **k: None))
    return PlainView(agent, console=console, show_reasoning=show_reasoning)


def text(plain: PlainView) -> str:
    return plain.console.file.getvalue()


def test_streamed_answer_is_printed_once():
    plain = view()

    plain.event("thinking")
    plain.event("assistant_delta", text="calc.py adds ")
    plain.event("assistant_delta", text="two numbers.")
    plain.event("final", text="calc.py adds two numbers.")

    assert text(plain).count("calc.py adds two numbers.") == 1


def test_unstreamed_answer_is_still_printed():
    plain = view()

    plain.event("thinking")
    plain.event("final", text="calc.py adds two numbers.")

    assert "answer>" in text(plain)
    assert "calc.py adds two numbers." in text(plain)


def test_streamed_reasoning_is_shown_apart_from_the_answer():
    plain = view()

    plain.event("thinking")
    plain.event("reasoning_delta", text="calc.py has ")
    plain.event("reasoning_delta", text="one function.")
    plain.event("assistant_delta", text="It adds two numbers.")
    plain.event("reasoning", text="calc.py has one function.")
    plain.event("final", text="It adds two numbers.")

    printed = text(plain)
    assert "think> calc.py has one function." in printed
    assert "model> It adds two numbers." in printed
    # Neither channel is printed twice by the events that report the whole turn.
    assert printed.count("calc.py has one function.") == 1
    assert printed.count("It adds two numbers.") == 1


def test_unstreamed_reasoning_is_still_printed():
    """--no-stream reports the turn's thinking only once it is over."""
    plain = view()

    plain.event("thinking")
    plain.event("reasoning", text="calc.py has one function.")
    plain.event("final", text="It adds two numbers.")

    printed = text(plain)
    assert "think> calc.py has one function." in printed
    assert "answer> It adds two numbers." in printed


def test_reasoning_can_be_turned_off():
    plain = view(show_reasoning=False)

    plain.event("thinking")
    plain.event("reasoning_delta", text="calc.py has one function.")
    plain.event("reasoning", text="calc.py has one function.")
    plain.event("final", text="It adds two numbers.")

    printed = text(plain)
    assert "calc.py has one function." not in printed
    assert "It adds two numbers." in printed


def test_streamed_body_is_not_repeated_when_the_call_is_announced():
    plain = view()

    plain.event("thinking")
    plain.event("tool_delta", name="write_file", args_text='{"path": "calc.py", "con')
    plain.event(
        "tool_delta",
        name="write_file",
        args_text='{"path": "calc.py", "content": "def add(a, b):\\n',
    )
    plain.event(
        "tool_delta",
        name="write_file",
        args_text='{"path": "calc.py", "content": "def add(a, b):\\n    return a + b\\n"}',
    )
    plain.event("tool_call", name="write_file", args={"path": "calc.py", "content": FILE})

    printed = text(plain)
    assert printed.count("def add(a, b):") == 1
    assert printed.count("return a + b") == 1
    # The call itself is still announced, with its short arguments.
    assert "→ write_file path=calc.py" in printed


def test_unstreamed_body_is_printed_as_a_block():
    plain = view()

    plain.event("thinking")
    plain.event("tool_call", name="write_file", args={"path": "calc.py", "content": FILE})

    printed = text(plain)
    assert "write_file · calc.py" in printed
    assert "return a + b" in printed


def test_tool_result_and_notices_are_shown():
    plain = view()

    plain.event("tool_call", name="read_file", args={"path": "calc.py"})
    plain.event("tool_result", name="read_file", result="1: def add(a, b):")
    plain.event("retry", notice="Runtime notice: no actionable output.")

    printed = text(plain)
    assert "← read_file" in printed
    assert "1: def add(a, b):" in printed
    assert "retrying: Runtime notice" in printed


def test_a_new_turn_forgets_what_the_last_one_streamed():
    plain = view()

    plain.event("thinking")
    plain.event("assistant_delta", text="reading the file")
    plain.event("tool_call", name="read_file", args={"path": "calc.py"})
    plain.event("tool_result", name="read_file", result="ok")
    plain.event("thinking")
    plain.event("final", text="It adds two numbers.")

    # The second turn streamed nothing, so its answer must be printed.
    assert "It adds two numbers." in text(plain)
