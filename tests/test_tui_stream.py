"""Tests for what the TUI does with each channel of a streamed turn.

A turn arrives on three channels -- the model's reasoning, the text it is
saying, and the tool call it is assembling -- and each has a place to be: the
live view while it is being written, and a block of its own in the log once
the turn is decided. Reasoning is the newest of the three: it used to go
nowhere, being neither the answer nor a tool call.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

from textual.widgets import RichLog, Static

from tui import MiniAgentApp

EVENTS = [
    ("thinking", {}),
    ("reasoning_delta", {"text": "calc.py only "}),
    ("reasoning_delta", {"text": "adds numbers."}),
    ("reasoning", {"text": "calc.py only adds numbers."}),
    ("final", {"text": "It adds two numbers."}),
]


def stub_agent(ask) -> SimpleNamespace:
    return SimpleNamespace(
        ask=ask,
        approval_policy="never",
        tools=SimpleNamespace(approval_fn=None),
        workspace=SimpleNamespace(branch="main", cwd="/tmp/x"),
        session=SimpleNamespace(id="test-session"),
        logger=SimpleNamespace(log=lambda *a, **k: None),
        memory_text=lambda: "Memory:",
        session_path="/tmp/x/session.json",
        log_path="/tmp/x/log.jsonl",
        reset=lambda: None,
    )


def widget_text(widget) -> str:
    """What a widget is currently showing, as text."""
    return "\n".join(
        strip.text.rstrip() for strip in widget.render_lines(widget.region.reset_offset)
    )


def run_app(show_reasoning: bool = True, events: list | None = None) -> str:
    """Run one turn of canned events and return what the log ended up with."""
    done = threading.Event()

    def ask(text, on_event=None):
        for event_type, data in events if events is not None else EVENTS:
            on_event(event_type, **data)
        done.set()
        return "done"

    app = MiniAgentApp(
        stub_agent(ask),
        model="m",
        context=4096,
        endpoint="127.0.0.1:8080",
        prompt="describe calc.py",
        show_reasoning=show_reasoning,
    )
    written: list = []

    async def drive():
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if done.is_set():
                    break
            # The worker thread hands its last writes to the UI thread.
            await pilot.pause()
            written.append(
                "\n".join(strip.text for strip in app.query_one("#log", RichLog).lines)
            )

    asyncio.run(drive())
    return written[0]


def test_reasoning_is_committed_to_the_log_as_its_own_block():
    log = run_app()

    assert "Agent · Reasoning" in log
    assert "calc.py only adds numbers." in log
    # The answer is still the answer, in a panel of its own.
    assert "It adds two numbers." in log


def test_reasoning_is_hidden_when_it_is_turned_off():
    log = run_app(show_reasoning=False)

    assert "Reasoning" not in log
    assert "calc.py only adds numbers." not in log
    assert "It adds two numbers." in log


def test_the_live_view_follows_the_turn_from_thinking_to_the_tool_body():
    """Each channel takes the view in turn; the body is what gets approved."""
    thought, wrote, seen = threading.Event(), threading.Event(), []
    args = '{"path": "calc.py", "content": "def add(a, b):'

    def ask(text, on_event=None):
        on_event("thinking")
        on_event("reasoning_delta", text="calc.py needs an add function.")
        # A repaint is throttled to one per tick, so give the view a tick to
        # catch up before looking at it -- and again once the call starts.
        time.sleep(0.1)
        thought.wait(2)
        on_event("tool_delta", name="write_file", args_text=args)
        time.sleep(0.1)
        on_event("tool_delta", name="write_file", args_text=args)
        wrote.wait(2)
        return "done"

    app = MiniAgentApp(
        stub_agent(ask),
        model="m",
        context=4096,
        endpoint="127.0.0.1:8080",
        prompt="write calc.py",
    )

    async def drive():
        async with app.run_test() as pilot:
            for gate in (thought, wrote):
                for _ in range(60):
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                    if widget_text(app.query_one("#stream", Static)).strip():
                        break
                seen.append(widget_text(app.query_one("#stream", Static)))
                gate.set()
                await pilot.pause()
                await asyncio.sleep(0.05)

    asyncio.run(drive())

    thinking, body = seen
    assert "thinking…" in thinking
    assert "calc.py needs an add function." in thinking
    # The call takes the view over, showing the file as it is being written.
    assert "write_file · content" in body
    assert "def add(a, b):" in body


def test_unstreamed_reasoning_still_reaches_the_log():
    """--no-stream reports the whole turn's thinking in one event."""
    log = run_app(
        events=[
            ("thinking", {}),
            ("reasoning", {"text": "calc.py only adds numbers."}),
            ("final", {"text": "It adds two numbers."}),
        ]
    )

    assert "Agent · Reasoning" in log
    assert "calc.py only adds numbers." in log
