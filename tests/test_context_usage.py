"""Tests for showing how full the model's context window is.

A small model's context is the resource a run actually runs out of, and it
does so silently: turns start being dropped and the only visible symptom is
the model forgetting something it read. So the number has to be on screen
while the run is going -- measured from the server's own token counts when it
reports them, and estimated from characters (marked as such) when it doesn't.
"""

import asyncio
import io
import threading
from types import SimpleNamespace

from rich.console import Console
from textual.widgets import Static

from agent import MiniAgent
from agent_logging import AgentLogger
from app_types import ContextUsage, ModelResponse
from plain import PlainView
from tui import MiniAgentApp
from utils import compact_tokens, format_context


# --- Formatting --------------------------------------------------------------


def test_a_measured_reading_is_shown_as_a_share_of_the_window():
    usage = ContextUsage(
        limit=32768, prompt_tokens=6200, completion_tokens=412, estimated=False
    )

    assert format_context(usage) == "ctx 6.2k/32k (19%)"


def test_an_estimated_reading_is_marked_as_a_guess():
    """A count derived from characters must not read as a measurement."""
    usage = ContextUsage(limit=32768, prompt_tokens=6200, estimated=True)

    assert format_context(usage) == "ctx ~6.2k/32k (19%)"


def test_without_a_known_window_only_what_was_sent_is_shown():
    """llama-server may report no n_ctx; a percentage of nothing is a lie."""
    usage = ContextUsage(limit=0, prompt_tokens=812, estimated=False)

    assert format_context(usage) == "ctx 812 tokens"


def test_a_session_that_has_sent_nothing_is_not_an_estimate():
    """The status bar is up before the first request; empty is exact."""
    assert format_context(ContextUsage(limit=32768)) == "ctx 0/32k (0%)"


def test_token_counts_stay_short_enough_for_a_status_line():
    assert compact_tokens(812) == "812"
    assert compact_tokens(6200) == "6.2k"
    # Floored, so a 32768-token window is the "32k" it is sold as.
    assert compact_tokens(32768) == "32k"
    assert compact_tokens(131072) == "131k"


# --- The agent's reading -----------------------------------------------------


def agent_stub(limit: int = 4096) -> MiniAgent:
    """A MiniAgent with only the fields the context reading touches."""
    agent = MiniAgent.__new__(MiniAgent)
    agent.context_limit = limit
    agent.schema_chars = 0
    agent.logger = AgentLogger(None, enabled=False)
    agent.context_usage = ContextUsage(limit=limit)
    return agent


def events(agent: MiniAgent, *args, **kwargs) -> list:
    seen: list = []
    agent._note_context(
        lambda event_type, **data: seen.append((event_type, data)), *args, **kwargs
    )
    return seen


def test_the_servers_own_token_counts_are_what_gets_shown():
    agent = agent_stub()
    response = ModelResponse(
        content="done",
        tool_calls=[],
        usage={"prompt_tokens": 1024, "completion_tokens": 64},
    )

    [(event_type, data)] = events(
        agent, [{"role": "user", "content": "x" * 9000}], response
    )
    usage = data["usage"]

    assert event_type == "context"
    assert data["phase"] == "response"
    assert (usage.prompt_tokens, usage.completion_tokens) == (1024, 64)
    assert usage.estimated is False
    # The characters that went out say something quite different; the point of
    # asking the server is not to have to guess.
    assert usage.percent == 25.0


def test_a_backend_that_reports_no_usage_falls_back_to_an_estimate():
    agent = agent_stub()
    response = ModelResponse(content="x" * 300, tool_calls=[], usage=None)

    [(_, data)] = events(agent, [{"role": "user", "content": "y" * 3000}], response)
    usage = data["usage"]

    assert usage.estimated is True
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0


def test_the_prompt_is_reported_before_the_model_answers():
    """A UI needs something to show while the turn is still being generated."""
    agent = agent_stub()

    [(_, data)] = events(agent, [{"role": "user", "content": "y" * 3000}])

    assert data["phase"] == "prompt"
    assert data["usage"].estimated is True
    assert data["usage"].completion_tokens == 0


def test_the_latest_reading_is_kept_for_a_front_end_to_read():
    agent = agent_stub()
    response = ModelResponse(content="", tool_calls=[], usage={"prompt_tokens": 2048})

    events(agent, [{"role": "user", "content": "hi"}], response)

    assert agent.context_usage.prompt_tokens == 2048


# --- Plain mode --------------------------------------------------------------


def plain_view() -> PlainView:
    console = Console(
        file=io.StringIO(), width=100, soft_wrap=True, highlight=False, no_color=True
    )
    agent = SimpleNamespace(logger=SimpleNamespace(log=lambda *a, **k: None))
    return PlainView(agent, console=console)


def test_the_transcript_reports_the_window_once_per_turn():
    plain = plain_view()
    usage = ContextUsage(
        limit=32768, prompt_tokens=6200, completion_tokens=412, estimated=False
    )

    plain.event("thinking")
    plain.event(
        "context", usage=ContextUsage(limit=32768, prompt_tokens=6100), phase="prompt"
    )
    plain.event("assistant_delta", text="calc.py adds two numbers.")
    plain.event("context", usage=usage, phase="response")
    plain.event("final", text="calc.py adds two numbers.")

    printed = plain.console.file.getvalue()
    # The estimate sent with the prompt is for a live UI, not for a transcript
    # that would then say the same thing twice per step.
    assert printed.count("ctx ") == 1
    assert "ctx 6.2k/32k (19%) · 412 out" in printed


# --- TUI ---------------------------------------------------------------------


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


def test_the_status_bar_follows_the_window_through_a_run():
    done = threading.Event()

    def ask(text, on_event=None):
        on_event("thinking")
        on_event(
            "context",
            usage=ContextUsage(limit=4096, prompt_tokens=1024, estimated=False),
            phase="response",
        )
        on_event("final", text="done")
        done.set()
        return "done"

    app = MiniAgentApp(
        stub_agent(ask), model="m", context=4096, endpoint="127.0.0.1:8080", prompt="hi"
    )
    seen: list = []

    async def drive():
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if done.is_set():
                    break
            await pilot.pause()
            status = app.query_one("#status", Static)
            seen.append(
                "\n".join(
                    strip.text
                    for strip in status.render_lines(status.region.reset_offset)
                )
            )

    asyncio.run(drive())

    assert "ctx 1.0k/4.1k (25%)" in seen[0]


def test_the_window_is_still_shown_while_the_agent_is_working():
    """Idle is the moment it matters least; a long run is when it fills up."""
    app = MiniAgentApp(
        stub_agent(lambda text, on_event=None: "done"),
        model="m",
        context=4096,
        endpoint="127.0.0.1:8080",
    )
    app._usage = ContextUsage(limit=4096, prompt_tokens=2048, estimated=False)
    app._busy = True

    assert "ctx 2.0k/4.1k (50%)" in app._status_text()
