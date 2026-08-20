"""Tests for what a request does when it runs out of tool steps.

The step limit is the harness's answer to a small model going in circles, so
raising it is never something the loop decides on its own: either the run stops
where it always did, or somebody is asked. These tests pin down both halves --
that nothing continues unasked, and that an answered "keep going" actually
buys more steps (or removes the limit entirely).
"""

import asyncio
import io
import threading
from types import SimpleNamespace

from rich.console import Console

from agent import MAX_STUCK_ATTEMPTS, NO_STEP_LIMIT, MiniAgent
from app_types import ModelResponse, ToolCall
from plain import PlainView
from session import SessionStore
from tui import MiniAgentApp
from workspace import WorkspaceContext


class StubModel:
    """A model whose every turn is scripted by `respond(turn_number)`."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.calls = 0
        self.model = "stub"
        self.ctx = 8192

    def complete(self, messages, max_new_tokens, tools=None, **sinks) -> ModelResponse:
        self.calls += 1
        return self.respond(self.calls)


def tool_turn(number: int) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=[
            ToolCall(id=f"c{number}", name="read_file", args={"path": "calc.py"})
        ],
    )


def answer_turn(text: str = "calc.py sets value to 1") -> ModelResponse:
    return ModelResponse(content=text, tool_calls=[])


def build_agent(tmp_path, respond, **options) -> MiniAgent:
    """A real agent over a throwaway workspace, driven by a scripted model."""
    (tmp_path / "calc.py").write_text("value = 1\n", encoding="utf-8")
    return MiniAgent(
        model_client=StubModel(respond),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / "sessions"),
        approval_policy="never",
        **options,
    )


def tools_until(count: int, text: str = "calc.py sets value to 1"):
    """Call a tool `count` times, then answer."""
    return lambda turn: tool_turn(turn) if turn <= count else answer_turn(text)


# --- Stopping ---


def test_a_run_stops_at_the_step_limit(tmp_path):
    agent = build_agent(tmp_path, tools_until(10), max_steps=2, on_step_limit="stop")

    final = agent.ask("read calc.py")

    assert agent.model_client.calls == 2
    assert "step limit (2 tool steps)" in final


def test_the_limit_is_not_raised_when_there_is_nobody_to_ask(tmp_path):
    """`--on-step-limit ask` in a headless run has no one to answer it."""
    agent = build_agent(tmp_path, tools_until(10), max_steps=2, on_step_limit="ask")

    final = agent.ask("read calc.py")

    assert agent.model_client.calls == 2
    assert "step limit" in final


def test_a_declined_question_stops_the_run(tmp_path):
    asked = []
    agent = build_agent(
        tmp_path,
        tools_until(10),
        max_steps=2,
        on_step_limit="ask",
        step_limit_fn=lambda steps, limit: asked.append((steps, limit)),
    )

    final = agent.ask("read calc.py")

    # `list.append` returns None: the front-end said stop.
    assert asked == [(2, 2)]
    assert "step limit" in final


def test_a_front_end_that_fails_to_ask_stops_the_run(tmp_path):
    def explode(steps, limit):
        raise RuntimeError("no UI to ask with")

    agent = build_agent(
        tmp_path,
        tools_until(10),
        max_steps=2,
        on_step_limit="ask",
        step_limit_fn=explode,
    )

    assert "step limit" in agent.ask("read calc.py")
    assert agent.model_client.calls == 2


def test_a_step_limit_policy_of_stop_never_asks(tmp_path):
    asked = []
    agent = build_agent(
        tmp_path,
        tools_until(10),
        max_steps=2,
        on_step_limit="stop",
        step_limit_fn=lambda steps, limit: asked.append(1) or limit,
    )

    agent.ask("read calc.py")

    assert asked == []


# --- Continuing ---


def test_granted_steps_continue_the_run(tmp_path):
    asked = []

    def grant(steps, limit):
        asked.append((steps, limit))
        return 2

    agent = build_agent(
        tmp_path,
        tools_until(3),
        max_steps=2,
        on_step_limit="ask",
        step_limit_fn=grant,
    )

    final = agent.ask("read calc.py")

    assert final == "calc.py sets value to 1"
    # Asked once at step 2; the four-step budget then covered the rest.
    assert asked == [(2, 2)]


def test_the_question_comes_back_when_the_granted_steps_run_out(tmp_path):
    asked = []

    def grant(steps, limit):
        asked.append((steps, limit))
        return 2

    agent = build_agent(
        tmp_path,
        tools_until(10),
        max_steps=2,
        on_step_limit="ask",
        step_limit_fn=grant,
    )

    agent.ask("read calc.py")

    # Each grant is spent before the next question, and the limit reported is
    # the one in force at the time.
    assert asked[:3] == [(2, 2), (4, 4), (6, 6)]


def test_dropping_the_limit_lets_the_run_finish(tmp_path):
    asked = []

    def forever(steps, limit):
        asked.append((steps, limit))
        return NO_STEP_LIMIT

    agent = build_agent(
        tmp_path,
        tools_until(7),
        max_steps=2,
        on_step_limit="ask",
        step_limit_fn=forever,
    )

    final = agent.ask("read calc.py")

    assert final == "calc.py sets value to 1"
    # Asked once: an unlimited run never comes back with the question.
    assert len(asked) == 1


def test_a_run_started_without_a_limit_is_never_asked(tmp_path):
    asked = []
    agent = build_agent(
        tmp_path,
        tools_until(9),
        max_steps=0,
        on_step_limit="ask",
        step_limit_fn=lambda steps, limit: asked.append(1) or limit,
    )

    assert agent.ask("read calc.py") == "calc.py sets value to 1"
    assert asked == []


def test_a_negative_limit_is_read_as_no_limit(tmp_path):
    agent = build_agent(tmp_path, tools_until(9), max_steps=-1)

    assert agent.max_steps == NO_STEP_LIMIT
    assert agent.ask("read calc.py") == "calc.py sets value to 1"


# --- Guards on an unlimited run ---


def test_an_unlimited_run_gives_up_on_a_model_saying_nothing(tmp_path):
    """No step budget still leaves a stuck model somewhere to stop."""
    agent = build_agent(tmp_path, lambda turn: answer_turn(""), max_steps=0)

    final = agent.ask("read calc.py")

    assert agent.model_client.calls == MAX_STUCK_ATTEMPTS
    assert "malformed model responses" in final


def test_useful_turns_clear_the_stuck_count(tmp_path):
    """A model that stumbles between real calls is not treated as stuck."""

    def stumble(turn: int) -> ModelResponse:
        if turn > 3 * MAX_STUCK_ATTEMPTS:
            return answer_turn()
        return answer_turn("") if turn % 2 else tool_turn(turn)

    agent = build_agent(tmp_path, stumble, max_steps=0)

    assert agent.ask("read calc.py") == "calc.py sets value to 1"


def test_stop_ends_a_running_request(tmp_path):
    """The escape hatch for a run with no step budget."""
    agent = build_agent(tmp_path, tools_until(50), max_steps=0)

    def respond(turn: int) -> ModelResponse:
        if turn == 3:
            agent.stop()
        return tool_turn(turn)

    agent.model_client.respond = respond
    final = agent.ask("read calc.py")

    assert agent.model_client.calls == 3
    assert "at your request" in final


def test_stopping_an_interrupted_run_does_not_carry_into_the_next(tmp_path):
    """Each request starts uninterrupted, whenever the flag was last set."""
    agent = build_agent(tmp_path, tools_until(50), max_steps=0)

    def respond(turn: int) -> ModelResponse:
        if turn == 2:
            agent.stop()
        return tool_turn(turn)

    agent.model_client.respond = respond
    assert "at your request" in agent.ask("read calc.py")

    agent.model_client.respond = tools_until(1)
    assert agent.ask("read calc.py again") == "calc.py sets value to 1"


# --- Front-ends ---


def plain_view(agent=None) -> PlainView:
    console = Console(
        file=io.StringIO(), width=100, soft_wrap=True, highlight=False, no_color=True
    )
    agent = agent or SimpleNamespace(logger=SimpleNamespace(log=lambda *a, **k: None))
    return PlainView(agent, console=console)


def test_plain_mode_reads_the_answer_from_stdin(monkeypatch):
    view = plain_view()

    monkeypatch.setattr("builtins.input", lambda: "u")
    assert view.step_limit(4, 4) == NO_STEP_LIMIT

    monkeypatch.setattr("builtins.input", lambda: "c")
    assert view.step_limit(4, 4) == 4

    monkeypatch.setattr("builtins.input", lambda: "")
    assert view.step_limit(4, 4) == 4

    monkeypatch.setattr("builtins.input", lambda: "s")
    assert view.step_limit(4, 4) is None


def test_plain_mode_stops_when_the_question_cannot_be_answered(monkeypatch):
    def closed():
        raise EOFError

    view = plain_view()
    monkeypatch.setattr("builtins.input", closed)

    assert view.step_limit(4, 4) is None


def test_plain_mode_asks_again_after_an_answer_it_cannot_read(monkeypatch):
    answers = iter(["maybe", "s"])
    view = plain_view()
    monkeypatch.setattr("builtins.input", lambda: next(answers))

    assert view.step_limit(4, 4) is None
    assert next(answers, "used up") == "used up"


def tui_agent(run) -> SimpleNamespace:
    return SimpleNamespace(
        ask=run,
        approval_policy="never",
        max_steps=2,
        on_step_limit="ask",
        step_limit_fn=None,
        stop=lambda: None,
        tools=SimpleNamespace(approval_fn=None),
        workspace=SimpleNamespace(branch="main", cwd="/tmp/x"),
        language=None,
        session=SimpleNamespace(id="test-session"),
        logger=SimpleNamespace(log=lambda *a, **k: None),
        memory_text=lambda: "Memory:",
        session_path="/tmp/x/session.json",
        log_path="/tmp/x/log.jsonl",
        reset=lambda: None,
    )


def run_tui_dialog(key: str):
    """Reach the step limit inside the TUI and answer its dialog with `key`."""
    answered: list = []

    def ask(text, on_event=None):
        # What MiniAgent.ask does when the budget runs out, from the same
        # worker thread the real one runs on.
        answered.append(agent.step_limit_fn(2, 2))
        return "done"

    agent = tui_agent(ask)
    app = MiniAgentApp(agent, model="m", context=4096, endpoint="x", prompt="go")

    async def drive():
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if app.screen_stack[-1].__class__.__name__ == "StepLimitScreen":
                    break
            await pilot.press(key)
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if answered:
                    break

    asyncio.run(drive())
    return answered


def test_the_tui_dialog_grants_another_budget():
    assert run_tui_dialog("c") == [2]


def test_the_tui_dialog_can_drop_the_limit():
    assert run_tui_dialog("u") == [NO_STEP_LIMIT]


def test_the_tui_dialog_can_stop_the_run():
    assert run_tui_dialog("s") == [None]


def press_while_running(key: str) -> bool:
    """Press `key` during a request and report whether the agent was stopped."""
    running = threading.Event()
    stopped = threading.Event()

    def ask(text, on_event=None):
        running.set()
        stopped.wait(2)
        return "done"

    agent = tui_agent(ask)
    agent.stop = stopped.set
    app = MiniAgentApp(agent, model="m", context=4096, endpoint="x", prompt="go")

    async def drive():
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if running.is_set():
                    break
            await pilot.press(key)
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if stopped.is_set():
                    break

    asyncio.run(drive())
    return stopped.is_set()


def test_the_interrupt_key_reaches_a_running_request():
    """The prompt Input owns ctrl+x and ctrl+c, so the key has to dodge both."""
    assert press_while_running("ctrl+g")


def test_escape_also_interrupts():
    assert press_while_running("escape")
