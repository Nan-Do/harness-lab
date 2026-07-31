"""Tests for a prompt handed to the TUI on the command line.

`--mode tui "do the thing"` used to drop the prompt on the floor: the TUI came
up empty and the request was never made. It should start the session with it
and then stay open for follow-ups, which is the difference between the TUI and
headless mode.
"""

import asyncio
import threading
from types import SimpleNamespace

from tui import MiniAgentApp


def stub_agent(asked: list) -> SimpleNamespace:
    done = threading.Event()

    def ask(text, on_event=None):
        asked.append(text)
        done.set()
        return "done"

    return SimpleNamespace(
        ask=ask,
        asked_once=done,
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


def run_app(prompt: str, wait: bool = True) -> list:
    """Start the TUI with `prompt` and return the requests the agent saw."""
    asked: list = []
    agent = stub_agent(asked)
    app = MiniAgentApp(
        agent, model="m", context=4096, endpoint="127.0.0.1:8080", prompt=prompt
    )

    async def drive():
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause()
                await asyncio.sleep(0.02)
                if agent.asked_once.is_set() or not wait:
                    break

    asyncio.run(drive())
    return asked


def test_command_line_prompt_is_run_at_startup():
    assert run_app("read calc.py") == ["read calc.py"]


def test_no_prompt_leaves_the_tui_idle():
    assert run_app("", wait=False) == []


def test_blank_prompt_is_not_treated_as_a_request():
    assert run_app("   ", wait=False) == []


def test_command_line_slash_command_is_handled_as_a_command():
    """`--mode tui /memory` is a command, not something to ask the model."""
    assert run_app("/memory", wait=False) == []
