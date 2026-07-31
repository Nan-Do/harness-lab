"""Tests for how the agent decides what the model still gets to see.

Content is sent whole or dropped whole -- never truncated -- so the questions
worth testing are which reads have gone out of date and whether the model is
told when something it relied on has left the context.
"""

import json

from agent import MiniAgent
from agent_logging import AgentLogger
from app_types import MessageEntry, ToolMessageEntry


def tool_entry(name: str, content: str = "ok", **args) -> ToolMessageEntry:
    return ToolMessageEntry(
        role="tool", name=name, args=args, content=content, created_at="now"
    )


def message(role: str = "user", content: str = "hello") -> MessageEntry:
    return MessageEntry(role=role, content=content, created_at="now")


def agent_stub(history_budget: int) -> MiniAgent:
    """A MiniAgent with only the fields the context helpers touch."""
    agent = MiniAgent.__new__(MiniAgent)
    agent.history_budget = history_budget
    agent.logger = AgentLogger(None, enabled=False)
    return agent


def tool_group(name: str, chars: int = 10, **args) -> list:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * chars},
    ]


# --- Stale reads ---


def test_a_later_whole_read_supersedes_an_earlier_one():
    history = [tool_entry("read_file", path="a.py"), tool_entry("read_file", path="a.py")]
    assert MiniAgent._stale_read_indices(history) == {0}


def test_reads_of_different_files_do_not_supersede_each_other():
    history = [tool_entry("read_file", path="a.py"), tool_entry("read_file", path="b.py")]
    assert MiniAgent._stale_read_indices(history) == set()


def test_consecutive_ranges_of_a_big_file_are_both_kept():
    # Working through a file too large to hold at once is the intended
    # workflow; neither half may be dropped as redundant.
    history = [
        tool_entry("read_file_range", path="big.py", start=1, end=200),
        tool_entry("read_file_range", path="big.py", start=201, end=400),
    ]
    assert MiniAgent._stale_read_indices(history) == set()


def test_a_range_contained_in_a_later_range_is_stale():
    history = [
        tool_entry("read_file_range", path="big.py", start=10, end=20),
        tool_entry("read_file_range", path="big.py", start=1, end=100),
    ]
    assert MiniAgent._stale_read_indices(history) == {0}


def test_a_whole_read_supersedes_earlier_ranges():
    history = [
        tool_entry("read_file_range", path="big.py", start=10, end=20),
        tool_entry("read_file", path="big.py"),
    ]
    assert MiniAgent._stale_read_indices(history) == {0}


def test_a_range_does_not_supersede_an_earlier_whole_read():
    history = [
        tool_entry("read_file", path="big.py"),
        tool_entry("read_file_range", path="big.py", start=1, end=10),
    ]
    assert MiniAgent._stale_read_indices(history) == set()


def test_write_file_supersedes_an_earlier_read_of_that_file():
    history = [
        tool_entry("read_file", path="a.py"),
        tool_entry("write_file", "wrote a.py", path="a.py"),
    ]
    assert MiniAgent._stale_read_indices(history) == {0}


def test_a_partial_edit_keeps_the_earlier_read():
    # patch_file leaves the read only partly out of date; dropping it would
    # leave the model with no view of the file at all.
    history = [
        tool_entry("read_file", path="a.py"),
        tool_entry("patch_file", "patched a.py", path="a.py"),
    ]
    assert MiniAgent._stale_read_indices(history) == set()


def test_failed_reads_are_not_treated_as_content():
    history = [
        tool_entry("read_file", "error: file not found: a.py", path="a.py"),
        tool_entry("read_file", path="a.py"),
    ]
    assert MiniAgent._stale_read_indices(history) == set()


def test_equivalent_path_spellings_compare_equal():
    history = [
        tool_entry("read_file", path="./a.py"),
        tool_entry("read_file", path="a.py"),
    ]
    assert MiniAgent._stale_read_indices(history) == {0}


# --- Eviction notice ---


def test_nothing_is_dropped_when_everything_fits():
    agent = agent_stub(history_budget=10_000)
    groups = [tool_group("read_file", path="a.py"), [{"role": "user", "content": "hi"}]]
    assert agent._fit_history_budget(list(groups)) == groups


def test_dropped_turns_are_announced_by_name():
    agent = agent_stub(history_budget=200)
    groups = [
        tool_group("read_file", chars=500, path="a.py"),
        [{"role": "user", "content": "current request"}],
    ]
    fitted = agent._fit_history_budget(groups)

    notice = fitted[0][0]
    assert notice["role"] == "user"
    assert "context notice" in notice["content"]
    assert "read_file a.py" in notice["content"]
    # The current request always survives.
    assert fitted[-1][0]["content"] == "current request"


def test_the_last_turn_is_never_dropped_even_when_oversized():
    agent = agent_stub(history_budget=10)
    groups = [[{"role": "user", "content": "x" * 5000}]]
    assert agent._fit_history_budget(list(groups)) == groups


def test_every_dropped_tool_is_named_once():
    agent = agent_stub(history_budget=100)
    groups = [
        tool_group("read_file", chars=300, path="a.py"),
        tool_group("search", chars=300, pattern="foo"),
        tool_group("read_file", chars=300, path="a.py"),
        [{"role": "user", "content": "now"}],
    ]
    content = agent._fit_history_budget(groups)[0][0]["content"]
    assert content.count("read_file a.py") == 1
    assert "search" in content


def test_notice_falls_back_when_only_messages_were_dropped():
    agent = agent_stub(history_budget=50)
    groups = [
        [{"role": "user", "content": "x" * 300}],
        [{"role": "user", "content": "now"}],
    ]
    content = agent._fit_history_budget(groups)[0][0]["content"]
    assert "earlier conversation" in content
