"""Tests for the HumanEval benchmark scripts.

A benchmark that is wrong is worse than no benchmark: it produces a number that
looks like an answer. So what is pinned here is everything that could quietly
turn a good run into a bad score -- how a candidate is executed and classified,
how the agent's workspace is isolated, what the harness is actually asked to
do, and the arithmetic that turns records into a percentage. No model and no
llama-server are involved: the harness itself is replaced by a one-line script
that behaves the way a real run would.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.humaneval import dataset, report, runner
from benchmarks.humaneval.execution import (
    build_program,
    check_candidate,
    estimate_pass_at_k,
)
from benchmarks.humaneval.runner import RunConfig

PROMPT = 'def add(a, b):\n    """Add two numbers."""\n'
TEST = "def check(candidate):\n    assert candidate(1, 2) == 3\n"


def problem(**overrides) -> dataset.Problem:
    fields = dict(
        task_id="HumanEval/0",
        prompt=PROMPT,
        entry_point="add",
        test=TEST,
        canonical_solution="    return a + b\n",
    )
    fields.update(overrides)
    return dataset.Problem(**fields)


# --- Executing a candidate ---


def test_a_correct_solution_passes():
    result = check_candidate(PROMPT + "    return a + b\n", "add", TEST)
    assert result.passed and result.status == "passed"


def test_a_wrong_answer_is_told_apart_from_code_that_raises():
    wrong = check_candidate(PROMPT + "    return a - b\n", "add", TEST)
    raised = check_candidate("def add(a, b):\n    raise ValueError('nope')\n", "add", TEST)

    assert (wrong.status, raised.status) == ("failed", "error")
    assert "AssertionError" in wrong.detail


def test_a_missing_entry_point_fails_by_name_rather_than_silently():
    result = check_candidate("def other(a, b):\n    return a + b\n", "add", TEST)
    assert result.status == "error"
    assert "add" in result.detail


def test_an_endless_solution_is_killed_and_reported_as_a_timeout():
    result = check_candidate(
        "def add(a, b):\n    while True:\n        pass\n", "add", TEST, timeout=2
    )
    assert result.status == "timeout"
    assert result.duration_s < 15  # killed, not waited out


def test_nothing_to_score_is_its_own_answer():
    assert check_candidate(None, "add", TEST).status == "no_solution"
    assert check_candidate("   \n", "add", TEST).status == "no_solution"


def test_a_demo_block_does_not_run_while_the_tests_do():
    """Small models add `if __name__ == "__main__":` blocks; importing ignores them."""
    candidate = (
        PROMPT
        + "    return a + b\n\n\n"
        + 'if __name__ == "__main__":\n    raise SystemExit("demo ran")\n'
    )
    assert check_candidate(candidate, "add", TEST).passed


def test_helpers_defined_beside_the_entry_point_are_in_the_tests_scope():
    """Some problems' tests call a helper from the prompt, not just the entry point."""
    test = "def check(candidate):\n    assert helper() == 7\n    assert candidate(1, 2) == 3\n"
    candidate = "def helper():\n    return 7\n\n\ndef add(a, b):\n    return a + b\n"

    assert "from solution import *" in build_program("add", test)
    assert check_candidate(candidate, "add", test).passed


def test_pass_at_k_matches_the_paper():
    assert estimate_pass_at_k(1, 1, 1) == 1.0
    assert estimate_pass_at_k(1, 0, 1) == 0.0
    assert estimate_pass_at_k(5, 1, 1) == pytest.approx(0.2)
    # One success in five samples is certain to appear in five draws.
    assert estimate_pass_at_k(5, 1, 5) == 1.0


# --- The workspace an agent is given ---


def test_the_problem_workspace_is_its_own_git_repository(tmp_path):
    """The harness takes `git rev-parse --show-toplevel` as its root, and an
    auto-approved agent can write anywhere inside it."""
    workspace = runner.prepare_workspace(tmp_path / "HumanEval_0", problem())

    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=workspace,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert Path(root).resolve() == workspace.resolve()


def test_the_seeded_solution_is_the_prompt_verbatim_and_importable(tmp_path):
    workspace = runner.prepare_workspace(tmp_path / "ws", problem())
    seeded = (workspace / "solution.py").read_text(encoding="utf-8")

    assert seeded == PROMPT
    # A docstring is a body, so an agent that does nothing leaves a file that
    # imports and returns None -- a wrong answer, not a crash.
    compile(seeded, "solution.py", "exec")


def test_a_rerun_starts_from_a_clean_workspace(tmp_path):
    workspace = tmp_path / "ws"
    runner.prepare_workspace(workspace, problem())
    (workspace / "leftover.py").write_text("stale\n", encoding="utf-8")

    runner.prepare_workspace(workspace, problem())

    assert not (workspace / "leftover.py").exists()


# --- What the harness is asked to do ---


def test_the_harness_is_run_headless_and_unattended(tmp_path):
    command = runner.harness_command(
        RunConfig(label="x"), tmp_path, "implement the function"
    )
    pairs = dict(zip(command, command[1:]))

    assert pairs["--mode"] == "headless"
    # Nobody is there to approve a write or to answer the step-limit question.
    assert pairs["--approval"] == "auto"
    assert pairs["--on-step-limit"] == "stop"
    assert pairs["--cwd"] == str(tmp_path)
    assert command[-1] == "implement the function"


def test_language_and_extra_arguments_reach_the_harness(tmp_path):
    command = runner.harness_command(
        RunConfig(label="x", language="python", extra_args=("--no-stream",)),
        tmp_path,
        "go",
    )
    assert "--language" in command and "python" in command
    # Passed through after the defaults, so an explicit flag wins over them.
    assert command.index("--no-stream") > command.index("--max-steps")


def test_the_request_names_the_function_and_forbids_the_usual_wreckage():
    request = runner.REQUEST.format(entry_point="add")
    assert "`add`" in request
    assert "input()" in request and "signature" in request


# --- One problem, end to end, with the harness stubbed out ---


def fake_harness(monkeypatch, script: str) -> None:
    """Replace the agent run with a script that acts like one outcome of it."""

    def harness_command(config, workspace, request):
        return [sys.executable, "-c", script.replace("@WS@", str(workspace))]

    monkeypatch.setattr(runner, "harness_command", harness_command)


WRITE = (
    "from pathlib import Path;"
    "Path(r'@WS@/solution.py').write_text("
    "'def add(a, b):\\n    return %s\\n')"
)


def test_a_solved_problem_is_scored_from_the_file_on_disk(tmp_path, monkeypatch):
    fake_harness(monkeypatch, WRITE % "a + b")

    record = runner.run_problem(problem(), RunConfig(label="x"), tmp_path / "ws")

    assert record["passed"] is True
    assert record["status"] == "passed"
    assert record["harness_status"] == "ok"


def test_a_wrong_solution_is_a_failure_not_an_error(tmp_path, monkeypatch):
    fake_harness(monkeypatch, WRITE % "a - b")

    record = runner.run_problem(problem(), RunConfig(label="x"), tmp_path / "ws")

    assert record["passed"] is False
    assert record["status"] == "failed"


def test_an_agent_that_wrote_nothing_is_reported_as_untouched(tmp_path, monkeypatch):
    fake_harness(monkeypatch, "pass")

    record = runner.run_problem(problem(), RunConfig(label="x"), tmp_path / "ws")

    assert record["status"] == "untouched"
    assert record["harness_status"] == "ok"


def test_a_dead_harness_is_not_blamed_on_the_model(tmp_path, monkeypatch):
    """`untouched` means the agent did nothing; this is the harness dying."""
    fake_harness(monkeypatch, "import sys; sys.stderr.write('boom'); sys.exit(1)")

    record = runner.run_problem(problem(), RunConfig(label="x"), tmp_path / "ws")

    assert record["status"] == "harness_crashed"
    assert "boom" in record["error"]


def test_a_harness_that_never_returns_is_killed(tmp_path, monkeypatch):
    fake_harness(monkeypatch, "import time; time.sleep(60)")

    record = runner.run_problem(
        problem(), RunConfig(label="x", harness_timeout=2), tmp_path / "ws"
    )

    assert record["status"] == "harness_timeout"


# --- Reading the harness's own log ---


def write_log(directory: Path, events: list) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "run-20260820-120000.jsonl"
    log.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    return log


def test_log_stats_carry_what_a_comparison_needs(tmp_path):
    write_log(
        tmp_path,
        [
            {"event": "tool_call", "name": "read_file"},
            {"event": "tool_outcome", "name": "read_file", "outcome": "ok"},
            {"event": "tool_call", "name": "write_file"},
            {"event": "tool_outcome", "name": "write_file", "outcome": "arg_error"},
            {"event": "retry", "attempt": 1},
            {"event": "malformed_tool_call", "name": "write_file"},
            {"event": "context_usage", "prompt_tokens": 4200, "percent": 12.8},
            {"event": "final", "reason": "answer", "tool_steps": 3, "attempts": 4},
        ],
    )

    stats = runner.read_log_stats(tmp_path)

    assert stats["steps"] == 3
    assert stats["final_reason"] == "answer"
    assert stats["tool_calls"] == 2
    assert stats["tool_outcomes"] == {"ok": 1, "arg_error": 1}
    assert (stats["retries"], stats["malformed"]) == (1, 1)
    assert stats["prompt_tokens"] == 4200


def test_a_missing_log_is_empty_stats_rather_than_a_crash(tmp_path):
    assert runner.read_log_stats(tmp_path / "nope")["steps"] == 0


# --- Suites ---


def stub_run_problem(monkeypatch, outcomes) -> list:
    """Drive run_suite with canned per-problem records."""
    seen = []

    def run_problem(problem_, config, workspace, sample=0):
        record = dict(outcomes[len(seen) % len(outcomes)])
        record.update({"task_id": problem_.task_id, "sample": sample})
        seen.append(record)
        return record

    monkeypatch.setattr(runner, "run_problem", run_problem)
    return seen


def problems(count: int) -> list:
    return [problem(task_id=f"HumanEval/{index}") for index in range(count)]


def test_a_suite_that_is_failing_identically_stops_early(tmp_path, monkeypatch):
    """A server the harness cannot talk to fails all 164 the same way."""
    stub_run_problem(
        monkeypatch,
        [{"passed": False, "harness_status": "crashed", "error": "HTTP 400 template"}],
    )

    outcome = runner.run_suite(
        problems(40), RunConfig(label="x"), tmp_path, jobs=1, abort_after=5
    )

    assert outcome.aborted
    assert "400" in outcome.abort_reason
    assert len(outcome.records) < 40


def test_ordinary_failures_do_not_trip_the_abort(tmp_path, monkeypatch):
    stub_run_problem(monkeypatch, [{"passed": False, "harness_status": "ok"}])

    outcome = runner.run_suite(
        problems(8), RunConfig(label="x"), tmp_path, jobs=1, abort_after=5
    )

    assert not outcome.aborted
    assert len(outcome.records) == 8


def test_results_are_written_as_they_land_and_can_be_resumed(tmp_path, monkeypatch):
    stub_run_problem(monkeypatch, [{"passed": True, "harness_status": "ok"}])
    config = RunConfig(label="x")

    runner.run_suite(problems(3), config, tmp_path, jobs=1)
    on_disk = runner.load_records(tmp_path / "results.jsonl")

    seen = stub_run_problem(monkeypatch, [{"passed": True, "harness_status": "ok"}])
    outcome = runner.run_suite(problems(3), config, tmp_path, jobs=1, resume=True)

    assert len(on_disk) == 3
    assert seen == []  # nothing was run a second time
    assert len(outcome.records) == 3


# --- The score ---


def records(*statuses) -> list:
    return [
        {
            "task_id": f"HumanEval/{index}",
            "passed": status == "passed",
            "status": status,
            "harness_status": "ok",
            "steps": 2,
            "duration_s": 1.0,
            "tool_calls": 2,
            "tool_outcomes": {"ok": 1, "arg_error": 1},
        }
        for index, status in enumerate(statuses)
    ]


def test_the_score_counts_problems_and_breaks_the_failures_down():
    summary = report.summarize(records("passed", "failed", "untouched", "passed"))

    assert summary.passed == 2
    assert summary.pass_at_1 == pytest.approx(50.0)
    assert summary.statuses["untouched"] == 1
    assert summary.tool_failures == 4

    block = report.format_summary(summary)
    assert "2 / 4 passed" in block and "(50.0%)" in block
    assert "untouched (agent never edited)" in block


def test_several_samples_score_the_problem_once_and_add_pass_at_k():
    attempts = [
        {"task_id": "HumanEval/0", "passed": True, "status": "passed", "sample": 0},
        {"task_id": "HumanEval/0", "passed": False, "status": "failed", "sample": 1},
    ]

    summary = report.summarize(attempts)

    assert summary.problems == 1
    assert summary.samples == 2
    assert summary.pass_at_1 == pytest.approx(50.0)
    assert summary.pass_at_k["pass@2"] == pytest.approx(100.0)


def test_an_empty_run_reports_zero_rather_than_dividing_by_it():
    summary = report.summarize([])
    assert summary.pass_at_1 == 0.0
    assert "0 / 0 passed" in report.format_summary(summary)


def test_the_comparison_is_ordered_best_first():
    better = report.summarize(records("passed", "passed"), label="better")
    worse = report.summarize(records("passed", "failed"), label="worse")

    table = report.format_comparison([worse, better])

    assert table.index("better") < table.index("worse")


# --- Problem selection ---


def test_problems_can_be_named_with_or_without_the_prefix():
    assert dataset.normalize_ids(["0", "HumanEval/2", "3,4"]) == [
        "HumanEval/0",
        "HumanEval/2",
        "HumanEval/3",
        "HumanEval/4",
    ]


def test_naming_problems_beats_a_leftover_limit():
    chosen = dataset.select(problems(10), only=["7", "2"], limit=1)
    assert [item.task_id for item in chosen] == ["HumanEval/7", "HumanEval/2"]


def test_a_typo_in_a_problem_id_is_refused_rather_than_ignored():
    with pytest.raises(KeyError):
        dataset.select(problems(3), only=["HumanEval/999"])


def test_limit_and_offset_slice_the_set():
    chosen = dataset.select(problems(10), limit=3, offset=5)
    assert [item.number for item in chosen] == [5, 6, 7]
