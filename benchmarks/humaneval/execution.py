"""Scoring one candidate: run the problem's own tests against it, contained.

HumanEval is scored by executing code a language model wrote, which is the one
step of this benchmark that can do damage. The containment here is deliberate
but modest, and worth stating plainly: a fresh interpreter per candidate, in a
throwaway directory, with a wall-clock timeout, a process group that is killed
as a unit, CPU/memory/file-size limits, no stdin, and an environment stripped
to almost nothing. That stops the accidents this benchmark actually produces --
runaway loops, memory bombs, a `while True` waiting on input -- and it is not a
sandbox against code that is trying to get out. Run a benchmark you do not
trust in a container or a VM.

The candidate is *imported* rather than concatenated into the test program, for
two reasons: `if __name__ == "__main__":` demo blocks (which small models love
to add) stay dormant under import, and the tests then exercise the module
exactly the way the agent's own tests would.
"""

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Enough headroom for the slowest legitimate solutions, and far short of what a
# runaway one needs to hurt the machine running the suite.
DEFAULT_TIMEOUT = 10.0
DEFAULT_MEMORY_MB = 1024
DEFAULT_MAX_FILE_MB = 16

# Applied inside the child instead of through preexec_fn: problems are scored
# from several threads at once, and a fork hook in a threaded parent is a
# documented way to deadlock.
_PRELUDE = """\
import resource
import sys


def _limit(name, soft):
    try:
        which = getattr(resource, name)
    except AttributeError:
        return
    try:
        hard = resource.getrlimit(which)[1]
        resource.setrlimit(which, (soft if hard < 0 else min(soft, hard), hard))
    except (ValueError, OSError):
        pass


_limit("RLIMIT_CPU", %(cpu_seconds)d)
_limit("RLIMIT_AS", %(memory_bytes)d)
_limit("RLIMIT_FSIZE", %(file_bytes)d)
# Several HumanEval problems have naturally recursive solutions.
sys.setrecursionlimit(10000)

"""


@dataclass(frozen=True)
class ExecutionResult:
    """What running the problem's tests against one candidate said."""

    passed: bool
    # passed | failed (assertion) | error (raised) | timeout | no_solution
    status: str
    detail: str
    duration_s: float


def build_program(
    entry_point: str,
    test: str,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> str:
    """The driver that imports the candidate and runs the problem's checks.

    Built by concatenation rather than str.format: HumanEval tests are full of
    dict and set literals, and every one of their braces would be a format
    placeholder.
    """
    prelude = _PRELUDE % {
        # The CPU limit is the backstop for a process that ignores the wall
        # clock (a busy loop that never yields); it is deliberately looser than
        # the timeout, which is what normally ends a run.
        "cpu_seconds": max(int(timeout) + 5, 5),
        "memory_bytes": memory_mb * 1024 * 1024,
        "file_bytes": max_file_mb * 1024 * 1024,
    }
    return (
        prelude
        # Star-import first: a handful of problems (32, 38, 50) put helper
        # functions in the prompt and then call them from the test, so the
        # tests' scope is the whole module, not just the entry point. The
        # explicit import after it is what fails loudly, and by name, when the
        # candidate never defined the function it was asked for.
        + "from solution import *  # noqa: F403\n"
        + f"from solution import {entry_point}  # noqa: F401\n\n"
        + test
        + f"\n\ncheck({entry_point})\n"
    )


def _child_env(work_dir: Path) -> dict:
    """A near-empty environment, with HOME and TMPDIR pointed at the scratch."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
    }


def _describe_failure(stderr: str) -> str:
    """The one line of a traceback worth carrying into the results file."""
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    # The last line is the exception; the line above it is usually the failing
    # assertion, which is the more useful half of "why did this fail".
    tail = lines[-1]
    if len(lines) > 1 and lines[-2].startswith("assert"):
        tail = f"{lines[-2]} -> {tail}"
    return tail[:300]


def check_candidate(
    solution_source: str | None,
    entry_point: str,
    test: str,
    python: str = sys.executable,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> ExecutionResult:
    """Run one problem's tests against one candidate solution."""
    if solution_source is None or not solution_source.strip():
        return ExecutionResult(False, "no_solution", "no solution file was written", 0.0)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="humaneval-") as scratch:
        work = Path(scratch)
        (work / "solution.py").write_text(solution_source, encoding="utf-8")
        (work / "program.py").write_text(
            build_program(entry_point, test, timeout=timeout, memory_mb=memory_mb),
            encoding="utf-8",
        )

        process = subprocess.Popen(
            # -E -s: ignore PYTHON* variables and the user site directory, so
            # what the candidate imports is the standard library and itself.
            # (-I would go further and drop the script's own directory from
            # sys.path, taking `solution` with it.)
            [python, "-E", "-s", "program.py"],
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=_child_env(work),
            # Its own process group, so a solution that spawns something is
            # killed as a unit rather than leaving an orphan behind.
            start_new_session=True,
        )
        try:
            _, stderr = process.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_group(process)
            _, stderr = process.communicate()
            timed_out = True

        duration = time.monotonic() - started
        if timed_out:
            return ExecutionResult(
                False, "timeout", f"still running after {timeout:g}s", duration
            )
        if process.returncode == 0:
            return ExecutionResult(True, "passed", "", duration)
        status = "failed" if "AssertionError" in stderr else "error"
        return ExecutionResult(False, status, _describe_failure(stderr), duration)


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), 9)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def estimate_pass_at_k(samples: int, correct: int, k: int) -> float:
    """Unbiased pass@k from the HumanEval paper.

    With one sample per problem this is just "did it pass", which is why the
    default run reports pass@1 and a plain count; the estimator earns its keep
    only once --samples asks for several attempts at each problem.
    """
    if k <= 0 or samples <= 0:
        return 0.0
    if samples - correct < k:
        return 1.0
    result = 1.0
    for index in range(samples - correct + 1, samples + 1):
        result *= 1.0 - k / index
    return 1.0 - result
