"""One HumanEval problem, end to end through the harness.

The unit of work is deliberately the whole harness, not the model: each problem
gets a workspace of its own holding `solution.py` seeded with the problem's
signature and docstring, and the agent is asked to implement it with the tools
it normally has. What is scored is the file that is on disk when the run ends.
That is the number this repository cares about -- a model's answer is worth
nothing here if the harness could not get it into a file -- and it means every
run leaves behind the same JSONL log any other run does, so a failure can be
read afterwards instead of guessed at.

Each workspace is its own git repository. That is not cosmetic: the harness
resolves its root with `git rev-parse --show-toplevel`, so a workspace nested
inside this checkout without a `.git` of its own would hand an agent running
under `--approval auto` write access to the harness itself.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from benchmarks.humaneval.dataset import Problem
from benchmarks.humaneval.execution import (
    DEFAULT_MEMORY_MB,
    DEFAULT_TIMEOUT,
    check_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / "main.py"

# What the agent is asked for. It names the file, the constraint that keeps the
# result scoreable (the signature must survive), and the two habits that turn a
# correct function into a failing one: demo blocks that ask for input, and
# rewritten docstrings that drop half the specification.
REQUEST = """\
\"solution.py defines the Python function `{entry_point}` with its docstring and \
no implementation. Read the file, then implement the function so that it does \
exactly what its docstring says.

- Keep the existing imports, signature and docstring unchanged.
- Write only what the function needs: no usage examples, no print(), no input().
- `from solution import {entry_point}` must keep working.

When the implementation is in the file, answer in one line.\""""


@dataclass(frozen=True)
class RunConfig:
    """One configuration under test: a model behind an endpoint, plus knobs."""

    label: str
    host: str = "127.0.0.1"
    port: str = "8080"
    model: str = ""
    tools: str = "standard"
    max_steps: int = 10
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    language: str | None = None
    # Anything else to hand the harness verbatim (--no-stream, --planning, ...).
    extra_args: tuple = ()
    harness_timeout: float = 300.0
    exec_timeout: float = DEFAULT_TIMEOUT
    memory_mb: int = DEFAULT_MEMORY_MB
    # The interpreter that runs the harness and the candidates; the harness
    # environment needs `openai`, the candidates need only the standard library.
    python: str = sys.executable

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def as_metadata(self) -> Dict:
        """The configuration as it goes into summary.json.

        The interpreter path is dropped: it is a fact about the machine, not
        about the configuration being compared.
        """
        data = {key: value for key, value in asdict(self).items() if key != "python"}
        data["extra_args"] = list(self.extra_args)
        return data


def server_model(config: RunConfig, timeout: float = 10.0) -> str:
    """Ask the endpoint which model it is serving.

    Doubles as the pre-flight check. A suite that starts against a server that
    is not up fails 164 times over an hour; this fails once, immediately.
    """
    url = f"http://{config.host}:{config.port}/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    entries = payload.get("data") or payload.get("models") or []
    if not entries:
        raise RuntimeError(f"{url} reported no models")
    first = entries[0]
    return str(first.get("id") or first.get("model") or first.get("name") or "unknown")


def check_endpoint(config: RunConfig) -> str:
    try:
        return server_model(config)
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"no llama-server answering at {config.endpoint} ({exc}). "
            "Start one (llama-server -m <model.gguf> --jinja) before running "
            "the benchmark, and use --host/--port to point at it."
        ) from exc


# --- One problem ---


def prepare_workspace(workspace: Path, problem: Problem) -> Path:
    """Create the problem's workspace: a git repo holding the seeded solution.

    The file is seeded with the prompt verbatim -- signature and docstring, no
    body -- which is valid Python (a docstring *is* a body), so a run where the
    agent does nothing at all scores as a wrong answer rather than a crash.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text(problem.prompt, encoding="utf-8")

    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=workspace,
        check=True,
        capture_output=True,
        timeout=30,
    )
    # Verify the boundary rather than trusting it: if this workspace is not its
    # own repository root, the harness would take the enclosing checkout as the
    # root and the agent could write anywhere in it.
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if not root or Path(root).resolve() != workspace.resolve():
        raise RuntimeError(
            f"{workspace} did not become its own git repository (git said "
            f"{root!r}); refusing to run an agent that would treat "
            f"{REPO_ROOT} as its workspace"
        )
    return workspace


def harness_command(config: RunConfig, workspace: Path, request: str) -> List[str]:
    """The headless harness invocation for one problem."""
    command = [
        config.python,
        str(MAIN),
        "--mode",
        "headless",
        "--cwd",
        str(workspace),
        # The agent has to be able to write without a human on the other end;
        # containment is the workspace boundary, not the approval prompt.
        "--approval",
        "auto",
        # Nobody to ask, so make the fallback explicit instead of implicit.
        "--on-step-limit",
        "stop",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--tools",
        config.tools,
        "--max-steps",
        str(config.max_steps),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--temperature",
        str(config.temperature),
        "--top-p",
        str(config.top_p),
        "--log-dir",
        str(workspace / ".harness-lab" / "logs"),
    ]
    if config.model:
        command += ["--model", config.model]
    if config.language:
        command += ["--language", config.language]
    # Passed through last so an explicit --harness-arg overrides a default above.
    command += list(config.extra_args)
    command.append(request)
    return command


def read_log_stats(log_dir: Path) -> Dict:
    """Distil the run's JSONL log into the numbers a comparison needs."""
    stats: Dict = {
        "steps": 0,
        "attempts": 0,
        "final_reason": "",
        "tool_calls": 0,
        "tool_outcomes": {},
        "retries": 0,
        "malformed": 0,
        "step_limit_hits": 0,
        "prompt_tokens": 0,
        "context_percent": 0.0,
        "log": "",
    }
    logs = sorted(log_dir.glob("run-*.jsonl")) if log_dir.is_dir() else []
    if not logs:
        return stats
    log = logs[-1]
    stats["log"] = str(log)

    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = event.get("event")
        if name == "final":
            stats["steps"] = event.get("tool_steps", stats["steps"])
            stats["attempts"] = event.get("attempts", stats["attempts"])
            stats["final_reason"] = event.get("reason", "")
        elif name == "tool_call":
            stats["tool_calls"] += 1
        elif name == "tool_outcome":
            outcome = str(event.get("outcome", "unknown"))
            stats["tool_outcomes"][outcome] = stats["tool_outcomes"].get(outcome, 0) + 1
        elif name == "retry":
            stats["retries"] += 1
        elif name == "malformed_tool_call":
            stats["malformed"] += 1
        elif name == "step_limit":
            stats["step_limit_hits"] += 1
        elif name == "context_usage":
            stats["prompt_tokens"] = max(
                stats["prompt_tokens"], int(event.get("prompt_tokens") or 0)
            )
            stats["context_percent"] = max(
                stats["context_percent"], float(event.get("percent") or 0.0)
            )
    return stats


def run_problem(
    problem: Problem,
    config: RunConfig,
    workspace: Path,
    sample: int = 0,
) -> Dict:
    """Run one problem through the harness and score what it left behind."""
    prepare_workspace(workspace, problem)
    request = REQUEST.format(entry_point=problem.entry_point)
    command = harness_command(config, workspace, request)

    started = time.monotonic()
    harness_status, returncode, answer, error = "ok", 0, "", ""
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=config.harness_timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1"},
        )
        returncode = completed.returncode
        answer = completed.stdout.strip()
        if returncode != 0:
            harness_status = "crashed"
            error = (completed.stderr.strip() or "no stderr")[-500:]
    except subprocess.TimeoutExpired as exc:
        harness_status = "timeout"
        returncode = -1
        error = f"harness did not finish within {config.harness_timeout:g}s"
        # Whatever it managed to print before the kill is the only account of
        # how far it got.
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        answer = partial.strip()
    duration = time.monotonic() - started

    solution_path = workspace / "solution.py"
    solution = (
        solution_path.read_text(encoding="utf-8", errors="replace")
        if solution_path.is_file()
        else None
    )
    # An untouched file is a distinct outcome from a wrong answer: it means the
    # agent never got as far as writing anything, which is a harness failure to
    # look at rather than a model that cannot solve the problem.
    untouched = solution is not None and solution.strip() == problem.prompt.strip()

    result = check_candidate(
        None if untouched else solution,
        problem.entry_point,
        problem.test,
        python=config.python,
        timeout=config.exec_timeout,
        memory_mb=config.memory_mb,
    )

    # Precedence matters for reading the results afterwards: a file that was
    # never written because the harness died is a harness failure, not a model
    # that could not solve the problem. A file that *was* written still gets
    # scored on its merits even if the run crashed after writing it -- what is
    # on disk is what the benchmark asked for.
    if untouched and harness_status != "ok":
        status, detail = f"harness_{harness_status}", error or "the agent run failed"
    elif untouched:
        status, detail = "untouched", "solution.py was never changed"
    else:
        status, detail = result.status, result.detail

    record = {
        "task_id": problem.task_id,
        "label": config.label,
        "sample": sample,
        "passed": result.passed,
        "status": status,
        "detail": detail,
        "harness_status": harness_status,
        "returncode": returncode,
        "error": error,
        "answer": answer[-500:],
        "duration_s": round(duration, 2),
        "exec_duration_s": round(result.duration_s, 3),
        "solution_chars": len(solution or ""),
        "workspace": str(workspace),
    }
    record.update(read_log_stats(workspace / ".harness-lab" / "logs"))
    return record


# --- The suite ---


# How many identical harness failures at the start of a suite are enough to
# call it a configuration problem rather than a hard problem set.
DEFAULT_ABORT_AFTER = 5


@dataclass
class SuiteOutcome:
    records: List[Dict] = field(default_factory=list)
    interrupted: bool = False
    aborted: bool = False
    abort_reason: str = ""


def crash_signature(record: Dict) -> str:
    """What makes two failed runs "the same failure" for the abort check."""
    if record.get("harness_status") == "ok":
        return ""
    return f"{record.get('harness_status')}:{str(record.get('error'))[:120]}"


def _key(record: Dict) -> tuple:
    return (record.get("task_id"), record.get("sample", 0))


def load_records(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def run_suite(
    problems: Sequence[Problem],
    config: RunConfig,
    run_dir: Path,
    jobs: int = 4,
    samples: int = 1,
    resume: bool = False,
    abort_after: int = DEFAULT_ABORT_AFTER,
    on_record: Callable[[Dict, int, int], None] | None = None,
) -> SuiteOutcome:
    """Run every problem (times `samples`) and stream the results to disk.

    Results are appended as they land rather than written at the end: a suite
    is an hours-long run against a local model, and one that cannot survive a
    Ctrl-C is one nobody dares start.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    done = {_key(record) for record in load_records(results_path)} if resume else set()
    if not resume and results_path.exists():
        results_path.unlink()

    tasks = [
        (problem, sample)
        for problem in problems
        for sample in range(samples)
        if (problem.task_id, sample) not in done
    ]
    outcome = SuiteOutcome(records=load_records(results_path) if resume else [])
    total = len(tasks)
    if not tasks:
        return outcome

    write_lock = threading.Lock()
    completed = 0
    early_signatures: List[str] = []

    def work(item) -> Dict:
        problem, sample = item
        workspace = run_dir / "workspaces" / problem.slug
        if samples > 1:
            workspace = workspace / f"sample-{sample}"
        try:
            return run_problem(problem, config, workspace, sample=sample)
        except Exception as exc:  # a broken workspace must not sink the suite
            return {
                "task_id": problem.task_id,
                "label": config.label,
                "sample": sample,
                "passed": False,
                "status": "runner_error",
                "detail": f"{type(exc).__name__}: {exc}"[:300],
                "harness_status": "error",
                "workspace": str(workspace),
            }

    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        futures = {pool.submit(work, item): item for item in tasks}
        try:
            for future in as_completed(futures):
                record = future.result()
                completed += 1
                with write_lock:
                    with results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    outcome.records.append(record)
                if on_record:
                    on_record(record, completed, total)

                # An endpoint that rejects this harness's requests fails every
                # problem identically, and finding that out after 164 of them
                # is finding it out too late.
                if abort_after and completed <= abort_after:
                    early_signatures.append(crash_signature(record))
                    if (
                        completed == abort_after
                        and len(set(early_signatures)) == 1
                        and early_signatures[0]
                    ):
                        outcome.aborted = True
                        outcome.abort_reason = (
                            f"the first {abort_after} runs all failed the same "
                            f"way ({early_signatures[0]}); stopping instead of "
                            "spending the whole suite on it"
                        )
                        break
        except KeyboardInterrupt:
            outcome.interrupted = True
        finally:
            if outcome.interrupted or outcome.aborted:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
    return outcome


def with_label(config: RunConfig, label: str) -> RunConfig:
    """Name a configuration after the model the server turned out to be serving."""
    return replace(config, label=label)
