"""Run the HumanEval suite against one harness configuration.

    uv run python benchmarks/humaneval/run_benchmark.py --port 8080

Every problem gets its own workspace, its own agent run, and its own JSONL log
under `runs/<stamp>-<label>/`; what is printed at the end is how many of the 164
passed. Point it at a different llama-server (`--host/--port`) to score a
different model, or change a knob (`--tools`, `--max-steps`, `--language`) to
score a different harness.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # allow `python benchmarks/humaneval/run_benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.humaneval import dataset, report  # noqa: E402
from benchmarks.humaneval.execution import check_candidate  # noqa: E402
from benchmarks.humaneval.runner import (  # noqa: E402
    RunConfig,
    check_endpoint,
    run_suite,
    with_label,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run the HumanEval benchmark through the harness and report how "
            "many problems passed."
        ),
    )
    server = parser.add_argument_group("llama-server")
    server.add_argument("--host", default="127.0.0.1", help="llama-server host.")
    server.add_argument("--port", default="8080", help="llama-server port.")
    server.add_argument(
        "--model",
        default="",
        help="Model name to request; llama-server falls back to what it loaded.",
    )
    server.add_argument(
        "--label",
        default="",
        help="Name for this run in the results (default: the model the server reports).",
    )

    harness = parser.add_argument_group("harness knobs")
    harness.add_argument("--tools", default="standard", help="--tools profile.")
    harness.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Tool steps a problem may spend before the run is cut off.",
    )
    harness.add_argument("--max-new-tokens", type=int, default=512)
    harness.add_argument("--temperature", type=float, default=0.2)
    harness.add_argument("--top-p", type=float, default=0.9)
    harness.add_argument(
        "--language",
        default=None,
        help="Pass --language to the harness (e.g. python) to give the agent "
        "its lint/type_check/write_tests tools.",
    )
    harness.add_argument(
        "--harness-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument handed to main.py verbatim; repeatable "
        "(--harness-arg --no-stream).",
    )
    harness.add_argument(
        "--harness-timeout",
        type=float,
        default=300.0,
        help="Seconds one problem's agent run may take before it is killed.",
    )

    suite = parser.add_argument_group("suite")
    suite.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Problems to run at once. llama-server needs matching --parallel "
        "slots, or requests will queue.",
    )
    suite.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Attempts per problem; >1 reports pass@k as well as pass@1.",
    )
    suite.add_argument(
        "--limit", type=int, default=0, help="Run only the first N problems (0 = all)."
    )
    suite.add_argument("--offset", type=int, default=0, help="Skip the first N problems.")
    suite.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="IDS",
        help="Run just these problems: --only 0,12 or --only HumanEval/32.",
    )
    suite.add_argument(
        "--out",
        default=None,
        help="Directory for this run's results (default: runs/<stamp>-<label>).",
    )
    suite.add_argument(
        "--abort-after",
        type=int,
        default=5,
        help="Stop early if this many runs in a row fail the same way (a dead "
        "endpoint or a model the harness cannot drive); 0 never aborts.",
    )
    suite.add_argument(
        "--resume",
        action="store_true",
        help="Keep the results already in --out and run only what is missing.",
    )

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument(
        "--exec-timeout",
        type=float,
        default=10.0,
        help="Seconds a candidate solution may run before it is killed.",
    )
    scoring.add_argument("--memory-mb", type=int, default=1024)
    scoring.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter that runs the harness and the candidate solutions.",
    )
    scoring.add_argument(
        "--self-test",
        action="store_true",
        help="Score the dataset's own reference solutions and exit. No model "
        "involved: it checks that this machine's scoring works before you "
        "blame an LLM for the number it produces.",
    )

    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", default=None, help="Path to HumanEval.jsonl.")
    data.add_argument("--data-dir", default=None, help="Where the dataset is cached.")
    return parser


def self_test(problems, args) -> int:
    """Run the reference solutions through the scorer; they must all pass."""
    print(f"scoring {len(problems)} reference solutions (no model involved)…")
    failures = []
    for problem in problems:
        result = check_candidate(
            problem.prompt + problem.canonical_solution,
            problem.entry_point,
            problem.test,
            python=args.python,
            timeout=args.exec_timeout,
            memory_mb=args.memory_mb,
        )
        if not result.passed:
            failures.append((problem.task_id, result.status, result.detail))

    passed = len(problems) - len(failures)
    print(f"self-test: {passed}/{len(problems)} reference solutions pass")
    for task_id, status, detail in failures:
        print(f"  {task_id:<16} {status:<10} {detail[:80]}")
    if failures:
        print(
            "\nThe scorer is wrong, not the model: fix this before running a "
            "benchmark, since every number below it inherits the bug."
        )
        return 1
    return 0


def make_progress(total: int):
    """One line per finished problem, carrying the running score."""
    state = {"passed": 0}

    def on_record(record: dict, done: int, _total: int) -> None:
        state["passed"] += 1 if record.get("passed") else 0
        verdict = "PASS" if record.get("passed") else "FAIL"
        note = "" if record.get("passed") else f"  {record.get('status', '')}"
        rate = 100.0 * state["passed"] / done
        print(
            f"[{done:>4}/{total}] {record.get('task_id', '?'):<15} {verdict}"
            f"  steps {record.get('steps', 0):>2}"
            f"  {record.get('duration_s', 0.0):>6.1f}s"
            f"  score {state['passed']}/{done} ({rate:.1f}%){note}",
            flush=True,
        )

    return on_record


def main() -> int:
    args = build_arg_parser().parse_args()

    problems = dataset.select(
        dataset.load_problems(
            path=Path(args.dataset) if args.dataset else None,
            data_dir=Path(args.data_dir) if args.data_dir else None,
        ),
        only=args.only,
        limit=args.limit,
        offset=args.offset,
    )
    if args.self_test:
        return self_test(problems, args)

    config = RunConfig(
        label=args.label,
        host=args.host,
        port=str(args.port),
        model=args.model,
        tools=args.tools,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        language=args.language,
        extra_args=tuple(args.harness_arg),
        harness_timeout=args.harness_timeout,
        exec_timeout=args.exec_timeout,
        memory_mb=args.memory_mb,
        python=args.python,
    )

    # Before anything expensive: is there a server, and what is it serving?
    try:
        served = check_endpoint(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    label = args.label or Path(served).stem or "run"
    config = with_label(config, label)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) if args.out else DEFAULT_RUNS_DIR / f"{stamp}-{label}"

    print(
        f"HumanEval · {len(problems)} problems × {args.samples} sample(s) "
        f"· {args.jobs} job(s)\n"
        f"  endpoint {config.endpoint} · serving {served}\n"
        f"  tools {config.tools} · max-steps {config.max_steps} "
        f"· language {config.language or 'none'}\n"
        f"  results  {run_dir}\n"
        "  note: candidate solutions are executed locally, contained but not "
        "sandboxed.\n",
        flush=True,
    )

    started = time.monotonic()
    outcome = run_suite(
        problems,
        config,
        run_dir,
        jobs=args.jobs,
        samples=args.samples,
        resume=args.resume,
        abort_after=args.abort_after,
        on_record=make_progress(len(problems) * args.samples),
    )
    elapsed = time.monotonic() - started

    summary = report.summarize(
        outcome.records,
        label=label,
        endpoint=config.endpoint,
        model=served,
        config=config.as_metadata(),
        wall_seconds=elapsed,
        jobs=args.jobs,
    )
    print(report.format_summary(summary))
    if summary.passed < summary.records:
        print("  did not pass")
        print(report.failures_table(outcome.records))
        print("")

    (run_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    print(f"  results {run_dir / 'results.jsonl'}")
    print(f"  summary {run_dir / 'summary.json'}\n")

    if outcome.aborted:
        print(f"aborted: {outcome.abort_reason}")
        print("  the failure above is the harness talking to the server, not a")
        print("  model getting the problems wrong; fix that and rerun.\n")
        return 3
    if outcome.interrupted:
        print("interrupted; rerun with --resume to finish the remaining problems")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
