"""Run the same problem set across several configurations and compare them.

    uv run python benchmarks/humaneval/sweep.py --config my-sweep.json

A configuration is a model behind an endpoint plus whatever harness knobs you
want measured, so the same script answers both questions this repository asks:
"which of these LLMs is better at HumanEval through the harness" (same knobs,
different endpoints) and "does this knob help" (same endpoint, different knobs).

Runs go one at a time on purpose. Two suites against the same llama-server
would queue behind each other and report each other's latency as their own;
`--jobs` is how you use a server that has parallel slots.

Nothing is ever started for you. If a configuration's endpoint is not up yet,
`--wait` holds the sweep there until it answers -- which is the whole
"benchmark model A, swap the server over to model B, benchmark that" workflow,
with the swap being the only thing you have to do by hand.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):  # allow `python benchmarks/humaneval/sweep.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.humaneval import dataset, report  # noqa: E402
from benchmarks.humaneval.runner import (  # noqa: E402
    RunConfig,
    check_endpoint,
    run_suite,
    server_model,
    with_label,
)

DEFAULT_SWEEPS_DIR = Path(__file__).resolve().parent / "runs"
# Keys a configuration entry may set, mapped onto RunConfig's own names.
CONFIG_KEYS = {
    "label",
    "host",
    "port",
    "model",
    "tools",
    "max_steps",
    "max_new_tokens",
    "temperature",
    "top_p",
    "language",
    "harness_args",
    "harness_timeout",
}


def parse_endpoint(value: str) -> Dict:
    """--endpoint 127.0.0.1:8081=qwen-4b -> a one-line configuration."""
    endpoint, _, label = value.partition("=")
    host, _, port = endpoint.partition(":")
    entry = {"host": host or "127.0.0.1", "port": port or "8080"}
    if label:
        entry["label"] = label
    return entry


def load_config(path: Path) -> List[Dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("runs", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must hold a list of run configurations")
    for entry in entries:
        unknown = set(entry) - CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"{path}: unknown key(s) {', '.join(sorted(unknown))}; "
                f"valid keys are {', '.join(sorted(CONFIG_KEYS))}"
            )
    return entries


def build_config(entry: Dict, args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        label=str(entry.get("label", "")),
        host=str(entry.get("host", "127.0.0.1")),
        port=str(entry.get("port", "8080")),
        model=str(entry.get("model", "")),
        tools=str(entry.get("tools", "standard")),
        max_steps=int(entry.get("max_steps", 10)),
        max_new_tokens=int(entry.get("max_new_tokens", 512)),
        temperature=float(entry.get("temperature", 0.2)),
        top_p=float(entry.get("top_p", 0.9)),
        language=entry.get("language"),
        extra_args=tuple(entry.get("harness_args", ())),
        harness_timeout=float(entry.get("harness_timeout", 300.0)),
        exec_timeout=args.exec_timeout,
        memory_mb=args.memory_mb,
        python=args.python,
    )


def wait_for_endpoint(config: RunConfig, seconds: float) -> str:
    """Block until the endpoint answers, or give up after `seconds`.

    Prints once and then waits quietly: this is where you go and swap the
    model over, and a scrolling wall of retries helps nobody.
    """
    deadline = time.monotonic() + max(seconds, 0.0)
    first = True
    while True:
        try:
            return server_model(config)
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"nothing answering at {config.endpoint} ({exc})"
                ) from exc
            if first:
                print(
                    f"  waiting for {config.endpoint} "
                    f"(up to {seconds:g}s) — start llama-server for this run…",
                    flush=True,
                )
                first = False
            time.sleep(5.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run HumanEval across several models or harness configurations and "
            "print a comparison."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="JSON file holding the list of configurations to run "
        "(see sweep.example.json).",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        metavar="HOST:PORT[=LABEL]",
        help="Shorthand for one configuration with default knobs; repeatable.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="Seconds to wait for a configuration's endpoint before giving up "
        "on it, so you can swap llama-server over between runs.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for the sweep (default: runs/sweep-<stamp>).",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=[], metavar="IDS")
    parser.add_argument("--abort-after", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exec-timeout", type=float, default=10.0)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    entries: List[Dict] = []
    if args.config:
        entries += load_config(Path(args.config))
    entries += [parse_endpoint(value) for value in args.endpoint]
    if not entries:
        print(
            "nothing to run: pass --config with a JSON list of configurations, "
            "or --endpoint host:port=label (repeatable).",
            file=sys.stderr,
        )
        return 2

    problems = dataset.select(
        dataset.load_problems(
            path=Path(args.dataset) if args.dataset else None,
            data_dir=Path(args.data_dir) if args.data_dir else None,
        ),
        only=args.only,
        limit=args.limit,
        offset=args.offset,
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = Path(args.out) if args.out else DEFAULT_SWEEPS_DIR / f"sweep-{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"sweep · {len(entries)} configuration(s) × {len(problems)} problems "
        f"× {args.samples} sample(s)\n  results {sweep_dir}\n"
    )

    summaries: List[report.Summary] = []
    for index, entry in enumerate(entries, start=1):
        config = build_config(entry, args)
        print(f"[{index}/{len(entries)}] {config.label or config.endpoint}")
        try:
            served = (
                wait_for_endpoint(config, args.wait)
                if args.wait
                else check_endpoint(config)
            )
        except RuntimeError as exc:
            # One dead endpoint must not throw away the runs that can still be
            # done -- especially the ones already finished.
            print(f"  skipped: {exc}\n", file=sys.stderr)
            continue

        label = config.label or Path(served).stem or f"run-{index}"
        config = with_label(config, label)
        run_dir = sweep_dir / f"{index:02d}-{label}"
        print(f"  serving {served} · results {run_dir}", flush=True)

        started = time.monotonic()
        outcome = run_suite(
            problems,
            config,
            run_dir,
            jobs=args.jobs,
            samples=args.samples,
            resume=args.resume,
            abort_after=args.abort_after,
            on_record=_progress(label, len(problems) * args.samples),
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
        summaries.append(summary)
        (run_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )
        print(report.format_summary(summary), flush=True)
        if outcome.aborted:
            print(f"  aborted: {outcome.abort_reason}\n")
        if outcome.interrupted:
            print("interrupted; the sweep stops here")
            break

    if not summaries:
        print("no configuration produced results", file=sys.stderr)
        return 1

    comparison = report.format_comparison(summaries)
    print("\nComparison\n")
    print(comparison)
    print("")
    (sweep_dir / "comparison.json").write_text(
        json.dumps([summary.to_dict() for summary in summaries], indent=2),
        encoding="utf-8",
    )
    (sweep_dir / "comparison.txt").write_text(comparison + "\n", encoding="utf-8")
    print(f"  {sweep_dir / 'comparison.txt'}\n")
    return 0


def _progress(label: str, total: int):
    state = {"passed": 0}

    def on_record(record: Dict, done: int, _total: int) -> None:
        state["passed"] += 1 if record.get("passed") else 0
        verdict = "PASS" if record.get("passed") else "FAIL"
        print(
            f"  [{done:>4}/{total}] {label} {record.get('task_id', '?'):<15} "
            f"{verdict}  score {state['passed']}/{done}",
            flush=True,
        )

    return on_record


if __name__ == "__main__":
    raise SystemExit(main())
