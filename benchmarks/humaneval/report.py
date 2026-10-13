"""Turning a pile of per-problem records into the number you were after.

The headline is deliberately plain -- how many of the 164 problems passed --
because that is the question the benchmark was run to answer. Everything under
it exists to keep that number honest: a run where eight problems never got a
solution written at all is not the same 62% as one where eight were solved
wrongly, and only the breakdown tells them apart.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from benchmarks.humaneval.execution import estimate_pass_at_k

# Failure statuses in the order a reader wants them: what the model got wrong
# first, then what the harness got wrong.
STATUS_ORDER = (
    "passed",
    "failed",
    "error",
    "timeout",
    "no_solution",
    "untouched",
    "harness_crashed",
    "harness_timeout",
    "runner_error",
)
STATUS_LABELS = {
    "failed": "failed (wrong answer)",
    "error": "error (raised)",
    "timeout": "timeout (ran too long)",
    "no_solution": "no solution written",
    "untouched": "untouched (agent never edited)",
    "harness_crashed": "harness crashed",
    "harness_timeout": "harness timed out",
    "runner_error": "runner error",
}


@dataclass
class Summary:
    label: str = ""
    endpoint: str = ""
    model: str = ""
    config: Dict = field(default_factory=dict)
    problems: int = 0
    samples: int = 1
    records: int = 0
    passed: int = 0
    pass_at_1: float = 0.0
    pass_at_k: Dict[str, float] = field(default_factory=dict)
    statuses: Dict[str, int] = field(default_factory=dict)
    harness_statuses: Dict[str, int] = field(default_factory=dict)
    avg_steps: float = 0.0
    step_limit_hits: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    tool_outcomes: Dict[str, int] = field(default_factory=dict)
    retries: int = 0
    malformed: int = 0
    agent_seconds: float = 0.0
    median_seconds: float = 0.0
    wall_seconds: float = 0.0
    jobs: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)


def load_records(path: Path) -> List[Dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(
    records: Sequence[Dict],
    label: str = "",
    endpoint: str = "",
    model: str = "",
    config: Dict | None = None,
    wall_seconds: float = 0.0,
    jobs: int = 1,
) -> Summary:
    """Aggregate one configuration's records into a Summary."""
    by_task: Dict[str, List[Dict]] = {}
    for record in records:
        by_task.setdefault(record.get("task_id", "?"), []).append(record)

    summary = Summary(
        label=label or (records[0].get("label", "") if records else ""),
        endpoint=endpoint,
        model=model,
        config=dict(config or {}),
        problems=len(by_task),
        records=len(records),
        wall_seconds=wall_seconds,
        jobs=jobs,
    )
    if not records:
        return summary

    summary.samples = max(len(group) for group in by_task.values())
    summary.passed = sum(1 for record in records if record.get("passed"))
    # pass@1 over problems rather than over records, so a --samples 5 run and a
    # --samples 1 run are still answering the same question.
    per_task_rate = [
        sum(1 for record in group if record.get("passed")) / len(group)
        for group in by_task.values()
    ]
    summary.pass_at_1 = 100.0 * statistics.fmean(per_task_rate)
    # The ks worth reporting: the usual ones, plus however many samples were
    # actually drawn -- a --samples 2 run should be able to say pass@2.
    for k in sorted({2, 5, 10, summary.samples}):
        if 1 < k <= summary.samples:
            summary.pass_at_k[f"pass@{k}"] = 100.0 * statistics.fmean(
                estimate_pass_at_k(
                    len(group), sum(1 for r in group if r.get("passed")), k
                )
                for group in by_task.values()
            )

    for record in records:
        status = record.get("status", "unknown")
        summary.statuses[status] = summary.statuses.get(status, 0) + 1
        harness_status = record.get("harness_status", "unknown")
        summary.harness_statuses[harness_status] = (
            summary.harness_statuses.get(harness_status, 0) + 1
        )
        summary.tool_calls += int(record.get("tool_calls") or 0)
        summary.retries += int(record.get("retries") or 0)
        summary.malformed += int(record.get("malformed") or 0)
        summary.step_limit_hits += int(record.get("step_limit_hits") or 0)
        for outcome, count in (record.get("tool_outcomes") or {}).items():
            summary.tool_outcomes[outcome] = summary.tool_outcomes.get(outcome, 0) + count

    summary.tool_failures = sum(
        count for outcome, count in summary.tool_outcomes.items() if outcome != "ok"
    )
    steps = [int(record.get("steps") or 0) for record in records]
    durations = [float(record.get("duration_s") or 0.0) for record in records]
    summary.avg_steps = statistics.fmean(steps) if steps else 0.0
    summary.agent_seconds = sum(durations)
    summary.median_seconds = statistics.median(durations) if durations else 0.0
    return summary


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def _outcome_note(summary: Summary) -> str:
    """The failed tool outcomes, biggest first: why calls did not land."""
    failures = sorted(
        ((count, name) for name, count in summary.tool_outcomes.items() if name != "ok"),
        reverse=True,
    )
    if not failures:
        return ""
    listed = ", ".join(f"{name} {count}" for count, name in failures[:4])
    return f"  (failed {summary.tool_failures}: {listed})"


def format_summary(summary: Summary) -> str:
    """The final score block, printed when a suite finishes."""
    total = summary.problems or 1
    passed_problems = round(summary.pass_at_1 * total / 100.0)
    lines = [
        "",
        f"HumanEval · {summary.label or 'run'}",
        f"  endpoint {summary.endpoint or '-'} · model {summary.model or '-'}",
    ]
    config = summary.config
    if config:
        lines.append(
            f"  tools {config.get('tools', '-')} · max-steps {config.get('max_steps', '-')} "
            f"· language {config.get('language') or 'none'} · samples {summary.samples}"
        )
    lines += [
        "",
        f"  SCORE   {passed_problems} / {summary.problems} passed"
        f"   ({summary.pass_at_1:.1f}%)",
    ]
    if summary.pass_at_k:
        extra = " · ".join(
            f"{name} {value:.1f}%" for name, value in summary.pass_at_k.items()
        )
        lines.append(f"          pass@1 {summary.pass_at_1:.1f}% · {extra}")
    lines.append("")

    failures = summary.records - summary.passed
    lines.append(f"  failures {failures}")
    for status in STATUS_ORDER:
        if status == "passed":
            continue
        count = summary.statuses.get(status, 0)
        if count:
            lines.append(f"    {STATUS_LABELS.get(status, status):<30} {count:>4}")
    for status, count in sorted(summary.statuses.items()):
        if status not in STATUS_ORDER:
            lines.append(f"    {status:<30} {count:>4}")

    crashed = summary.harness_statuses.get("crashed", 0)
    timed_out = summary.harness_statuses.get("timeout", 0)
    lines += [
        "  harness",
        f"    {'avg tool steps':<30} {summary.avg_steps:>4.1f}",
        f"    {'step-limit stops':<30} {summary.step_limit_hits:>4}",
        f"    {'tool calls':<30} {summary.tool_calls:>4}{_outcome_note(summary)}",
        f"    {'malformed tool calls':<30} {summary.malformed:>4}",
        f"    {'retries':<30} {summary.retries:>4}",
        f"    {'crashed / timed out':<30} {crashed:>4} / {timed_out}",
        "  time",
        f"    {'wall clock':<30} {format_duration(summary.wall_seconds):>4}"
        f"  ({summary.jobs} job{'s' if summary.jobs != 1 else ''})",
        f"    {'agent time, summed':<30} {format_duration(summary.agent_seconds):>4}",
        f"    {'median per problem':<30} {summary.median_seconds:>4.1f}s",
        "",
    ]
    return "\n".join(lines)


COMPARISON_COLUMNS = (
    ("label", 24),
    ("model", 26),
    ("pass@1", 8),
    ("passed", 10),
    ("steps", 7),
    ("toolfail", 9),
    ("wall", 8),
)


def _cell(value: str, width: int) -> str:
    value = str(value)
    return value if len(value) <= width else value[: width - 1] + "…"


def format_comparison(summaries: Sequence[Summary]) -> str:
    """The table that answers "which of these is better", best first."""
    header = "  ".join(name.ljust(width) for name, width in COMPARISON_COLUMNS)
    rows = [header, "-" * len(header)]
    for summary in sorted(summaries, key=lambda item: item.pass_at_1, reverse=True):
        passed_problems = round(summary.pass_at_1 * (summary.problems or 0) / 100.0)
        values = (
            summary.label,
            summary.model or summary.endpoint,
            f"{summary.pass_at_1:.1f}%",
            f"{passed_problems}/{summary.problems}",
            f"{summary.avg_steps:.1f}",
            str(summary.tool_failures),
            format_duration(summary.wall_seconds),
        )
        rows.append(
            "  ".join(
                _cell(value, width).ljust(width)
                for value, (_, width) in zip(values, COMPARISON_COLUMNS)
            )
        )
    return "\n".join(rows)


def failures_table(records: Sequence[Dict], limit: int = 20) -> str:
    """The problems that did not pass, for the "what now" after a run."""
    failed = [record for record in records if not record.get("passed")]
    if not failed:
        return "  every problem passed"
    failed.sort(key=lambda record: record.get("task_id", ""))
    lines = []
    for record in failed[:limit]:
        detail = str(record.get("detail") or record.get("error") or "").replace("\n", " ")
        lines.append(
            f"  {record.get('task_id', '?'):<16} {record.get('status', '?'):<12} "
            f"{detail[:80]}"
        )
    if len(failed) > limit:
        lines.append(f"  ... and {len(failed) - limit} more (see results.jsonl)")
    return "\n".join(lines)
