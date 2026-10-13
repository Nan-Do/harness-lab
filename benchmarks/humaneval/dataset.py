"""The HumanEval problem set: fetching it once, and handing it out.

The canonical file is the gzipped JSONL that ships with openai/human-eval, and
it is downloaded rather than vendored: 164 problems whose exact text is the
benchmark are worth keeping byte-identical to the source everyone else reports
numbers against. It is cached under `data/` after the first fetch, so a run
needs the network once and never again.

Run this module directly to prime the cache:

    uv run python benchmarks/humaneval/dataset.py
"""

import argparse
import gzip
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_NAME = "HumanEval.jsonl"
# What the published set contains. A short read means a truncated download, and
# a benchmark run against 12 problems reported as "HumanEval" is worse than a
# failed one.
EXPECTED_PROBLEMS = 164


@dataclass(frozen=True)
class Problem:
    """One HumanEval task, exactly as the dataset states it."""

    task_id: str  # "HumanEval/0"
    prompt: str  # imports, signature and docstring -- no implementation
    entry_point: str  # the function the tests call
    test: str  # defines check(candidate)
    canonical_solution: str  # the reference body, used by --self-test

    @property
    def slug(self) -> str:
        """A task id that can be a directory name: HumanEval/0 -> HumanEval_0."""
        return self.task_id.replace("/", "_")

    @property
    def number(self) -> int:
        return int(self.task_id.rsplit("/", 1)[-1])


def dataset_path(data_dir: Path | None = None) -> Path:
    return (data_dir or DEFAULT_DATA_DIR) / DATASET_NAME


def fetch(data_dir: Path | None = None, force: bool = False) -> Path:
    """Download and unpack the problem set, unless it is already cached."""
    target = dataset_path(data_dir)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(".jsonl.gz")
    with urllib.request.urlopen(HUMANEVAL_URL, timeout=60) as response:
        archive.write_bytes(response.read())
    # Unpacked on disk so the file can be read, diffed and edited like any
    # other input; the archive is kept as proof of what was downloaded.
    with gzip.open(archive, "rb") as packed, target.open("wb") as unpacked:
        shutil.copyfileobj(packed, unpacked)
    return target


def load_problems(
    path: Path | None = None,
    data_dir: Path | None = None,
    allow_fetch: bool = True,
) -> List[Problem]:
    """Load the problem set, fetching it first if it is not cached yet."""
    source = Path(path) if path else dataset_path(data_dir)
    if not source.exists():
        if not allow_fetch:
            raise FileNotFoundError(
                f"{source} is missing; run "
                "`uv run python benchmarks/humaneval/dataset.py` to fetch it"
            )
        source = fetch(data_dir)

    problems = [
        Problem(
            task_id=row["task_id"],
            prompt=row["prompt"],
            entry_point=row["entry_point"],
            test=row["test"],
            canonical_solution=row.get("canonical_solution", ""),
        )
        for row in (
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    if not problems:
        raise ValueError(f"{source} holds no problems; delete it and fetch again")
    return problems


def normalize_ids(values: Iterable[str]) -> List[str]:
    """Accept 'HumanEval/12', '12', and comma-separated runs of either."""
    ids: List[str] = []
    for value in values:
        for part in str(value).replace(" ", "").split(","):
            if not part:
                continue
            ids.append(part if "/" in part else f"HumanEval/{part}")
    return ids


def select(
    problems: Sequence[Problem],
    only: Iterable[str] = (),
    limit: int = 0,
    offset: int = 0,
) -> List[Problem]:
    """Narrow the set the way the CLI flags describe it.

    `only` wins over `offset`/`limit`: naming problems is how a failure gets
    re-run, and a leftover --limit silently dropping one of them would be a
    confusing way to lose an afternoon.
    """
    wanted = normalize_ids(only)
    if wanted:
        by_id = {problem.task_id: problem for problem in problems}
        missing = [task_id for task_id in wanted if task_id not in by_id]
        if missing:
            raise KeyError(f"no such problem(s): {', '.join(missing)}")
        return [by_id[task_id] for task_id in wanted]

    chosen = list(problems)[offset:]
    return chosen[:limit] if limit > 0 else chosen


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fetch the HumanEval problem set into the local cache.",
    )
    parser.add_argument(
        "--data-dir", default=None, help="Where to cache the dataset."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if it is cached."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    path = fetch(data_dir, force=args.force)
    problems = load_problems(path)
    print(f"{path} · {len(problems)} problems")
    if len(problems) != EXPECTED_PROBLEMS:
        print(
            f"warning: expected {EXPECTED_PROBLEMS} problems, found "
            f"{len(problems)}; the download may be incomplete",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
