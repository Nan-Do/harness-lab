import json
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

from agent_logging import AgentLogger
from workspace import WorkspaceContext
from utils import (
    BACKUP_DIR_NAME,
    DEFAULT_MAX_NOISY_OUTPUT,
    IGNORED_PATH_NAMES,
    STATE_DIR_NAME,
    TOOL_PROFILES,
)
from tool_support import (
    ArgError,
    DeniedError,
    GuardError,
    NoMatchError,
    NotFoundError,
    ToolError,
    apply_arg_aliases,
    clip,
    coerce_list_content,
    find_fuzzy_match,
    match_lines,
    nearest_block,
    normalize_content,
    outline_generic,
    outline_python,
    resolve_tool_name,
    strip_line_numbers,
    unwrap_envelope,
    validate_syntax,
)
from app_types import (
    HistoryEntry,
    ToolMessageEntry,
    Tools,
    ToolDescriptionEntry,
)


# --- Global Tool Catalog & Decorator ---

_TOOL_CATALOG = {}
_TOOL_EXAMPLES = {}

# Tools whose output legitimately differs between identical calls, so an
# identical repeat is not necessarily a loop (re-running tests after an edit is
# the normal workflow).
_VOLATILE_TOOLS = set()

# Tools that put file content in front of the model; used by the
# read-before-overwrite guard and by stale-read supersession.
READ_TOOLS = ("read_file", "read_file_range")
# Tools that change a file's content, so a prior read of it is no longer current.
WRITE_TOOLS = ("write_file", "patch_file", "append_file", "replace_lines")


def agent_tool(
    name: str,
    description: str,
    schema: Dict[str, str],
    risky: bool = False,
    example: str = "",
    profile: str = "standard",
    volatile: bool = False,
):
    """Decorator to register a tool function into the global catalog.

    Tool functions validate their arguments eagerly and return a zero-argument
    callable that performs the actual work, so the registry can reject
    malformed calls before asking for approval.

    `profile` is the smallest tool set this tool appears in (see
    utils.TOOL_PROFILES); `volatile` marks tools whose identical repeat is not a
    loop.
    """

    def decorator(func: Callable):
        _TOOL_CATALOG[name] = {
            "description": description,
            "schema": schema,
            "risky": risky,
            "func": func,
            "profile": profile,
        }
        if example:
            _TOOL_EXAMPLES[name] = example
        if volatile:
            _VOLATILE_TOOLS.add(name)
        return func

    return decorator


# --- Navigation ---


@agent_tool(
    name="list_files",
    description="List files in the workspace.",
    schema={
        "path": "str='.'|Directory to list, relative to the repo root",
        "depth": "int=1|How many directory levels to descend (1 = this directory only)",
    },
    example='arguments: {"path": ".", "depth": 1}',
    profile="minimal",
)
def list_files_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._path(args.get("path", "."))
    if path.is_file():
        raise ArgError(
            f"{path.relative_to(registry.root)} is a file, not a directory; "
            "use read_file to read it"
        )
    if not path.is_dir():
        raise NotFoundError(f"directory not found: {args.get('path', '.')}")
    depth = max(int(args.get("depth", 1)), 1)

    def execute() -> str:
        limit = 300
        entries: List[str] = []
        truncated = False

        def walk(current: Path, level: int) -> None:
            nonlocal truncated
            if level > depth or truncated:
                return
            items = sorted(
                (
                    item
                    for item in current.iterdir()
                    if item.name not in IGNORED_PATH_NAMES
                ),
                key=lambda item: (item.is_file(), item.name.lower()),
            )
            for item in items:
                if len(entries) >= limit:
                    truncated = True
                    return
                relative = item.relative_to(registry.root)
                if item.is_dir():
                    entries.append(f"[D] {relative}/")
                    walk(item, level + 1)
                else:
                    try:
                        size = item.stat().st_size
                    except OSError:
                        size = 0
                    entries.append(f"[F] {relative} ({size} bytes)")

        walk(path, 1)
        body = "\n".join(entries) or "(empty)"
        if truncated:
            # Silently cutting the list makes the model believe it saw
            # everything; say so and name the way to narrow the request.
            body += (
                f"\n[stopped at {limit} entries; list a subdirectory "
                "or use find_files to narrow the search]"
            )
        return body

    return execute


@agent_tool(
    name="read_file",
    description="Read a whole UTF-8 file.",
    schema={
        "path": "str|File path relative to the repo root",
    },
    example='arguments: {"path": "README.md"}',
    profile="minimal",
)
def read_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._existing_file(args.get("path"))

    def execute() -> str:
        # Never clipped: an edit made against a partial view of a file is how
        # content gets destroyed. Whole files are dropped from context as a
        # unit when they no longer fit (agent.MiniAgent._fit_history_budget).
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(
            f"{number:>4}: {line}" for number, line in enumerate(lines, start=1)
        )
        header = f"# {path.relative_to(registry.root)} ({len(lines)} lines)"
        return f"{header}\n{body}" if lines else f"{header}\n(empty file)"

    return execute


@agent_tool(
    name="read_file_range",
    description=(
        "Read part of a UTF-8 file by line range. Prefer this over read_file "
        "for large files: read the range you need, then edit it with replace_lines."
    ),
    schema={
        "path": "str|File path relative to the repo root",
        "start": "int=1|First line to read (1-based)",
        "end": "int=200|Last line to read (inclusive)",
    },
    example='arguments: {"path": "README.md", "start": 1, "end": 80}',
    profile="minimal",
)
def read_file_range_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._existing_file(args.get("path"))
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1:
        raise ArgError("start must be 1 or greater")
    if end < start:
        raise ArgError(f"invalid line range: end ({end}) is before start ({start})")

    def execute() -> str:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        relative = path.relative_to(registry.root)
        if start > total:
            # An empty body reads as "the file has nothing there"; say what the
            # file actually contains so the next call can be right.
            return (
                f"# {relative} ({total} lines)\n"
                f"(no lines: requested {start}-{end} but the file ends at line {total})"
            )
        stop = min(end, total)
        body = "\n".join(
            f"{number:>4}: {line}"
            for number, line in enumerate(lines[start - 1 : stop], start=start)
        )
        header = f"# {relative} (lines {start}-{stop} of {total})"
        return f"{header}\n{body}"

    return execute


@agent_tool(
    name="outline_file",
    description=(
        "List the classes and functions in a file with their line numbers. "
        "Use it to find the part of a large file you need before reading it."
    ),
    schema={"path": "str|File path relative to the repo root"},
    example='arguments: {"path": "tools.py"}',
    profile="standard",
)
def outline_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._existing_file(args.get("path"))

    def execute() -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(registry.root)
        total = len(text.splitlines())
        if path.suffix == ".py":
            try:
                lines = outline_python(text)
            except SyntaxError as exc:
                return (
                    f"# {relative} ({total} lines)\n"
                    f"cannot outline: invalid Python at line {exc.lineno}: {exc.msg}"
                )
        else:
            lines = outline_generic(text)
        if not lines:
            return (
                f"# {relative} ({total} lines)\n"
                "(no declarations found; read the file with read_file_range)"
            )
        return f"# {relative} ({total} lines)\n" + "\n".join(lines)

    return execute


@agent_tool(
    name="find_files",
    description="Find files whose name matches a glob pattern.",
    schema={
        "pattern": "str|Filename or glob, e.g. 'tools.py' or '*.md'",
        "path": "str='.'|Directory to search under, relative to the repo root",
    },
    example='arguments: {"pattern": "*.py", "path": "."}',
    profile="standard",
)
def find_files_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ArgError("pattern must not be empty")
    path = registry._path(args.get("path", "."))
    if not path.is_dir():
        raise NotFoundError(f"directory not found: {args.get('path', '.')}")
    # A bare name is what models mean by "find the file called X".
    glob = pattern if any(char in pattern for char in "*?[") else f"*{pattern}*"

    def execute() -> str:
        matches = []
        for item in path.rglob(glob):
            if not item.is_file():
                continue
            relative = item.relative_to(registry.root)
            if any(part in IGNORED_PATH_NAMES for part in relative.parts):
                continue
            matches.append(str(relative))
            if len(matches) >= 200:
                break
        return "\n".join(sorted(matches)) or f"(no files matching {glob})"

    return execute


@agent_tool(
    name="search",
    description="Search file contents in the workspace with rg or a simple fallback.",
    schema={
        "pattern": "str|Text to search for",
        "path": "str='.'|File or directory to search in, relative to the repo root",
        "glob": "str=|Optional file glob to restrict the search, e.g. '*.py'",
    },
    example='arguments: {"pattern": "binary_search", "path": "."}',
    profile="minimal",
)
def search_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ArgError("pattern must not be empty")
    path = registry._path(args.get("path", "."))
    glob = str(args.get("glob", "")).strip()

    def execute() -> str:
        hint = "narrow the pattern or pass a glob"
        # Search a repo-relative target so rg reports repo-relative paths, the
        # same way the fallback does -- otherwise the same call answers with
        # absolute paths or relative ones depending on whether rg is installed.
        target = "." if path == registry.root else str(path.relative_to(registry.root))
        if shutil.which("rg"):
            command = [
                "rg",
                "-n",
                # rg drops the filename when given a single file; always show
                # it so every match line has the path the model needs to act on.
                "--with-filename",
                "--smart-case",
                "--max-count",
                "20",
                # One minified line must not be able to flood the context.
                "--max-columns",
                "300",
                "--max-columns-preview",
            ]
            if glob:
                command += ["-g", glob]
            # Searching the root means passing no path at all, so rg reports
            # "tools.py:12:" rather than "./tools.py:12:".
            result = subprocess.run(
                [*command, pattern, *([] if target == "." else [target])],
                cwd=registry.root,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            output = result.stdout.strip()
            if not output:
                return result.stderr.strip() or "(no matches)"
            return registry.clip(output, hint)

        # Fallback: keep the result comparable to rg's by skipping the same
        # noise directories and refusing binary files, so the same call does
        # not answer differently depending on whether rg is installed.
        matches: List[str] = []
        files = (
            [path]
            if path.is_file()
            else [
                item
                for item in path.rglob(glob or "*")
                if item.is_file()
                and not any(
                    part in IGNORED_PATH_NAMES
                    for part in item.relative_to(registry.root).parts
                )
            ]
        )
        for file_path in files:
            try:
                raw = file_path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.lower() in line.lower():
                    relative = file_path.relative_to(registry.root)
                    matches.append(f"{relative}:{number}:{line[:300]}")
                    if len(matches) >= 200:
                        return registry.clip("\n".join(matches), hint)
        return registry.clip("\n".join(matches), hint) or "(no matches)"

    return execute


# --- Shell & verification ---

# Commands that hang forever, need a human, or reach outside the workspace.
# Each entry is (substring, why/what to do instead).
_SHELL_DENYLIST = (
    ("sudo", "no privileged commands"),
    ("rm -rf /", "refusing a filesystem-wide delete"),
    ("mkfs", "refusing to format a device"),
    ("dd if=", "refusing raw device writes"),
    (":(){", "refusing a fork bomb"),
    ("shutdown", "refusing to power off the machine"),
    ("reboot", "refusing to reboot the machine"),
    ("git push", "this workspace is not for publishing; leave commits local"),
    ("pip install", "do not change the environment; use the existing one"),
    ("vim ", "no interactive editors; use read_file and patch_file"),
    ("nano ", "no interactive editors; use read_file and patch_file"),
    ("emacs ", "no interactive editors; use read_file and patch_file"),
    ("less ", "no pagers; use read_file"),
    ("more ", "no pagers; use read_file"),
    ("top", "no interactive monitors"),
    ("htop", "no interactive monitors"),
)

# Anything that prompts, pages, or colours its output wastes the timeout or the
# context; force a strictly non-interactive environment.
_SHELL_ENV = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "CI": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "DEBIAN_FRONTEND": "noninteractive",
}


def _shell_result(registry: "ToolRegistry", command: str, timeout: int) -> str:
    import os

    try:
        result = subprocess.run(
            command,
            cwd=registry.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            # A command that waits on stdin would otherwise burn the whole
            # timeout instead of failing immediately.
            stdin=subprocess.DEVNULL,
            env={**os.environ, **_SHELL_ENV},
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        note = ""
    except subprocess.TimeoutExpired as exc:
        # The partial output is usually where the reason for the hang is, so
        # keep it instead of reporting a bare timeout.
        exit_code = -1
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        note = (
            f"\nnote: timed out after {timeout}s and was killed. Raise timeout, "
            "or run something that finishes on its own (no servers, no watchers)."
        )

    hint = "re-run with a narrower command or pipe through 'tail'"
    return "\n".join(
        [
            f"exit_code: {exit_code}",
            "stdout:",
            registry.clip(stdout.strip(), hint) or "(empty)",
            "stderr:",
            registry.clip(stderr.strip(), hint) or "(empty)",
        ]
    ) + note


@agent_tool(
    name="run_shell",
    description="Run a shell command in the repo root.",
    schema={
        "command": "str|Shell command executed at the repo root",
        "timeout": "int=20|Timeout in seconds, between 1 and 120",
    },
    risky=True,
    example='arguments: {"command": "uv run --with pytest python -m pytest -q", "timeout": 20}',
    profile="minimal",
    volatile=True,
)
def run_shell_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    command = str(args.get("command", "")).strip()
    if not command:
        raise ArgError("command must not be empty")

    lowered = command.lower()
    for needle, reason in _SHELL_DENYLIST:
        if needle in lowered:
            raise DeniedError(f"refused command containing '{needle.strip()}': {reason}")
    if command.rstrip().endswith("&"):
        raise DeniedError(
            "refused a backgrounded command; it would outlive this step. "
            "Run something that finishes and returns its output."
        )

    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ArgError(f"timeout must be in [1, 120], got {timeout}")

    return lambda: _shell_result(registry, command, timeout)


@agent_tool(
    name="run_tests",
    description="Run the project's test suite and report the result.",
    schema={
        "path": "str=|Optional test file or directory to limit the run to",
        "timeout": "int=120|Timeout in seconds, between 1 and 600",
    },
    risky=True,
    example='arguments: {"path": "tests/test_tools.py"}',
    profile="standard",
    volatile=True,
)
def run_tests_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    root = registry.root
    target = str(args.get("path", "")).strip()
    if target:
        # Validate before running so a typo is a cheap error, not a red suite.
        registry._path(target)

    venv_python = root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    if (
        (root / "pyproject.toml").exists()
        or (root / "pytest.ini").exists()
        or (root / "tests").is_dir()
    ):
        command = f"{python} -m pytest -q"
    elif (root / "package.json").exists():
        command = "npm test --silent"
    elif (root / "Cargo.toml").exists():
        command = "cargo test"
    else:
        raise ArgError(
            "could not detect a test command for this workspace; "
            "use run_shell with the command you want"
        )
    if target:
        command += f" {target}"

    timeout = int(args.get("timeout", 120))
    if timeout < 1 or timeout > 600:
        raise ArgError(f"timeout must be in [1, 600], got {timeout}")

    return lambda: f"$ {command}\n" + _shell_result(registry, command, timeout)


@agent_tool(
    name="git_status",
    description="Show which files in the workspace have been modified.",
    schema={},
    example="arguments: {}",
    profile="standard",
    volatile=True,
)
def git_status_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    def execute() -> str:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=registry.root,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        return result.stdout.strip() or result.stderr.strip() or "(clean)"

    return execute


@agent_tool(
    name="git_diff",
    description="Show uncommitted changes as a diff.",
    schema={"path": "str=|Optional file to diff, relative to the repo root"},
    example='arguments: {"path": "tools.py"}',
    profile="standard",
    volatile=True,
)
def git_diff_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    target = str(args.get("path", "")).strip()
    if target:
        registry._path(target)

    def execute() -> str:
        command = ["git", "diff"]
        if target:
            command += ["--", target]
        result = subprocess.run(
            command,
            cwd=registry.root,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        output = result.stdout.strip()
        if not output:
            return result.stderr.strip() or "(no uncommitted changes)"
        return registry.clip(output, "diff a single file instead")

    return execute


# --- Writing ---


@agent_tool(
    name="write_file",
    description=(
        "Write a text file, replacing any existing content. "
        "For long files write only the first part, then continue with append_file."
    ),
    schema={
        "path": "str|File path relative to the repo root",
        "content": "str|Full text content of the file",
    },
    risky=True,
    example='arguments: {"path": "binary_search.py", "content": "def binary_search(nums, target):\\n    return -1\\n"}',
    profile="minimal",
)
def write_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._write_path(args.get("path"))
    if path.is_dir():
        raise ArgError(f"{path.relative_to(registry.root)} is a directory")
    if "content" not in args:
        raise ArgError("missing content")

    content, notes = normalize_content(str(args["content"]), path.suffix)
    existing = (
        path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    )

    if existing is not None and existing.strip():
        if existing == content:
            raise ArgError(
                f"{path.relative_to(registry.root)} already has exactly this content; "
                "nothing to do"
            )
        registry._require_seen(path, len(existing.splitlines()))

    def execute() -> str:
        backup = registry._backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        parts = [
            f"wrote {path.relative_to(registry.root)} "
            f"({len(content.splitlines())} lines, {len(content)} chars)"
        ]
        check = validate_syntax(path.name, content)
        if check:
            parts.append(check)
        if backup:
            parts.append(f"previous version saved to {backup}")
        return "\n".join(parts + notes)

    return execute


@agent_tool(
    name="patch_file",
    description="Replace one exact text block in a file.",
    schema={
        "path": "str|File path relative to the repo root",
        "old_text": "str|Exact existing text to replace; must occur exactly once",
        "new_text": "str|Replacement text",
    },
    risky=True,
    example='arguments: {"path": "binary_search.py", "old_text": "return -1", "new_text": "return mid"}',
    profile="minimal",
)
def patch_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._existing_file(args.get("path"))
    registry._guard_write_target(path)

    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ArgError("old_text must not be empty")
    if "new_text" not in args:
        raise ArgError("missing new_text")
    new_text = str(args["new_text"])
    if old_text == new_text:
        raise ArgError("old_text and new_text are identical; nothing to change")

    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(registry.root)
    note = ""

    def locate(needle: str) -> tuple[int, int] | None:
        starts = match_lines(text, needle)
        if len(starts) == 1:
            index = text.find(needle)
            return index, index + len(needle)
        if len(starts) > 1:
            raise NoMatchError(
                f"old_text occurs {len(starts)} times in {relative} "
                f"(lines {', '.join(str(line) for line in starts)}); "
                "include surrounding lines to make it unique, "
                "or use replace_lines with the line range you mean"
            )
        return None

    span = locate(old_text)

    # Retry without the "  12: " prefixes rather than stripping them up front:
    # a file that genuinely contains such text still patches correctly.
    if span is None:
        stripped = strip_line_numbers(old_text)
        if stripped != old_text:
            span = locate(stripped)
            if span is not None:
                old_text = stripped
                note = "note: ignored line-number prefixes in old_text"

    # Indentation is what small models get wrong most; match on words first,
    # then splice using the file's own text.
    if span is None:
        span = find_fuzzy_match(text, old_text)
        if span is not None:
            note = (
                "note: matched old_text ignoring indentation; "
                "copy text exactly from read_file to avoid this"
            )

    if span is None:
        detail = f"old_text not found in {relative}"
        closest = nearest_block(text, old_text)
        if closest:
            detail += f". Closest text in the file:\n{closest}\nCopy it exactly."
        else:
            detail += ". Read the file first and copy the text exactly."
        raise NoMatchError(detail)

    start, end = span
    fingerprint = text[start:end]

    def execute() -> str:
        # Re-read at execution time: approval can sit in front of a human for
        # a while, and the validation snapshot may be stale by now.
        current = path.read_text(encoding="utf-8")
        if current[start:end] != fingerprint:
            index = current.find(fingerprint)
            if index == -1 or current.count(fingerprint) != 1:
                raise NoMatchError(
                    f"{relative} changed since this patch was prepared; "
                    "read it again and retry"
                )
            local_start, local_end = index, index + len(fingerprint)
        else:
            local_start, local_end = start, end

        backup = registry._backup(path)
        updated = current[:local_start] + new_text + current[local_end:]
        path.write_text(updated, encoding="utf-8")
        parts = [
            f"patched {relative} at line {current.count(chr(10), 0, local_start) + 1}"
        ]
        check = validate_syntax(path.name, updated)
        if check:
            parts.append(check)
        if backup:
            parts.append(f"previous version saved to {backup}")
        if note:
            parts.append(note)
        return "\n".join(parts)

    return execute


@agent_tool(
    name="replace_lines",
    description=(
        "Replace a range of lines in a file with new text. Use the line numbers "
        "shown by read_file or read_file_range; easier to get right than patch_file."
    ),
    schema={
        "path": "str|File path relative to the repo root",
        "start": "int|First line to replace (1-based, inclusive)",
        "end": "int|Last line to replace (inclusive)",
        "content": "str|Replacement text for those lines, without line numbers",
    },
    risky=True,
    example='arguments: {"path": "binary_search.py", "start": 4, "end": 5, "content": "    return mid\\n"}',
    profile="standard",
)
def replace_lines_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._existing_file(args.get("path"))
    registry._guard_write_target(path)

    for field in ("start", "end"):
        if field not in args:
            raise ArgError(f"missing {field}")
    start = int(args["start"])
    end = int(args["end"])
    if "content" not in args:
        raise ArgError("missing content")

    content, notes = normalize_content(str(args["content"]), path.suffix)

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    total = len(lines)
    relative = path.relative_to(registry.root)

    if start < 1:
        raise ArgError("start must be 1 or greater")
    if end < start:
        raise ArgError(f"end ({end}) is before start ({start})")
    if start > total:
        raise ArgError(
            f"start ({start}) is past the end of {relative}, which has {total} lines; "
            "use append_file to add to the end"
        )
    if end > total:
        notes.append(f"note: clamped end from {end} to {total}, the last line")
        end = total

    replaced = "".join(lines[start - 1 : end])
    if content and not content.endswith("\n"):
        content += "\n"

    def execute() -> str:
        current = path.read_text(encoding="utf-8")
        current_lines = current.splitlines(keepends=True)
        if "".join(current_lines[start - 1 : end]) != replaced:
            raise NoMatchError(
                f"{relative} changed since these line numbers were read; "
                "read it again and retry"
            )
        backup = registry._backup(path)
        updated = "".join(current_lines[: start - 1] + [content] + current_lines[end:])
        path.write_text(updated, encoding="utf-8")
        parts = [
            f"replaced lines {start}-{end} of {relative} "
            f"({end - start + 1} lines -> {len(content.splitlines())} lines)"
        ]
        check = validate_syntax(path.name, updated)
        if check:
            parts.append(check)
        if backup:
            parts.append(f"previous version saved to {backup}")
        return "\n".join(parts + notes)

    return execute


@agent_tool(
    name="append_file",
    description=(
        "Append text to the end of a file, creating it if missing. "
        "Use it to build long files piece by piece after an initial write_file."
    ),
    schema={
        "path": "str|File path relative to the repo root",
        "content": "str|Text appended verbatim to the end of the file",
    },
    risky=True,
    example='arguments: {"path": "binary_search.py", "content": "\\n\\ndef test_empty():\\n    assert binary_search([], 1) == -1\\n"}',
    profile="minimal",
)
def append_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._write_path(args.get("path"))
    if path.is_dir():
        raise ArgError(f"{path.relative_to(registry.root)} is a directory")
    if "content" not in args:
        raise ArgError("missing content")

    content, notes = normalize_content(str(args["content"]), path.suffix)
    if not content:
        raise ArgError("content must not be empty")

    def execute() -> str:
        backup = registry._backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        total = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        parts = [
            f"appended {len(content)} chars to {path.relative_to(registry.root)} "
            f"(now {total} lines)"
        ]
        # A file assembled across several appends is only valid once the last
        # one lands, so a warning here is expected mid-build and useful at the end.
        check = validate_syntax(path.name, path.read_text(encoding="utf-8"))
        if check:
            parts.append(check)
        if backup:
            parts.append(f"previous version saved to {backup}")
        return "\n".join(parts + notes)

    return execute


@agent_tool(
    name="move_file",
    description="Move or rename a file inside the workspace.",
    schema={
        "path": "str|Existing file path relative to the repo root",
        "destination": "str|New path relative to the repo root",
    },
    risky=True,
    example='arguments: {"path": "old.py", "destination": "new.py"}',
    profile="standard",
)
def move_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    source = registry._existing_file(args.get("path"))
    registry._guard_write_target(source)
    destination = registry._write_path(args.get("destination"))
    if destination.exists():
        raise ArgError(
            f"{destination.relative_to(registry.root)} already exists; "
            "delete it first or choose another name"
        )

    def execute() -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return (
            f"moved {source.relative_to(registry.root)} -> "
            f"{destination.relative_to(registry.root)}"
        )

    return execute


@agent_tool(
    name="delete_file",
    description="Delete a file from the workspace.",
    schema={"path": "str|File path relative to the repo root"},
    risky=True,
    example='arguments: {"path": "scratch.py"}',
    profile="standard",
)
def delete_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._existing_file(args.get("path"))
    registry._guard_write_target(path)

    def execute() -> str:
        backup = registry._backup(path)
        path.unlink()
        message = f"deleted {path.relative_to(registry.root)}"
        return f"{message}\nsaved to {backup}" if backup else message

    return execute


@agent_tool(
    name="revert_file",
    description="Restore a file from the most recent automatic backup.",
    schema={"path": "str|File path relative to the repo root"},
    risky=True,
    example='arguments: {"path": "binary_search.py"}',
    profile="full",
)
def revert_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._write_path(args.get("path"))
    relative = path.relative_to(registry.root)
    backups = registry._backups_for(path)
    if not backups:
        raise NotFoundError(f"no backup found for {relative}")
    latest = backups[-1]

    def execute() -> str:
        shutil.copy2(latest, path)
        return (
            f"restored {relative} from "
            f"{latest.relative_to(registry.root)} "
            f"({len(path.read_text(encoding='utf-8', errors='replace').splitlines())} lines)"
        )

    return execute


@agent_tool(
    name="delegate",
    description="Ask a bounded read-only child agent to investigate.",
    schema={
        "task": "str|Question for the child agent to investigate",
        "max_steps": "int=3|Tool budget for the child agent",
    },
    example='arguments: {"task": "inspect README.md", "max_steps": 3}',
    profile="full",
)
def delegate_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    if registry.delegate_fn is None:
        raise ArgError("delegate function not configured")

    task = str(args.get("task", "")).strip()
    if not task:
        raise ArgError("task must not be empty")

    max_steps = int(args.get("max_steps", 3))

    def execute() -> str:
        return registry.delegate_fn(task, max_steps)

    return execute


# --- Core Tool Registry ---


class ToolRegistry:
    def __init__(
        self,
        workspace: WorkspaceContext,
        root: Path,
        approval_policy: str,
        read_only: bool,
        depth: int,
        max_depth: int,
        get_history: Callable[[], List[HistoryEntry]],
        delegate_fn: Callable[[str, int], str] | None = None,
        logger: AgentLogger | None = None,
        approval_fn: Callable[[str, Dict], bool] | None = None,
        profile: str = "standard",
        max_noisy_output: int = DEFAULT_MAX_NOISY_OUTPUT,
        require_read_before_overwrite: bool = True,
    ) -> None:
        self.workspace = workspace
        self.root = root
        self.approval_policy = approval_policy
        self.read_only = read_only
        self.depth = depth
        self.max_depth = max_depth
        self.get_history = get_history
        self.delegate_fn = delegate_fn
        self.logger = logger or AgentLogger(None, enabled=False)
        self.approval_fn = approval_fn
        self.profile = profile if profile in TOOL_PROFILES else "standard"
        self.max_noisy_output = max_noisy_output
        self.require_read_before_overwrite = require_read_before_overwrite
        self._registry: Tools = self._build()

    def items(self):
        return self._registry.items()

    def clip(self, text: str, hint: str = "") -> str:
        """Bound output that is noise rather than payload. Never used on reads."""
        return clip(text, self.max_noisy_output, hint)

    _JSON_TYPES = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }

    @staticmethod
    def _split_spec(spec: str) -> tuple[str, str, bool, str]:
        """Split a "type[=default]|description" field spec.

        Returns (type token, default token, required, description).
        """
        head, _, description = str(spec).partition("|")
        type_token, sep, default = head.partition("=")
        return type_token.strip(), default.strip(), sep == "", description.strip()

    @classmethod
    def _parse_default(cls, json_type: str, token: str):
        try:
            if json_type == "integer":
                return int(token)
            if json_type == "number":
                return float(token)
        except ValueError:
            # A default that does not parse as its own type would put a string
            # on an integer property and teach the model the wrong shape.
            return 0
        if json_type == "boolean":
            return token.lower() == "true"
        return token.strip("'\"")

    @classmethod
    def _field_schema(cls, spec: str) -> tuple[dict, bool]:
        """Convert a "type[=default]|description" field spec into a JSON-schema
        property.

        Returns the property schema and whether the field is required.
        """
        type_token, default, required, description = cls._split_spec(spec)
        json_type = cls._JSON_TYPES.get(type_token, "string")
        prop = {"type": json_type}
        if description:
            prop["description"] = description
        if not required:
            prop["default"] = cls._parse_default(json_type, default)
        return prop, required

    def schemas(self) -> List[Dict]:
        """Return the registered tools as OpenAI-style JSON-schema definitions."""
        definitions = []
        for name, tool in self._registry.items():
            properties = {}
            required = []
            for field, spec in tool.schema.items():
                prop, is_required = self._field_schema(spec)
                properties[field] = prop
                if is_required:
                    required.append(field)
            # Priming with an example beats correcting after a failure.
            description = tool.description
            example = _TOOL_EXAMPLES.get(name)
            if example:
                description += f" Example {example}"
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return definitions

    def _coerce_args(self, tool: ToolDescriptionEntry, args: Dict) -> Dict:
        """Coerce scalar arguments to their declared types.

        Small models frequently send "20" for an int or 1 for a str; a
        wrong-but-unambiguous type should not burn a retry. Values that cannot
        be coerced are left as-is so the tool's own validation reports them.
        """
        coerced = dict(args)
        for field, spec in tool.schema.items():
            if field not in coerced:
                continue
            type_token = self._split_spec(spec)[0]
            value = coerced[field]
            try:
                if type_token == "int" and isinstance(value, str):
                    coerced[field] = int(value.strip())
                elif (
                    type_token == "int"
                    and isinstance(value, float)
                    and value.is_integer()
                ):
                    coerced[field] = int(value)
                elif type_token == "float" and isinstance(value, (str, int)):
                    coerced[field] = float(value)
                elif type_token == "bool" and isinstance(value, str):
                    coerced[field] = value.strip().lower() in {"true", "yes", "1"}
                elif type_token == "str" and isinstance(value, (int, float, bool)):
                    coerced[field] = str(value)
            except ValueError:
                continue
        return coerced

    def normalize_args(self, tool: ToolDescriptionEntry, args: Dict) -> tuple[Dict, List[str]]:
        """Repair the shape of a tool call before validating it.

        Unwraps argument envelopes, renames known aliases onto the fields this
        tool declares, joins content sent as a list of lines, and coerces
        scalar types. Returns the repaired arguments and notes describing what
        was changed, so the model can learn the right shape.
        """
        notes: List[str] = []
        fields = list(tool.schema.keys())

        unwrapped = unwrap_envelope(args)
        if unwrapped is not args and unwrapped != args:
            notes.append("unwrapped the arguments envelope")

        aliased, renames = apply_arg_aliases(fields, unwrapped)
        for alias, field in renames:
            notes.append(f"read '{alias}' as '{field}'")

        if "content" in aliased:
            joined, note = coerce_list_content(aliased["content"])
            if note:
                aliased["content"] = joined
                notes.append(note)

        return self._coerce_args(tool, aliased), notes

    def run(self, name: str, args: Dict) -> str:
        """Resolve, repair, validate, approve and execute one tool call."""
        result, outcome = self._run(name, args or {})
        self.logger.log(
            "tool_outcome", name=name, outcome=outcome, chars=len(result)
        )
        return result

    def _run(self, name: str, args: Dict) -> tuple[str, str]:
        resolved, note = resolve_tool_name(name, list(self._registry.keys()))
        if not resolved:
            self.logger.log("tool_unknown", name=name, args=args, message=note)
            return note, "unknown_tool"
        if note:
            self.logger.log("tool_name_recovered", requested=name, resolved=resolved)

        tool = self._registry[resolved]
        try:
            args, arg_notes = self.normalize_args(tool, args)
        except Exception as exc:
            return self._tool_error(resolved, args, exc), "arg_error"
        if arg_notes:
            self.logger.log(
                "tool_args_normalized", name=resolved, notes=arg_notes, args=args
            )

        prefix = "\n".join(filter(None, [note, *(f"note: {item}" for item in arg_notes)]))
        prefix = prefix + "\n" if prefix else ""

        repeat = self._repeated_call(resolved, args)
        if repeat:
            self.logger.log(
                "tool_blocked", name=resolved, args=args, reason="repeated_call"
            )
            return prefix + repeat, "repeated"

        try:
            # Validate the arguments first so approval is only requested for
            # calls that can actually execute.
            execute = tool.run(args)
        except Exception as exc:
            return prefix + self._tool_error(resolved, args, exc), self._outcome(exc)

        if tool.risky:
            approved = self._approve(resolved, args)
            self.logger.log(
                "tool_approval",
                name=resolved,
                args=args,
                risky=True,
                policy=self.approval_policy,
                read_only=self.read_only,
                granted=approved,
            )
            if not approved:
                reason = (
                    "this session is read-only"
                    if self.read_only
                    else "the user did not approve it"
                )
                return (
                    f"{prefix}error: approval denied for {resolved} ({reason}). "
                    "Do not retry; use a read-only tool or give your final answer.",
                    "denied",
                )

        try:
            return prefix + execute(), "ok"
        except Exception as exc:
            return prefix + self._tool_error(resolved, args, exc), self._outcome(exc)

    @staticmethod
    def _outcome(exc: Exception) -> str:
        return getattr(exc, "outcome", "error")

    def _tool_error(self, name: str, args: Dict, exc: Exception) -> str:
        self.logger.log(
            "tool_error",
            name=name,
            args=args,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        message = f"error: tool {name} failed: {exc}"
        # An argument mistake is worth answering with the tool's actual shape;
        # a not-found or no-match error already carries its own guidance.
        if isinstance(exc, ArgError) or not isinstance(exc, ToolError):
            tool = self._registry.get(name)
            if tool is not None:
                fields = ", ".join(
                    f"{field} ({self._split_spec(spec)[0]}"
                    + ("" if self._split_spec(spec)[2] else ", optional")
                    + ")"
                    for field, spec in tool.schema.items()
                )
                message += f"\narguments: {fields}"
            example = _TOOL_EXAMPLES.get(name, "")
            if example:
                message += f"\nexample: {example}"
        return message

    def _build(self) -> Tools:
        tools = {}
        allowed = TOOL_PROFILES[: TOOL_PROFILES.index(self.profile) + 1]
        for name, definition in _TOOL_CATALOG.items():
            if definition["profile"] not in allowed:
                continue
            if name == "delegate" and self.delegate_fn is None:
                continue

            func = definition["func"]

            # Closure to inject the registry instance into the tool function
            def make_run(f):
                def run_wrapper(args: Dict) -> Callable[[], str]:
                    args = args or {}
                    return f(args, registry=self)

                return run_wrapper

            tools[name] = ToolDescriptionEntry(
                schema=definition["schema"],
                risky=definition["risky"],
                description=definition["description"],
                run=make_run(func),
            )
        return tools

    # --- Paths ---

    def _path_is_within_root(self, resolved: Path) -> bool:
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def _path(self, raw_path) -> Path:
        raw = str(raw_path if raw_path is not None else "").strip()
        if not raw:
            raise ArgError("missing path")
        candidate = Path(raw).expanduser()
        resolved = (
            candidate if candidate.is_absolute() else self.root / candidate
        ).resolve()

        # Models often prefix the repo directory name onto a repo-relative
        # path ("harness-lab/tools.py"); retry without it before failing.
        if not resolved.exists() and candidate.parts[:1] == (self.root.name,):
            retry = (self.root / Path(*candidate.parts[1:])).resolve()
            if retry.exists() and self._path_is_within_root(retry):
                return retry

        if not self._path_is_within_root(resolved):
            raise ArgError(
                f"path escapes the workspace: {raw}. "
                f"Use a path relative to the repo root ({self.root.name})."
            )
        return resolved

    def _existing_file(self, raw_path) -> Path:
        """Resolve a path that must already be a file, with a helpful miss."""
        path = self._path(raw_path)
        if path.is_file():
            return path
        relative_raw = str(raw_path).strip()
        if path.is_dir():
            raise ArgError(
                f"{path.relative_to(self.root)} is a directory, not a file; "
                "use list_files to see what is in it"
            )
        # "path is not a file" leaves the model guessing; name real candidates.
        candidates = []
        for item in self.root.rglob(Path(relative_raw).name):
            if not item.is_file():
                continue
            item_relative = item.relative_to(self.root)
            if any(part in IGNORED_PATH_NAMES for part in item_relative.parts):
                continue
            candidates.append(str(item_relative))
            if len(candidates) >= 5:
                break
        message = f"file not found: {relative_raw}"
        if candidates:
            message += f". Did you mean: {', '.join(sorted(candidates))}?"
        else:
            message += ". Use list_files or find_files to locate it."
        raise NotFoundError(message)

    def _guard_write_target(self, path: Path) -> None:
        """Refuse writes to git internals and harness state."""
        try:
            parts = path.relative_to(self.root).parts
        except ValueError:
            raise ArgError(f"path escapes the workspace: {path}")
        for blocked in (".git", STATE_DIR_NAME):
            if blocked in parts:
                raise DeniedError(
                    f"refusing to modify {blocked}/; it is not part of the project's "
                    "source. Edit project files instead."
                )

    def _write_path(self, raw_path) -> Path:
        path = self._path(raw_path)
        self._guard_write_target(path)
        return path

    # --- Write safety ---

    def _read_paths(self) -> set:
        """Files whose content this session has already put in front of the model."""
        seen = set()
        for item in self.get_history():
            if not isinstance(item, ToolMessageEntry):
                continue
            if item.name not in READ_TOOLS + WRITE_TOOLS + ("outline_file",):
                continue
            if str(item.content).startswith("error:"):
                continue
            raw = item.args.get("path")
            if not raw:
                continue
            try:
                seen.add(self._path(raw))
            except ToolError:
                continue
        return seen

    def _require_seen(self, path: Path, line_count: int) -> None:
        """Refuse to blind-overwrite a file this session has never looked at.

        Replacing a file the model has not read is how existing work gets
        silently destroyed: the model writes what it imagines the file contains.
        Disable with --no-write-guard to measure the difference.
        """
        if not self.require_read_before_overwrite:
            return
        if path in self._read_paths():
            return
        relative = path.relative_to(self.root)
        raise GuardError(
            f"{relative} already exists with {line_count} lines and has not been "
            "read in this session; writing it now would discard content you have "
            f"not seen. Read it first (read_file {relative}), then edit it with "
            "patch_file or replace_lines, or write_file once you know what is there."
        )

    def _backup_dir(self) -> Path:
        return self.root / STATE_DIR_NAME / BACKUP_DIR_NAME

    @staticmethod
    def _backup_stem(relative: Path) -> str:
        return str(relative).replace("/", "__")

    def _backup(self, path: Path) -> str:
        """Snapshot a file before changing it; returns the relative backup path.

        Cheap insurance that also makes a bad run diffable after the fact
        instead of only visible as damage.
        """
        if not path.is_file():
            return ""
        try:
            relative = path.relative_to(self.root)
            destination = self._backup_dir()
            destination.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            target = destination / f"{self._backup_stem(relative)}.{stamp}"
            shutil.copy2(path, target)
            return str(target.relative_to(self.root))
        except OSError as exc:
            self.logger.log("backup_failed", path=str(path), error=str(exc))
            return ""

    def _backups_for(self, path: Path) -> List[Path]:
        directory = self._backup_dir()
        if not directory.is_dir():
            return []
        stem = self._backup_stem(path.relative_to(self.root))
        return sorted(item for item in directory.glob(f"{stem}.*") if item.is_file())

    # --- Approval & loop breaking ---

    def _approve(self, name: str, args: Dict) -> bool:
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        if self.approval_fn is not None:
            return bool(self.approval_fn(name, args))
        try:
            answer = input(
                f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] "
            )
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    def _repeated_call(self, name: str, args: Dict) -> str:
        """Detect a call that cannot teach the model anything new.

        Catches an immediate identical repeat on the *second* call rather than
        the third, and the A-B-A-B two-cycle that a single-step check misses.
        Tools whose output legitimately changes between calls (tests, git,
        shell) are exempt from the cycle check -- re-running tests after an
        edit is the intended workflow.
        """
        events = [
            item for item in self.get_history() if isinstance(item, ToolMessageEntry)
        ]
        if not events:
            return ""

        def same(event: ToolMessageEntry, other_name: str, other_args: Dict) -> bool:
            return event.name == other_name and event.args == other_args

        if same(events[-1], name, args):
            return (
                f"error: {name} was just called with these exact arguments and "
                "returned the result above; calling it again cannot tell you "
                "anything new. Use a different tool or give your final answer."
            )

        if name not in _VOLATILE_TOOLS and len(events) >= 3:
            if (
                same(events[-2], name, args)
                and not same(events[-1], name, args)
                and events[-3].name == events[-1].name
                and events[-3].args == events[-1].args
            ):
                return (
                    f"error: you are alternating between {events[-1].name} and "
                    f"{name} with the same arguments without making progress. "
                    "Change what you are doing or give your final answer."
                )
        return ""
