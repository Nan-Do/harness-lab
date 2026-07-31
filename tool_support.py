"""Argument normalization, output shaping, and error types for the tool layer.

Small models get the *shape* of a tool call wrong far more often than they get
the intent wrong: the right idea arrives as {"file_path": ...} instead of
{"path": ...}, wrapped in an {"arguments": {...}} envelope, with the replacement
text still carrying the "  12: " prefixes the model just read out of
`read_file`. Every helper here turns one of those near-misses into a valid call
instead of a burned step, and every failure it cannot repair is reported with
the exact text needed to get the next call right.

The functions are deliberately pure so they can be tested without a model or a
workspace; the registry in tools.py wires them together.
"""

import ast
import difflib
import json
import re
from typing import Dict, List, Sequence, Tuple


# --- Error types -------------------------------------------------------------
# Subclasses of ValueError so existing `except ValueError` paths still catch
# them; `outcome` classifies the failure for the run log so logviz can report a
# per-tool failure rate broken down by *kind* of failure.


class ToolError(ValueError):
    outcome = "error"


class ArgError(ToolError):
    outcome = "arg_error"


class NotFoundError(ToolError):
    outcome = "not_found"


class NoMatchError(ToolError):
    outcome = "no_match"


class GuardError(ToolError):
    outcome = "guard_blocked"


class DeniedError(ToolError):
    outcome = "denied"


# --- Tool name recovery ------------------------------------------------------

# What small models reach for when they have not internalised this tool set:
# shell muscle memory (cat/ls/grep) and names from other harnesses
# (str_replace_editor, apply_patch). Resolving these costs nothing and saves a
# whole step plus a retry.
TOOL_NAME_ALIASES = {
    "cat": "read_file",
    "view": "read_file",
    "view_file": "read_file",
    "open": "read_file",
    "open_file": "read_file",
    "read": "read_file",
    "show": "read_file",
    "show_file": "read_file",
    "get_file": "read_file",
    "file_read": "read_file",
    "ls": "list_files",
    "dir": "list_files",
    "list": "list_files",
    "list_dir": "list_files",
    "list_directory": "list_files",
    "tree": "list_files",
    "grep": "search",
    "rg": "search",
    "ripgrep": "search",
    "search_files": "search",
    "search_text": "search",
    "find_text": "search",
    "find_in_files": "search",
    "find": "find_files",
    "glob": "find_files",
    "find_file": "find_files",
    "locate": "find_files",
    "bash": "run_shell",
    "sh": "run_shell",
    "shell": "run_shell",
    "exec": "run_shell",
    "execute": "run_shell",
    "terminal": "run_shell",
    "run": "run_shell",
    "run_command": "run_shell",
    "command": "run_shell",
    "write": "write_file",
    "create_file": "write_file",
    "new_file": "write_file",
    "save_file": "write_file",
    "put_file": "write_file",
    "file_write": "write_file",
    "edit": "patch_file",
    "edit_file": "patch_file",
    "str_replace": "patch_file",
    "str_replace_editor": "patch_file",
    "apply_patch": "patch_file",
    "modify_file": "patch_file",
    "replace_text": "patch_file",
    "append": "append_file",
    "add_to_file": "append_file",
    "outline": "outline_file",
    "symbols": "outline_file",
    "summarize_file": "outline_file",
    "replace_range": "replace_lines",
    "edit_lines": "replace_lines",
    "test": "run_tests",
    "tests": "run_tests",
    "pytest": "run_tests",
    "run_test": "run_tests",
    "diff": "git_diff",
    "status": "git_status",
    "rm": "delete_file",
    "del": "delete_file",
    "remove_file": "delete_file",
    "mv": "move_file",
    "rename": "move_file",
    "rename_file": "move_file",
}


def resolve_tool_name(name: str, known: Sequence[str]) -> Tuple[str, str]:
    """Map a model-invented tool name onto a registered one.

    Returns (resolved name or "", note). The note is prepended to the tool
    result so the model learns the canonical name instead of silently relying
    on the alias.
    """
    raw = str(name or "")
    if raw in known:
        return raw, ""

    canonical = raw.strip().strip("/").lower().replace("-", "_").replace(" ", "_")
    if canonical in known:
        return canonical, f"note: tool names are case-sensitive; used '{canonical}'."

    target = TOOL_NAME_ALIASES.get(canonical)
    if target and target in known:
        return (
            target,
            f"note: '{raw}' is not a tool here; ran '{target}' instead. "
            f"Use '{target}' next time.",
        )

    close = difflib.get_close_matches(canonical, list(known), n=3, cutoff=0.6)
    if close:
        return "", f"error: unknown tool '{raw}'; did you mean {' or '.join(close)}?"
    return (
        "",
        f"error: unknown tool '{raw}'. Available tools: {', '.join(sorted(known))}.",
    )


# --- Argument normalization --------------------------------------------------

# Wrappers models add when they confuse the function-calling envelope with the
# arguments themselves.
ENVELOPE_KEYS = (
    "arguments",
    "args",
    "input",
    "inputs",
    "parameters",
    "params",
    "kwargs",
    "tool_input",
    "payload",
)

# Alias -> canonical, applied only when the canonical field belongs to the tool
# being called, so "search" can mean `pattern` for search and `old_text` for
# patch_file without ambiguity.
ARG_ALIASES: Dict[str, Tuple[str, ...]] = {
    "path": (
        "file_path",
        "filepath",
        "filename",
        "file_name",
        "file",
        "pathname",
        "target",
        "target_file",
        "directory",
        "dir",
        "folder",
    ),
    "content": (
        "text",
        "body",
        "data",
        "contents",
        "file_content",
        "file_contents",
        "new_content",
        "source",
        "code",
    ),
    "pattern": (
        "query",
        "regex",
        "q",
        "search",
        "search_pattern",
        "search_string",
        "text",
        "needle",
        "string",
    ),
    "command": (
        "cmd",
        "shell_command",
        "command_line",
        "commandline",
        "script",
        "bash_command",
    ),
    "old_text": (
        "old",
        "old_str",
        "old_string",
        "search",
        "find",
        "from",
        "before",
        "original",
        "target_text",
    ),
    "new_text": (
        "new",
        "new_str",
        "new_string",
        "replace",
        "replacement",
        "to",
        "after",
        "replace_with",
    ),
    "start": ("start_line", "from_line", "begin", "first_line", "line_start", "line"),
    "end": ("end_line", "to_line", "last_line", "line_end", "stop"),
    "task": ("question", "prompt", "instruction", "goal", "request"),
    "max_steps": ("steps", "budget", "step_budget"),
    "timeout": ("timeout_seconds", "timeout_sec", "seconds", "time_limit"),
    "depth": ("levels", "recursion", "recursive_depth", "max_levels"),
    "glob": ("include", "file_glob", "filter", "extension"),
    "destination": ("dest", "to_path", "new_path", "target_path", "dst"),
}


def unwrap_envelope(args: Dict) -> Dict:
    """Unwrap {"arguments": {...}} style envelopes, including JSON-string ones."""
    current = args
    for _ in range(3):
        if not isinstance(current, dict) or len(current) != 1:
            break
        key, value = next(iter(current.items()))
        if str(key).lower() not in ENVELOPE_KEYS:
            break
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                break
        if not isinstance(value, dict):
            break
        current = value
    return current if isinstance(current, dict) else args


def apply_arg_aliases(
    fields: Sequence[str], args: Dict
) -> Tuple[Dict, List[Tuple[str, str]]]:
    """Rename known alias keys onto the fields this tool actually declares."""
    result = dict(args)
    renames: List[Tuple[str, str]] = []
    for field in fields:
        if result.get(field) not in (None, ""):
            continue
        for alias in ARG_ALIASES.get(field, ()):
            # Never steal a key the tool declares in its own right.
            if alias in fields or alias not in result:
                continue
            value = result.pop(alias)
            if value in (None, ""):
                continue
            result[field] = value
            renames.append((alias, field))
            break
    return result, renames


# Matches the "  12: " / "12| " prefixes that read_file and read_file_range put
# in front of every line.
LINE_NUMBER_RE = re.compile(r"^\s*(\d+)\s*[:|]\s?")


def strip_line_numbers(text: str) -> str:
    """Remove a line-number prefix from every line that carries one."""
    return "\n".join(LINE_NUMBER_RE.sub("", line) for line in str(text).split("\n"))


def looks_line_numbered(text: str) -> bool:
    """True when text is clearly a paste of numbered tool output.

    Requires at least two numbered lines with consecutively increasing numbers,
    so genuine content like "3: see appendix" is not mangled. Single-line cases
    are handled by retrying the match instead of guessing (see
    tools.patch_file_tool).
    """
    numbers: List[int] = []
    for line in str(text).split("\n"):
        if not line.strip():
            continue
        match = LINE_NUMBER_RE.match(line)
        if not match:
            return False
        numbers.append(int(match.group(1)))
    if len(numbers) < 2:
        return False
    return all(
        later == earlier + 1 for earlier, later in zip(numbers, numbers[1:])
    )


# Non-greedy so the captured body keeps its own trailing newline: a file should
# end with one, and the fence is the only thing being removed.
FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)```[ \t]*$", re.DOTALL)


def strip_code_fence(text: str, suffix: str = "") -> str:
    """Unwrap a whole-content markdown code fence.

    Models often answer "write this file" with a fenced block. Skipped for
    markdown targets, where a file that is one big fence is plausible content.
    """
    if suffix.lower() in {".md", ".markdown", ".mdx", ".rst"}:
        return text
    match = FENCE_RE.match(str(text).strip())
    return match.group(1) if match else text


def normalize_content(text: str, suffix: str = "") -> Tuple[str, List[str]]:
    """Repair the two ways models mangle file content, reporting what changed."""
    notes: List[str] = []
    result = str(text)

    unfenced = strip_code_fence(result, suffix)
    if unfenced != result:
        result = unfenced
        notes.append("removed a wrapping ``` code fence from content")

    if looks_line_numbered(result):
        result = strip_line_numbers(result)
        notes.append("stripped line-number prefixes from content")

    return result, notes


def coerce_list_content(value: object) -> Tuple[object, str]:
    """Join content sent as a list of lines into a single string."""
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float)) for item in value
    ):
        joined = "\n".join(str(item) for item in value)
        if joined and not joined.endswith("\n"):
            joined += "\n"
        return joined, "joined a list of lines into a single string"
    return value, ""


# --- Call presentation -------------------------------------------------------
# A front-end has to show what a tool is about to do without dumping the file
# it is about to write into a one-line label or a modal dialog. The split is
# always by size, never by argument name: whatever is short enough to read at a
# glance stays on the call line, and a body goes to a block of its own that the
# front-end renders (and streams) where there is room for it.


# Arguments known to carry a body, most-recently-written first. Only used to
# decide which one to follow while a call is still streaming; what counts as a
# body is decided by size below.
BODY_ARGS = ("content", "new_text", "old_text", "command", "task")

# Longer than this, or spanning lines, and an argument is a body, not a label.
INLINE_ARG_LIMIT = 120


def size_note(text: str) -> str:
    """Human-readable size of a body: what a reader needs to judge it."""
    body = str(text)
    lines = body.count("\n") + (1 if body and not body.endswith("\n") else 0)
    size = len(body.encode("utf-8", "replace"))
    unit = f"{size} B" if size < 1024 else f"{size / 1024:.1f} kB"
    return f"{lines} line{'s' if lines != 1 else ''}, {unit}"


def split_args(args: Dict) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split arguments into (inline labels, bodies worth their own block)."""
    inline: List[Tuple[str, str]] = []
    bodies: List[Tuple[str, str]] = []
    for key, value in (args or {}).items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if isinstance(value, str) and ("\n" in value or len(value) > INLINE_ARG_LIMIT):
            bodies.append((key, value))
        else:
            inline.append((key, text))
    return inline, bodies


def describe_call(name: str, args: Dict) -> str:
    """One line naming the call and its arguments, with bodies as sizes only.

    This is the text an approval prompt asks about: enough to decide on, and
    never the payload itself -- that is shown in the conversation, where it can
    be read as code instead of as a quoted JSON blob.
    """
    inline, bodies = split_args(args)
    parts = [f"{key}={value}" for key, value in inline]
    parts += [f"{key}: {size_note(body)}" for key, body in bodies]
    if not parts:
        return name
    return f"{name} " + " · ".join(parts)


def _decode_partial(body: str) -> str:
    """Decode a JSON string body that may be cut off mid-value or mid-escape."""
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    out: List[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == '"':
            break
        if char != "\\":
            out.append(char)
            index += 1
            continue
        if index + 1 >= len(body):
            break  # escape cut off by the chunk boundary; it arrives next time
        marker = body[index + 1]
        if marker in escapes:
            out.append(escapes[marker])
            index += 2
        elif marker == "u":
            if index + 6 > len(body):
                break
            try:
                out.append(chr(int(body[index + 2 : index + 6], 16)))
            except ValueError:
                break
            index += 6
        else:
            break
    return "".join(out)


def streaming_body(raw_args: str) -> Tuple[str, str]:
    """(argument name, text so far) of the body a streamed call is writing.

    The arguments arrive as JSON assembled a few characters at a time, so it is
    never parseable until the call is complete -- but the file being written is
    exactly what someone about to approve the write wants to watch. This reads
    the value being filled in right now straight out of the half-finished blob,
    and returns ("", "") until one starts.
    """
    raw = str(raw_args)
    found = [(raw.rfind(f'"{key}"'), key) for key in BODY_ARGS]
    at, key = max(found)
    if at < 0:
        return "", ""
    rest = raw[at + len(key) + 2 :]
    colon = rest.find(":")
    if colon < 0:
        return "", ""
    rest = rest[colon + 1 :].lstrip()
    if not rest.startswith('"'):
        return "", ""
    return key, _decode_partial(rest[1:])


# --- Output shaping ----------------------------------------------------------


def clip(text: str, limit: int, hint: str = "") -> str:
    """Keep the head and tail of noisy output within `limit` characters.

    Used only for output that is *noise* rather than payload -- shell logs and
    match floods -- never for file reads, which must reach the model whole (see
    agent.MiniAgent._fit_history_budget: content is sent whole or dropped
    whole, never truncated).
    """
    body = str(text)
    if limit <= 0 or len(body) <= limit:
        return body
    keep = max(limit - 200, limit // 2)
    head = keep * 2 // 3
    tail = keep - head
    dropped = len(body) - head - tail
    notice = f"\n\n[clipped {dropped} chars of {len(body)}"
    if hint:
        notice += f"; {hint}"
    notice += "]\n\n"
    return body[:head] + notice + body[-tail:]


# --- Post-write validation ---------------------------------------------------


def validate_syntax(name: str, text: str) -> str:
    """Cheap syntax check for a file just written.

    A write that came out truncated (a very common small-model failure, since
    the model is also fighting a token limit) otherwise looks like a success
    and is only discovered several steps later when something runs.
    """
    lowered = name.lower()
    try:
        if lowered.endswith(".py"):
            compile(text, name, "exec")
        elif lowered.endswith(".json"):
            json.loads(text)
        else:
            return ""
    except SyntaxError as exc:
        return f"WARNING invalid Python: line {exc.lineno}: {exc.msg}"
    except json.JSONDecodeError as exc:
        return f"WARNING invalid JSON: line {exc.lineno}: {exc.msg}"
    except ValueError as exc:
        return f"WARNING invalid content: {exc}"
    return "syntax OK"


# --- Structural outline ------------------------------------------------------

OUTLINE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?"
    r"(?:def|class|function|fn|struct|impl|interface|type|const|var|let|public|private|protected)\b"
    r"|^\s*[A-Za-z_$][\w$]*\s*[:=]\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>)"
)


def outline_python(text: str) -> List[str]:
    """Outline a Python file via ast: classes and their methods, and functions.

    Function bodies are not descended into: inner closures (every tool in
    tools.py defines an `execute`) would double the size of the outline without
    helping the model find anything.
    """
    tree = ast.parse(text)
    lines: List[str] = []

    def walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                lines.append(f"{'  ' * depth}{child.lineno:>5}: class {child.name}")
                walk(child, depth + 1)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ", ".join(item.arg for item in child.args.args)
                keyword = (
                    "async def" if isinstance(child, ast.AsyncFunctionDef) else "def"
                )
                lines.append(
                    f"{'  ' * depth}{child.lineno:>5}: {keyword} {child.name}({args})"
                )

    walk(tree, 0)
    return lines


def outline_generic(text: str) -> List[str]:
    """Regex outline for non-Python files: declaration-looking lines."""
    lines: List[str] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if len(line) > 200 or not line.strip():
            continue
        if OUTLINE_RE.match(line):
            lines.append(f"{number:>5}: {line.strip()[:120]}")
    return lines


# --- Patch diagnostics -------------------------------------------------------


def match_lines(text: str, needle: str) -> List[int]:
    """1-based line numbers where each occurrence of needle starts."""
    starts: List[int] = []
    index = text.find(needle)
    while index != -1:
        starts.append(text.count("\n", 0, index) + 1)
        index = text.find(needle, index + 1)
    return starts


def normalize_whitespace(text: str) -> str:
    """Drop trailing whitespace per line and collapse leading indentation."""
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def find_fuzzy_match(text: str, needle: str) -> Tuple[int, int] | None:
    """Locate needle in text ignoring indentation and trailing whitespace.

    Small models reproduce the *words* of a block reliably and its indentation
    unreliably, so an indentation-insensitive second pass converts a large
    class of dead-end failures into successful edits. Returns the (start, end)
    character span in the original text, or None when there is no unique match.
    """
    target = normalize_whitespace(needle)
    if not target:
        return None
    file_lines = text.split("\n")
    needle_height = len(needle.split("\n"))
    spans: List[Tuple[int, int]] = []

    for start in range(len(file_lines)):
        for height in {needle_height, needle_height + 1, max(needle_height - 1, 1)}:
            window = file_lines[start : start + height]
            if len(window) < height:
                continue
            if normalize_whitespace("\n".join(window)) != target:
                continue
            offset = sum(len(line) + 1 for line in file_lines[:start])
            spans.append((offset, offset + len("\n".join(window))))
            break

    unique = sorted(set(spans))
    return unique[0] if len(unique) == 1 else None


def nearest_block(text: str, needle: str, context: int = 2) -> str:
    """The region of text most similar to needle, with line numbers.

    Reporting "found 0 occurrences" teaches the model nothing; showing the
    closest actual text lets the next call copy it verbatim.
    """
    file_lines = text.split("\n")
    needle_lines = [line for line in needle.split("\n") if line.strip()]
    if not needle_lines or not file_lines:
        return ""

    anchor = needle_lines[0].strip()
    best_index, best_score = 0, 0.0
    for index, line in enumerate(file_lines):
        score = difflib.SequenceMatcher(None, line.strip(), anchor).ratio()
        if score > best_score:
            best_index, best_score = index, score
    if best_score < 0.5:
        return ""

    start = max(best_index - context, 0)
    end = min(best_index + len(needle_lines) + context, len(file_lines))
    return "\n".join(
        f"{number:>5}: {file_lines[number - 1]}" for number in range(start + 1, end + 1)
    )
