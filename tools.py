import json
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Callable, Dict, List

from agent_logging import AgentLogger
from workspace import WorkspaceContext
from utils import IGNORED_PATH_NAMES, clip
from app_types import (
    HistoryEntry,
    ToolMessageEntry,
    Tools,
    ToolDescriptionEntry,
)


# --- Global Tool Catalog & Decorator ---

_TOOL_CATALOG = {}
_TOOL_EXAMPLES = {}


def agent_tool(
    name: str,
    description: str,
    schema: Dict[str, str],
    risky: bool = False,
    example: str = "",
):
    """Decorator to register a tool function into the global catalog.

    Tool functions validate their arguments eagerly and return a zero-argument
    callable that performs the actual work, so the registry can reject
    malformed calls before asking for approval.
    """

    def decorator(func: Callable):
        _TOOL_CATALOG[name] = {
            "description": description,
            "schema": schema,
            "risky": risky,
            "func": func,
        }
        if example:
            _TOOL_EXAMPLES[name] = example
        return func

    return decorator


# --- Tool Implementations ---


@agent_tool(
    name="list_files",
    description="List files in the workspace.",
    schema={"path": "str='.'|Directory to list, relative to the repo root"},
    example='arguments: {"path": "."}',
)
def list_files_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    path = registry._path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")

    def execute() -> str:
        entries = [
            item
            for item in sorted(
                path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())
            )
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = []
        for entry in entries[:200]:
            kind = "[D]" if entry.is_dir() else "[F]"
            lines.append(f"{kind} {entry.relative_to(registry.root)}")
        return "\n".join(lines) or "(empty)"

    return execute


@agent_tool(
    name="read_file",
    description="Read a UTF-8 file by line range.",
    schema={
        "path": "str|File path relative to the repo root",
        "start": "int=1|First line to read (1-based)",
        "end": "int=200|Last line to read (inclusive)",
    },
    example='arguments: {"path": "README.md", "start": 1, "end": 80}',
)
def read_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    if "path" not in args:
        raise ValueError("missing path")
    path = registry._path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")

    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")

    def execute() -> str:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(
            f"{number:>4}: {line}"
            for number, line in enumerate(lines[start - 1 : end], start=start)
        )
        return f"# {path.relative_to(registry.root)}\n{body}"

    return execute


@agent_tool(
    name="search",
    description="Search the workspace with rg or a simple fallback.",
    schema={
        "pattern": "str|Text to search for",
        "path": "str='.'|File or directory to search in, relative to the repo root",
    },
    example='arguments: {"pattern": "binary_search", "path": "."}',
)
def search_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = registry._path(args.get("path", "."))

    def execute() -> str:
        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
                cwd=registry.root,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"

        matches = []
        files = (
            [path]
            if path.is_file()
            else [
                item
                for item in path.rglob("*")
                if item.is_file()
                and not any(
                    part in IGNORED_PATH_NAMES
                    for part in item.relative_to(registry.root).parts
                )
            ]
        )
        for file_path in files:
            for number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if pattern.lower() in line.lower():
                    matches.append(
                        f"{file_path.relative_to(registry.root)}:{number}:{line}"
                    )
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    return execute


@agent_tool(
    name="run_shell",
    description="Run a shell command in the repo root.",
    schema={
        "command": "str|Shell command executed at the repo root",
        "timeout": "int=20|Timeout in seconds, between 1 and 120",
    },
    risky=True,
    example='arguments: {"command": "uv run --with pytest python -m pytest -q", "timeout": 20}',
)
def run_shell_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")

    def execute() -> str:
        result = subprocess.run(
            command,
            cwd=registry.root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "\n".join(
            [
                f"exit_code: {result.returncode}",
                "stdout:",
                result.stdout.strip() or "(empty)",
                "stderr:",
                result.stderr.strip() or "(empty)",
            ]
        )

    return execute


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
)
def write_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    if "path" not in args:
        raise ValueError("missing path")
    path = registry._path(args["path"])

    if path.exists() and path.is_dir():
        raise ValueError("path is a directory")
    if "content" not in args:
        raise ValueError("missing content")

    content = str(args["content"])

    def execute() -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {path.relative_to(registry.root)} ({len(content)} chars)"

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
)
def patch_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    if "path" not in args:
        raise ValueError("missing path")
    path = registry._path(args["path"])

    if not path.is_file():
        raise ValueError("path is not a file")

    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    new_text = str(args["new_text"])

    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")

    def execute() -> str:
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"patched {path.relative_to(registry.root)}"

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
)
def append_file_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    if "path" not in args:
        raise ValueError("missing path")
    path = registry._path(args["path"])

    if path.exists() and path.is_dir():
        raise ValueError("path is a directory")
    if "content" not in args:
        raise ValueError("missing content")

    content = str(args["content"])

    def execute() -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return (
            f"appended {len(content)} chars to {path.relative_to(registry.root)}"
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
)
def delegate_tool(args: Dict, registry: "ToolRegistry") -> Callable[[], str]:
    if registry.delegate_fn is None:
        raise ValueError("delegate function not configured")

    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")

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
        self._registry: Tools = self._build()

    def items(self):
        return self._registry.items()

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

    @staticmethod
    def _parse_default(json_type: str, token: str):
        try:
            if json_type == "integer":
                return int(token)
            if json_type == "number":
                return float(token)
        except ValueError:
            return token
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

    def run(self, name: str, args: Dict) -> str:
        tool = self._registry.get(name, None)
        if tool is None:
            self.logger.log("tool_unknown", name=name, args=args)
            return f"error: unknown tool '{name}'"

        if self._repeated_call(name, args):
            self.logger.log(
                "tool_blocked", name=name, args=args, reason="repeated_call"
            )
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"

        args = self._coerce_args(tool, args or {})

        try:
            # Validate the arguments first so approval is only requested for
            # calls that can actually execute.
            execute = tool.run(args)
        except Exception as exc:
            return self._tool_error(name, args, exc)

        if tool.risky:
            approved = self._approve(name, args)
            self.logger.log(
                "tool_approval",
                name=name,
                args=args,
                risky=True,
                policy=self.approval_policy,
                read_only=self.read_only,
                granted=approved,
            )
            if not approved:
                return f"error: approval denied for {name}"

        try:
            return clip(execute())
        except Exception as exc:
            return self._tool_error(name, args, exc)

    def _tool_error(self, name: str, args: Dict, exc: Exception) -> str:
        self.logger.log(
            "tool_error",
            name=name,
            args=args,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        example = _TOOL_EXAMPLES.get(name, "")
        message = f"error: tool {name} failed: {exc}"
        if example:
            message += f"\nexample: {example}"
        return message

    def _build(self) -> Tools:
        tools = {}
        for name, definition in _TOOL_CATALOG.items():
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

    def _path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        if not self._path_is_within_root(resolved):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved

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

    def _repeated_call(self, name: str, args: Dict) -> bool:
        tool_events = [
            item for item in self.get_history() if isinstance(item, ToolMessageEntry)
        ]
        if len(tool_events) < 2:
            return False
        recent = tool_events[-2:]
        return all(item.name == name and item.args == args for item in recent)
