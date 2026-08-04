from datetime import datetime, timezone

from app_types import ContextUsage


DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
HELP_TEXT = "/help, /memory, /session, /log, /reset, /clear, /exit"
APP_NAME = (
    r" _   _                                          _          _     ",
    r"| | | | __ _ _ __ _ __   ___  ___ ___          | |    __ _| |__  ",
    r"| |_| |/ _` | '__| '_ \ / _ \/ __/ __|  _____  | |   / _` | '_ \ ",
    r"|  _  | (_| | |  | | | |  __/\__ \__ \ |_____| | |__| (_| | |_) |",
    r"|_| |_|\__,_|_|  |_| |_|\___||___/___/         |_____\__,_|_.__/ ",
)
HELP_DETAILS = "\n".join(
    [
        "Commands:",
        "/help    Show this help message.",
        "/memory  Show the agent's distilled working memory.",
        "/session Show the path to the saved session file.",
        "/log     Show the path to the current JSONL run log.",
        "/reset   Clear the current session history and memory.",
        "/clear   Clear the conversation view (keeps the session).",
        "/exit    Exit the agent.",
    ]
)
IGNORED_PATH_NAMES = {
    ".git",
    ".harness-lab",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "node_modules",
}
# Workspace-relative home for harness state (logs, sessions, write backups).
STATE_DIR_NAME = ".harness-lab"
BACKUP_DIR_NAME = "backups"
# Tool profiles, smallest first. Each tool declares the profile it first
# appears in; a bigger tool set gives the model more reach but measurably more
# wrong-tool picks on small models, so which profile wins is an experiment
# (--tools), not a fixed answer.
TOOL_PROFILES = ("minimal", "standard", "full")
# Character budget for output that is noise rather than payload -- shell logs
# and search-match floods. File reads are never clipped: content reaches the
# model whole or is dropped whole (see agent.MiniAgent._fit_history_budget).
# 0 disables clipping entirely (--max-tool-output 0).
DEFAULT_MAX_NOISY_OUTPUT = 6000
# Fallback total-history budget (chars) used to decide when to start
# dropping old turns, for when the backend doesn't report a model context
# size (see context_chars below).
DEFAULT_HISTORY_BUDGET = 12000
# Conservative chars-per-token estimate for sizing that budget off a model's
# real n_ctx. Code and tool output (line numbers, punctuation, JSON) tokenize
# denser than prose, so this errs low to avoid overflowing the context.
CHARS_PER_TOKEN = 3.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def context_chars(n_ctx: int, reserve_tokens: int, floor: int) -> int:
    """Translate a model's reported token context window into a character
    budget for prompt assembly.

    llama-server reports the real n_ctx of the loaded model; hardcoding a
    single char budget regardless of that would starve large-context models
    of headroom they actually have, while risking overflow on tiny ones.
    `reserve_tokens` is held back for the system prompt, tool schemas, and
    the model's own reply. `floor` is used when n_ctx is unknown (0) or too
    small to beat it.
    """
    if n_ctx <= 0:
        return floor
    return max(floor, int((n_ctx - reserve_tokens) * CHARS_PER_TOKEN))


def compact_tokens(count: int) -> str:
    """A token count short enough to sit on a status line: 812, 6.2k, 131k.

    Thousands are floored rather than rounded, so a 32768-token window reads
    as the "32k" its owner asked for instead of a puzzling 33k.
    """
    if count < 1000:
        return str(count)
    if count < 10000:
        return f"{count / 1000:.1f}k"
    return f"{count // 1000}k"


def format_context(usage: ContextUsage) -> str:
    """One line of "how full is the window", shared by the front-ends.

    A `~` marks counts estimated from characters because the backend reported
    no usage, so a guess is never read as a measurement -- but nothing sent
    yet is exactly nothing, not an estimate of it. Without a known context
    size there is no percentage to give, only what was sent.
    """
    mark = "~" if usage.estimated and usage.prompt_tokens else ""
    used = f"{mark}{compact_tokens(usage.prompt_tokens)}"
    if usage.limit <= 0:
        return f"ctx {used} tokens"
    # The share is parenthesised rather than another `·` field: on the status
    # line it belongs to the count beside it, not next to `branch` and
    # `approval`.
    return f"ctx {used}/{compact_tokens(usage.limit)} ({usage.percent:.0f}%)"


def middle(text: str, limit: int) -> str:
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]
