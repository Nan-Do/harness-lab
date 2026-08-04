from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeAlias, Union


@dataclass
class ToolDescriptionEntry:
    schema: dict
    risky: bool
    description: str
    run: Callable


Tools: TypeAlias = Dict[str, ToolDescriptionEntry]


@dataclass
class Memory:
    task: str
    files: List[str]
    notes: List[str]


@dataclass
class MessageEntry:
    role: str  # "user", "assistant" or "system" (runtime notices)
    content: str
    created_at: str


@dataclass
class ToolMessageEntry:
    role: str  # always "tool"
    name: str
    args: Dict[str, Any]
    content: str
    created_at: str
    # Id assigned by the model backend, echoed back as tool_call_id.
    call_id: str = ""
    # Plan/reasoning text the model emitted alongside this tool call; kept so
    # the transcript preserves the model's own thread of thought.
    assistant_text: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class MalformedToolCall:
    id: str
    name: str
    raw_args: str
    error: str


@dataclass
class ModelResponse:
    content: str
    tool_calls: List[ToolCall]
    malformed_tool_calls: List[MalformedToolCall] = field(default_factory=list)
    # True when generation stopped at the token limit mid tool call, so the
    # arguments are cut off rather than genuinely malformed.
    truncated: bool = False
    # The model's thinking for this turn, kept apart from `content`: it is
    # shown to the user but never sent back as part of the next prompt.
    reasoning: str = ""
    # Token counts as the server reported them ({"prompt_tokens": ...}), or
    # None from a backend that doesn't send usage at all.
    usage: Dict[str, Any] | None = None


@dataclass
class ContextUsage:
    """How full the model's context window was for the last prompt sent.

    `limit` is the model's own `n_ctx`, 0 when the backend doesn't report one.
    `prompt_tokens` is the whole prompt -- system rules, tool schemas, and
    whatever history survived the budget -- which is what "used" means here:
    the room the conversation is taking up before the model writes anything.
    `estimated` marks counts derived from characters because the server sent
    no usage, so a guess is never shown as a measurement.
    """

    limit: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = True

    @property
    def percent(self) -> float:
        if self.limit <= 0:
            return 0.0
        return 100.0 * self.prompt_tokens / self.limit


HistoryEntry: TypeAlias = Union[MessageEntry, ToolMessageEntry]


@dataclass
class Session:
    id: str
    created_at: str
    workspace_root: str
    history: List[HistoryEntry]
    memory: Memory
