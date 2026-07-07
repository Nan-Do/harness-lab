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
    role: str  # "user" or "assistant"
    content: str
    created_at: str


@dataclass
class ToolMessageEntry:
    role: str  # always "tool"
    name: str
    args: Dict[str, Any]
    content: str
    created_at: str


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


HistoryEntry: TypeAlias = Union[MessageEntry, ToolMessageEntry]


@dataclass
class Session:
    id: str
    created_at: str
    workspace_root: str
    history: List[HistoryEntry]
    memory: Memory
