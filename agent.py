import json
import uuid
from datetime import datetime
from pathlib import Path

from agent_logging import AgentLogger
from model_clients import LlamaCppModelClient
from session import SessionStore
from tools import ToolRegistry
from typing import Callable, Dict, List, Self, Tuple
from workspace import WorkspaceContext
from app_types import (
    HistoryEntry,
    Memory,
    MessageEntry,
    Session,
    ToolMessageEntry,
)
from utils import (
    MAX_HISTORY,
    now,
    clip,
)


class MiniAgent:
    def __init__(
        self: Self,
        model_client: LlamaCppModelClient,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session: Session | None = None,
        approval_policy: str = "ask",
        max_steps: int = 6,
        max_new_tokens: int = 512,
        depth: int = 0,
        max_depth: int = 1,
        read_only: bool = False,
        logger: AgentLogger | None = None,
        approval_fn: Callable[[str, Dict], bool] | None = None,
    ) -> None:
        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.approval_fn = approval_fn
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.session = session or Session(
            id=datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            created_at=now(),
            workspace_root=workspace.repo_root,
            history=[],
            memory=Memory(task="", files=[], notes=[]),
        )
        base_logger = logger or AgentLogger(None, enabled=False)
        self.logger = base_logger.child(session=self.session.id, depth=self.depth)
        self.logger.log(
            "session_start",
            workspace_root=self.session.workspace_root,
            approval_policy=self.approval_policy,
            read_only=self.read_only,
            resumed=session is not None,
            history_len=len(self.session.history),
        )
        self.tools = ToolRegistry(
            workspace=self.workspace,
            root=self.root,
            approval_policy=self.approval_policy,
            read_only=self.read_only,
            depth=self.depth,
            max_depth=self.max_depth,
            get_history=lambda: self.session.history,
            delegate_fn=self._make_delegate if self.depth < self.max_depth else None,
            logger=self.logger,
            approval_fn=self.approval_fn,
        )
        self.prefix = self.build_prefix()
        self.session_path = self.session_store.save(self.session)

    @classmethod
    def from_session(
        cls: type[Self],
        model_client: LlamaCppModelClient,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session_id: str,
        **kwargs,
    ) -> Self:
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    @staticmethod
    def remember(bucket: List[str], item: str, limit: int) -> None:
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def _make_delegate(self: Self, task: str, max_steps: int) -> str:
        child = MiniAgent(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            approval_policy="never",
            max_steps=max_steps,
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
            logger=self.logger,
        )
        child.session.memory.task = task
        child.session.memory.notes = [clip(self.history_text(), 300)]
        return "delegate_result:\n" + child.ask(task)

    def build_prefix(self: Self) -> str:
        rules = "\n".join(
            [
                "- Use the provided tools instead of guessing about the workspace.",
                "- Call tools through the function-calling interface; never describe a tool call in plain text.",
                "- When you are done, reply with the final answer as plain text and do not call a tool.",
                "- Never invent tool results.",
                "- Keep answers concise and concrete.",
                "- If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
                "- Before writing tests for existing code, read the implementation first.",
                "- When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
                "- New files should be complete and runnable, including obvious imports.",
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or give a final answer.",
                "- Required tool arguments must not be empty.",
            ]
        )
        return "\n\n".join(
            [
                "You are Mini-Coding-Agent, a small local coding agent running through llama-server.",
                "Rules:\n" + rules,
            ]
        )

    def memory_text(self: Self) -> str:
        memory: Memory = self.session.memory
        notes = "\n".join(f"- {note}" for note in memory.notes) or "- none"
        return "\n".join(
            [
                "Memory:",
                f"- task: {memory.task or '-'}",
                f"- files: {', '.join(memory.files) or '-'}",
                "- notes:",
                notes,
            ]
        )

    def history_text(self: Self) -> str:
        history: List[HistoryEntry] = self.session.history
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if isinstance(item, ToolMessageEntry) and item.name in (
                "write_file",
                "patch_file",
            ):
                path = str(item.args.get("path", ""))
                seen_reads.discard(path)
            if (
                isinstance(item, ToolMessageEntry)
                and item.name == "read_file"
                and not recent
            ):
                path = str(item.args.get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if isinstance(item, ToolMessageEntry):
                limit = 900 if recent else 180
                lines.append(
                    f"[tool:{item.name}] {json.dumps(item.args, sort_keys=True)}"
                )
                lines.append(clip(item.content, limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item.role}] {clip(item.content, limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def prompt(self: Self, user_message: str) -> Tuple[str, str]:
        return self.prefix, "\n\n".join(
            [
                self.workspace.text(),
                self.memory_text(),
                "Transcript:\n" + self.history_text(),
                "Current user request:\n" + user_message,
            ]
        )

    def memory_snapshot(self: Self) -> Dict[str, object]:
        memory = self.session.memory
        return {
            "task": memory.task,
            "files": list(memory.files),
            "notes": list(memory.notes),
        }

    def log_memory(self: Self, reason: str) -> None:
        self.logger.log("memory_update", reason=reason, memory=self.memory_snapshot())

    def record(self: Self, item: HistoryEntry) -> None:
        self.session.history.append(item)
        if isinstance(item, ToolMessageEntry):
            self.logger.log(
                "history_append",
                entry="tool",
                index=len(self.session.history) - 1,
                name=item.name,
                args=item.args,
                content=clip(item.content, 2000),
            )
        else:
            self.logger.log(
                "history_append",
                entry="message",
                index=len(self.session.history) - 1,
                role=item.role,
                content=clip(item.content, 2000),
            )
        self.session_path = self.session_store.save(self.session)

    def note_tool(self: Self, name: str, args: Dict[str, str], result: str) -> None:
        memory = self.session.memory
        path = args.get("path")
        if name in {"read_file", "write_file", "patch_file"} and path:
            self.remember(memory.files, str(path), 8)
        note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
        self.remember(memory.notes, note, 5)
        self.log_memory(f"note_tool:{name}")

    def _emit(
        self: Self,
        on_event: Callable[..., None] | None,
        event_type: str,
        **data: object,
    ) -> None:
        if on_event is None:
            return
        try:
            on_event(event_type, **data)
        except Exception:
            # The UI callback must never break the agent loop.
            self.logger.log("event_callback_error", event_type=event_type)

    def ask(
        self: Self,
        user_message: str,
        on_event: Callable[..., None] | None = None,
    ) -> str:
        memory = self.session.memory
        if not memory.task:
            memory.task = clip(user_message.strip(), 300)
        self.logger.log(
            "request_start",
            max_steps=self.max_steps,
            max_new_tokens=self.max_new_tokens,
            user_message=clip(user_message, 2000),
        )
        self.log_memory("request_start")
        self.record(MessageEntry(role="user", content=user_message, created_at=now()))

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            self._emit(on_event, "thinking", attempt=attempts, step=tool_steps)
            system_prompt, user_prompt = self.prompt(user_message)
            self.logger.log(
                "prompt_built",
                attempt=attempts,
                tool_step=tool_steps,
                system_len=len(system_prompt),
                user_len=len(user_prompt),
                memory_text=self.memory_text(),
                history_text=self.history_text(),
            )
            response = self.model_client.complete(
                system_prompt,
                user_prompt,
                self.max_new_tokens,
                tools=self.tools.schemas(),
            )
            self.logger.log(
                "model_output",
                attempt=attempts,
                tool_step=tool_steps,
                tool_calls=[call.name for call in response.tool_calls],
                malformed_tool_calls=[
                    {"name": bad.name, "error": bad.error}
                    for bad in response.malformed_tool_calls
                ],
                content=clip(response.content, 2000),
            )

            if response.malformed_tool_calls:
                for bad in response.malformed_tool_calls:
                    self.logger.log(
                        "malformed_tool_call",
                        attempt=attempts,
                        name=bad.name,
                        error=bad.error,
                        raw_args=clip(bad.raw_args, 500),
                    )
                    self._emit(
                        on_event,
                        "malformed_tool_call",
                        name=bad.name,
                        error=bad.error,
                    )
                    self.remember(
                        memory.notes,
                        f"malformed tool call to '{bad.name or 'unknown'}' "
                        f"({bad.error}); arguments must be a single valid JSON object",
                        5,
                    )
                self.log_memory("malformed_tool_call")

            if response.tool_calls:
                for call in response.tool_calls:
                    tool_steps += 1
                    name = call.name
                    args = call.args
                    self.logger.log("tool_call", name=name, args=args, step=tool_steps)
                    self._emit(
                        on_event, "tool_call", name=name, args=args, step=tool_steps
                    )
                    result = self.tools.run(name, args)
                    self.logger.log(
                        "tool_result",
                        name=name,
                        step=tool_steps,
                        result=clip(result, 2000),
                    )
                    self._emit(
                        on_event,
                        "tool_result",
                        name=name,
                        result=result,
                        step=tool_steps,
                    )
                    self.record(
                        ToolMessageEntry(
                            role="tool",
                            name=name,
                            args=args,
                            content=result,
                            created_at=now(),
                        )
                    )
                    self.note_tool(name, args, result)
                    if tool_steps >= self.max_steps:
                        break
                continue

            final = response.content.strip()
            if not final:
                if response.malformed_tool_calls:
                    problems = "; ".join(
                        f"{bad.name or 'unknown'}: {bad.error}"
                        for bad in response.malformed_tool_calls
                    )
                    notice = self.retry_notice(
                        f"model produced malformed tool calls ({problems})"
                    )
                else:
                    notice = self.retry_notice(
                        "model returned no tool call and no answer"
                    )
                self.logger.log("retry", attempt=attempts, notice=clip(notice, 500))
                self._emit(on_event, "retry", notice=notice)
                self.record(
                    MessageEntry(role="assistant", content=notice, created_at=now())
                )
                continue

            self.record(MessageEntry(role="assistant", content=final, created_at=now()))
            self.remember(memory.notes, clip(final, 220), 5)
            self.log_memory("final")
            self.logger.log(
                "final",
                reason="answer",
                tool_steps=tool_steps,
                attempts=attempts,
                final=clip(final, 2000),
            )
            self._emit(on_event, "final", text=final)
            return final

        if attempts >= max_attempts and tool_steps < self.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            reason = "max_attempts"
        else:
            final = "Stopped after reaching the step limit without a final answer."
            reason = "max_steps"
        self.record(MessageEntry(role="assistant", content=final, created_at=now()))
        self.logger.log(
            "final",
            reason=reason,
            tool_steps=tool_steps,
            attempts=attempts,
            final=clip(final, 2000),
        )
        self._emit(on_event, "final", text=final, reason=reason)
        return final

    @staticmethod
    def retry_notice(problem: str | None = None) -> str:
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned no actionable output"
        return (
            f"{prefix}. Call one of the available tools through the function-calling "
            "interface, or reply with a non-empty plain-text final answer."
        )

    def reset(self: Self) -> None:
        self.session.history = []
        self.session.memory = Memory(task="", files=[], notes=[])
        self.session_store.save(self.session)
        self.logger.log("reset")

    @property
    def log_path(self: Self) -> str:
        return str(self.logger.path) if self.logger.path else "(logging disabled)"
