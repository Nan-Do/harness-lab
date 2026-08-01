import json
import uuid
from datetime import datetime
from pathlib import Path

from agent_logging import AgentLogger
from model_clients import LlamaCppModelClient
from session import SessionStore
from tools import READ_TOOLS, WRITE_TOOLS, ToolRegistry
from typing import Callable, Dict, List, Self
from workspace import WorkspaceContext
from app_types import (
    HistoryEntry,
    Memory,
    MessageEntry,
    Session,
    ToolMessageEntry,
)
from utils import (
    CHARS_PER_TOKEN,
    DEFAULT_HISTORY_BUDGET,
    DEFAULT_MAX_NOISY_OUTPUT,
    context_chars,
    now,
)

# Tools whose `path` argument names a file worth remembering as in-play.
FILE_TOOLS = frozenset(READ_TOOLS + WRITE_TOOLS + ("outline_file", "move_file"))


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
        tool_profile: str = "standard",
        max_noisy_output: int = DEFAULT_MAX_NOISY_OUTPUT,
        require_read_before_overwrite: bool = True,
        add_planning: bool = False,
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
        self.tool_profile = tool_profile
        self.max_noisy_output = max_noisy_output
        self.require_read_before_overwrite = require_read_before_overwrite
        self.add_planning = add_planning
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
            profile=self.tool_profile,
            max_noisy_output=self.max_noisy_output,
            require_read_before_overwrite=self.require_read_before_overwrite,
        )
        self.prefix = self.build_prefix()

        # llama-server reports the loaded model's real context window
        # (n_ctx); size the history budget off of that instead of a fixed
        # guess, so a large-context model gets to keep more turns and a
        # small one still fits. Falls back to a conservative fixed budget
        # when n_ctx is unavailable. This budget is enforced by dropping
        # whole old turns (see _fit_history_budget), never by truncating
        # content -- tool output and messages are always sent in full.
        #
        # What has to be held back is measured rather than guessed: the tool
        # schemas alone run to thousands of tokens and grow with --tools, so a
        # fixed reserve would quietly overcommit the window on the larger
        # profiles and overflow the model's real context.
        overhead_chars = len(self.prefix) + len(json.dumps(self.tools.schemas()))
        reserve_tokens = (
            self.max_new_tokens + int(overhead_chars / CHARS_PER_TOKEN) + 500
        )
        self.history_budget = context_chars(
            getattr(model_client, "ctx", 0), reserve_tokens, DEFAULT_HISTORY_BUDGET
        )
        self.logger.log(
            "context_budget",
            n_ctx=getattr(model_client, "ctx", 0),
            tool_profile=self.tool_profile,
            tool_count=len(self.tools.schemas()),
            overhead_chars=overhead_chars,
            reserve_tokens=reserve_tokens,
            history_budget_chars=self.history_budget,
        )
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
            tool_profile=self.tool_profile,
            max_noisy_output=self.max_noisy_output,
            require_read_before_overwrite=self.require_read_before_overwrite,
        )
        child.session.memory.task = task
        child.session.memory.notes = [self.history_text()]
        return "delegate_result:\n" + child.ask(task)

    def build_prefix(self: Self) -> str:
        planning = ""
        if self.add_planning:
            planning = "\n".join(
                [
                    "Planning:",
                    "- Break down the problem into logical requirements.",
                    "- Outline the core algorithm, data structures, and edge cases to consider.",
                    "- List the sequential steps you will take to implement the solution.",
                    "Solution:",
                    "- Implement the complete, clean code based on your plan.",
                    "- Include any necessary execution details or usage examples.",
                ]
            )

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
                "- To create a long file, write the first part with write_file and continue with append_file; do not try to fit it all in one call.",
                "- To change an existing file, edit it with patch_file or replace_lines; use write_file only for a new file or a full rewrite of a file you have read.",
                "- For a large file, find the part you need with outline_file, read it with read_file_range, and edit it with replace_lines instead of reading or rewriting the whole file.",
                "- Copy old_text for patch_file exactly as read_file showed it, without the leading line numbers.",
                "- Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or give a final answer.",
                "- Required tool arguments must not be empty.",
            ]
        )

        return "\n\n".join(
            [
                "You are Harness Lab, a small local coding agent running through llama-server.",
                planning,
                "Rules:\n" + rules,
                self.workspace.text(),
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

    @staticmethod
    def _norm_path(raw: object) -> str:
        """Normalize a tool path argument so "./x" and "x" compare equal."""
        text = str(raw or "").strip()
        return Path(text).as_posix() if text else ""

    @staticmethod
    def _covers(outer: tuple, inner: tuple) -> bool:
        """True when line range `outer` fully contains `inner` (None = to EOF)."""
        outer_start, outer_end = outer
        inner_start, inner_end = inner
        outer_end = float("inf") if outer_end is None else outer_end
        inner_end = float("inf") if inner_end is None else inner_end
        return outer_start <= inner_start and outer_end >= inner_end

    @classmethod
    def _stale_read_indices(cls, history: List[HistoryEntry]) -> set:
        """Indices of file reads that no longer carry current content.

        A read is stale when a later read of the same path covers its line
        range, or when a later write_file replaced the whole file. Range-aware
        on purpose: reading lines 1-200 and then 201-400 of a big file leaves
        both results useful, so neither may be dropped -- that sequence is the
        intended way to work through a file too large to hold at once. Partial
        edits (patch_file, replace_lines, append_file) deliberately do not
        supersede a read: the read is only partly out of date, and dropping it
        would leave the model with no view of the file at all.
        """
        reads: List[tuple] = []
        replaced: Dict[str, int] = {}
        for index, item in enumerate(history):
            if not isinstance(item, ToolMessageEntry):
                continue
            path = cls._norm_path(item.args.get("path"))
            if not path:
                continue
            if str(item.content).startswith("error:"):
                continue
            if item.name == "read_file":
                reads.append((index, path, 1, None))
            elif item.name == "read_file_range":
                try:
                    start = int(item.args.get("start", 1))
                    end = int(item.args.get("end", 200))
                except (TypeError, ValueError):
                    start, end = 1, None
                reads.append((index, path, start, end))
            elif item.name == "write_file":
                replaced[path] = index

        stale = set()
        for position, (index, path, start, end) in enumerate(reads):
            if replaced.get(path, -1) > index:
                stale.add(index)
                continue
            for later_index, later_path, later_start, later_end in reads[
                position + 1 :
            ]:
                if later_path != path:
                    continue
                if cls._covers((later_start, later_end), (start, end)):
                    stale.add(index)
                    break
        return stale

    def history_text(self: Self) -> str:
        history: List[HistoryEntry] = self.session.history
        if not history:
            return "- empty"

        stale_reads = self._stale_read_indices(history)
        blocks: List[str] = []
        for index, item in enumerate(history):
            if index in stale_reads:
                continue

            if isinstance(item, ToolMessageEntry):
                blocks.append(
                    f"[tool:{item.name}] {json.dumps(item.args, sort_keys=True)}"
                    f"\n{item.content}"
                )
            else:
                blocks.append(f"[{item.role}] {item.content}")

        if not blocks:
            return "- empty"

        total = sum(len(block) for block in blocks)
        while len(blocks) > 1 and total > self.history_budget:
            total -= len(blocks.pop(0))

        return "\n".join(blocks)

    @staticmethod
    def _group_chars(group: List[Dict]) -> int:
        return sum(
            len(str(message.get("content") or ""))
            + len(json.dumps(message.get("tool_calls", "")))
            for message in group
        )

    @staticmethod
    def _group_labels(groups: List[List[Dict]]) -> List[str]:
        """Short "tool path" labels for the tool calls inside these groups."""
        labels: List[str] = []
        for group in groups:
            for message in group:
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    name = function.get("name", "tool")
                    try:
                        args = json.loads(function.get("arguments") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    path = args.get("path") if isinstance(args, dict) else None
                    labels.append(f"{name} {path}" if path else str(name))
        return labels

    def _eviction_notice(self: Self, dropped: List[List[Dict]]) -> str:
        """Tell the model what it can no longer see.

        Content is sent whole or dropped whole, never truncated -- but a read
        that silently disappears one turn later is worse than a visible gap:
        the model goes on editing a file from a memory it no longer has. Naming
        what went makes the gap recoverable.
        """
        labels = self._group_labels(dropped)
        detail = ", ".join(dict.fromkeys(labels)) if labels else "earlier conversation"
        return (
            "[context notice] Older turns were dropped to fit the context window: "
            f"{detail}. That output is no longer visible to you. Re-read anything "
            "you still need (read_file_range for a specific part of a large file) "
            "instead of relying on remembering it."
        )

    def _fit_history_budget(self: Self, groups: List[List[Dict]]) -> List[List[Dict]]:
        """Drop the oldest whole turns until the prompt fits history_budget.

        A turn (a tool call + its result, or a single message) is either
        sent whole or dropped whole -- content is never truncated. The most
        recent turn (the current request) is never dropped, so the request
        always goes through even if it alone exceeds the budget. Whatever gets
        dropped is replaced by a one-line notice so the loss is visible to the
        model rather than silent.
        """
        total = sum(self._group_chars(group) for group in groups)
        dropped: List[List[Dict]] = []
        while len(groups) > 1 and total > self.history_budget:
            removed = groups.pop(0)
            total -= self._group_chars(removed)
            dropped.append(removed)
        if dropped:
            self.logger.log(
                "history_window",
                dropped_turns=len(dropped),
                dropped_labels=self._group_labels(dropped),
                kept_turns=len(groups),
                total_chars=total,
                budget_chars=self.history_budget,
            )
            groups.insert(
                0, [{"role": "user", "content": self._eviction_notice(dropped)}]
            )
        return groups

    def build_messages(self: Self, user_message: str) -> List[Dict]:
        """Build the native chat-message list sent to the model.

        Tool results go back as role="tool" messages tied to assistant
        tool_calls turns, matching the format the model was trained on,
        instead of a flattened text transcript. Stale reads are dropped so a
        superseded snapshot of a file doesn't shadow the current one, and
        the oldest whole turns are dropped -- never truncated -- once the
        prompt would exceed the model's context budget.
        """
        history: List[HistoryEntry] = self.session.history
        stale_reads = self._stale_read_indices(history)

        groups: List[List[Dict]] = []
        for index, item in enumerate(history):
            if isinstance(item, ToolMessageEntry):
                if index in stale_reads:
                    continue
                call_id = item.call_id or f"call_{index}"
                groups.append(
                    [
                        {
                            "role": "assistant",
                            "content": item.assistant_text or "",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": item.name,
                                        "arguments": json.dumps(item.args),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": item.content,
                        },
                    ]
                )
            else:
                # Runtime notices are stored as "system" but sent as user
                # turns: many chat templates only accept a leading system
                # message, and correction should not come in the model's own
                # (assistant) voice.
                role = "user" if item.role == "system" else item.role
                groups.append([{"role": role, "content": item.content}])

        # The model always ends on a user turn restating memory and the
        # current request. On the first iteration that replaces the freshly
        # recorded user message instead of duplicating it.
        context = self.memory_text() + "\n\nCurrent user request:\n" + user_message
        if (
            groups
            and isinstance(history[-1], MessageEntry)
            and history[-1].role == "user"
        ):
            groups[-1] = [{"role": "user", "content": context}]
        else:
            groups.append([{"role": "user", "content": context}])

        groups = self._fit_history_budget(groups)

        messages: List[Dict] = [{"role": "system", "content": self.prefix}]
        for group in groups:
            for message in group:
                # Merge adjacent user turns (e.g. a runtime notice followed by
                # the context refresher) for strict-alternation templates.
                if message["role"] == "user" and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n\n" + message["content"]
                else:
                    messages.append(message)
        return messages

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
                content=item.content,
            )
        else:
            self.logger.log(
                "history_append",
                entry="message",
                index=len(self.session.history) - 1,
                role=item.role,
                content=item.content,
            )
        self.session_path = self.session_store.save(self.session)

    def note_tool(self: Self, name: str, args: Dict[str, str], result: str) -> None:
        memory = self.session.memory
        path = args.get("path")
        if name in FILE_TOOLS and path:
            self.remember(memory.files, str(path), 8)
        note = f"{name}: {str(result).replace(chr(10), ' ')}"
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

    def _stream_sinks(
        self: Self, on_event: Callable[..., None] | None
    ) -> tuple[Callable | None, Callable | None]:
        """Bridge streamed output to the front-end, as text and as tool calls.

        Both are None when nobody is listening, so a headless or delegated run
        doesn't pay for a callback per token. The tool sink carries the raw
        arguments assembled so far -- a front-end that wants to show the file
        being written decodes it with `tool_support.streaming_body`.
        """
        if on_event is None:
            return None, None

        def on_text(text: str) -> None:
            self._emit(on_event, "assistant_delta", text=text)

        def on_tool(name: str, args_text: str) -> None:
            self._emit(on_event, "tool_delta", name=name, args_text=args_text)

        return on_text, on_tool

    def ask(
        self: Self,
        user_message: str,
        on_event: Callable[..., None] | None = None,
    ) -> str:
        memory = self.session.memory
        if not memory.task:
            memory.task = user_message.strip()
        self.logger.log(
            "request_start",
            max_steps=self.max_steps,
            max_new_tokens=self.max_new_tokens,
            user_message=user_message,
        )
        self.log_memory("request_start")
        self.record(MessageEntry(role="user", content=user_message, created_at=now()))

        tool_steps = 0
        attempts = 0
        max_attempts = max(self.max_steps * 3, self.max_steps + 4)

        while tool_steps < self.max_steps and attempts < max_attempts:
            attempts += 1
            self._emit(on_event, "thinking", attempt=attempts, step=tool_steps)
            messages = self.build_messages(user_message)
            self.logger.log(
                "prompt_built",
                attempt=attempts,
                tool_step=tool_steps,
                message_count=len(messages),
                roles=[message["role"] for message in messages],
                chars=sum(
                    len(str(message.get("content") or "")) for message in messages
                ),
                memory_text=self.memory_text(),
            )
            on_text, on_tool = self._stream_sinks(on_event)
            response = self.model_client.complete(
                messages,
                self.max_new_tokens,
                tools=self.tools.schemas(),
                on_delta=on_text,
                on_tool_delta=on_tool,
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
                content=response.content,
            )

            if response.malformed_tool_calls:
                for bad in response.malformed_tool_calls:
                    self.logger.log(
                        "malformed_tool_call",
                        attempt=attempts,
                        name=bad.name,
                        error=bad.error,
                        raw_args=bad.raw_args,
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
                # Keep the model's plan text attached to its first tool call
                # so the transcript preserves its thread of thought.
                assistant_text = response.content.strip()
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
                        result=result,
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
                            call_id=call.id or "",
                            assistant_text=assistant_text,
                        )
                    )
                    assistant_text = ""
                    self.note_tool(name, args, result)
                    if tool_steps >= self.max_steps:
                        break
                if response.malformed_tool_calls:
                    self.record_retry(
                        on_event, attempts, self.malformed_notice(response)
                    )
                continue

            final = response.content.strip()
            if response.malformed_tool_calls:
                # A broken tool call plus text like "I'll read the file now"
                # is an intent to act, not an answer; keep the plan in the
                # transcript but force a corrected retry.
                if final:
                    self.record(
                        MessageEntry(role="assistant", content=final, created_at=now())
                    )
                self.record_retry(on_event, attempts, self.malformed_notice(response))
                continue

            if not final:
                self.record_retry(
                    on_event,
                    attempts,
                    self.retry_notice("model returned no tool call and no answer"),
                )
                continue

            self.record(MessageEntry(role="assistant", content=final, created_at=now()))
            self.remember(memory.notes, final, 5)
            self.log_memory("final")
            self.logger.log(
                "final",
                reason="answer",
                tool_steps=tool_steps,
                attempts=attempts,
                final=final,
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
            final=final,
        )
        self._emit(on_event, "final", text=final, reason=reason)
        return final

    def record_retry(
        self: Self,
        on_event: Callable[..., None] | None,
        attempts: int,
        notice: str,
    ) -> None:
        """Record a corrective notice and surface it to the UI.

        Stored with role "system" so the correction does not appear in the
        model's own (assistant) voice, which small models tend to imitate.
        """
        self.logger.log("retry", attempt=attempts, notice=notice)
        self._emit(on_event, "retry", notice=notice)
        self.record(MessageEntry(role="system", content=notice, created_at=now()))

    def malformed_notice(self: Self, response) -> str:
        problems = "; ".join(
            f"{bad.name or 'unknown'}: {bad.error}"
            for bad in response.malformed_tool_calls
        )
        if response.truncated:
            return (
                "Runtime notice: the tool call was cut off by the output token "
                f"limit ({problems}). Produce a shorter call: write the first "
                "part of the file with write_file, then extend it with "
                "append_file."
            )
        return self.retry_notice(f"model produced malformed tool calls ({problems})")

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
