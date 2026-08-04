import json
import re

from dataclasses import dataclass, field
from typing import Self
from collections.abc import Callable

from openai import OpenAI, OpenAIError

from agent_logging import AgentLogger
from app_types import MalformedToolCall, ModelResponse, ToolCall


# --- Reasoning ---------------------------------------------------------------
# A thinking model's reasoning reaches the harness one of two ways, depending on
# how the server is configured, and never both:
#
#   * as its own field on the message or delta -- what llama-server sends with
#     `--reasoning-format deepseek` (its default for models that think), and
#     what most OpenAI-compatible backends do. Servers disagree on the name.
#   * inline in the content, wrapped in the `<think>` tags the model itself
#     emits -- what arrives with `--reasoning-format none`, or from a server
#     that doesn't parse reasoning at all.
#
# Both are pulled out here so a front-end sees one reasoning stream regardless
# of the backend, and so the reasoning stops landing in the answer text.

REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")

THINK_TAGS = ("think", "thinking")
OPEN_TAGS = tuple(f"<{tag}>" for tag in THINK_TAGS)
OPEN_TAG_RE = re.compile("<(" + "|".join(THINK_TAGS) + ")>")

# The two channels a turn is written on, as tagged on each piece of output.
CONTENT = "content"
REASONING = "reasoning"


def reasoning_field(payload: object) -> str:
    """Reasoning the server sent as its own field, whatever it calls it."""
    extra = getattr(payload, "model_extra", None) or {}
    for name in REASONING_FIELDS:
        value = getattr(payload, name, None) or extra.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _safe_end(text: str, candidates: tuple[str, ...]) -> int:
    """How much of `text` cannot be the start of one of `candidates`.

    A tag is split across chunk boundaries as readily as any other text, so
    the tail that could still turn into one has to wait for the next chunk.
    """
    longest = max(len(candidate) for candidate in candidates)
    for index in range(max(0, len(text) - longest + 1), len(text)):
        fragment = text[index:]
        if any(candidate.startswith(fragment) for candidate in candidates):
            return index
    return len(text)


class ThinkSplitter:
    """Sorts streamed content onto the two channels as it arrives.

    Text is fed in whatever pieces the server sends and comes back as
    (channel, text) pairs *in the order it was written*: a single chunk can
    finish a think block and start the answer, and a front-end painting the
    two as they arrive has to keep them in sequence. Anything that might still
    turn out to be a `<think>` tag is held back until the next piece settles
    it, and `flush` releases the remainder once the turn is over. The open-tag
    state survives across rounds, so reasoning cut off at the token limit is
    still read as reasoning when generation continues.
    """

    def __init__(self: Self) -> None:
        self.buffer = ""
        self.open_tag = ""

    def feed(self: Self, text: str) -> list[tuple[str, str]]:
        self.buffer += text
        pieces: list[tuple[str, str]] = []
        while True:
            if self.open_tag:
                closing = f"</{self.open_tag}>"
                at = self.buffer.find(closing)
                if at < 0:
                    break
                pieces.append((REASONING, self.buffer[:at]))
                self.buffer = self.buffer[at + len(closing) :]
                self.open_tag = ""
                continue
            match = OPEN_TAG_RE.search(self.buffer)
            if match is None:
                break
            pieces.append((CONTENT, self.buffer[: match.start()]))
            self.open_tag = match.group(1)
            self.buffer = self.buffer[match.end() :]

        candidates = (f"</{self.open_tag}>",) if self.open_tag else OPEN_TAGS
        keep = _safe_end(self.buffer, candidates)
        released, self.buffer = self.buffer[:keep], self.buffer[keep:]
        pieces.append((REASONING if self.open_tag else CONTENT, released))
        return [piece for piece in pieces if piece[1]]

    def flush(self: Self) -> list[tuple[str, str]]:
        """Release the held-back tail; nothing more is coming this turn."""
        tail, self.buffer = self.buffer, ""
        if not tail:
            return []
        return [(REASONING if self.open_tag else CONTENT, tail)]


def fold(pieces: list[tuple[str, str]]) -> tuple[str, str]:
    """Collapse ordered pieces into the whole (content, reasoning) of a turn."""
    return (
        "".join(text for channel, text in pieces if channel == CONTENT),
        "".join(text for channel, text in pieces if channel == REASONING),
    )


@dataclass
class RawToolCall:
    """A tool call exactly as the backend sent it, before argument parsing."""

    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class Turn:
    """One round trip with the server, however it arrived.

    Streamed and non-streamed responses are normalized into this shape so the
    rest of the client -- parsing, continuation, logging -- has a single path.
    """

    content: str = ""
    tool_calls: list[RawToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict | None = None
    reasoning: str = ""


class LlamaCppModelClient:
    def __check_model(self: Self):
        try:
            models = self.client.models.list()
        except OpenAIError as exc:
            raise RuntimeError(
                "Could not reach LlamaCpp.\n"
                "Make sure `llama-server` is running.\n"
                f"URL: {self.base_url}\n"
                f"Model: {self.model}"
            ) from exc

        available = list(models.data)
        if not available:
            raise RuntimeError("Llama-server reported no available models")

        chosen = next((m for m in available if m.id == self.model), available[0])
        self.model = chosen.id
        meta = getattr(chosen, "model_extra", None) or {}
        self.ctx = (meta.get("meta") or {}).get("n_ctx", 0)

    def __init__(
        self: Self,
        model: str,
        host: str,
        port: int,
        temperature: float,
        top_p: float,
        timeout: int,
        logger: AgentLogger | None = None,
        stream: bool = True,
    ):
        self.base_url = f"http://{host}:{port}/v1"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.model = model
        self.stream = stream
        self.logger = logger or AgentLogger(None, enabled=False)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="sk-no-key-required",
            timeout=timeout,
        )
        self.__check_model()

    def __create(self, messages, tools, max_new_tokens, stream: bool):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_new_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            # One call per turn: the agent loop feeds each result back before
            # the model picks its next action, so a second parallel call would
            # be chosen blind.
            kwargs["parallel_tool_calls"] = False
        if stream:
            kwargs["stream"] = True
            # Token counts normally ride on the response body, which a stream
            # doesn't have; this asks for them on a final usage-only chunk.
            # A server that doesn't support the option just omits it and usage
            # stays None.
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return self.client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise RuntimeError(f"LlamaCpp chat completion failed: {exc}") from exc

    def __complete_whole(
        self: Self, messages, tools, max_new_tokens, splitter: ThinkSplitter
    ) -> Turn:
        """Wait for the server to finish, then read the response in one piece."""
        completion = self.__create(messages, tools, max_new_tokens, stream=False)
        choice = completion.choices[0]
        message = choice.message
        content, reasoning = fold(
            splitter.feed(message.content or "") + splitter.flush()
        )
        return Turn(
            content=content,
            reasoning=reasoning_field(message) + reasoning,
            tool_calls=[
                RawToolCall(
                    id=call.id or "",
                    name=call.function.name or "",
                    arguments=call.function.arguments or "",
                )
                for call in message.tool_calls or []
            ],
            finish_reason=choice.finish_reason or "",
            usage=completion.usage.model_dump() if completion.usage else None,
        )

    def __complete_streamed(
        self: Self,
        messages,
        tools,
        max_new_tokens,
        on_delta: Callable[[str], None] | None,
        on_tool_delta: Callable[[str, str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        splitter: ThinkSplitter | None = None,
    ) -> Turn:
        """Consume the SSE stream, handing output to the callbacks as it lands.

        Text arrives token by token and tool calls arrive as fragments keyed by
        `index` -- name first, then the arguments JSON a few characters at a
        time -- so both are reassembled here. `on_tool_delta` gets the call's
        name and the arguments assembled so far, which is how a front-end can
        show the file a write is building before it asks to approve it.
        Reasoning goes to `on_reasoning_delta` on a channel of its own, whether
        the server sent it as a field or inline in the content.

        The turn is only complete once the stream is exhausted, which is why
        the caller still sees a whole `Turn`: streaming changes when the user
        sees the output, not what the agent loop gets to decide on.
        """
        turn = Turn()
        splitter = splitter or ThinkSplitter()
        slots: dict[int, RawToolCall] = {}

        def deliver(pieces: list[tuple[str, str]]) -> None:
            """Route already-split output to its channel, in writing order."""
            for channel, text in pieces:
                if channel == REASONING:
                    turn.reasoning += text
                    if on_reasoning_delta is not None:
                        on_reasoning_delta(text)
                else:
                    turn.content += text
                    if on_delta is not None:
                        on_delta(text)

        stream = self.__create(messages, tools, max_new_tokens, stream=True)
        try:
            for chunk in stream:
                if chunk.usage:
                    turn.usage = chunk.usage.model_dump()
                if not chunk.choices:
                    # The trailing usage-only chunk carries no choices.
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    turn.finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                thought = reasoning_field(delta)
                if thought:
                    deliver([(REASONING, thought)])
                if delta.content:
                    deliver(splitter.feed(delta.content))
                for call in delta.tool_calls or []:
                    slot = slots.setdefault(
                        getattr(call, "index", 0) or 0, RawToolCall()
                    )
                    if call.id:
                        slot.id = call.id
                    if call.function is None:
                        continue
                    if call.function.name:
                        slot.name += call.function.name
                    if call.function.arguments:
                        slot.arguments += call.function.arguments
                    if on_tool_delta is not None:
                        on_tool_delta(slot.name, slot.arguments)
        except OpenAIError as exc:
            raise RuntimeError(
                f"LlamaCpp chat completion stream failed: {exc}"
            ) from exc
        deliver(splitter.flush())
        turn.tool_calls = [slots[index] for index in sorted(slots)]
        return turn

    def __parse_tool_calls(
        self: Self, raw_calls: list[RawToolCall]
    ) -> tuple[list[ToolCall], list[MalformedToolCall]]:
        calls: list[ToolCall] = []
        malformed: list[MalformedToolCall] = []
        for call in raw_calls:
            name = call.name
            raw_args = call.arguments or "{}"
            error = ""
            args = None
            if not name:
                error = "missing tool name"
            else:
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    error = f"arguments are not valid JSON ({exc})"
                else:
                    if not isinstance(args, dict):
                        error = (
                            "arguments are not a JSON object "
                            f"(got {type(args).__name__})"
                        )
            if error:
                self.logger.log(
                    "malformed_tool_call",
                    id=call.id,
                    name=name,
                    raw_args=raw_args,
                    error=error,
                )
                malformed.append(
                    MalformedToolCall(
                        id=call.id, name=name, raw_args=raw_args, error=error
                    )
                )
                continue
            calls.append(ToolCall(id=call.id, name=name, args=args))
        return calls, malformed

    def complete(
        self: Self,
        messages: list[dict],
        max_new_tokens: int,
        tools: list[dict] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_tool_delta: Callable[[str, str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Run one model turn, continuing it if it stops at the token limit.

        With streaming on, `on_delta` is called with each piece of text as the
        server produces it, `on_reasoning_delta` with the model's thinking, and
        `on_tool_delta` with each tool call as it is assembled; all three are
        views onto output being written, so the return value is the same either
        way.
        """
        self.logger.log(
            "llm_request",
            backend="llama-server",
            url=self.base_url + "/chat/completions",
            model=self.model,
            stream=self.stream,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_new_tokens,
            tools=[tool["function"]["name"] for tool in (tools or [])],
            messages=messages,
        )

        assistant_message = ""
        reasoning_message = ""
        tool_calls: list[ToolCall] = []
        malformed_tool_calls: list[MalformedToolCall] = []
        # The last round that reported token counts, kept so the caller can
        # show how full the context window is. A continuation round measures
        # the same conversation plus what has been written since, so the
        # latest one is the one to keep; it stays None on a backend that
        # doesn't report usage.
        last_usage: dict | None = None
        truncated = False
        round_index = 0
        has_more_data = True
        # Shared across continuation rounds so a `<think>` block cut off at the
        # token limit is still read as reasoning when generation resumes.
        splitter = ThinkSplitter()
        while has_more_data:
            if self.stream:
                turn = self.__complete_streamed(
                    messages,
                    tools,
                    max_new_tokens,
                    on_delta,
                    on_tool_delta,
                    on_reasoning_delta,
                    splitter,
                )
            else:
                turn = self.__complete_whole(messages, tools, max_new_tokens, splitter)
            round_index += 1

            chunk = turn.content
            assistant_message += chunk
            reasoning_message += turn.reasoning
            tool_calls, malformed_tool_calls = self.__parse_tool_calls(turn.tool_calls)
            finish_reason = turn.finish_reason
            usage = turn.usage
            last_usage = usage or last_usage
            self.logger.log(
                "llm_response",
                round=round_index,
                stream=self.stream,
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=[
                    {"name": call.name, "args": call.args} for call in tool_calls
                ],
                malformed_tool_calls=[
                    {"name": bad.name, "error": bad.error, "raw_args": bad.raw_args}
                    for bad in malformed_tool_calls
                ],
                reasoning=turn.reasoning,
                content=chunk,
            )

            # Only plain-text answers are continued; tool calls finish a turn.
            # A turn that spent its whole budget thinking counts as output too,
            # so a long think block is resumed rather than dropped as empty.
            if (
                finish_reason == "length"
                and not tool_calls
                and not malformed_tool_calls
                and (chunk or turn.reasoning)
            ):
                # The model resumes from its own partial output; when that is
                # all reasoning so far, the reasoning is what it needs back.
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": assistant_message or reasoning_message,
                    }
                ]
                self.logger.log(
                    "llm_continuation",
                    round=round_index,
                    reason="finish_reason=length; requesting continuation",
                    messages=messages,
                )
            else:
                # A tool call cut off at the token limit produces unparseable
                # arguments; flag it so the agent can give targeted feedback
                # instead of a generic "invalid JSON" notice.
                truncated = finish_reason == "length" and bool(
                    tool_calls or malformed_tool_calls
                )
                has_more_data = False

        return ModelResponse(
            content=assistant_message,
            tool_calls=tool_calls,
            malformed_tool_calls=malformed_tool_calls,
            truncated=truncated,
            reasoning=reasoning_message,
            usage=last_usage,
        )
