import json

from dataclasses import dataclass, field
from typing import Self
from collections.abc import Callable

from openai import OpenAI, OpenAIError

from agent_logging import AgentLogger
from app_types import MalformedToolCall, ModelResponse, ToolCall


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

    def __complete_whole(self: Self, messages, tools, max_new_tokens) -> Turn:
        """Wait for the server to finish, then read the response in one piece."""
        completion = self.__create(messages, tools, max_new_tokens, stream=False)
        choice = completion.choices[0]
        message = choice.message
        return Turn(
            content=message.content or "",
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
    ) -> Turn:
        """Consume the SSE stream, handing text to `on_delta` as it lands.

        Text arrives token by token and tool calls arrive as fragments keyed by
        `index` -- name first, then the arguments JSON a few characters at a
        time -- so both are reassembled here. The turn is only complete once
        the stream is exhausted, which is why the caller still sees a whole
        `Turn`: streaming changes when the user sees the text, not what the
        agent loop gets to decide on.
        """
        turn = Turn()
        slots: dict[int, RawToolCall] = {}
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
                if delta.content:
                    turn.content += delta.content
                    if on_delta is not None:
                        on_delta(delta.content)
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
        except OpenAIError as exc:
            raise RuntimeError(
                f"LlamaCpp chat completion stream failed: {exc}"
            ) from exc
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
    ) -> ModelResponse:
        """Run one model turn, continuing it if it stops at the token limit.

        With streaming on, `on_delta` is called with each piece of text as the
        server produces it; it is only a view onto the answer being written,
        so the return value is the same either way.
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
        tool_calls: list[ToolCall] = []
        malformed_tool_calls: list[MalformedToolCall] = []
        truncated = False
        round_index = 0
        has_more_data = True
        while has_more_data:
            if self.stream:
                turn = self.__complete_streamed(
                    messages, tools, max_new_tokens, on_delta
                )
            else:
                turn = self.__complete_whole(messages, tools, max_new_tokens)
            round_index += 1

            chunk = turn.content
            assistant_message += chunk
            tool_calls, malformed_tool_calls = self.__parse_tool_calls(turn.tool_calls)
            finish_reason = turn.finish_reason
            usage = turn.usage
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
                content=chunk,
            )

            # Only plain-text answers are continued; tool calls finish a turn.
            if (
                finish_reason == "length"
                and not tool_calls
                and not malformed_tool_calls
                and chunk
            ):
                messages = messages + [
                    {"role": "assistant", "content": assistant_message}
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
        )
