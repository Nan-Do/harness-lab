import json

from typing import Dict, List, Self, Tuple

from openai import OpenAI, OpenAIError

from agent_logging import AgentLogger
from app_types import MalformedToolCall, ModelResponse, ToolCall


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
    ):
        self.base_url = f"http://{host}:{port}/v1"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.model = model
        self.logger = logger or AgentLogger(None, enabled=False)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="sk-no-key-required",
            timeout=timeout,
        )
        self.__check_model()

    def __create(self, messages, tools, max_new_tokens):
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
        try:
            return self.client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise RuntimeError(f"LlamaCpp chat completion failed: {exc}") from exc

    def __parse_tool_calls(
        self: Self, message
    ) -> Tuple[List[ToolCall], List[MalformedToolCall]]:
        calls: List[ToolCall] = []
        malformed: List[MalformedToolCall] = []
        for call in message.tool_calls or []:
            name = call.function.name or ""
            raw_args = call.function.arguments or "{}"
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
        messages: List[Dict],
        max_new_tokens: int,
        tools: List[Dict] | None = None,
    ) -> ModelResponse:
        self.logger.log(
            "llm_request",
            backend="llama-server",
            url=self.base_url + "/chat/completions",
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_new_tokens,
            tools=[tool["function"]["name"] for tool in (tools or [])],
            messages=messages,
        )

        assistant_message = ""
        tool_calls: List[ToolCall] = []
        malformed_tool_calls: List[MalformedToolCall] = []
        truncated = False
        round_index = 0
        has_more_data = True
        while has_more_data:
            completion = self.__create(messages, tools, max_new_tokens)
            round_index += 1

            choice = completion.choices[0]
            message = choice.message
            chunk = message.content or ""
            assistant_message += chunk
            tool_calls, malformed_tool_calls = self.__parse_tool_calls(message)
            finish_reason = choice.finish_reason
            usage = completion.usage.model_dump() if completion.usage else None
            self.logger.log(
                "llm_response",
                round=round_index,
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
