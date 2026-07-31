"""Tests for reassembling a streamed completion.

Streaming only changes how the answer arrives: text lands token by token and a
tool call arrives as fragments spread over many chunks. What the agent loop
gets to decide on has to come out identical to the non-streamed response, so
these tests pin both paths to the same result.
"""

from types import SimpleNamespace
from typing import List

from agent_logging import AgentLogger
from model_clients import LlamaCppModelClient


def client(stream: bool = True) -> LlamaCppModelClient:
    """A client with only the fields the completion paths touch."""
    model = LlamaCppModelClient.__new__(LlamaCppModelClient)
    model.base_url = "http://127.0.0.1:8080/v1"
    model.model = "test-model"
    model.temperature = 0.2
    model.top_p = 0.9
    model.stream = stream
    model.logger = AgentLogger(None, enabled=False)
    return model


def usage(total: int = 12) -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda: {"total_tokens": total})


def chunk(
    content: str = "",
    tool_calls: List[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content or None, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def call_fragment(
    index: int = 0,
    call_id: str = "",
    name: str = "",
    arguments: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=call_id or None,
        function=SimpleNamespace(name=name or None, arguments=arguments or None),
    )


def whole_response(
    content: str = "",
    tool_calls: List[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    message = SimpleNamespace(content=content or None, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage(),
    )


def whole_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def feed(model: LlamaCppModelClient, chunks: List[SimpleNamespace]) -> None:
    """Answer the next request with this canned stream (or single response)."""

    def create(messages, tools, max_new_tokens, stream):
        return iter(chunks) if stream else chunks[0]

    model._LlamaCppModelClient__create = create


def test_streamed_text_is_forwarded_and_reassembled():
    model = client()
    seen: List[str] = []
    feed(
        model,
        [
            chunk("Reading "),
            chunk("the file"),
            chunk(".", finish_reason="stop"),
        ],
    )

    response = model.complete([], 128, on_delta=seen.append)

    assert seen == ["Reading ", "the file", "."]
    assert response.content == "Reading the file."
    assert response.tool_calls == []


def test_streamed_tool_call_fragments_are_joined():
    model = client()
    feed(
        model,
        [
            chunk(tool_calls=[call_fragment(call_id="call_1", name="read_file")]),
            chunk(tool_calls=[call_fragment(arguments='{"path": ')]),
            chunk(tool_calls=[call_fragment(arguments='"main.py"}')]),
            chunk(finish_reason="tool_calls"),
        ],
    )

    response = model.complete([], 128, tools=[])

    assert [(call.id, call.name, call.args) for call in response.tool_calls] == [
        ("call_1", "read_file", {"path": "main.py"})
    ]
    assert response.malformed_tool_calls == []
    assert response.truncated is False


def test_usage_only_final_chunk_is_not_mistaken_for_content():
    model = client()
    tail = SimpleNamespace(choices=[], usage=usage(42))
    feed(model, [chunk("done", finish_reason="stop"), tail])

    response = model.complete([], 128)

    assert response.content == "done"


def test_streamed_tool_call_cut_off_at_the_token_limit_is_flagged():
    model = client()
    feed(
        model,
        [
            chunk(tool_calls=[call_fragment(call_id="call_1", name="write_file")]),
            chunk(
                tool_calls=[call_fragment(arguments='{"path": "main.py", "cont')],
                finish_reason="length",
            ),
        ],
    )

    response = model.complete([], 128, tools=[])

    assert response.truncated is True
    assert [bad.name for bad in response.malformed_tool_calls] == ["write_file"]


def test_streamed_answer_continues_past_the_token_limit():
    """A plain-text answer stopped at the limit is resumed, streamed included."""
    model = client()
    rounds = [
        [chunk("first half", finish_reason="length")],
        [chunk(" and second", finish_reason="stop")],
    ]
    seen: List[str] = []

    def create(messages, tools, max_new_tokens, stream):
        return iter(rounds.pop(0))

    model._LlamaCppModelClient__create = create

    response = model.complete([], 128, on_delta=seen.append)

    assert seen == ["first half", " and second"]
    assert response.content == "first half and second"
    assert rounds == []


def test_non_streamed_response_parses_the_same_way():
    model = client(stream=False)
    feed(
        model,
        [
            whole_response(
                content="reading",
                tool_calls=[whole_call("call_1", "read_file", '{"path": "main.py"}')],
                finish_reason="tool_calls",
            )
        ],
    )

    response = model.complete([], 128, tools=[])

    assert response.content == "reading"
    assert [(call.id, call.name, call.args) for call in response.tool_calls] == [
        ("call_1", "read_file", {"path": "main.py"})
    ]
