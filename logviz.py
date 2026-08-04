#!/usr/bin/env python3
"""
logviz.py - pretty-printer / viewer for harness-lab jsonl log events.

Usage:
    logviz.py                          # render all *.jsonl in cwd, narrative view
    logviz.py run-foo.jsonl run-bar.jsonl
    logviz.py --all                    # include verbose/internal bookkeeping events
    logviz.py --event tool_call,tool_result
    logviz.py --session c8f95e
    logviz.py --grep mandelbrot
    logviz.py --stats                  # per-session summary table instead of the stream
    logviz.py --compact                # one line per event
    logviz.py --full                   # do not truncate long fields
    logviz.py --max-lines 20           # lines shown per text block (0 = unlimited)
    logviz.py -f                       # follow (tail -f) new lines as they are appended
    logviz.py --raw                    # dump raw indented JSON instead of narrative

Events carrying a chat transcript (llm_request, llm_continuation) or tool
arguments are rendered as a readable conversation rather than as JSON: message
bodies are printed as text, and tool call arguments -- which travel as a JSON
string nested inside the JSON record -- are decoded first.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------
# color handling
# --------------------------------------------------------------------------


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


_NO_COLOR = False

# Max lines shown per text block (message body, file content, traceback...)
# before the rest is elided. 0 means unlimited; --full ignores it entirely.
_MAX_LINES = 6

# --compact promises one line per event, so the block layouts below collapse.
_COMPACT = False


def col(text, *codes):
    if _NO_COLOR or not codes:
        return text
    return "".join(codes) + text + C.RESET


_ANSI = re.compile(r"\033\[[0-9;]*m")


def plain_len(text):
    """Visible length of a string that may already carry color codes."""
    return len(_ANSI.sub("", str(text)))


# --------------------------------------------------------------------------
# event metadata
# --------------------------------------------------------------------------

# (short tag, color codes)
EVENT_STYLE = {
    "session_start": ("SESSION", (C.BOLD, C.MAGENTA)),
    "repl_input": ("INPUT", (C.CYAN,)),
    "request_start": ("REQUEST", (C.CYAN, C.BOLD)),
    "memory_update": ("MEMORY", (C.GRAY,)),
    "history_append": ("HISTORY", (C.GRAY,)),
    "prompt_built": ("PROMPT", (C.GRAY,)),
    "llm_request": ("LLM ->", (C.GRAY,)),
    "llm_response": ("LLM <-", (C.YELLOW,)),
    "llm_continuation": ("LLM ...", (C.YELLOW,)),
    "model_output": ("OUTPUT", (C.GRAY,)),
    "tool_call": ("TOOL", (C.BLUE, C.BOLD)),
    "tool_result": ("RESULT", (C.BLUE,)),
    "tool_outcome": ("OUTCOME", (C.GRAY,)),
    "tool_approval": ("APPROVAL", (C.MAGENTA,)),
    "tool_error": ("TOOL ERR", (C.RED, C.BOLD)),
    "tool_unknown": ("NO TOOL", (C.RED,)),
    "tool_blocked": ("BLOCKED", (C.YELLOW,)),
    "tool_name_recovered": ("RENAMED", (C.GRAY,)),
    "tool_args_normalized": ("ARGS FIX", (C.GRAY,)),
    "malformed_tool_call": ("MALFORMED", (C.RED,)),
    "context_budget": ("BUDGET", (C.GRAY,)),
    "context_usage": ("CTX", (C.GRAY,)),
    "history_window": ("EVICTED", (C.YELLOW,)),
    "repl_command": ("COMMAND", (C.CYAN,)),
    "repl_error": ("REPL ERR", (C.RED,)),
    "retry": ("RETRY", (C.YELLOW,)),
    "reset": ("RESET", (C.MAGENTA,)),
    "final": ("FINAL", (C.GREEN, C.BOLD)),
}

# chat roles, as they appear inside the `messages` payloads
ROLE_STYLE = {
    "system": (C.MAGENTA,),
    "user": (C.CYAN,),
    "assistant": (C.YELLOW,),
    "tool": (C.BLUE,),
}

# events that are internal bookkeeping / duplicate what llm_response,
# tool_call and tool_result already show. Hidden unless --all or explicitly
# requested with --event.
VERBOSE_EVENTS = {
    "memory_update",
    "history_append",
    "prompt_built",
    "llm_request",
    "model_output",
    # Two readings a step, the second of which is the `usage` llm_response
    # already carries; ask for them with --event context_usage when the
    # question is how full the window got.
    "context_usage",
}

SESSION_START_EVENTS = {"session_start"}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def truncate(s, width, full=False):
    s = "" if s is None else str(s)
    if full or width <= 0 or len(s) <= width:
        return s
    return s[:width] + col(f"... [+{len(s) - width} chars]", C.GRAY)


def one_line(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def fmt_ts(ts):
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%H:%M:%S.%f")[:-3]
    except (ValueError, TypeError):
        return str(ts)


def fmt_json_compact(obj, width, full=False):
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(obj)
    return truncate(one_line(s), width, full)


def short_session(session):
    if not session:
        return "-"
    # sessions look like 20260724-233204-c8f95e ; the trailing hex is the
    # useful discriminator when several sessions are interleaved.
    return session.split("-")[-1] if "-" in session else session


# --------------------------------------------------------------------------
# pretty-printing of payloads (messages, tool arguments, long text)
# --------------------------------------------------------------------------


def decode_json_string(value):
    """Return the object encoded in a JSON *string*, else None.

    Tool call arguments cross the wire as a JSON string nested inside the JSON
    record, so without this step they render as one escaped blob.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "{[" or stripped[-1] not in "}]":
        return None
    try:
        obj = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, (dict, list)) else None


def text_lines(text, w, full, gutter="", codes=()):
    """Lay text out as display lines, clipping each line and the line count."""
    if _COMPACT:
        return [col(truncate(one_line(text), w, full), *codes)]
    lines = str(text).split("\n")
    if len(lines) > 1 and not lines[-1].strip():
        # a trailing newline is the norm for file bodies; don't spend a line on it
        lines.pop()
    dropped = 0
    if not full and _MAX_LINES > 0 and len(lines) > _MAX_LINES:
        dropped = len(lines) - _MAX_LINES
        lines = lines[:_MAX_LINES]
    out = [gutter + col(truncate(line, w, full), *codes) for line in lines]
    if dropped:
        out.append(gutter + col(f"... [+{dropped} more lines]", C.GRAY))
    return out


def labeled_text(label, text, w, full, gutter="  | ", codes=()):
    """A label above its indented body -- folded onto one line when compact."""
    if _COMPACT:
        return [f"{label} {col(truncate(one_line(text), w, full), *codes)}"]
    return [label] + text_lines(text, w, full, gutter=gutter, codes=codes)


def fmt_json_block(obj, w, full, gutter=""):
    """Indented multi-line JSON, for values with no better textual form."""
    try:
        pretty = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        pretty = str(obj)
    return text_lines(pretty, w, full, gutter=gutter)


def is_block_value(value, w):
    """True when a value wants its own lines instead of sitting on the header."""
    if _COMPACT:
        return False
    if isinstance(value, str):
        return "\n" in value or (w > 0 and len(value) > w // 2)
    return isinstance(value, (dict, list)) and bool(value)


def fmt_scalar(value, w, full):
    try:
        return truncate(json.dumps(value, ensure_ascii=False, default=str), w, full)
    except (TypeError, ValueError):
        return truncate(str(value), w, full)


def fmt_args_inline(args, w, full):
    """`key=value, ...` on one line, or None when the values need their own."""
    if args is None:
        return ""
    if not isinstance(args, dict):
        return fmt_json_compact(args, w, full)
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        if is_block_value(value, w):
            return None
        # colors are dropped in compact mode: the line is truncated afterwards
        # and a cut through an escape sequence would bleed into the terminal
        label = key if _COMPACT else col(key, C.GRAY)
        parts.append(f"{label}={fmt_scalar(value, w, full)}")
    line = ", ".join(parts)
    if _COMPACT:
        return truncate(one_line(line), w, full)
    if not full and w > 0 and plain_len(line) > w:
        return None
    return line


def fmt_args_block(args, w, full, indent=""):
    """One key per line; text and nested structures get an indented body."""
    if not isinstance(args, dict):
        return fmt_json_block(args, w, full, gutter=indent)
    lines = []
    for key, value in args.items():
        decoded = decode_json_string(value)
        if decoded is not None:
            value = decoded
        label = f"{indent}{col(key, C.GRAY)}:"
        if isinstance(value, str) and is_block_value(value, w):
            lines.append(label)
            lines.extend(text_lines(value, w, full, gutter=indent + "  | "))
        elif isinstance(value, (dict, list)) and value:
            lines.append(label)
            lines.extend(fmt_json_block(value, w, full, gutter=indent + "  | "))
        else:
            lines.append(f"{label} {fmt_scalar(value, w, full)}")
    return lines


def fmt_call(name, args, w, full, indent="", label="", suffix=""):
    """`name(args)` on one line when it fits, else a header plus an arg block."""
    head = f"{indent}{label}{col(str(name), C.BOLD)}"
    tail = f"  {col(suffix, C.GRAY)}" if suffix else ""
    inline = fmt_args_inline(args, w, full)
    if inline is not None:
        return [f"{head}({inline}){tail}"]
    # Indentation delimits the arguments here, so parentheses would only add
    # a stray line above and below the block.
    return [head + tail] + fmt_args_block(args, w, full, indent + "    ")


def fmt_tool_call_message(call, w, full, indent=""):
    """One entry of a chat message's `tool_calls` array."""
    function = call.get("function") if isinstance(call, dict) else None
    function = function if isinstance(function, dict) else {}
    name = function.get("name") or call.get("name") or "?"
    call_id = call.get("id") or ""
    suffix = f"id={call_id}" if call_id else ""
    label = col("-> ", C.BLUE)

    raw = function.get("arguments", call.get("args"))
    args = raw if isinstance(raw, dict) else decode_json_string(raw)
    if args is None and raw not in (None, ""):
        # Unparseable arguments are exactly what a truncated call looks like,
        # so show the raw text instead of hiding the call.
        body = col(truncate(one_line(raw), w, full), C.RED)
        return [f"{indent}{label}{col(str(name), C.BOLD)}({body})"]
    return fmt_call(name, args or {}, w, full, indent=indent, label=label, suffix=suffix)


def fmt_message(index, message, w, full, indent=""):
    """Render one chat message: a header line plus its body and tool calls."""
    if not isinstance(message, dict):
        return [indent + truncate(one_line(message), w, full)]

    role = str(message.get("role") or "?")
    meta = []
    if message.get("name"):
        meta.append(f"name={message['name']}")
    if message.get("tool_call_id"):
        meta.append(f"id={message['tool_call_id']}")
    content = message.get("content")
    if isinstance(content, str) and content:
        meta.append(f"{len(content)} chars")

    header = f"{indent}{col(f'[{index}]', C.GRAY)} {col(role, *ROLE_STYLE.get(role, (C.GRAY,)))}"
    if meta:
        header += "  " + col(", ".join(meta), C.GRAY)
    lines = [header]

    body = indent + "    "
    if isinstance(content, str) and content.strip():
        lines.extend(text_lines(content, w, full, gutter=body + "| "))
    elif content:
        lines.extend(fmt_json_block(content, w, full, gutter=body + "| "))

    for call in message.get("tool_calls") or []:
        lines.extend(fmt_tool_call_message(call, w, full, indent=body))
    return lines


def messages_digest(messages):
    """`8 messages (system=1 user=3 assistant=2 tool=2, 4,210 chars)`."""
    counts = {}
    chars = 0
    for message in messages:
        role = str(message.get("role") or "?") if isinstance(message, dict) else "?"
        counts[role] = counts.get(role, 0) + 1
        if isinstance(message, dict):
            content = message.get("content")
            chars += len(content) if isinstance(content, str) else 0
    roles = " ".join(f"{role}={n}" for role, n in counts.items())
    return f"{len(messages)} messages ({roles}, {chars:,} chars)"


def fmt_messages(messages, w, full, indent=""):
    """A chat transcript as a conversation.

    Only the digest is shown by default -- every request re-sends the whole
    history, so the transcript is worth screen space only when asked for with
    --full.
    """
    if not messages:
        return []
    lines = [indent + col(messages_digest(messages), C.GRAY)]
    if not full:
        return lines
    for index, message in enumerate(messages, 1):
        lines.extend(fmt_message(index, message, w, full, indent))
    return lines


# --------------------------------------------------------------------------
# per-event rendering (returns: header_suffix:str, detail_lines:list[str])
# --------------------------------------------------------------------------


def render_session_start(d, w, full):
    hdr = (
        f"workspace={d.get('workspace_root')} policy={d.get('approval_policy')} "
        f"read_only={d.get('read_only')} resumed={d.get('resumed')} "
        f"history_len={d.get('history_len')}"
    )
    return hdr, []


def render_repl_input(d, w, full):
    return f"mode={d.get('mode')}", text_lines(d.get("text"), w, full, gutter="> ")


def render_request_start(d, w, full):
    hdr = f"max_steps={d.get('max_steps')} max_new_tokens={d.get('max_new_tokens')}"
    return hdr, text_lines(d.get("user_message"), w, full, gutter="> ")


def render_memory_update(d, w, full):
    mem = d.get("memory", {}) or {}
    task = mem.get("task")
    files = mem.get("files") or []
    notes = mem.get("notes") or []
    hdr = f"reason={d.get('reason')} files={len(files)} notes={len(notes)}"
    lines = []
    if full:
        if task:
            lines.append(f"{col('task:', C.GRAY)} {task}")
        if files:
            lines.append(f"{col('files:', C.GRAY)} " + ", ".join(str(f) for f in files))
        lines.extend(f"{col('note:', C.GRAY)} {note}" for note in notes)
    elif notes:
        lines.append(f"latest note: {truncate(notes[-1], w, full)}")
    return hdr, lines


def render_history_append(d, w, full):
    # tool entries carry name/args, message entries carry role
    who = d.get("role") or d.get("name") or d.get("entry")
    hdr = f"entry={d.get('entry')} {who} idx={d.get('index')}"
    lines = []
    if d.get("args") is not None:
        lines.extend(fmt_call(d.get("name"), d.get("args"), w, full))
    content = d.get("content")
    if content:
        lines.extend(text_lines(content, w, full))
    return hdr, lines


def render_prompt_built(d, w, full):
    hdr = (
        f"attempt={d.get('attempt')} tool_step={d.get('tool_step')} "
        f"messages={d.get('message_count')} roles={','.join(d.get('roles') or [])} "
        f"chars={d.get('chars')}"
    )
    memory = d.get("memory_text")
    lines = text_lines(memory, w, full) if full and memory else []
    return hdr, lines


def render_llm_request(d, w, full):
    hdr = (
        f"model={d.get('model')} backend={d.get('backend')} "
        f"stream={d.get('stream')} max_tokens={d.get('max_tokens')} "
        f"tools={len(d.get('tools') or [])}"
    )
    return hdr, fmt_messages(d.get("messages") or [], w, full)


def render_llm_continuation(d, w, full):
    hdr = f"round={d.get('round')} {d.get('reason')}"
    return hdr, fmt_messages(d.get("messages") or [], w, full)


def _fmt_usage(usage):
    if not usage:
        return ""
    p = usage.get("prompt_tokens")
    c = usage.get("completion_tokens")
    t = usage.get("total_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    s = f"tokens: prompt={p} completion={c} total={t}"
    if cached:
        s += f" cached={cached}"
    return s


def render_llm_response(d, w, full):
    hdr = f"round={d.get('round')} finish={d.get('finish_reason')} {_fmt_usage(d.get('usage'))}"
    lines = []
    for tc in d.get("tool_calls") or []:
        lines.extend(
            fmt_call(tc.get("name"), tc.get("args"), w, full, label=col("call: ", C.BLUE))
        )
    for mc in d.get("malformed_tool_calls") or []:
        lines.append(
            col("malformed: ", C.RED)
            + f"{mc.get('name')}: {truncate(mc.get('error'), w, full)}"
        )
        if mc.get("raw_args"):
            lines.extend(
                text_lines(mc["raw_args"], w, full, gutter="  | ", codes=(C.RED,))
            )
    if d.get("reasoning"):
        lines.extend(
            labeled_text(col("thinks:", C.MAGENTA), d["reasoning"], w, full, gutter="  ")
        )
    if d.get("content"):
        lines.extend(
            labeled_text(col("says:", C.YELLOW), d["content"], w, full, gutter="  ")
        )
    return hdr, lines


def render_model_output(d, w, full):
    hdr = f"attempt={d.get('attempt')} tool_step={d.get('tool_step')}"
    lines = []
    calls = d.get("tool_calls") or []
    if calls:
        lines.append("tool_calls: " + ", ".join(calls))
    for mc in d.get("malformed_tool_calls") or []:
        lines.append(
            col("malformed: ", C.RED)
            + f"{mc.get('name')}: {truncate(mc.get('error'), w, full)}"
        )
    if d.get("reasoning"):
        lines.extend(
            labeled_text(col("thinks:", C.MAGENTA), d["reasoning"], w, full, gutter="  ")
        )
    if d.get("content"):
        lines.extend(text_lines(d.get("content"), w, full))
    return hdr, lines


def render_tool_call(d, w, full):
    hdr = f"step={d.get('step')}"
    return hdr, fmt_call(d.get("name"), d.get("args"), w, full)


def render_tool_result(d, w, full):
    hdr = f"step={d.get('step')} name={d.get('name')}"
    result = d.get("result")
    is_error = isinstance(result, str) and result.startswith("error:")
    return hdr, text_lines(result, w, full, codes=(C.RED,) if is_error else ())


def render_tool_approval(d, w, full):
    granted = d.get("granted")
    granted_s = col("granted", C.GREEN) if granted else col("DENIED", C.RED, C.BOLD)
    risky_s = col("risky", C.RED, C.BOLD) if d.get("risky") else "safe"
    hdr = f"{granted_s} policy={d.get('policy')} {risky_s} read_only={d.get('read_only')}"
    return hdr, fmt_call(d.get("name"), d.get("args"), w, full)


def render_tool_error(d, w, full):
    hdr = f"{d.get('name')}: {d.get('error_type')}"
    lines = text_lines(d.get("error"), w, full, codes=(C.RED,))
    if d.get("args") is not None:
        lines.extend(fmt_call(d.get("name"), d.get("args"), w, full))
    if full and d.get("traceback"):
        lines.extend(text_lines(d.get("traceback"), w, full, gutter="  ", codes=(C.GRAY,)))
    return hdr, lines


def render_malformed_tool_call(d, w, full):
    hdr = f"attempt={d.get('attempt')} name={d.get('name')}"
    lines = text_lines(d.get("error"), w, full, codes=(C.RED,))
    if d.get("raw_args"):
        lines.extend(labeled_text(col("raw_args:", C.GRAY), d["raw_args"], w, full))
    return hdr, lines


def render_retry(d, w, full):
    hdr = f"attempt={d.get('attempt')}"
    return hdr, text_lines(d.get("notice"), w, full)


def render_final(d, w, full):
    hdr = f"reason={d.get('reason')} attempts={d.get('attempts')} tool_steps={d.get('tool_steps')}"
    return hdr, text_lines(d.get("final"), w, full)


RENDERERS = {
    "session_start": render_session_start,
    "repl_input": render_repl_input,
    "request_start": render_request_start,
    "memory_update": render_memory_update,
    "history_append": render_history_append,
    "prompt_built": render_prompt_built,
    "llm_request": render_llm_request,
    "llm_response": render_llm_response,
    "llm_continuation": render_llm_continuation,
    "model_output": render_model_output,
    "tool_call": render_tool_call,
    "tool_result": render_tool_result,
    "tool_approval": render_tool_approval,
    "tool_error": render_tool_error,
    "malformed_tool_call": render_malformed_tool_call,
    "retry": render_retry,
    "final": render_final,
}


SKIP_FIELDS = {"event", "ts", "session", "depth", "_source", "_rsession"}


def render_generic(d, w, full):
    """Fallback for events with no dedicated renderer.

    Scalars go on the header; a chat transcript, a nested structure or any
    multi-line text gets its own indented block rather than being flattened
    into escaped JSON.
    """
    parts = []
    lines = []
    for key, value in d.items():
        if key in SKIP_FIELDS:
            continue
        if key == "messages" and isinstance(value, list):
            lines.extend(fmt_messages(value, w, full))
        elif key == "args" and isinstance(value, dict):
            lines.extend(fmt_call(d.get("name", "args"), value, w, full))
        elif is_block_value(value, w):
            label = col(key + ":", C.GRAY)
            if isinstance(value, str):
                lines.extend(labeled_text(label, value, w, full))
            else:
                lines.append(label)
                lines.extend(fmt_json_block(value, w, full, gutter="  | "))
        else:
            parts.append(f"{key}={fmt_scalar(value, w, full)}")
    return " ".join(parts), lines


# --------------------------------------------------------------------------
# stream rendering
# --------------------------------------------------------------------------


def render_event(d, width, full, compact, source=None):
    event = d.get("event", "?")
    tag, codes = EVENT_STYLE.get(event, (event.upper(), (C.GRAY,)))
    ts = fmt_ts(d.get("ts"))
    sess = short_session(d.get("_rsession") or d.get("session"))
    depth = d.get("depth") or 0
    indent = "  " * depth

    renderer = RENDERERS.get(event, render_generic)
    hdr, lines = renderer(d, width, full)

    tag_s = col(f"{tag:>10}", *codes)
    prefix = f"{col(ts, C.GRAY)} {col(sess, C.CYAN if not _NO_COLOR else '')} {tag_s}"
    if source:
        prefix = f"{col('[' + source + ']', C.GRAY)} {prefix}"

    first = f"{indent}{prefix}  {hdr}".rstrip()

    if compact:
        extra = " | " + " ; ".join(one_line(line) for line in lines) if lines else ""
        return first + extra

    out = [first]
    for line in lines:
        for sub in str(line).split("\n"):
            out.append(f"{indent}              {sub}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def iter_events(path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"{path}:{lineno}: skipping invalid json line: {e}\n")


def load_all(paths):
    events = []
    multi = len(paths) > 1
    for p in paths:
        base = os.path.basename(p)
        for d in iter_events(p):
            if multi:
                d["_source"] = base
            events.append(d)
    events.sort(key=lambda d: (d.get("ts") or ""))
    resolve_sessions(events)
    return events


def resolve_sessions(events):
    """Some events (llm_request/llm_response, the raw malformed_tool_call
    emitted below the session layer) carry no 'session' field. Forward-fill
    the last known session per source file so display/stats can still
    attribute them correctly."""
    last = {}
    for d in events:
        key = d.get("_source")
        if d.get("session"):
            last[key] = d["session"]
        d["_rsession"] = d.get("session") or last.get(key)


def default_files():
    files = sorted(glob.glob("*.jsonl"))
    if not files:
        sys.stderr.write("no *.jsonl files found in current directory\n")
        sys.exit(1)
    return files


# --------------------------------------------------------------------------
# stats mode
# --------------------------------------------------------------------------


def print_stats(events):
    sessions = {}
    order = []
    for d in events:
        sid = d.get("_rsession") or d.get("session") or "-"
        if sid not in sessions:
            sessions[sid] = {
                "events": 0,
                "tool_calls": {},
                "errors": 0,
                "malformed": 0,
                "retries": 0,
                "approvals_denied": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "start": None,
                "end": None,
                "final": None,
                "final_reason": None,
                "user_message": None,
            }
            order.append(sid)
        s = sessions[sid]
        s["events"] += 1
        ts = d.get("ts")
        if ts:
            s["start"] = s["start"] or ts
            s["end"] = ts
        ev = d.get("event")
        if ev == "request_start":
            s["user_message"] = d.get("user_message")
        elif ev == "tool_call":
            name = d.get("name", "?")
            s["tool_calls"][name] = s["tool_calls"].get(name, 0) + 1
        elif ev == "tool_error":
            s["errors"] += 1
        elif ev == "malformed_tool_call":
            s["malformed"] += 1
        elif ev == "retry":
            s["retries"] += 1
        elif ev == "tool_approval" and not d.get("granted", True):
            s["approvals_denied"] += 1
        elif ev == "llm_response":
            usage = d.get("usage") or {}
            s["prompt_tokens"] += usage.get("prompt_tokens") or 0
            s["completion_tokens"] += usage.get("completion_tokens") or 0
        elif ev == "final":
            s["final"] = d.get("final")
            s["final_reason"] = d.get("reason")

    for sid in order:
        s = sessions[sid]
        print(col(f"=== session {sid} ===", C.BOLD, C.MAGENTA))
        if s["user_message"]:
            print(f"  task:      {truncate(one_line(s['user_message']), 100)}")
        if s["start"] and s["end"]:
            try:
                t0 = datetime.fromisoformat(s["start"])
                t1 = datetime.fromisoformat(s["end"])
                print(f"  duration:  {(t1 - t0).total_seconds():.1f}s")
            except ValueError:
                pass
        print(f"  events:    {s['events']}")
        if s["tool_calls"]:
            tc = ", ".join(f"{k}={v}" for k, v in sorted(s["tool_calls"].items()))
            print(f"  tools:     {tc}")
        print(
            f"  tokens:    prompt={s['prompt_tokens']} completion={s['completion_tokens']} "
            f"total={s['prompt_tokens'] + s['completion_tokens']}"
        )
        flags = []
        if s["errors"]:
            flags.append(col(f"errors={s['errors']}", C.RED))
        if s["malformed"]:
            flags.append(col(f"malformed={s['malformed']}", C.RED))
        if s["retries"]:
            flags.append(col(f"retries={s['retries']}", C.YELLOW))
        if s["approvals_denied"]:
            flags.append(col(f"denied={s['approvals_denied']}", C.RED))
        if flags:
            print("  issues:    " + " ".join(flags))
        if s["final"]:
            reason_col = C.GREEN if s["final_reason"] == "final_answer" else C.YELLOW
            print(f"  outcome:   {col(s['final_reason'], reason_col)}")
            print(f"  answer:    {truncate(one_line(s['final']), 160)}")
        print()


# --------------------------------------------------------------------------
# follow mode
# --------------------------------------------------------------------------


def follow(paths, args):
    handles = {}
    last_session = {}
    for p in paths:
        f = open(p, "r", encoding="utf-8")
        f.seek(0, os.SEEK_END)
        handles[p] = f
    sys.stderr.write(f"following {len(paths)} file(s)... (ctrl-c to stop)\n")
    try:
        while True:
            progressed = False
            for p, f in handles.items():
                line = f.readline()
                if line:
                    progressed = True
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("session"):
                        last_session[p] = d["session"]
                    d["_rsession"] = d.get("session") or last_session.get(p)
                    if should_show(d, args):
                        src = os.path.basename(p) if len(paths) > 1 else None
                        print(
                            render_event(
                                d, args.width, args.full, args.compact, source=src
                            )
                        )
                        sys.stdout.flush()
            if not progressed:
                time.sleep(0.3)
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------


def should_show(d, args):
    event = d.get("event")
    if args.event:
        return event in args.event
    if not args.all and event in VERBOSE_EVENTS:
        return False
    if args.session and args.session not in (d.get("_rsession") or d.get("session") or ""):
        return False
    if args.grep:
        blob = json.dumps(d, ensure_ascii=False)
        if not args.grep.search(blob):
            return False
    return True


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Pretty-print / visualize harness-lab jsonl log events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("files", nargs="*", help="jsonl files (default: *.jsonl in cwd)")
    p.add_argument(
        "-e",
        "--event",
        help="comma-separated list of event types to show (overrides --all filtering)",
    )
    p.add_argument("-s", "--session", help="only show events whose session contains this substring")
    p.add_argument("-g", "--grep", help="only show events whose raw JSON matches this regex")
    p.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="include verbose bookkeeping events (memory_update, history_append, "
        "prompt_built, llm_request, model_output)",
    )
    p.add_argument("-c", "--compact", action="store_true", help="one line per event")
    p.add_argument(
        "--full",
        action="store_true",
        help="do not truncate long fields; print chat transcripts in full",
    )
    p.add_argument("-w", "--width", type=int, default=180, help="truncation width (default 180)")
    p.add_argument(
        "--max-lines",
        type=int,
        default=_MAX_LINES,
        metavar="N",
        help=f"lines shown per text block, 0 for unlimited (default {_MAX_LINES})",
    )
    p.add_argument("--stats", action="store_true", help="print per-session summary and exit")
    p.add_argument("--raw", action="store_true", help="dump raw indented JSON, no narrative")
    p.add_argument("-f", "--follow", action="store_true", help="tail -f the given/default files")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return p


def main(argv=None):
    global _NO_COLOR, _MAX_LINES, _COMPACT
    args = build_parser().parse_args(argv)

    _MAX_LINES = max(0, args.max_lines)
    _COMPACT = args.compact

    if args.event:
        args.event = {e.strip() for e in args.event.split(",") if e.strip()}
    if args.grep:
        args.grep = re.compile(args.grep, re.IGNORECASE)

    _NO_COLOR = args.no_color or not sys.stdout.isatty() or os.environ.get("NO_COLOR")

    files = args.files or default_files()
    for p in files:
        if not os.path.isfile(p):
            sys.stderr.write(f"no such file: {p}\n")
            sys.exit(1)

    if args.follow:
        follow(files, args)
        return

    events = load_all(files)

    if args.stats:
        print_stats(events)
        return

    multi = len(files) > 1
    shown = 0
    for d in events:
        if not should_show(d, args):
            continue
        if args.raw:
            clean = {k: v for k, v in d.items() if k not in ("_source", "_rsession")}
            print(json.dumps(clean, indent=2, ensure_ascii=False))
        else:
            src = d.get("_source") if multi else None
            print(render_event(d, args.width, args.full, args.compact, source=src))
        shown += 1

    if shown == 0:
        sys.stderr.write("no events matched the given filters\n")


if __name__ == "__main__":
    main()
