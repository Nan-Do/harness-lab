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
    logviz.py -f                       # follow (tail -f) new lines as they are appended
    logviz.py --raw                    # dump raw indented JSON instead of narrative
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


def col(text, *codes):
    if _NO_COLOR or not codes:
        return text
    return "".join(codes) + text + C.RESET


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
    "model_output": ("OUTPUT", (C.GRAY,)),
    "tool_call": ("TOOL", (C.BLUE, C.BOLD)),
    "tool_result": ("RESULT", (C.BLUE,)),
    "tool_approval": ("APPROVAL", (C.MAGENTA,)),
    "tool_error": ("TOOL ERR", (C.RED, C.BOLD)),
    "malformed_tool_call": ("MALFORMED", (C.RED,)),
    "retry": ("RETRY", (C.YELLOW,)),
    "final": ("FINAL", (C.GREEN, C.BOLD)),
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


def indent_block(text, prefix="      "):
    return "\n".join(prefix + line for line in str(text).split("\n"))


def short_session(session):
    if not session:
        return "-"
    # sessions look like 20260724-233204-c8f95e ; the trailing hex is the
    # useful discriminator when several sessions are interleaved.
    return session.split("-")[-1] if "-" in session else session


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
    return f"mode={d.get('mode')}", [f'"{truncate(d.get("text"), w, full)}"']


def render_request_start(d, w, full):
    hdr = f"max_steps={d.get('max_steps')} max_new_tokens={d.get('max_new_tokens')}"
    return hdr, [f'"{truncate(d.get("user_message"), w, full)}"']


def render_memory_update(d, w, full):
    mem = d.get("memory", {}) or {}
    task = mem.get("task")
    files = mem.get("files") or []
    notes = mem.get("notes") or []
    hdr = f"reason={d.get('reason')} files={len(files)} notes={len(notes)}"
    lines = []
    if full:
        lines.append(fmt_json_compact(mem, w, full=True))
    elif notes:
        lines.append(f"latest note: {truncate(notes[-1], w, full)}")
    return hdr, lines


def render_history_append(d, w, full):
    hdr = f"role={d.get('role')} name={d.get('name')} idx={d.get('index')}"
    content = d.get("content")
    lines = [truncate(one_line(content), w, full)] if content else []
    return hdr, lines


def render_prompt_built(d, w, full):
    hdr = (
        f"attempt={d.get('attempt')} tool_step={d.get('tool_step')} "
        f"messages={d.get('message_count')} roles={','.join(d.get('roles') or [])} "
        f"chars={d.get('chars')}"
    )
    return hdr, []


def render_llm_request(d, w, full):
    hdr = f"model={d.get('model')} backend={d.get('backend')} tools={len(d.get('tools') or [])}"
    lines = []
    if full:
        for m in d.get("messages") or []:
            lines.append(f"[{m.get('role')}] {fmt_json_compact(m, w, full=True)}")
    return hdr, lines


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
        lines.append(
            col("call: ", C.BLUE)
            + f"{tc.get('name')}({fmt_json_compact(tc.get('args'), w, full)})"
        )
    for mc in d.get("malformed_tool_calls") or []:
        lines.append(
            col("malformed: ", C.RED)
            + f"{mc.get('name')}: {truncate(mc.get('error'), w, full)}"
        )
    if d.get("content"):
        lines.append(col("says: ", C.YELLOW) + truncate(d.get("content"), w, full))
    return hdr, lines


def render_model_output(d, w, full):
    hdr = f"attempt={d.get('attempt')} tool_step={d.get('tool_step')}"
    lines = []
    calls = d.get("tool_calls") or []
    if calls:
        lines.append("tool_calls: " + ", ".join(calls))
    if d.get("content"):
        lines.append(truncate(d.get("content"), w, full))
    return hdr, lines


def render_tool_call(d, w, full):
    hdr = f"step={d.get('step')}"
    return hdr, [f"{d.get('name')}({fmt_json_compact(d.get('args'), w, full)})"]


def render_tool_result(d, w, full):
    hdr = f"step={d.get('step')} name={d.get('name')}"
    result = d.get("result")
    is_error = isinstance(result, str) and result.startswith("error:")
    line = truncate(result, w, full)
    if is_error:
        line = col(line, C.RED)
    return hdr, [line]


def render_tool_approval(d, w, full):
    granted = d.get("granted")
    granted_s = col("granted", C.GREEN) if granted else col("DENIED", C.RED, C.BOLD)
    risky_s = col("risky", C.RED, C.BOLD) if d.get("risky") else "safe"
    hdr = f"{granted_s} policy={d.get('policy')} {risky_s} read_only={d.get('read_only')}"
    return hdr, [f"{d.get('name')}({fmt_json_compact(d.get('args'), w, full)})"]


def render_tool_error(d, w, full):
    hdr = f"{d.get('name')}: {d.get('error_type')}"
    lines = [truncate(d.get("error"), w, full)]
    if full and d.get("traceback"):
        lines.append(indent_block(d.get("traceback")))
    return hdr, lines


def render_malformed_tool_call(d, w, full):
    hdr = f"attempt={d.get('attempt')} name={d.get('name')}"
    lines = [truncate(d.get("error"), w, full)]
    if full:
        lines.append("raw_args: " + truncate(d.get("raw_args"), w, full))
    return hdr, lines


def render_retry(d, w, full):
    hdr = f"attempt={d.get('attempt')}"
    return hdr, [truncate(d.get("notice"), w, full)]


def render_final(d, w, full):
    hdr = f"reason={d.get('reason')} attempts={d.get('attempts')} tool_steps={d.get('tool_steps')}"
    return hdr, [truncate(d.get("final"), w, full)]


RENDERERS = {
    "session_start": render_session_start,
    "repl_input": render_repl_input,
    "request_start": render_request_start,
    "memory_update": render_memory_update,
    "history_append": render_history_append,
    "prompt_built": render_prompt_built,
    "llm_request": render_llm_request,
    "llm_response": render_llm_response,
    "model_output": render_model_output,
    "tool_call": render_tool_call,
    "tool_result": render_tool_result,
    "tool_approval": render_tool_approval,
    "tool_error": render_tool_error,
    "malformed_tool_call": render_malformed_tool_call,
    "retry": render_retry,
    "final": render_final,
}


def render_generic(d, w, full):
    skip = {"event", "ts", "session", "depth", "_source", "_rsession"}
    parts = [f"{k}={fmt_json_compact(v, w, full)}" for k, v in d.items() if k not in skip]
    return " ".join(parts), []


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
        extra = " | " + " ; ".join(one_line(l) for l in lines) if lines else ""
        return first + extra

    out = [first]
    for l in lines:
        for sub in str(l).split("\n"):
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
    p.add_argument("--full", action="store_true", help="do not truncate long fields")
    p.add_argument("-w", "--width", type=int, default=180, help="truncation width (default 180)")
    p.add_argument("--stats", action="store_true", help="print per-session summary and exit")
    p.add_argument("--raw", action="store_true", help="dump raw indented JSON, no narrative")
    p.add_argument("-f", "--follow", action="store_true", help="tail -f the given/default files")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return p


def main(argv=None):
    global _NO_COLOR
    args = build_parser().parse_args(argv)

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
