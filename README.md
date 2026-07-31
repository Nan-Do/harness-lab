# Harness Lab

Harness Lab is a small, readable coding-agent harness for local models
served through `llama-server` (llama.cpp). It exists to let you experiment
with the pieces that make an agent harness work — the system prompt and
rules, the tool set, how context and memory are compacted, and the approval
policy — and see how each choice changes what a small local model can
actually get done.

Every run writes a full structured log (prompt sent, model response, tool
calls, memory updates), so you can change one knob, rerun the same task, and
diff the two runs to see what the change actually did.

&nbsp;

## Repository layout

| File | Role |
| --- | --- |
| `main.py` | CLI entry point: argument parsing, front-end dispatch |
| `agent.py` | The agent loop: builds prompts, calls the model, runs tools, tracks memory |
| `tools.py` | Tool catalog, JSON-schema generation, approval gating, path sandboxing, write safety |
| `tool_support.py` | Repairing malformed tool calls, tool/argument aliases, patch diagnostics, output shaping |
| `model_clients.py` | OpenAI-compatible client for `llama-server`: streamed and whole completions, tool-call parsing |
| `app_types.py` | Dataclasses shared across the codebase (history entries, tool calls, session) |
| `workspace.py` | One-shot collection of repo facts (branch, status, recent commits, project docs) |
| `session.py` | Loads/saves session transcripts as JSON under `.harness-lab/sessions/` |
| `agent_logging.py` | Structured JSONL logger used everywhere in the loop |
| `log_viewer.py` | Standalone CLI to pretty-print and filter a run's JSONL log |
| `tui.py` | Textual-based interactive front-end |
| `plain.py` | Plain-text front-end: the whole interaction as text on stdout |
| `utils.py` | Shared constants and helpers: context budgets, tool profiles, ignored paths, formatting |

&nbsp;

## How the harness fits together

- **Workspace context** (`workspace.py`) — collected once when the agent
  starts: repo root, current branch, `git status --short`, the last 5
  commits, and the contents of any `AGENTS.md` / `README.md` /
  `pyproject.toml` / `package.json` found in the repo root or working
  directory. This becomes part of the system prompt.
- **Prompt and message shape** (`agent.py: build_prefix`, `build_messages`)
  — the model is sent a native OpenAI-style chat message list: one `system`
  message (rules + workspace context, stable across a session), then
  `user` / `assistant` / `tool` turns rebuilt from the session transcript
  on every step. Tool calls and their results are paired using
  `tool_call_id`, matching the format most tool-calling models are trained
  on. Tool output is never truncated: content reaches the model whole or is
  dropped whole. Reads that have gone out of date (a later read covering the
  same lines, or a `write_file` that replaced the file) are dropped, and when
  old turns have to go to fit the context the model is told what it can no
  longer see (`agent.py: _eviction_notice`).
- **Model transport** (`model_clients.py`) — an OpenAI-compatible client
  against `llama-server`. With `--stream` (default) the response is consumed
  as it is generated and text is handed to the front-end token by token;
  otherwise it is read in one piece. Either way the agent loop only sees a
  finished turn, so streaming changes when you see the answer, not how it is
  decided.
- **Tools** (`tools.py`) — the model acts only through named, schema-checked
  tools (see the table below). Every path argument is resolved and checked
  against the workspace root so a tool can't escape it.
- **Call repair** (`tool_support.py`) — a call whose *shape* is wrong is
  repaired instead of rejected: invented tool names are mapped onto real ones
  (`cat` → `read_file`), alias arguments are renamed (`file_path` → `path`),
  `{"arguments": {...}}` envelopes are unwrapped, loose scalars are coerced
  (`"20"` for an int), content sent as a list of lines is joined, and pasted
  line-number prefixes and code fences are stripped. Every repair is reported
  back to the model so it learns the right shape, and logged as
  `tool_args_normalized`.
- **Write safety** (`tools.py: ToolRegistry`) — every write is preceded by an
  automatic backup under `.harness-lab/backups/`, followed by a syntax check
  of the result (Python and JSON), and `write_file` refuses to replace an
  existing file this session has never read (disable with `--no-write-guard`).
- **Approval** (`tools.py: ToolRegistry._approve`) — tools marked `risky`
  (shell, file writes) are gated by `--approval ask|auto|never`; the TUI
  routes `ask` through a modal dialog instead of a terminal prompt.
- **Memory** (`app_types.Memory`, `agent.py: note_tool`/`remember`) — a
  small distilled summary (current task, up to 8 recently touched files, up
  to 5 recent notes) kept separately from the full transcript, so the model
  keeps track of what matters even after old transcript entries are
  clipped away.
- **Sessions and transcripts** (`session.py`) — the full interaction is
  persisted as JSON after every step, so a session can be resumed later
  with `--resume`.
- **Structured logging** (`agent_logging.py`, `log_viewer.py`) — every
  prompt, model response, tool call, and memory change is written to a
  JSONL run log for offline analysis and run-to-run comparison.
- **Delegation** (`delegate` tool, `agent.py: _make_delegate`) — the model
  can spin up a bounded, read-only child agent (own step budget, one level
  of nesting by default) to investigate something without spending the
  parent's own step budget.

&nbsp;

## Requirements

- Python 3.10+
- `llama.cpp` installed (`llama-server`)

Optional:

- `uv` for environment management and the `harness-lab` CLI entry point

This project has no runtime dependency beyond `openai` (used as an
OpenAI-compatible HTTP client against `llama-server`) and `textual` (TUI), so
you can also run it directly with `python main.py` if you don't want to use
`uv`.

&nbsp;

## Install llama.cpp

Install llama.cpp on your machine so the `llama-server` command is available
in your shell.

Llama.cpp link: [llama.cpp](https://github.com/ggml-org/llama.cpp)

Then verify:

```bash
llama-server --help
```

Start a server with a tool-calling-capable model, for example:

```bash
llama-server -m Qwen3.5-4B-Q4_K_M.gguf --jinja
```

`--jinja` (or an equivalent chat-template flag) is required for
`llama-server` to expose OpenAI-style function calling — this agent talks to
the model exclusively through the `tools` / `tool_calls` API, not through a
custom text format.

&nbsp;

## Project setup

Clone the repo or your fork and change into it:

```bash
git clone https://github.com/Nan-Do/mini-coding-agent.git
cd mini-coding-agent
```

Install dependencies:

```bash
uv sync
```

&nbsp;

## Basic usage

Start the interactive TUI:

```bash
uv run harness-lab
```

Run a single one-shot request and print the answer (no TUI):

```bash
uv run harness-lab "list the files in this repo"
```

Piped input is also treated as a one-shot headless request:

```bash
echo "summarize README.md" | uv run harness-lab
```

`--mode` overrides this auto-detection: `--mode tui` always launches the
interactive UI, `--mode headless` always runs one request and exits, and
`--mode plain` runs it while printing the whole interaction as text (see
[Plain mode](#plain-mode)).

A prompt given on the command line is never dropped: whichever mode is chosen
starts with it. In the TUI it runs as the first request and the session then
stays open for follow-ups, which is the whole difference from headless mode:

```bash
uv run harness-lab --mode tui "add a docstring to calc.py"
```

Without `uv`, run the script directly:

```bash
python main.py
```

By default it uses:

- approval: `ask`
- mode: `auto` (TUI if interactive, headless if a prompt/stdin is given)

For a concrete usage example, see [EXAMPLE.md](EXAMPLE.md).

&nbsp;

## Approval modes

Risky tools — `run_shell`, `run_tests`, `write_file`, `patch_file`,
`append_file`, `replace_lines`, `move_file`, `delete_file`, `revert_file` —
are gated by approval.

- `--approval ask`
  prompts before risky actions (default and recommended)
- `--approval auto`
  allows risky actions automatically, including arbitrary command execution
  and file writes by the model; use only with trusted prompts and trusted
  repositories
- `--approval never`
  denies risky actions

Example:

```bash
uv run harness-lab --approval auto
```

What the prompt shows is deliberately not the payload. A write is answered in
two parts: the file itself goes to the conversation first — streamed as the
model writes it, so it is already on screen by the time the call completes —
and the question that follows names the tool, its short arguments, and how big
the body is:

```text
  ⋯ write_file
def greet(name):
    """Return a greeting string for the given name."""
    return f"Hello, {name}!"

  → write_file path=greet.py
approve write_file path=greet.py · content: 3 lines, 101 B? [y/N]
```

The split is by size, not by argument name: anything short enough to read at a
glance (a path, a line range, a shell command) stays on the question, and
anything longer or spanning lines is shown as a block and summarized in the
prompt. A 500-line file and a one-line command therefore both produce a
readable dialog.

&nbsp;

## Resume sessions

The agent saves sessions under the target workspace root in:

```text
.harness-lab/sessions/
```

Resume the latest session:

```bash
uv run harness-lab --resume latest
```

Resume a specific session:

```bash
uv run harness-lab --resume 20260401-144025-2dd0aa
```

&nbsp;

## Interactive commands

These slash commands are handled by the TUI (and by plain mode's read-ask
loop) directly instead of being sent to the model as a task. Headless mode
runs a single request and has no command handling.

- `/help` — shows the list of available interactive commands
- `/memory` — prints the distilled session memory: current task, tracked
  files, and notes
- `/session` — prints the path to the current saved session JSON file
- `/log` — prints the path to the current JSONL run log
- `/reset` — clears the current session history and distilled memory but
  keeps you in the TUI
- `/clear` — clears the visible conversation log without touching the saved
  session
- `/exit` / `/quit` — exits the interactive session

&nbsp;

## Main CLI flags

```bash
uv run harness-lab --help
```

Without `uv`:

```bash
python main.py --help
```

- `prompt` — optional prompt words; on their own they run a headless one-shot,
  and with `--mode tui` / `--mode plain` they start that session instead
- `--mode` — `auto` (default), `tui`, `headless`, or `plain`
- `--cwd` — workspace directory the agent inspects and modifies; default `.`
- `--model` — model name/id requested from `llama-server`; if it doesn't
  match an available model, the server's first reported model is used
  instead; default `Qwen3.5-4B-Q4_K_M.gguf`
- `--host` — `llama-server` host address; default `127.0.0.1`
- `--port` — `llama-server` port; default `8080`
- `--llama-timeout` — request timeout in seconds; default `300`
- `--stream` / `--no-stream` — stream tokens from `llama-server` as they are
  generated instead of waiting for the whole completion; default on. See
  [Streaming](#streaming)
- `--resume` — resume a saved session by id, or `latest`; default: start a
  new session
- `--approval` — `ask`, `auto`, or `never`; default `ask`
- `--max-steps` — maximum tool/model iterations per request; default `6`
- `--max-new-tokens` — maximum model output tokens per step; default `512`
- `--temperature` — sampling temperature sent to `llama-server`; default `0.2`
- `--top-p` — nucleus sampling value sent to `llama-server`; default `0.9`
- `--log-dir` — directory for JSONL run logs; default
  `<workspace>/.harness-lab/logs`
- `--no-log` — disables structured logging entirely

&nbsp;

## Streaming

By default the harness streams from `llama-server`: the response is consumed
as it is generated instead of arriving in one piece when generation ends. On
a small local model producing a few hundred tokens a step, that is the
difference between a blank screen for several seconds and watching the answer
being written.

In the TUI, streamed text appears in a live view above the status bar while
it arrives. What happens to it when the turn ends depends on how the model
ended it:

- a **final answer** is rendered as the usual `agent` panel in the log
- text that preceded a **tool call** (the model thinking out loud on its way
  to acting) is committed to the log as an `agent · thinking` panel, so the
  reasoning stays readable after the live view is gone
- text cut short by an **error** mid-generation is kept too — it is the only
  record of how far the model got

Once the model starts assembling a **tool call**, the live view switches to
the body that call is writing — decoded straight out of the half-finished
arguments — and the finished body is written to the log as a syntax-
highlighted block. That is what makes the approval question that follows
answerable without putting the file inside the dialog (see
[Approval modes](#approval-modes)).

Headless mode is unchanged: it prints the final answer once, so piping the
output stays scriptable. [Plain mode](#plain-mode) streams the same
interaction as text.

Streaming is a view onto generation, not a change to the loop. The agent
still decides on a whole turn: tool calls are reassembled from their
fragments and only run once the stream is complete, and a turn that stops at
the token limit is continued exactly as before (the continuation streams
too). Running the same prompt with and without `--stream` should produce the
same `tool_call` and `final` events.

Use `--no-stream` to go back to one-shot completions — worth trying if a
backend's streamed tool calls arrive malformed, since it isolates whether a
failure comes from generation or from stream reassembly.

Notes:

- `llm_request` and `llm_response` log lines carry a `stream` field, so a run
  log records which transport produced it.
- Token counts for a streamed round come from `stream_options:
  {"include_usage": true}`; a server that doesn't send that final usage chunk
  logs `usage: null` while everything else keeps working.

&nbsp;

## Plain mode

`--mode plain` runs a request the way headless mode does, but prints the whole
interaction instead of only the answer: the model's text as it is generated,
every tool call with the body it carries, every result, and every corrective
notice sent back to the model.

```bash
uv run harness-lab --mode plain "add a docstring to calc.py"
```

```text
harness-lab · plain mode
model Qwen3.5-4B-Q4_K_M.gguf · endpoint 127.0.0.1:8080 · workspace /tmp/demo · approval ask

you> Create greet.py with a function greet(name) returning a greeting string.

  ⋯ write_file
def greet(name):
    """Return a greeting string for the given name."""
    return f"Hello, {name}!"

  → write_file path=greet.py
approve write_file path=greet.py · content: 3 lines, 101 B? [y/N] y
  ← write_file
wrote greet.py (3 lines, 101 chars)
syntax OK

model> Done. Created `greet.py` with a `greet(name)` function.
```

- Given a prompt (or piped stdin) it runs once and exits, like headless mode.
  Started on a terminal with no prompt, it drops into a bare read-ask loop
  that takes the same [slash commands](#interactive-commands) as the TUI.
- Output is text, not a UI: it is colored on a terminal and plain when
  redirected, so `--mode plain … > run.txt` gives a transcript that can be
  diffed against another run — the same knob-changing workflow as the JSONL
  logs, without a viewer.
- Nothing is deliberately hidden and nothing is clipped, which makes it the
  mode to reach for when a small model is doing something inexplicable and the
  TUI's panels are in the way.

`--mode plain` is never chosen by `auto`: its extra output would surprise
anything piping the answer somewhere, so it is always an explicit request.

&nbsp;

## Tools

Which tools the model sees is set by `--tools minimal|standard|full`. More
tools give the model more reach but cost accuracy on small models, so the
profile is an experiment knob rather than a fixed answer.

| Tool | Profile | Risky | Purpose |
| --- | --- | --- | --- |
| `list_files` | minimal | no | List a workspace directory, with an optional `depth` |
| `read_file` | minimal | no | Read a whole UTF-8 file, never clipped |
| `read_file_range` | minimal | no | Read part of a file by 1-based line range |
| `search` | minimal | no | Search file contents with `rg` (falls back to a substring scan if `rg` isn't installed) |
| `run_shell` | minimal | yes | Run a shell command at the repo root, with a timeout |
| `write_file` | minimal | yes | Write/replace a file's full content |
| `patch_file` | minimal | yes | Replace one exact, unique text block in a file |
| `append_file` | minimal | yes | Append text to a file, creating it if missing; used to build long files across multiple calls |
| `outline_file` | standard | no | List a file's classes and functions with line numbers |
| `find_files` | standard | no | Find files by name or glob |
| `replace_lines` | standard | yes | Replace a range of lines by number |
| `git_status` | standard | no | Show which files have been modified |
| `git_diff` | standard | no | Show uncommitted changes |
| `run_tests` | standard | yes | Detect and run the project's test suite |
| `move_file` | standard | yes | Move or rename a file |
| `delete_file` | standard | yes | Delete a file (backed up first) |
| `revert_file` | full | yes | Restore a file from its most recent automatic backup |
| `delegate` | full | no | Hand a bounded, read-only investigation to a child agent |

Notes:

- All path arguments are resolved relative to the repo root and rejected if
  they resolve outside it. Writes into `.git/` and `.harness-lab/` are
  refused even though they are inside the root.
- The model is limited to one tool call per step (`parallel_tool_calls`
  disabled), so each tool result is fed back before the next decision is
  made.
- Working on a file too large to hold in context is meant to go
  `outline_file` → `read_file_range` → `replace_lines`: every byte the model
  sees is complete, and it never needs the whole file at once.
- A tool call identical to the one before it is short-circuited with an
  error, as is an A-B-A-B alternation between two calls that aren't making
  progress. Tools whose output legitimately changes between calls
  (`run_shell`, `run_tests`, `git_status`, `git_diff`) are exempt from the
  alternation check, since re-running tests after an edit is the point.
- Failures answer with what to do next: an ambiguous `patch_file` lists the
  line numbers it matched, a missed one prints the closest real text in the
  file, and a missing path suggests files that do exist.
- `patch_file` retries a failed match without pasted line numbers and then
  ignoring indentation before giving up, so the two most common ways a small
  model mangles a copied block still land.
- Tool schemas are generated from a compact `"type[=default]|description"`
  spec per argument (see `tools.py: ToolRegistry._field_schema`); adding a
  new tool is a matter of writing a function and decorating it with
  `@agent_tool(...)`.

&nbsp;

## Logging

By default each run writes a structured [JSON Lines](https://jsonlines.org)
log to `.harness-lab/logs/run-<timestamp>.jsonl` (under the workspace
root, so it is already git-ignored). It records everything the agent stores
and exchanges, in order:

- `session_start` — session id, workspace, approval policy, whether it resumed
- `request_start` / `final` — the lifecycle of a single `ask()`
- `memory_update` — a full snapshot of the distilled memory (task, files,
  notes) every time it changes
- `history_append` — each message or tool entry added to the transcript
- `prompt_built` — the message roles and character counts sent to the model
  for that step
- `llm_request` / `llm_response` / `llm_continuation` — the raw messages,
  model params, and completions exchanged with the `llama-server` backend,
  including whether the round was streamed
- `tool_call` / `tool_result` — tool invocations and their output
- `tool_outcome` — how each call ended (`ok`, `arg_error`, `not_found`,
  `no_match`, `denied`, `guard_blocked`, `repeated`, `unknown_tool`), which
  is what makes a per-tool failure rate measurable across runs
- `tool_args_normalized` / `tool_name_recovered` — calls that arrived
  malformed and were repaired, and what was wrong with them
- `context_budget` — the measured prompt overhead and the resulting history
  budget for the chosen `--tools` profile
- `history_window` — turns dropped to fit the context, and what they were
- `model_output` / `retry` / `malformed_tool_call` — raw model output and
  how it was parsed, plus any corrective notice sent back to the model

Each line is one self-contained JSON object tagged with a UTC timestamp,
session id, and agent `depth` (nested `delegate` agents log to the same
file). Inspect a run with `jq`:

```bash
cat .harness-lab/logs/run-*.jsonl | jq 'select(.event=="llm_request")'
```

or with the bundled viewer, which pretty-prints and can filter by event
type or summarize event counts:

```bash
uv run python log_viewer.py .harness-lab/logs/run-*.jsonl --filter llm_request
uv run python log_viewer.py .harness-lab/logs/run-*.jsonl --show_events
```

&nbsp;

## Experimenting with the harness

Since the point of this project is to see how harness design affects
output quality, here's where each piece lives:

- **System prompt / rules** — `agent.py: MiniAgent.build_prefix`
- **How much workspace context is exposed** —
  `workspace.py: WorkspaceContext.build/text`, and `DOC_NAMES` in `utils.py`
- **Tool set** — `--tools minimal|standard|full`, or add, remove and reword
  tools in `tools.py` (`@agent_tool(...)`); a worse or better tool
  description/schema/example directly changes how reliably a small model
  calls it correctly
- **Call repair** — the alias tables and repair rules in `tool_support.py`;
  turning one off shows how much of a model's apparent competence comes from
  the harness rather than the model
- **Write friction** — `--no-write-guard` removes the read-before-overwrite
  requirement
- **Context/memory budget** — the history budget is derived from the model's
  real `n_ctx` in `agent.py: MiniAgent.__init__` (see `context_chars` in
  `utils.py`), spent by dropping whole turns in `_fit_history_budget`, and
  bounded by the stale-read rules in `_stale_read_indices`; file/note
  retention lives in `agent.py: note_tool`
- **Noisy-output budget** — `--max-tool-output` (0 disables it); applies to
  shell and search output only, never to file reads
- **Generation and step budget** — `--max-steps`, `--max-new-tokens`,
  `--temperature`, `--top-p`
- **Transport** — `--stream` / `--no-stream`; changes when output is visible,
  and is worth flipping when a backend's streamed tool calls look suspect
- **Friction on risky actions** — `--approval`; `--mode plain` makes a whole
  run readable as text when the question is why a change was approved

Since every run is fully logged, a natural workflow is: change one of the
above, rerun the same prompt against the same workspace, and diff the two
JSONL logs (or compare `final` events, tool-call counts, and `retry` /
`malformed_tool_call` events) to see the effect.

&nbsp;

## Example

See [EXAMPLE.md](EXAMPLE.md)

&nbsp;

## Notes & tips

- The agent talks to the model exclusively through the OpenAI-compatible
  `tools` / `tool_calls` function-calling API exposed by `llama-server`, not
  a custom text tag format — the model must be served with a chat template
  that supports tool calling.
- Weaker or smaller models will produce more `malformed_tool_call` and
  `retry` events; these are visible in the run log and are a useful signal
  when comparing models or prompt changes.
- The agent is intentionally small and optimized for readability and
  experimentation, not production robustness.

&nbsp;

## Origin

Originally forked from [`rasbt/mini-coding-agent`](https://github.com/rasbt/mini-coding-agent)
(Apache-2.0; see [LICENSE](LICENSE)) and substantially extended since —
native function calling instead of a custom tag format, a Textual TUI and
headless mode, more tools, structured JSONL logging with a viewer, and
transcript/memory compaction.
