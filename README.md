# Mini Coding Agent

Mini Coding Agent is a small, readable coding-agent harness for local models
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
| `main.py` | CLI entry point: argument parsing, headless/TUI dispatch |
| `agent.py` | The agent loop: builds prompts, calls the model, runs tools, tracks memory |
| `tools.py` | Tool catalog, JSON-schema generation, approval gating, path sandboxing |
| `model_clients.py` | OpenAI-compatible client for `llama-server`, tool-call parsing |
| `app_types.py` | Dataclasses shared across the codebase (history entries, tool calls, session) |
| `workspace.py` | One-shot collection of repo facts (branch, status, recent commits, project docs) |
| `session.py` | Loads/saves session transcripts as JSON under `.mini-coding-agent/sessions/` |
| `agent_logging.py` | Structured JSONL logger used everywhere in the loop |
| `log_viewer.py` | Standalone CLI to pretty-print and filter a run's JSONL log |
| `tui.py` | Textual-based interactive front-end |
| `utils.py` | Shared constants and helpers: clipping limits, ignored paths, formatting |

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
  on. Older turns are clipped harder than the last few, and superseded
  `read_file` results are dropped so re-reading a file doesn't bloat the
  context with stale content.
- **Tools** (`tools.py`) — the model acts only through named, schema-checked
  tools (see the table below). Arguments are validated and loosely-typed
  values are coerced (e.g. `"20"` for an int) before a tool runs. Every
  path argument is resolved and checked against the workspace root so a
  tool can't escape it.
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

- `uv` for environment management and the `mini-coding-agent` CLI entry point

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
uv run mini-coding-agent
```

Run a single one-shot request and print the answer (no TUI):

```bash
uv run mini-coding-agent "list the files in this repo"
```

Piped input is also treated as a one-shot headless request:

```bash
echo "summarize README.md" | uv run mini-coding-agent
```

`--mode` overrides this auto-detection: `--mode tui` always launches the
interactive UI, `--mode headless` always runs one request and exits.

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

Risky tools — `run_shell`, `write_file`, `patch_file`, `append_file` — are
gated by approval.

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
uv run mini-coding-agent --approval auto
```

&nbsp;

## Resume sessions

The agent saves sessions under the target workspace root in:

```text
.mini-coding-agent/sessions/
```

Resume the latest session:

```bash
uv run mini-coding-agent --resume latest
```

Resume a specific session:

```bash
uv run mini-coding-agent --resume 20260401-144025-2dd0aa
```

&nbsp;

## Interactive commands

These slash commands are handled by the TUI directly instead of being sent
to the model as a task (headless mode runs a single request and has no
command handling).

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
uv run mini-coding-agent --help
```

Without `uv`:

```bash
python main.py --help
```

- `prompt` — optional one-shot prompt words; if present, runs headless
- `--mode` — `auto` (default), `tui`, or `headless`
- `--cwd` — workspace directory the agent inspects and modifies; default `.`
- `--model` — model name/id requested from `llama-server`; if it doesn't
  match an available model, the server's first reported model is used
  instead; default `Qwen3.5-4B-Q4_K_M.gguf`
- `--host` — `llama-server` host address; default `127.0.0.1`
- `--port` — `llama-server` port; default `8080`
- `--llama-timeout` — request timeout in seconds; default `300`
- `--resume` — resume a saved session by id, or `latest`; default: start a
  new session
- `--approval` — `ask`, `auto`, or `never`; default `ask`
- `--max-steps` — maximum tool/model iterations per request; default `6`
- `--max-new-tokens` — maximum model output tokens per step; default `512`
- `--temperature` — sampling temperature sent to `llama-server`; default `0.2`
- `--top-p` — nucleus sampling value sent to `llama-server`; default `0.9`
- `--log-dir` — directory for JSONL run logs; default
  `<workspace>/.mini-coding-agent/logs`
- `--no-log` — disables structured logging entirely

&nbsp;

## Tools

| Tool | Risky | Purpose |
| --- | --- | --- |
| `list_files` | no | List entries in a workspace directory |
| `read_file` | no | Read a UTF-8 file by 1-based line range |
| `search` | no | Search the workspace with `rg` (falls back to a plain substring scan if `rg` isn't installed) |
| `run_shell` | yes | Run a shell command at the repo root, with a timeout |
| `write_file` | yes | Write/replace a file's full content |
| `patch_file` | yes | Replace one exact, unique text block in a file |
| `append_file` | yes | Append text to a file, creating it if missing; used to build long files across multiple calls |
| `delegate` | no | Hand a bounded, read-only investigation to a child agent |

Notes:

- All path arguments are resolved relative to the repo root and rejected if
  they resolve outside it.
- The model is limited to one tool call per step (`parallel_tool_calls`
  disabled), so each tool result is fed back before the next decision is
  made.
- If the model repeats the exact same tool call twice in a row, the tool
  registry short-circuits it with an error instead of running it again, to
  break simple loops.
- Tool schemas are generated from a compact `"type[=default]|description"`
  spec per argument (see `tools.py: ToolRegistry._field_schema`); adding a
  new tool is a matter of writing a function and decorating it with
  `@agent_tool(...)`.

&nbsp;

## Logging

By default each run writes a structured [JSON Lines](https://jsonlines.org)
log to `.mini-coding-agent/logs/run-<timestamp>.jsonl` (under the workspace
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
  model params, and completions exchanged with the `llama-server` backend
- `tool_call` / `tool_result` — tool invocations and their output
- `model_output` / `retry` / `malformed_tool_call` — raw model output and
  how it was parsed, plus any corrective notice sent back to the model

Each line is one self-contained JSON object tagged with a UTC timestamp,
session id, and agent `depth` (nested `delegate` agents log to the same
file). Inspect a run with `jq`:

```bash
cat .mini-coding-agent/logs/run-*.jsonl | jq 'select(.event=="llm_request")'
```

or with the bundled viewer, which pretty-prints and can filter by event
type or summarize event counts:

```bash
uv run python log_viewer.py .mini-coding-agent/logs/run-*.jsonl --filter llm_request
uv run python log_viewer.py .mini-coding-agent/logs/run-*.jsonl --show_events
```

&nbsp;

## Experimenting with the harness

Since the point of this project is to see how harness design affects
output quality, here's where each piece lives:

- **System prompt / rules** — `agent.py: MiniAgent.build_prefix`
- **How much workspace context is exposed** —
  `workspace.py: WorkspaceContext.build/text`, and `DOC_NAMES` in `utils.py`
- **Tool set** — add, remove, or reword tools in `tools.py`
  (`@agent_tool(...)`); a worse or better tool description/schema/example
  directly changes how reliably a small model calls it correctly
- **Context/memory budget** — `MAX_HISTORY` / `MAX_TOOL_OUTPUT` in
  `utils.py`, the stale-read pruning and per-turn clip limits in
  `agent.py: build_messages`, and the file/note retention limits in
  `agent.py: note_tool`
- **Generation and step budget** — `--max-steps`, `--max-new-tokens`,
  `--temperature`, `--top-p`
- **Friction on risky actions** — `--approval`

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
