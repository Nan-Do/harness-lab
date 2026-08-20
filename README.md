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
| `model_clients.py` | OpenAI-compatible client for `llama-server`: streamed and whole completions, tool-call parsing, reasoning extraction |
| `app_types.py` | Dataclasses shared across the codebase (history entries, tool calls, session) |
| `workspace.py` | One-shot collection of repo facts (branch, status, recent commits, project docs) |
| `languages.py` | Language packs: the label and prompt rules `--language` adds, and the key its tools are gated on |
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
  directory. This becomes part of the system prompt, and carries the
  `- language:` line when a run named one (see
  [Language packs](#language-packs)).
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
  decided. A thinking model's reasoning is separated from its answer here and
  carried on a channel of its own (see [Reasoning](#reasoning)).
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
- **Language packs** (`languages.py`) — `--language python` tells the model
  what it is writing and hands it the tools that can check that: `lint`
  (ruff), `type_check` (mypy), and `write_tests`, which writes a pytest file
  and runs it in the same step. Without the flag the harness stays
  language-agnostic, exactly as before. See
  [Language packs](#language-packs).
- **Write safety** (`tools.py: ToolRegistry`) — every write is preceded by an
  automatic backup under `.harness-lab/backups/`, followed by a syntax check
  of the result (Python and JSON), and `write_file` refuses to replace an
  existing file this session has never read (disable with `--no-write-guard`).
- **Approval** (`tools.py: ToolRegistry._approve`) — tools marked `risky`
  (shell, file writes) are gated by `--approval ask|auto|never`; the TUI
  routes `ask` through a modal dialog instead of a terminal prompt.
- **Step budget** (`agent.py: ask`, `_grant_more_steps`) — a request may spend
  `--max-steps` tool calls, which is what stops a small model going in
  circles. When the budget runs out the run does not simply die: with
  `--on-step-limit ask` (default) the front-end asks whether to continue —
  for another budget's worth or with no limit at all — and `--max-steps 0`
  starts a run unbounded. See [Step limit](#step-limit).
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

## Step limit

A request is allowed `--max-steps` tool calls (default `6`). The limit exists
because small models circle: they re-read the same file, re-run the same
command, and announce the same plan without ever finishing it. Cutting the run
off is how the harness keeps that from going on forever.

The trouble is that a run which is genuinely getting somewhere looks exactly
the same from the outside — out of steps, no answer — and only the person
watching the transcript can tell the two apart. So the limit is not a silent
verdict. When a request spends its budget, `--on-step-limit` decides:

- `--on-step-limit ask`
  the question goes to whoever is driving (default)
- `--on-step-limit stop`
  the request ends with a notice, as it always did

There are three answers:

- **continue** — grants another `--max-steps` worth and asks again when those
  are spent, so a long task can be nursed along a budget at a time
- **no limit** — drops the limit for the rest of this request; nothing more is
  asked and the run ends when the model answers (or you stop it)
- **stop** — ends the request where it stands

In the TUI the question is a modal dialog (`c` / `u` / `s`); in plain mode it
is a prompt on stdin:

```text
  ! step limit reached after 6 of 6 tool steps
  continue?  [c] +6 steps · [u] no limit · [s] stop >
```

To start a run with no limit at all, pass `--max-steps 0`:

```bash
uv run harness-lab --max-steps 0 --mode plain "port the tests to pytest"
```

An unlimited run needs a way out, and it has two. **ctrl+g** in the TUI (or
**Ctrl-C** in plain and headless mode) ends the request after the step in
flight, so the tool call already running still finishes and is recorded. And a
model that stops producing anything usable — malformed calls, empty replies —
is given up on after 8 consecutive turns, since an unbounded loop against a
stuck model would otherwise hammer `llama-server` indefinitely. A bounded run
has its own allowance for those turns, derived from the step budget.

Nothing continues unasked. `--on-step-limit ask` needs a front-end that can
put the question to someone, and headless mode, piped input, and delegated
child agents have nobody to answer — so there the policy falls back to
stopping, and an unattended run behaves exactly as it did before. The
decision is logged either way:

```text
STEPS      tool_steps=6 limit=6 policy=ask asked=True granted=no limit
```

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
- `--reasoning` / `--no-reasoning` — show the model's thinking in the TUI and
  plain mode, separately from its answer; default on. See
  [Reasoning](#reasoning)
- `--resume` — resume a saved session by id, or `latest`; default: start a
  new session
- `--approval` — `ask`, `auto`, or `never`; default `ask`
- `--max-steps` — maximum tool steps per request; `0` runs without a limit;
  default `6`. See [Step limit](#step-limit)
- `--on-step-limit` — what happens when a request spends its step budget:
  `ask` offers to continue, `stop` ends it with a notice; default `ask`
- `--language` — the language the workspace is worked in (`python`); adds the
  language to the system prompt and unlocks its verification tools; default:
  no language. See [Language packs](#language-packs)
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
  plan stays readable after the live view is gone
- a thinking model's **reasoning** streams on a channel of its own and lands
  in an `agent · reasoning` panel, whichever way the turn ended (see
  [Reasoning](#reasoning))
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
  logs `usage: null` while everything else keeps working — the context
  display falls back to an estimate (see [Context usage](#context-usage)).

&nbsp;

## Reasoning

Thinking models spend most of a turn reasoning before they say anything, and
on a small local model that reasoning is often the only explanation of why the
next tool call is what it is. The harness treats it as a channel of its own —
shown, but never confused with the answer.

It arrives from the server one of two ways, and which one depends on how
`llama-server` was started, not on the model:

- **as its own field** on the message or delta (`reasoning_content`, or
  `reasoning` / `thinking` on other backends) — what you get with
  `--reasoning-format deepseek`, the default for templates that support it
- **inline in the content**, wrapped in the `<think>` … `</think>` tags the
  model emits — what you get with `--reasoning-format none`, or from a server
  that doesn't parse reasoning at all

`model_clients.py` normalizes both (`reasoning_field` and `ThinkSplitter`), so
a front-end sees the same thing either way: reasoning on `reasoning_delta`
while it is being written, then a `reasoning` event with the whole turn's
thinking once the turn is decided. Tags never reach a front-end, and inline
reasoning no longer lands in the answer — `final`, the session transcript, and
the model's own history hold what it said, not what it thought.

Where it shows up:

- **TUI** — the live view shows `thinking…` with the tail of the reasoning
  while it arrives, and hands the view over to the answer (or to the body of
  a tool call) as soon as one starts. The finished reasoning is committed to
  the log as an `Agent · Reasoning` panel, kept even when the turn ends in a
  final answer, since the answer panel never contains it.
- **Plain mode** — a dimmed `think>` block, printed apart from the `model>`
  text it led to. A turn that crosses between the two more than once gets a
  fresh header each time, so the transcript stays readable when redirected to
  a file.
- **Headless mode** — unchanged: it prints the final answer and nothing else,
  so piping it stays scriptable.

Reasoning is shown, not remembered: it is never sent back to the model and is
not written to the session transcript, so a long think block costs nothing in
context on the next step. (The plain text a model writes *before a tool call*
is a different thing — that is ordinary content, and it is kept and replayed;
see `assistant_text` in `app_types.py`.)

`--no-reasoning` hides it in both front-ends. It is still logged either way:
`llm_response` and `model_output` carry a `reasoning` field, and the log
viewer prints it as `thinks:` above what the model said.

&nbsp;

## Context usage

The context window is the resource a run actually runs out of, and it runs
out quietly: once the prompt stops fitting, whole turns are dropped
(`history_window`) and the only symptom is a model that has forgotten
something it read. So every mode that shows anything shows how full the
window is, beside how big it is:

- **TUI** — on the status bar, idle and working alike:
  `⠹ working…   ctx 6.2k/32k (19%)`. It moves twice a step — an estimate when
  the prompt goes out, so the number is current while the model is still
  generating, then the server's own count when the turn lands.
- **Plain mode** — a dim line after each model turn, with what that turn
  generated: `ctx 6.2k/32k (19%) · 412 out`. The startup banner names the
  window itself (`context 32768 tokens`).
- **Headless mode** — unchanged: it prints the final answer and nothing else,
  so piping it stays scriptable. The numbers are in the run log either way.

What is counted as used is the prompt that was sent — the system rules, the
tool schemas, and whatever history survived the budget — since that is what
the conversation occupies before the model writes a token.

The counts come from the backend's own `usage` (a streamed round asks for it
with `stream_options: {"include_usage": true}`). A server that reports none
does not leave the display blank: the count is estimated from characters and
marked with a `~` — `ctx ~6.2k/32k (19%)` — so a guess is never read as a
measurement. The estimate deliberately errs high, on the same
characters-per-token ratio the history budget is sized with. If `n_ctx` is
unknown too, only what was sent is shown (`ctx 6.2k tokens`): a percentage of
an unknown window would be invented.

Every reading is logged as `context_usage` — phase, `n_ctx`, prompt and
completion tokens, percentage, and whether it was estimated — so "how full
did this run get, and when" is comparable across runs like everything else.

&nbsp;

## Plain mode

`--mode plain` runs a request the way headless mode does, but prints the whole
interaction instead of only the answer: the model's reasoning and text as they
are generated, every tool call with the body it carries, every result, and
every corrective notice sent back to the model.

```bash
uv run harness-lab --mode plain "add a docstring to calc.py"
```

```text
harness-lab · plain mode
model Qwen3.5-4B-Q4_K_M.gguf · endpoint 127.0.0.1:8080 · context 32768 tokens · workspace /tmp/demo · approval ask

you> Create greet.py with a function greet(name) returning a greeting string.

think> A new file with one function; write_file covers it in a single call.

  ⋯ write_file
def greet(name):
    """Return a greeting string for the given name."""
    return f"Hello, {name}!"

  ctx 3.1k/32k (9%) · 96 out

  → write_file path=greet.py
approve write_file path=greet.py · content: 3 lines, 101 B? [y/N] y
  ← write_file
wrote greet.py (3 lines, 101 chars)
syntax OK

model> Done. Created `greet.py` with a `greet(name)` function.
  ctx 3.4k/32k (10%) · 21 out
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
  TUI's panels are in the way. `--no-reasoning` is the one thing that trims
  it, for a transcript of what the model did without how it talked itself
  into it.

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
| `lint` | `--language python` | no | Lint the code with ruff and report what it found |
| `type_check` | `--language python` | no | Type-check the code with mypy and report what it found |
| `write_tests` | `--language python` | yes | Write a pytest file for a module and run it in the same step |

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
- The last three are gated on `--language`, not on `--tools`: they appear in
  every profile of a run that named their language and in none of a run that
  did not (see [Language packs](#language-packs)).

&nbsp;

## Language packs

Everything above is language-agnostic, which is the right default for a
harness pointed at whatever repository you have — and the wrong one when you
know what you are working on. `--language python` says so, and a run that says
so gets three things.

**The model is told.** The workspace block gains a line:

```text
Workspace:
- language: Python
- cwd: /tmp/demo
```

and the system prompt gains a short set of rules that name the tools below and
say when to reach for them — that new code is checked and covered, and that
code is not finished until the checks come back clean.

**The verification tools appear.**

| Tool | Runs | Risky |
| --- | --- | --- |
| `lint` | `ruff check --no-cache --output-format concise <path>` | no |
| `type_check` | `mypy --ignore-missing-imports --check-untyped-defs <path>` | no |
| `write_tests` | writes a pytest file, then runs it | yes |

`lint` and `type_check` only read the code, so they are not gated by approval:
a check the model has to ask permission for is a check it stops running. Each
reports the checker's own output plus a one-line verdict, since `exit_code: 1`
and a wall of findings is not a conclusion a small model reliably draws:

```text
  → lint path=calc.py
  ← lint
$ ruff check --no-cache --output-format concise calc.py
exit_code: 1
stdout:
calc.py:2:12: F401 [*] `os` imported but unused
stderr:
(empty)
ruff reported issues; fix them and run lint again
```

Both flags on the `mypy` line are there because the alternative is a checker
that says nothing useful about freshly written code: missing third-party stubs
are the loudest thing mypy has to say about a small project and none of it is
about what the model wrote, and a model writing plain Python annotates
nothing — which by default means mypy skips those function bodies entirely and
reports success without having looked inside them.

**Tests are generated and run in one step.** `write_tests` takes the module to
cover and the test source the model wrote for it, writes it where the project
keeps tests (`tests/test_<module>.py`, or beside the module if there is no
`tests/`), and immediately runs pytest over it:

```text
  → write_tests path=calc.py
approve write_tests path=calc.py · content: 5 lines, 65 B? [y/N] y
  ← write_tests
wrote tests/test_calc.py (5 lines, 65 chars)
syntax OK
$ /usr/bin/python -m pytest -q tests/test_calc.py
exit_code: 0
stdout:
1 passed in 0.01s
stderr:
(empty)
tests passed: calc.py does what these tests say
```

Writing and running are one tool rather than two because the interesting
result is the second one, and a small model that has just written a test file
will happily call it a day. The same reasoning shapes what it refuses: content
with no `test_*` function, and a filename pytest would not collect, are both
turned down before anything is written, because a suite that collects nothing
passes — and a pass is what the model reads as proof. `pytest` exiting with
"collected nothing" or "no module named pytest" is reported as *neither* a
pass nor a failing test, for the same reason. Re-run the tests later with
`run_tests`; `write_tests` is for writing them.

The checkers are found in the order that respects the workspace: its own
`.venv` first (invoked as `python -m <tool>`, which survives a moved
virtualenv where the console script's absolute shebang does not), then a
binary on `PATH`, then the environment the harness itself runs in — so
`uv sync` here is enough to check a project that has no environment of its
own. A checker that is nowhere to be found is reported as a checker that is
missing, with what to do instead, rather than as code that failed a check.

```bash
uv run harness-lab --language python --mode plain "write factorial.py and prove it works"
```

Nothing about the flag is load-bearing for the rest of the harness: a run
without it behaves exactly as it did before, and adding a language is adding
an entry to `languages.py` plus its tools in `tools.py`, declared with
`language="<key>"`.

&nbsp;

## Logging

By default each run writes a structured [JSON Lines](https://jsonlines.org)
log to `.harness-lab/logs/run-<timestamp>.jsonl` (under the workspace
root, so it is already git-ignored). It records everything the agent stores
and exchanges, in order:

- `session_start` — session id, workspace, approval policy, the `--language`
  the run was started with, whether it resumed
- `request_start` / `final` — the lifecycle of a single `ask()`
- `memory_update` — a full snapshot of the distilled memory (task, files,
  notes) every time it changes
- `history_append` — each message or tool entry added to the transcript
- `prompt_built` — the message roles and character counts sent to the model
  for that step
- `llm_request` / `llm_response` / `llm_continuation` — the raw messages,
  model params, and completions exchanged with the `llama-server` backend,
  including whether the round was streamed and the `reasoning` the model
  produced for it
- `tool_call` / `tool_result` — tool invocations and their output
- `tool_outcome` — how each call ended (`ok`, `arg_error`, `not_found`,
  `no_match`, `denied`, `guard_blocked`, `repeated`, `unknown_tool`), which
  is what makes a per-tool failure rate measurable across runs
- `tool_args_normalized` / `tool_name_recovered` — calls that arrived
  malformed and were repaired, and what was wrong with them
- `context_budget` — the measured prompt overhead and the resulting history
  budget for the chosen `--tools` profile
- `context_usage` — how full the window was for each prompt sent and each
  turn that came back (see [Context usage](#context-usage))
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
- **Knowing the language** — `--language python` adds the language to the
  prompt and the checkers to the tool set; running the same task with and
  without it is the cleanest measure of whether a small model uses
  verification it is given, or only claims to
  (see [Language packs](#language-packs))
- **Context/memory budget** — the history budget is derived from the model's
  real `n_ctx` in `agent.py: MiniAgent.__init__` (see `context_chars` in
  `utils.py`), spent by dropping whole turns in `_fit_history_budget`, and
  bounded by the stale-read rules in `_stale_read_indices`; file/note
  retention lives in `agent.py: note_tool`. How much of the window a run is
  actually spending is on the screen while it runs and in the `context_usage`
  log lines afterwards (see [Context usage](#context-usage)), which is what
  makes a change to any of this measurable rather than a matter of taste
- **Noisy-output budget** — `--max-tool-output` (0 disables it); applies to
  shell and search output only, never to file reads
- **Generation and step budget** — `--max-steps`, `--on-step-limit`,
  `--max-new-tokens`, `--temperature`, `--top-p`; how often a run hits the
  limit, and whether the extra steps granted after it actually finish the
  task, is itself a measure of whether a model can hold a task together
- **Transport** — `--stream` / `--no-stream`; changes when output is visible,
  and is worth flipping when a backend's streamed tool calls look suspect
- **Visible thinking** — `--reasoning` / `--no-reasoning`; changes only what
  the front-end shows, since reasoning is logged either way and never reaches
  the next prompt
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
