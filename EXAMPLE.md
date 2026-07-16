&nbsp;

# Interactive Example

This is a hands-on walkthrough of `mini-coding-agent` against `llama-server`
on a small Python project.

The flow is:

1. create a fresh repo
2. launch the agent
3. implement `binary_search.py`
4. edit the implementation
5. add `pytest` tests
6. run tests
7. fix anything that fails

This example assumes:

- `llama-server` is already running with a tool-calling-capable model
  (see [README.md](README.md#install-llamacpp))
- you already cloned or forked `Nan-Do/mini-coding-agent`
- you already ran `uv sync` in your local `mini-coding-agent` folder

&nbsp;

## 1. Create a fresh repo

```bash
cd mini-coding-agent
mkdir -p ./tmp/binary-search-repo
cd ./tmp/binary-search-repo
git init
```

At this point the repo is basically empty:

```bash
ls -la
```

&nbsp;

## 2. Launch the agent

Open the agent from your `mini-coding-agent` clone, but point it at the new
repo:

```bash
cd mini-coding-agent
uv run mini-coding-agent \
  --cwd ./tmp/binary-search-repo \
  --model "Qwen3.5-4B-Q4_K_M.gguf"
```

This starts the interactive TUI. The status bar at the bottom shows the
active model, context size, current branch, approval policy, and session id.

&nbsp;

## 3. Ask it to implement binary search

At the prompt, type:

```text
Inspect this repository and create a minimal binary_search.py file. Implement an iterative binary_search(nums, target) function for a sorted list of integers. Return the index if the target exists and -1 if it does not. Keep the code very small.
```

The agent will typically call `list_files` and/or `read_file` to look
around, then call `write_file` to create `binary_search.py`. If approval is
set to `ask` (the default), you'll be prompted to approve the `write_file`
call before it happens.

After the agent finishes, inspect the result in another terminal or code
editor:

```bash
cat ./tmp/binary-search-repo/binary_search.py
```

&nbsp;

## 4. Ask it to edit the implementation

Now make a small follow-up change. Back in the agent, type:

```text
Update binary_search.py so it raises ValueError if the input list is not sorted in ascending order. Keep the implementation simple.
```

This exercises `read_file` followed by `patch_file` (a targeted edit rather
than a full rewrite). Check the file again:

```bash
cat ./tmp/binary-search-repo/binary_search.py
```

&nbsp;

## 5. Ask it to add unit tests

Back in the agent, type:

```text
Create test_binary_search.py with pytest tests for found, missing, empty list, first element, last element, and unsorted input raising ValueError. Keep the tests small and readable.
```

Inspect the new test file:

```bash
cat ./tmp/binary-search-repo/test_binary_search.py
```

&nbsp;

## 6. Ask it to run the tests

Back in the agent, type:

```text
Run pytest for this repo. If any test fails, fix the code or tests and rerun until everything passes.
```

This exercises `run_shell` (also gated by approval when `--approval ask` is
active). You can also verify manually in a different terminal window:

```bash
uv run --with pytest python -m pytest tmp/binary-search-repo
```

&nbsp;

## 7. Inspect the final repo state

Check what changed:

```bash
cd mini-coding-agent
cd ./tmp/binary-search-repo
git status --short
```

You should now have:

- `binary_search.py`
- `test_binary_search.py`

&nbsp;

## 8. Inspect the run log

Every step above was recorded as structured JSONL. Find the path with
`/log` inside the agent, or list the log directory directly:

```bash
ls .mini-coding-agent/logs/
uv run python log_viewer.py .mini-coding-agent/logs/run-<timestamp>.jsonl --show_events
```

This is the basis for comparing harness changes: rerun the same prompts
after editing the rules in `agent.py: build_prefix`, a tool's description in
`tools.py`, or the `--max-steps` / `--max-new-tokens` flags, and diff the
resulting logs.

&nbsp;

## 9. Useful interactive commands

While the agent is running, these commands are available:

- `/help` shows the available slash commands and what each one does.
- `/memory` prints the agent's distilled working memory for the current
  session.
- `/session` shows the path to the saved session JSON file on disk.
- `/log` shows the path to the current JSONL run log.
- `/reset` clears the current conversation history and working memory.
- `/clear` clears the visible conversation view without touching the saved
  session.
- `/exit` (or `/quit`) leaves the interactive agent.
