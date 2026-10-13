# HumanEval through the harness

Scores a model on the 164 [HumanEval](https://github.com/openai/human-eval)
problems *as the harness runs it*: each problem gets its own workspace with
`solution.py` seeded from the problem's signature and docstring, the agent is
asked to implement it with the tools it normally has, and what gets scored is
the file on disk when the run ends.

That is a deliberately different number from a completion benchmark. A model
that knows the answer but cannot get it into a file through a tool call scores
zero here, and that is the point — this repository is about the harness, so the
benchmark measures the harness and the model together.

&nbsp;

## Requirements

- A running `llama-server` with a tool-calling chat template
  (`llama-server -m <model.gguf> --jinja`)
- `uv sync` in the repository root (the benchmark uses only the standard
  library, but the harness it drives needs `openai`)
- Network access the first time, to fetch the dataset

&nbsp;

## Quick start

```bash
# 1. cache the problem set (164 problems, ~300 KB)
uv run python benchmarks/humaneval/dataset.py

# 2. check the scorer on this machine -- no model involved
uv run python benchmarks/humaneval/run_benchmark.py --self-test

# 3. score whatever llama-server is serving
uv run python benchmarks/humaneval/run_benchmark.py --port 8080
```

Step 2 runs the dataset's own reference solutions through the same execution
path a model's code takes; it must print `164/164`. If it doesn't, the scoring
is broken on this machine and every number after it would inherit the bug.

Start small the first time — `--limit 5` — to see how long a problem takes on
your hardware before committing to all 164. A local 12B model takes roughly a
minute per problem at `--jobs 1`, so the full set is an evening's run unless
your server has parallel slots.

&nbsp;

## What it prints

```text
[  1/164] HumanEval/0     PASS  steps  3    24.1s  score 1/1 (100.0%)
[  2/164] HumanEval/1     FAIL  steps 10    61.7s  score 1/2 (50.0%)  failed

HumanEval · qwen3.5-4b
  endpoint 127.0.0.1:8080 · model Qwen3.5-4B-Q4_K_M.gguf
  tools standard · max-steps 10 · language none · samples 1

  SCORE   103 / 164 passed   (62.8%)

  failures 61
    failed (wrong answer)            48
    error (raised)                    5
    untouched (agent never edited)    8
  harness
    avg tool steps                  3.4
    step-limit stops                  6
    tool calls                      561  (failed 42: arg_error 27, not_found 15)
    malformed tool calls              3
    retries                           9
    crashed / timed out               0 / 1
```

The breakdown under the score is what keeps it honest. Eight problems where the
agent never wrote anything is not the same benchmark result as eight wrong
answers: the first is a harness failure to go and read the logs about, the
second is the model not knowing the answer. Each line maps to a status in
`results.jsonl`:

| Status | Means |
| --- | --- |
| `passed` | the problem's own tests passed |
| `failed` | tests ran, an assertion failed |
| `error` | the candidate raised (bad import, syntax, exception) |
| `timeout` | the candidate was still running after `--exec-timeout` |
| `untouched` | the run finished but `solution.py` was never edited |
| `harness_crashed` / `harness_timeout` | the agent run died before writing anything |
| `runner_error` | the benchmark itself failed to set the problem up |

&nbsp;

## Comparing models

`sweep.py` runs the same problems across several configurations and prints one
table. Since `llama-server` serves one model per process, there are two ways to
give it more than one model.

**Several servers at once**, one per port:

```bash
uv run python benchmarks/humaneval/sweep.py \
    --endpoint 127.0.0.1:8080=qwen3.5-4b \
    --endpoint 127.0.0.1:8081=gemma-4-12b
```

**One server, swapped between runs** — `--wait` holds each configuration until
its endpoint answers, so the sweep pauses while you restart `llama-server` with
the next model:

```bash
uv run python benchmarks/humaneval/sweep.py --config my-sweep.json --wait 900
```

A configuration file is a list of runs; every key mirrors a `run_benchmark.py`
flag (see `sweep.example.json`):

```json
{
  "runs": [
    {"label": "qwen3.5-4b", "port": "8080", "max_steps": 10},
    {"label": "qwen3.5-4b +language", "port": "8080", "max_steps": 14, "language": "python"}
  ]
}
```

The same file is how you measure a harness knob instead of a model: point two
runs at one endpoint and change one thing between them.

```text
Comparison

label                     model                       pass@1    passed      steps    toolfail   wall
------------------------------------------------------------------------------------------------------
qwen3.5-4b +language      Qwen3.5-4B-Q4_K_M.gguf      66.5%     109/164     5.1      14         55m
qwen3.5-4b                Qwen3.5-4B-Q4_K_M.gguf      62.8%     103/164     3.4      42         41m
```

&nbsp;

## Output layout

```text
runs/20260820-2114-qwen3.5-4b/
  results.jsonl                    one record per problem, written as it lands
  summary.json                     the score block as data
  workspaces/HumanEval_0/
    solution.py                    what the agent left behind
    .harness-lab/logs/run-*.jsonl  that problem's full harness log
```

Every problem keeps its own workspace and its own JSONL log, so a failure is
something you can read afterwards rather than guess at:

```bash
uv run python log_viewer.py runs/*/workspaces/HumanEval_11/.harness-lab/logs/run-*.jsonl
```

`results.jsonl` carries per-problem harness stats alongside the verdict — tool
calls, failed tool outcomes, retries, malformed calls, step-limit hits, peak
context — which is what makes "why did this model score worse" answerable:

```bash
jq -r 'select(.passed | not) | [.task_id, .status, .steps, .detail] | @tsv' \
    runs/*/results.jsonl | head
```

&nbsp;

## Knobs that matter

- `--jobs` (default 4) — problems run concurrently. `llama-server` needs
  matching parallel slots (`--parallel 4`) or the requests simply queue and
  every problem's timing includes the wait.
- `--max-steps` (default 10) — the agent's tool budget per problem. Read the
  file, write it, check it: too low and a capable model shows up as
  `untouched`, too high and a stuck one burns your evening.
- `--max-new-tokens` (default 512) — the single knob most likely to make a
  capable model look incapable. A model that thinks out loud before acting can
  spend the whole budget on prose and never reach its tool call, which scores
  as `untouched` however well it understood the problem; the same model with
  `--max-new-tokens 2048` writes the file. If a run comes back mostly
  `untouched`, raise this before concluding anything about the model.
- `--harness-timeout` (default 300s) — wall clock per problem, the backstop for
  a model that never stops generating.
- `--language python` — gives the agent `lint`, `type_check` and `write_tests`,
  which is a different benchmark (self-verification costs steps and may buy
  accuracy); worth running as its own configuration rather than as the default.
- `--samples N` — several attempts per problem, which turns the score into
  pass@k as well as pass@1.
- `--resume` — re-runs only what is missing from an existing `--out` directory,
  so an interrupted suite (Ctrl-C included) picks up where it stopped.
- `--abort-after` (default 5) — stop early when the first runs all fail the
  same way. A server that rejects the harness's requests fails all 164
  identically, and finding that out at the end is finding it out too late.

&nbsp;

## Executing model-written code

Scoring HumanEval means running code a language model wrote. Each candidate
runs in a fresh interpreter, in a throwaway directory, in its own process
group, with a wall-clock timeout, CPU/memory/file-size limits, no stdin, and
almost no environment. That contains the accidents this benchmark actually
produces — runaway loops, memory bombs, a `while True` waiting on input — and
it is **not** a sandbox against code that is trying to escape one. Benchmark a
model you do not trust inside a container or a VM.

The agent itself runs with `--approval auto`, since there is nobody to answer
an approval prompt. Its containment is the workspace: every problem directory
is initialised as its own git repository, because the harness resolves its root
with `git rev-parse --show-toplevel` and a workspace without a `.git` of its
own would take *this* checkout as its root — handing an auto-approved agent
write access to the harness. `prepare_workspace` verifies that boundary and
refuses to run the problem if it is not in place.

&nbsp;

## Files

| File | Role |
| --- | --- |
| `dataset.py` | Fetch, cache and select the 164 problems |
| `execution.py` | Run a candidate against a problem's tests, contained; pass@k |
| `runner.py` | One problem end to end: workspace, agent run, score, log stats |
| `report.py` | Records to a score block, a comparison table, a failure list |
| `run_benchmark.py` | CLI: score one configuration |
| `sweep.py` | CLI: score several and compare them |
