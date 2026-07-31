import argparse
import sys
from datetime import datetime
from pathlib import Path

from agent import MiniAgent
from agent_logging import AgentLogger
from model_clients import LlamaCppModelClient
from session import SessionStore
from tui import run_tui
from utils import DEFAULT_MAX_NOISY_OUTPUT, TOOL_PROFILES
from workspace import WorkspaceContext


def build_logger(args: argparse.Namespace, repo_root: str) -> AgentLogger:
    if args.no_log:
        return AgentLogger(None, enabled=False)
    log_dir = Path(args.log_dir or Path(repo_root) / ".harness-lab" / "logs")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return AgentLogger(log_dir / f"run-{run_id}.jsonl")


def build_agent(args: argparse.Namespace) -> MiniAgent:
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(Path(workspace.repo_root) / ".harness-lab" / "sessions")
    logger = build_logger(args, workspace.repo_root)
    model = LlamaCppModelClient(
        model=args.model,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.llama_timeout,
        logger=logger,
    )
    options = dict(
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        logger=logger,
        tool_profile=args.tools,
        max_noisy_output=max(args.max_tool_output, 0),
        require_read_before_overwrite=not args.no_write_guard,
    )
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return MiniAgent.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            **options,
        )
    return MiniAgent(
        model_client=model,
        workspace=workspace,
        session_store=store,
        **options,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Harness Lab: an experimentation harness for coding agents backed by llama-server.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument(
        "--mode",
        choices=("auto", "tui", "headless"),
        default="auto",
        help=(
            "Interface mode. 'tui' launches the interactive Textual UI, "
            "'headless' runs a single request and prints the answer, "
            "'auto' picks headless when a prompt/piped input is present and the "
            "TUI otherwise."
        ),
    )
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument(
        "--model",
        default="Qwen3.5-4B-Q4_K_M.gguf",
        help="Model name (if the suggested model doesn't exist it will use the one provided by llama-server).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="llama-server host address."
    )
    parser.add_argument("--port", default=8080, help="llama-server port.")
    parser.add_argument(
        "--llama-timeout",
        type=int,
        default=300,
        help="Llama request timeout in seconds.",
    )
    parser.add_argument(
        "--resume", default=None, help="Session id to resume or 'latest'."
    )
    parser.add_argument(
        "--approval",
        choices=("ask", "auto", "never"),
        default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum tool/model iterations per request.",
    )
    parser.add_argument(
        "--tools",
        choices=TOOL_PROFILES,
        default="standard",
        help=(
            "Tool set exposed to the model. 'minimal' is the core read/write/shell "
            "set, 'standard' adds navigation and verification tools, 'full' adds "
            "delegation and revert. More tools give more reach but cost accuracy "
            "on small models."
        ),
    )
    parser.add_argument(
        "--max-tool-output",
        type=int,
        default=DEFAULT_MAX_NOISY_OUTPUT,
        help=(
            "Character budget for noisy tool output (shell logs, search floods); "
            "0 disables clipping. File reads are never clipped."
        ),
    )
    parser.add_argument(
        "--no-write-guard",
        action="store_true",
        help=(
            "Allow write_file to replace an existing file the model has not read "
            "in this session (the guard is on by default)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum model output tokens per step.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature sent to llama-server.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value sent to llama-server.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for JSONL run logs (default: <repo>/.harness-lab/logs).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable structured logging of memory, history, and llama-server traffic.",
    )
    return parser


def resolve_mode(mode: str, prompt: str) -> str:
    """Decide between the headless and TUI front-ends."""
    if mode != "auto":
        return mode
    # A prompt argument or piped stdin means a non-interactive, scriptable run.
    if prompt or not sys.stdin.isatty():
        return "headless"
    return "tui"


def read_prompt(args: argparse.Namespace) -> str:
    prompt = " ".join(args.prompt).strip()
    if prompt:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def run_headless(agent: MiniAgent, prompt: str) -> int:
    if not prompt:
        print("headless mode needs a prompt argument or piped stdin", file=sys.stderr)
        return 2
    agent.logger.log("repl_input", mode="headless", text=prompt)
    try:
        print(agent.ask(prompt))
    except RuntimeError as exc:
        agent.logger.log("repl_error", mode="headless", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    prompt = read_prompt(args)
    mode = resolve_mode(args.mode, prompt)

    agent = build_agent(args)
    endpoint = f"{args.host}:{args.port}"

    if mode == "headless":
        return run_headless(agent, prompt)

    return run_tui(
        agent,
        model=agent.model_client.model,
        context=agent.model_client.ctx,
        endpoint=endpoint,
    )


if __name__ == "__main__":
    sys.exit(main())
