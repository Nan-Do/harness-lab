"""Plain-text front-end: the whole interaction with the model, on stdout.

Headless mode prints the answer and nothing else, which is what you want in a
pipeline and useless when the question is *how* the model got there. This mode
prints the conversation the TUI shows -- streamed model text, each tool call
with the body it carries, each result, every corrective notice -- as ordinary
text, so a run can be read in a terminal, piped into a file, or diffed against
another run without going through the JSONL log.

Rich is used only for styling: attached to a terminal it colors the trace, and
redirected to a file it writes plain text.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from rich.console import Console
from rich.text import Text

from agent import MiniAgent
from tool_support import size_note, split_args, streaming_body
from utils import HELP_DETAILS


class PlainView:
    """Renders agent events as a readable plain-text transcript."""

    def __init__(self, agent: MiniAgent, console: Console | None = None) -> None:
        self.agent = agent
        self.console = console or Console(soft_wrap=True, highlight=False)
        self._reset_turn()

    def _reset_turn(self) -> None:
        """Forget what the previous model turn put on screen.

        Everything here answers one question -- has this already been shown? --
        so that a streamed turn is not printed twice and an unstreamed one is
        not left out.
        """
        self._streamed_text = False
        self._body_key = ""
        self._body_shown = ""
        self._streamed_keys: set[str] = set()
        self._call = ""

    # --- Output helpers ----------------------------------------------------

    def _line(self, text: str, style: str = "") -> None:
        self.console.print(Text(text, style=style))

    def _write(self, text: str, style: str = "") -> None:
        """Emit text with no line break, for output still being generated."""
        self.console.print(Text(text, style=style), end="")

    def banner(self, endpoint: str) -> None:
        agent = self.agent
        self._line("harness-lab · plain mode", "bold cyan")
        self._line(
            f"model {agent.model_client.model} · endpoint {endpoint} · "
            f"workspace {agent.workspace.cwd} · approval {agent.approval_policy}",
            "dim",
        )

    def _write_body(self, name: str, key: str, body: str, path: str) -> None:
        """Print a tool-call body: what the call is about to commit, indented.

        Deliberately not syntax-highlighted. A highlighted block pads every
        line out to the terminal width, and this transcript is meant to be
        redirected to a file and diffed against another run, where that
        padding is just trailing whitespace.
        """
        self._line(f"  {self._body_label(name, key, path)}  ({size_note(body)})", "dim")
        indented = "\n".join(
            f"    {line}" if line.strip() else "" for line in body.splitlines()
        )
        self._line(indented, "dim")

    @staticmethod
    def _body_label(name: str, key: str, path: str) -> str:
        label = f"{name} · {path}" if path else name
        if key != "content":
            label += f" · {key}"
        return label

    # --- Agent events ------------------------------------------------------

    def event(self, event_type: str, **data: Any) -> None:
        if event_type == "thinking":
            self._reset_turn()
        elif event_type == "assistant_delta":
            if not self._streamed_text:
                self._write("\nmodel> ", "bold green")
                self._streamed_text = True
            self._write(data.get("text", ""))
        elif event_type == "tool_delta":
            self._on_tool_delta(data.get("name", ""), data.get("args_text", ""))
        elif event_type == "tool_call":
            self._on_tool_call(data.get("name", ""), data.get("args", {}))
        elif event_type == "tool_result":
            self._on_tool_result(data.get("name", ""), data.get("result", ""))
        elif event_type == "malformed_tool_call":
            self._line(
                f"\n  ! malformed tool call: {data.get('name') or 'unknown'} "
                f"({data.get('error', '')})",
                "yellow",
            )
        elif event_type == "retry":
            self._line(f"\n  ! retrying: {data.get('notice', '')}", "yellow")
        elif event_type == "final":
            self._on_final(data.get("text", ""))

    def _on_tool_delta(self, name: str, args_text: str) -> None:
        """Print a call's body as the model writes it.

        This is what makes the approval question that follows answerable: by
        the time it is asked, the file has already gone past on screen. A call
        that fills in several bodies (patch_file's old and new text) switches
        between them, so each one is announced when it starts.
        """
        self._call = name or self._call
        key, body = streaming_body(args_text)
        if not body:
            return
        if key != self._body_key:
            self._body_key = key
            self._body_shown = ""
            self._streamed_keys.add(key)
            self._line(f"\n  ⋯ {self._body_label(self._call, key, '')}", "dim")
        fresh = body[len(self._body_shown) :]
        if fresh:
            self._body_shown = body
            self._write(fresh, "dim")

    def _on_tool_call(self, name: str, args: Dict[str, Any]) -> None:
        inline, bodies = split_args(args)
        labels = " ".join(f"{key}={value}" for key, value in inline)
        self.console.print(
            Text.assemble(
                ("\n  → ", "bold blue"),
                (name, "bold"),
                (f" {labels}" if labels else ""),
            )
        )
        # A body that streamed is already on screen in full; only one that
        # never streamed (--no-stream) still needs printing.
        path = str(args.get("path") or "")
        for key, body in bodies:
            if key not in self._streamed_keys:
                self._write_body(name, key, body, path)

    def _on_tool_result(self, name: str, result: str) -> None:
        style = "red" if result.startswith("error:") else "blue"
        self._line(f"  ← {name}", style)
        self._line(result, "dim")

    def _on_final(self, text: str) -> None:
        if self._streamed_text:
            # The answer was printed as it was generated; just close the line.
            self.console.print()
            return
        self._write("\nanswer> ", "bold green")
        self._line(text)

    # --- Driving the agent -------------------------------------------------

    def ask(self, prompt: str, echo: bool = True) -> None:
        if echo:
            self._write("\nyou> ", "bold cyan")
            self._line(prompt)
        self.agent.logger.log("repl_input", mode="plain", text=prompt)
        try:
            self.agent.ask(prompt, on_event=self.event)
        except RuntimeError as exc:  # network/runtime failures from llama-server
            self.agent.logger.log("repl_error", mode="plain", error=str(exc))
            self._line(f"\nerror: {exc}", "red")

    def command(self, command: str) -> bool:
        """Handle a slash command; False means the session should end."""
        self.agent.logger.log("repl_command", command=command, mode="plain")
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self._line(HELP_DETAILS, "dim")
        elif command == "/memory":
            self._line(self.agent.memory_text(), "dim")
        elif command == "/session":
            self._line(self.agent.session_path, "cyan")
        elif command == "/log":
            self._line(self.agent.log_path, "cyan")
        elif command in {"/reset", "/clear"}:
            self.agent.reset()
            self._line("session reset", "green")
        else:
            self._line(f"unknown command: {command}", "red")
        return True


def run_plain(agent: MiniAgent, prompt: str, endpoint: str) -> int:
    """One-shot with a prompt, or a read-ask loop on an interactive terminal."""
    view = PlainView(agent)
    view.banner(endpoint)

    if prompt:
        view.ask(prompt)
        return 0

    if not sys.stdin.isatty():
        print("plain mode needs a prompt argument or piped stdin", file=sys.stderr)
        return 2

    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            view.console.print()
            return 0
        if not text:
            continue
        if text.startswith("/"):
            if not view.command(text):
                return 0
            continue
        view.ask(text, echo=False)
