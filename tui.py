"""Textual-based TUI front-end for Harness Lab."""

from __future__ import annotations

import threading
from time import monotonic
from typing import Any, Dict

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

from agent import MiniAgent
from tool_support import size_note, split_args, streaming_body
from utils import HELP_DETAILS, WELCOME_ART

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# Repainting on every token would spend more time in the UI than in the model,
# so streamed text lands in a buffer and the live view catches up on a tick.
_STREAM_REFRESH = 0.05
# The live view is a peephole onto text still being written; it keeps the tail
# in sight and lets the finished turn land in the log as a whole block.
_STREAM_TAIL_LINES = 8


def _body_syntax(key: str, body: str, path: str) -> Syntax:
    """Highlight a tool-call body the way its destination would be read."""
    if key == "command":
        lexer = "bash"
    elif path:
        lexer = Syntax.guess_lexer(path, code=body)
    else:
        lexer = "text"
    return Syntax(
        body,
        lexer,
        theme="ansi_dark",
        background_color="default",
        word_wrap=True,
    )


class ApprovalScreen(ModalScreen[bool]):
    """Blocking yes/no dialog used for the `ask` approval policy.

    Asks about the call, never with it: the file a write is about to commit
    has already scrolled past in the conversation, so the dialog names the
    tool, its short arguments, and how big the body is -- the things the
    decision actually turns on -- and stays readable at any payload size.
    """

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("n,escape", "deny", "Deny"),
    ]

    def __init__(self, name: str, args: Dict[str, Any]) -> None:
        super().__init__()
        self._name = name
        self._args = args

    def compose(self) -> ComposeResult:
        inline, bodies = split_args(self._args)
        body = Text("Approve risky tool\n\n", style="bold")
        body.append(self._name, style="bold yellow")
        for key, value in inline:
            body.append(f"\n  {key}  ", style="dim")
            body.append(value)
        for key, payload in bodies:
            body.append(f"\n  {key}  ", style="dim")
            body.append(size_note(payload), style="italic")
        if bodies:
            body.append("\n\nThe full text is shown above.", style="dim")
        yield Static(body, id="approval-body")
        with Horizontal(id="approval-buttons"):
            yield Button("Approve (y)", variant="success", id="approve")
            yield Button("Deny (n)", variant="error", id="deny")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#approve")
    def _on_approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def _on_deny(self) -> None:
        self.dismiss(False)


class MiniAgentApp(App):
    """A modern conversational front-end around :class:`MiniAgent`."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #log {
        height: 1fr;
        border: round $primary 30%;
        padding: 0 1;
        background: $surface;
    }
    #stream {
        display: none;
        height: auto;
        max-height: 10;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
        border-left: outer $success 50%;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }
    #prompt {
        border: round $accent;
        margin: 0;
    }
    ApprovalScreen {
        align: center middle;
    }
    ApprovalScreen > #approval-body {
        width: 70;
        padding: 1 2;
        border: round $warning;
        background: $panel;
    }
    #approval-buttons {
        width: 70;
        height: auto;
        align: center middle;
        padding: 1 0;
    }
    #approval-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+r", "reset_session", "Reset"),
    ]

    def __init__(
        self,
        agent: MiniAgent,
        model: str,
        context: int,
        endpoint: str,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.model = model
        self.context = context
        self.endpoint = endpoint
        self._busy = False
        self._spinner_index = 0
        # Written from the agent worker thread, rendered from the UI thread.
        self._stream_text = ""
        self._stream_call = ""
        self._stream_args = ""
        self._stream_painted = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="log", wrap=True, markup=True, highlight=False)
        yield Static("", id="stream")
        yield Static(self._status_text(), id="status")
        yield Input(placeholder="Ask the agent…  (/help for commands)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Harness Lab"
        self.sub_title = f"{self.model} · {self.endpoint}"
        # Route the `ask` approval policy through a modal dialog.
        if self.agent.approval_policy == "ask":
            self.agent.tools.approval_fn = self._approval_callback
        self.set_interval(0.1, self._tick_spinner)
        self._write_banner()
        self.query_one("#prompt", Input).focus()

    # --- Rendering helpers -------------------------------------------------

    @property
    def _logview(self) -> RichLog:
        return self.query_one("#log", RichLog)

    def _status_text(self) -> str:
        if self._busy:
            frame = _SPINNER[self._spinner_index % len(_SPINNER)]
            return f"{frame} working…"
        return (
            f"● ready   model {self.model} · ctx {self.context} · "
            f"branch {self.agent.workspace.branch} · "
            f"approval {self.agent.approval_policy} · "
            f"session {self.agent.session.id}"
        )

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    def _tick_spinner(self) -> None:
        if self._busy:
            self._spinner_index += 1
            self._refresh_status()

    def _write_banner(self) -> None:
        art = "\n".join(WELCOME_ART)
        self._logview.write(
            Panel(
                Text.assemble(
                    (art + "\n\n", "cyan"),
                    ("Harness Lab\n", "bold cyan"),
                    (f"workspace  {self.agent.workspace.cwd}\n", ""),
                    (f"model      {self.model}\n", ""),
                    (f"endpoint   {self.endpoint}\n", ""),
                    (f"approval   {self.agent.approval_policy}\n", ""),
                    ("\nType a request and press Enter. /help lists commands.", "dim"),
                ),
                border_style="cyan",
                title="welcome",
            )
        )

    def _write_user(self, text: str) -> None:
        self._logview.write(Panel(Text(text), title="you", border_style="cyan"))

    def _write_agent(self, text: str) -> None:
        self._logview.write(Panel(Markdown(text), title="agent", border_style="green"))

    def _write_tool_call(self, name: str, args: Dict[str, Any]) -> None:
        """Announce the call on one line, then show any body it carries.

        The body is what the user is about to approve, so it is rendered as
        code where it can be read, instead of being packed into the call line
        (or, worse, into the approval dialog).
        """
        inline, bodies = split_args(args)
        line = Text.assemble(("  → ", "bold blue"), (name, "bold"))
        for key, value in inline:
            line.append(f" {key}=", style="dim")
            line.append(value)
        self._logview.write(line)

        path = str(args.get("path") or "")
        for key, body in bodies:
            label = f"{name} · {path}" if path else name
            if key != "content":
                label += f" · {key}"
            self._logview.write(
                Panel(
                    _body_syntax(key, body, path),
                    title=label,
                    subtitle=size_note(body),
                    border_style="dim blue",
                    title_align="left",
                    subtitle_align="right",
                )
            )

    def _write_tool_result(self, name: str, result: str) -> None:
        is_error = result.startswith("error:")
        style = "red" if is_error else "blue"
        self._logview.write(
            Panel(
                Text(result),
                title=f"{name} result",
                border_style=style,
                title_align="left",
            )
        )

    def _write_notice(self, text: str, style: str = "yellow") -> None:
        self._logview.write(Text(text, style=style))

    # --- Streaming live view -----------------------------------------------

    @property
    def _streamview(self) -> Static:
        return self.query_one("#stream", Static)

    def _render_stream(self) -> None:
        """Repaint the live view with the tail of what is being written.

        Once a tool call starts assembling, the call takes over the view: the
        file a write is building is what matters at that point, and seeing it
        arrive is what makes the approval question that follows answerable.
        """
        view = Text()
        if self._stream_args:
            key, body = streaming_body(self._stream_args)
            header = self._stream_call or "tool call"
            if body:
                view.append(f"{header} · {key}\n", style="bold")
                view.append("\n".join(body.splitlines()[-_STREAM_TAIL_LINES:]))
            else:
                view.append(f"{header}…", style="bold")
        else:
            view.append("\n".join(self._stream_text.splitlines()[-_STREAM_TAIL_LINES:]))

        if not view.plain.strip():
            self._streamview.display = False
            return
        self._streamview.update(view)
        self._streamview.display = True

    def _reset_stream(self) -> None:
        self._stream_text = ""
        self._stream_call = ""
        self._stream_args = ""
        self._streamview.update("")
        self._streamview.display = False

    def _end_stream(self, commit: bool) -> None:
        """Close out the live view once the model turn is decided.

        A turn that ends in a tool call is the model thinking out loud on its
        way to acting; `commit` keeps that text in the log so the reasoning
        survives after the live view is gone. The final answer is committed by
        `_write_agent` instead, so it is not written twice.
        """
        text = self._stream_text.strip()
        self._reset_stream()
        if commit and text:
            self._logview.write(
                Panel(
                    Markdown(text),
                    title="agent · thinking",
                    border_style="dim green",
                    title_align="left",
                )
            )

    # --- Event handling ----------------------------------------------------

    @on(Input.Submitted, "#prompt")
    def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            return
        event.input.value = ""
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._write_user(text)
        self._start_task(text)

    def _handle_command(self, command: str) -> None:
        self.agent.logger.log("repl_command", command=command, mode="tui")
        if command in {"/exit", "/quit"}:
            self.exit()
        elif command == "/help":
            self._logview.write(Panel(Text(HELP_DETAILS), title="help", border_style="dim"))
        elif command == "/memory":
            self._logview.write(
                Panel(Text(self.agent.memory_text()), title="memory", border_style="dim")
            )
        elif command == "/session":
            self._write_notice(self.agent.session_path, style="cyan")
        elif command == "/log":
            self._write_notice(self.agent.log_path, style="cyan")
        elif command in {"/reset", "/clear"}:
            if command == "/reset":
                self.action_reset_session()
            else:
                self.action_clear_log()
        else:
            self._write_notice(f"unknown command: {command}", style="red")

    def _start_task(self, text: str) -> None:
        self._busy = True
        self._refresh_status()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = True
        self.agent.logger.log("repl_input", mode="tui", text=text)
        self._run_agent(text)

    @work(thread=True, exclusive=True)
    def _run_agent(self, text: str) -> None:
        try:
            self.agent.ask(text, on_event=self._on_agent_event)
        except Exception as exc:  # network/runtime failures from llama-server
            self.agent.logger.log("repl_error", mode="tui", error=str(exc))
            # Keep whatever was streamed before the failure: on a mid-stream
            # error it is the only record of how far the model got.
            self.call_from_thread(self._end_stream, True)
            self.call_from_thread(self._write_notice, f"error: {exc}", "red")
        finally:
            self.call_from_thread(self._finish_task)

    def _finish_task(self) -> None:
        self._reset_stream()
        self._busy = False
        self._refresh_status()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

    def _on_agent_event(self, event_type: str, **data: Any) -> None:
        """Called from the worker thread; marshal back onto the UI thread."""
        if event_type == "assistant_delta":
            self._on_stream_delta(text=data.get("text", ""))
        elif event_type == "tool_delta":
            self._on_stream_delta(
                call=data.get("name", ""), args=data.get("args_text", "")
            )
        elif event_type == "thinking":
            self.call_from_thread(self._reset_stream)
        elif event_type == "tool_call":
            self.call_from_thread(self._end_stream, True)
            self.call_from_thread(
                self._write_tool_call, data["name"], data.get("args", {})
            )
        elif event_type == "tool_result":
            self.call_from_thread(
                self._write_tool_result, data["name"], data.get("result", "")
            )
        elif event_type == "malformed_tool_call":
            self.call_from_thread(self._end_stream, True)
            self.call_from_thread(
                self._write_notice,
                f"malformed tool call: {data.get('name') or 'unknown'} "
                f"({data.get('error', '')})",
                "yellow",
            )
        elif event_type == "retry":
            self.call_from_thread(self._end_stream, True)
            self.call_from_thread(
                self._write_notice, "retrying: " + data.get("notice", ""), "yellow"
            )
        elif event_type == "final":
            self.call_from_thread(self._end_stream, False)
            self.call_from_thread(self._write_agent, data.get("text", ""))

    def _on_stream_delta(self, text: str = "", call: str = "", args: str = "") -> None:
        """Buffer streamed output on the worker thread, repainting on a tick.

        Every repaint is a cross-thread hop, so output accumulates between
        them; whatever the last hop missed is picked up when the turn ends.
        Tool arguments arrive already accumulated, so they replace rather than
        extend, and decoding them is left to the (throttled) repaint.
        """
        if text:
            self._stream_text += text
        if call or args:
            self._stream_call = call or self._stream_call
            self._stream_args = args
        elapsed = monotonic() - self._stream_painted
        if elapsed < _STREAM_REFRESH:
            return
        self._stream_painted = monotonic()
        self.call_from_thread(self._render_stream)

    def _approval_callback(self, name: str, args: Dict[str, Any]) -> bool:
        """Blocking approval prompt invoked from the agent worker thread."""
        done = threading.Event()
        result: Dict[str, bool] = {"approved": False}

        def request() -> None:
            def on_result(approved: bool | None) -> None:
                result["approved"] = bool(approved)
                done.set()

            self.push_screen(ApprovalScreen(name, args), on_result)

        self.call_from_thread(request)
        done.wait()
        return result["approved"]

    # --- Actions -----------------------------------------------------------

    def action_clear_log(self) -> None:
        self._reset_stream()
        self._logview.clear()
        self._write_banner()

    def action_reset_session(self) -> None:
        self.agent.reset()
        self._reset_stream()
        self._logview.clear()
        self._write_banner()
        self._write_notice("session reset", style="green")
        self._refresh_status()


def run_tui(agent: MiniAgent, model: str, context: int, endpoint: str) -> int:
    MiniAgentApp(agent, model=model, context=context, endpoint=endpoint).run()
    return 0
