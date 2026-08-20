"""Tests for --language: the flag that tells a run what it is writing.

Naming the language does three things, and each is here: it puts the language
in front of the model (a workspace line and a set of rules), it unlocks the
tools that can check that language's code, and it keeps those tools out of
every run that did not ask for them. The checks themselves are driven through
real files wherever the checker is cheap to run (pytest) and through a scripted
`_run_command` where it would mean depending on mypy or ruff being installed on
whatever machine runs this suite.
"""

import io

import pytest
from rich.console import Console

import tools
from agent import MiniAgent
from app_types import ModelResponse
from languages import resolve_language
from plain import PlainView
from session import SessionStore
from tools import ToolRegistry
from workspace import WorkspaceContext


PYTHON_TOOLS = {"lint", "type_check", "write_tests"}


@pytest.fixture
def workspace(tmp_path):
    """A registry factory over a temp repo holding one small module."""
    root = tmp_path.resolve()
    (root / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    history: list = []

    def build(**kwargs) -> ToolRegistry:
        options = dict(
            workspace=None,
            root=root,
            approval_policy="auto",
            read_only=False,
            depth=0,
            max_depth=1,
            get_history=lambda: history,
            language="python",
        )
        options.update(kwargs)
        return ToolRegistry(**options)

    return root, history, build


class StubModel:
    """Enough of a model client for MiniAgent to build itself."""

    model = "stub"
    ctx = 8192

    def complete(self, messages, max_new_tokens, tools=None, **sinks):
        return ModelResponse(content="done", tool_calls=[])


def build_agent(tmp_path, **options) -> MiniAgent:
    return MiniAgent(
        model_client=StubModel(),
        workspace=WorkspaceContext.build(tmp_path, language=options.get("language")),
        session_store=SessionStore(tmp_path / "sessions"),
        approval_policy="never",
        **options,
    )


# --- What the model is told ---


def test_the_workspace_names_the_language_when_one_was_given(tmp_path):
    text = WorkspaceContext.build(tmp_path, language="python").text()
    assert "- language: Python" in text


def test_the_workspace_says_nothing_about_language_by_default(tmp_path):
    text = WorkspaceContext.build(tmp_path).text()
    assert "language" not in text


def test_the_system_prompt_carries_the_language_rules(tmp_path):
    prefix = build_agent(tmp_path, language="python").prefix
    assert "- language: Python" in prefix
    assert "Python:" in prefix
    # The rules exist to point at the tools the flag just unlocked.
    for name in PYTHON_TOOLS:
        assert name in prefix


def test_a_run_without_a_language_gets_no_language_rules(tmp_path):
    prefix = build_agent(tmp_path).prefix
    assert "Python:" not in prefix
    assert "write_tests" not in prefix


# --- Which tools exist ---


def test_the_python_tools_appear_only_for_a_python_run(workspace):
    _, _, build = workspace
    assert PYTHON_TOOLS <= set(dict(build().items()))
    assert PYTHON_TOOLS.isdisjoint(dict(build(language=None).items()))


def test_the_language_tools_ignore_the_tool_profile(workspace):
    """--language is their gate, so asking for a language is enough on its own."""
    _, _, build = workspace
    for profile in ("minimal", "standard", "full"):
        assert PYTHON_TOOLS <= set(dict(build(profile=profile).items()))


def test_an_unknown_language_leaves_the_run_language_agnostic(workspace):
    _, _, build = workspace
    registry = build(language="cobol")
    assert registry.language is None
    assert PYTHON_TOOLS.isdisjoint(dict(registry.items()))


def test_the_language_key_is_resolved_rather_than_matched_literally(workspace):
    _, _, build = workspace
    assert build(language="Python").language == "python"


def test_the_agent_hands_its_language_to_the_registry(tmp_path):
    assert build_agent(tmp_path, language="python").tools.language == "python"
    assert build_agent(tmp_path).tools.language is None


def test_a_python_call_by_its_checker_name_is_recovered(workspace):
    """A model that asks for 'mypy' means type_check; that costs a step to learn."""
    _, _, build = workspace
    registry = build()
    resolved = dict(registry.items())
    assert "mypy" not in resolved
    result = registry.run("mypy", {"path": "does-not-exist.py"})
    assert "ran 'type_check' instead" in result


# --- Finding the checkers ---


def test_a_workspace_virtualenv_wins_and_is_run_as_a_module(tmp_path):
    """A venv console script carries an absolute shebang; -m does not."""
    binaries = tmp_path / ".venv" / "bin"
    binaries.mkdir(parents=True)
    (binaries / "python").write_text("", encoding="utf-8")
    (binaries / "ruff").write_text("", encoding="utf-8")

    command = tools._python_tool_command(tmp_path, "ruff")
    assert command == f"{binaries / 'python'} -m ruff"


def test_a_checker_on_the_path_is_used_when_the_workspace_has_no_venv(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert tools._python_tool_command(tmp_path, "ruff") == "/usr/bin/ruff"


def test_a_missing_checker_is_reported_as_something_to_work_around(
    workspace, monkeypatch
):
    _, _, build = workspace
    monkeypatch.setattr(tools, "_python_tool_command", lambda root, module: "")

    result = build().run("lint", {"path": "calc.py"})

    assert "ruff is not installed" in result
    # A dead end the model can do nothing with is a wasted run; name the way on.
    assert "run its tests" in result


# --- What the checkers report ---


def scripted_run(monkeypatch, exit_code: int, output: str = "output"):
    """Replace command execution with a fixed result, capturing the command."""
    seen: list = []

    def fake(registry, command, timeout):
        seen.append((command, timeout))
        return exit_code, output

    monkeypatch.setattr(tools, "_run_command", fake)
    monkeypatch.setattr(tools, "_python_tool_command", lambda root, module: module)
    return seen


def test_lint_runs_ruff_over_the_path_and_reports_a_clean_result(
    workspace, monkeypatch
):
    _, _, build = workspace
    seen = scripted_run(monkeypatch, 0)

    result = build().run("lint", {"path": "calc.py"})

    command, _ = seen[0]
    assert command.startswith("ruff check ")
    # Repo-relative, so the checker's output is not one workspace prefix per line.
    assert command.endswith(" calc.py")
    assert "no lint issues" in result


def test_lint_turns_a_failing_exit_code_into_an_instruction(workspace, monkeypatch):
    _, _, build = workspace
    scripted_run(monkeypatch, 1, "calc.py:2:12: F401 `os` imported but unused")

    result = build().run("lint", {"path": "calc.py"})

    assert "F401" in result
    assert "fix them and run lint again" in result


def test_type_check_runs_mypy_and_looks_inside_unannotated_code(
    workspace, monkeypatch
):
    root, _, build = workspace
    seen = scripted_run(monkeypatch, 0)

    result = build().run("type_check", {})

    command, _ = seen[0]
    assert command.startswith("mypy ")
    # Without this, mypy skips the body of every unannotated function -- which
    # is most of what a small model writes -- and reports success unread.
    assert "--check-untyped-defs" in command
    # The cache belongs to the harness, not to the project being worked on.
    assert str(root / ".harness-lab") in command
    assert command.endswith(" .")
    assert "no type errors" in result


def test_a_checker_path_outside_the_workspace_is_refused(workspace):
    _, _, build = workspace
    assert "escapes the workspace" in build().run("lint", {"path": "../etc"})


def test_a_checker_timeout_is_bounded(workspace):
    _, _, build = workspace
    assert "timeout must be in [1, 600]" in build().run(
        "type_check", {"path": "calc.py", "timeout": 9000}
    )


# --- Generated tests, and running them ---


def test_write_tests_writes_the_tests_and_reports_that_they_pass(workspace):
    root, _, build = workspace

    result = build().run(
        "write_tests",
        {
            "path": "calc.py",
            "content": "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        },
    )

    written = root / "test_calc.py"
    assert written.is_file()
    assert "wrote test_calc.py" in result
    assert "tests passed" in result


def test_generated_tests_land_in_the_projects_tests_directory(workspace):
    root, _, build = workspace
    (root / "tests").mkdir()

    result = build().run(
        "write_tests",
        {
            "path": "calc.py",
            "content": "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        },
    )

    assert (root / "tests" / "test_calc.py").is_file()
    assert "tests passed" in result


def test_tests_that_the_code_fails_are_reported_as_a_failure(workspace):
    root, _, build = workspace
    (root / "broken.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )

    result = build().run(
        "write_tests",
        {
            "path": "broken.py",
            "content": "from broken import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        },
    )

    assert "tests failed" in result
    # The run itself has to be readable, or there is nothing to fix from.
    assert "assert" in result


def test_content_with_no_test_functions_is_refused(workspace):
    """A suite pytest collects nothing from passes, which reads as proof."""
    _, _, build = workspace

    result = build().run(
        "write_tests", {"path": "calc.py", "content": "from calc import add\n"}
    )

    assert "no test functions found" in result
    assert "test_*" in result


def test_a_collection_of_nothing_is_not_reported_as_a_pass(workspace, monkeypatch):
    _, _, build = workspace
    # pytest's exit code 5: it ran, it was happy, it found no tests.
    scripted_run(monkeypatch, 5)

    result = build().run(
        "write_tests",
        {
            "path": "calc.py",
            "content": "class Helper:\n    def test_add(self):\n        assert True\n",
        },
    )

    assert "no tests ran" in result
    assert "tests passed" not in result


def test_a_missing_pytest_is_not_reported_as_a_failing_test(workspace, monkeypatch):
    _, _, build = workspace
    scripted_run(monkeypatch, 1, "/usr/bin/python: No module named pytest")

    result = build().run(
        "write_tests",
        {"path": "calc.py", "content": "def test_add():\n    assert True\n"},
    )

    assert "pytest is not installed" in result
    assert "tests failed" not in result


def test_a_test_file_pytest_would_not_collect_is_refused(workspace):
    _, _, build = workspace

    result = build().run(
        "write_tests",
        {
            "path": "calc.py",
            "test_path": "checks.py",
            "content": "def test_add():\n    assert True\n",
        },
    )

    assert "pytest will not collect checks.py" in result


def test_write_tests_refuses_to_overwrite_the_module_under_test(workspace):
    _, _, build = workspace

    result = build().run(
        "write_tests",
        {
            "path": "calc.py",
            "test_path": "calc.py",
            "content": "def test_add():\n    assert True\n",
        },
    )

    assert "is the module under test" in result
    assert (
        "def add" in (build().root / "calc.py").read_text(encoding="utf-8")
    )


def test_write_tests_only_covers_python_modules(workspace):
    root, _, build = workspace
    (root / "notes.md").write_text("# notes\n", encoding="utf-8")

    result = build().run(
        "write_tests",
        {"path": "notes.md", "content": "def test_add():\n    assert True\n"},
    )

    assert "not a .py file" in result


def test_replacing_unread_tests_is_guarded_like_any_other_overwrite(workspace):
    root, history, build = workspace
    (root / "test_calc.py").write_text(
        "def test_old():\n    assert True\n", encoding="utf-8"
    )

    result = build().run(
        "write_tests",
        {"path": "calc.py", "content": "def test_add():\n    assert True\n"},
    )

    assert "has not been read in this session" in result
    assert "test_old" in (root / "test_calc.py").read_text(encoding="utf-8")
    assert history == []


def test_the_test_source_is_cleaned_up_the_way_a_write_is(workspace):
    """Fenced, line-numbered content is what a small model actually sends."""
    root, _, build = workspace

    result = build().run(
        "write_tests",
        {
            "path": "calc.py",
            "content": (
                "```python\n"
                "   1: from calc import add\n"
                "   2: \n"
                "   3: \n"
                "   4: def test_add():\n"
                "   5:     assert add(1, 2) == 3\n"
                "```"
            ),
        },
    )

    assert "tests passed" in result
    assert (root / "test_calc.py").read_text(encoding="utf-8").startswith(
        "from calc import add"
    )


def test_write_tests_is_gated_by_approval_like_every_other_write(workspace):
    _, _, build = workspace
    registry = build(approval_policy="ask", approval_fn=lambda name, args: False)

    result = registry.run(
        "write_tests",
        {"path": "calc.py", "content": "def test_add():\n    assert True\n"},
    )

    assert "approval denied for write_tests" in result
    assert not (registry.root / "test_calc.py").exists()


# --- The pack itself ---


def test_a_language_pack_resolves_from_its_key_and_from_itself():
    python = resolve_language("python")
    assert python is not None
    assert resolve_language(python) is python
    assert resolve_language(None) is None
    assert resolve_language("") is None


def test_the_plain_banner_names_the_language_only_when_there_is_one(tmp_path):
    """Which knobs a run was started with is the first thing a transcript needs."""

    def banner(**options) -> str:
        console = Console(
            file=io.StringIO(), width=200, soft_wrap=True, highlight=False, no_color=True
        )
        agent = build_agent(tmp_path, **options)
        PlainView(agent, console=console).banner("127.0.0.1:8080")
        return console.file.getvalue()

    assert "language Python" in banner(language="python")
    assert "language" not in banner()
