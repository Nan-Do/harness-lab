"""Tests for the tool registry against a real temporary workspace.

The focus is the behaviour that exists specifically to absorb small-model
mistakes: recovering a call whose shape is wrong, refusing one that would
destroy work, and answering a failed one with something the model can act on.
"""

import pytest

from app_types import ToolMessageEntry
from tools import ToolRegistry


def entry(name: str, content: str = "ok", **args) -> ToolMessageEntry:
    return ToolMessageEntry(
        role="tool", name=name, args=args, content=content, created_at="now"
    )


@pytest.fixture
def workspace(tmp_path):
    """A registry over a temp repo, plus the history list the agent would own."""
    root = tmp_path.resolve()
    (root / "app.py").write_text("def f():\n    return -1\n", encoding="utf-8")
    (root / "notes.md").write_text("# notes\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
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
        )
        options.update(kwargs)
        return ToolRegistry(**options)

    return root, history, build


# --- Reads are never clipped ---


def test_read_file_returns_whole_file_regardless_of_output_budget(workspace):
    root, _, build = workspace
    body = "\n".join(f"line {index}" for index in range(500))
    (root / "big.txt").write_text(body, encoding="utf-8")

    # A budget this small clips shell and search output; reads must ignore it,
    # because editing against a partial view is how content gets destroyed.
    registry = build(max_noisy_output=100)
    result = registry.run("read_file", {"path": "big.txt"})

    assert "clipped" not in result
    assert "line 0" in result and "line 499" in result
    assert "(500 lines)" in result


def test_read_file_range_past_end_explains_instead_of_returning_nothing(workspace):
    _, _, build = workspace
    result = build().run("read_file_range", {"path": "app.py", "start": 50, "end": 60})
    assert "ends at line 2" in result


def test_read_file_range_clamps_and_reports_the_real_span(workspace):
    _, _, build = workspace
    result = build().run("read_file_range", {"path": "app.py", "start": 1, "end": 900})
    assert "lines 1-2 of 2" in result


# --- Call-shape recovery ---


def test_shell_tool_name_is_resolved_and_reported(workspace):
    _, _, build = workspace
    result = build().run("cat", {"path": "app.py"})
    assert "read_file" in result
    assert "def f():" in result


def test_alias_argument_names_are_accepted(workspace):
    _, _, build = workspace
    result = build().run("read_file", {"file_path": "app.py"})
    assert "def f():" in result
    assert "read 'file_path' as 'path'" in result


def test_argument_envelope_is_unwrapped(workspace):
    _, _, build = workspace
    result = build().run("read_file", {"arguments": {"path": "app.py"}})
    assert "def f():" in result


def test_repo_name_prefix_on_a_path_is_recovered(workspace):
    root, _, build = workspace
    result = build().run("read_file", {"path": f"{root.name}/app.py"})
    assert "def f():" in result


def test_unknown_tool_suggests_a_real_one(workspace):
    _, _, build = workspace
    result = build().run("reed_file", {"path": "app.py"})
    assert "read_file" in result


def test_missing_file_names_candidates(workspace):
    _, _, build = workspace
    result = build().run("read_file", {"path": "mod.py"})
    assert "pkg/mod.py" in result


def test_argument_error_reports_the_tools_real_shape(workspace):
    _, _, build = workspace
    result = build().run("patch_file", {"path": "app.py", "old_text": ""})
    assert "arguments:" in result and "new_text" in result


# --- Write safety ---


def test_write_file_refuses_to_replace_an_unread_file(workspace):
    root, _, build = workspace
    result = build().run("write_file", {"path": "app.py", "content": "x = 1\n"})
    assert "has not been read in this session" in result
    assert (root / "app.py").read_text() == "def f():\n    return -1\n"


def test_write_file_allowed_once_the_file_has_been_read(workspace):
    root, history, build = workspace
    history.append(entry("read_file", path="app.py"))
    result = build().run("write_file", {"path": "app.py", "content": "x = 1\n"})
    assert "wrote app.py" in result
    assert (root / "app.py").read_text() == "x = 1\n"


def test_write_guard_can_be_disabled_for_comparison(workspace):
    root, _, build = workspace
    registry = build(require_read_before_overwrite=False)
    assert "wrote app.py" in registry.run(
        "write_file", {"path": "app.py", "content": "x = 1\n"}
    )
    assert (root / "app.py").read_text() == "x = 1\n"


def test_new_files_are_never_blocked_by_the_guard(workspace):
    root, _, build = workspace
    result = build().run("write_file", {"path": "fresh.py", "content": "x = 1\n"})
    assert "wrote fresh.py" in result
    assert (root / "fresh.py").exists()


def test_writing_identical_content_is_reported_not_repeated(workspace):
    _, history, build = workspace
    history.append(entry("read_file", path="app.py"))
    result = build().run(
        "write_file", {"path": "app.py", "content": "def f():\n    return -1\n"}
    )
    assert "already has exactly this content" in result


def test_writes_into_git_internals_are_refused(workspace):
    root, _, build = workspace
    (root / ".git").mkdir(exist_ok=True)
    result = build().run("write_file", {"path": ".git/config", "content": "x"})
    assert "refusing to modify" in result


def test_paths_outside_the_workspace_are_refused(workspace):
    _, _, build = workspace
    result = build().run("read_file", {"path": "../../etc/passwd"})
    assert "escapes the workspace" in result


# --- Post-write feedback ---


def test_truncated_python_is_reported_immediately(workspace):
    _, _, build = workspace
    result = build().run(
        "write_file", {"path": "broken.py", "content": "def f():\n    return (1,\n"}
    )
    assert "WARNING invalid Python" in result


def test_valid_python_is_confirmed(workspace):
    _, _, build = workspace
    result = build().run("write_file", {"path": "ok.py", "content": "x = 1\n"})
    assert "syntax OK" in result


def test_fenced_content_is_unwrapped_before_writing(workspace):
    root, _, build = workspace
    build().run(
        "write_file", {"path": "fenced.py", "content": "```python\nx = 1\n```"}
    )
    assert (root / "fenced.py").read_text() == "x = 1\n"


def test_overwrite_backs_up_the_previous_version_and_can_be_reverted(workspace):
    root, history, build = workspace
    original = (root / "app.py").read_text()
    history.append(entry("read_file", path="app.py"))

    registry = build(profile="full")
    assert "previous version saved to" in registry.run(
        "write_file", {"path": "app.py", "content": "x = 1\n"}
    )
    assert registry._backups_for(root / "app.py")

    assert "restored app.py" in registry.run("revert_file", {"path": "app.py"})
    assert (root / "app.py").read_text() == original


# --- patch_file diagnostics ---


def test_patch_file_ignores_pasted_line_numbers(workspace):
    root, _, build = workspace
    result = build().run(
        "patch_file",
        {"path": "app.py", "old_text": "   2:     return -1", "new_text": "    return 1"},
    )
    assert "patched app.py" in result
    assert (root / "app.py").read_text() == "def f():\n    return 1\n"


def test_patch_file_recovers_from_wrong_indentation(workspace):
    root, _, build = workspace
    result = build().run(
        "patch_file",
        {"path": "app.py", "old_text": "return -1", "new_text": "return 1"},
    )
    assert "patched app.py" in result
    assert "return 1" in (root / "app.py").read_text()


def test_patch_file_reports_every_ambiguous_line(workspace):
    root, _, build = workspace
    (root / "dup.py").write_text("a = 1\nb = 1\n", encoding="utf-8")
    result = build().run(
        "patch_file", {"path": "dup.py", "old_text": "= 1", "new_text": "= 2"}
    )
    assert "occurs 2 times" in result
    assert "lines 1, 2" in result


def test_patch_file_shows_the_closest_real_text_when_it_misses(workspace):
    _, _, build = workspace
    result = build().run(
        "patch_file",
        {"path": "app.py", "old_text": "    return -999", "new_text": "    return 1"},
    )
    assert "Closest text in the file" in result
    assert "return -1" in result


def test_patch_file_rejects_a_no_op(workspace):
    _, _, build = workspace
    result = build().run(
        "patch_file", {"path": "app.py", "old_text": "return -1", "new_text": "return -1"}
    )
    assert "identical" in result


# --- replace_lines ---


def test_replace_lines_edits_by_line_number(workspace):
    root, _, build = workspace
    result = build().run(
        "replace_lines",
        {"path": "app.py", "start": 2, "end": 2, "content": "    return 1"},
    )
    assert "replaced lines 2-2" in result
    assert (root / "app.py").read_text() == "def f():\n    return 1\n"


def test_replace_lines_strips_pasted_line_numbers(workspace):
    root, _, build = workspace
    build().run(
        "replace_lines",
        {
            "path": "app.py",
            "start": 1,
            "end": 2,
            "content": "   1: def f():\n   2:     return 7",
        },
    )
    assert (root / "app.py").read_text() == "def f():\n    return 7\n"


def test_replace_lines_past_end_points_at_append(workspace):
    _, _, build = workspace
    result = build().run(
        "replace_lines", {"path": "app.py", "start": 40, "end": 41, "content": "x"}
    )
    assert "append_file" in result


# --- Shell hardening ---


def test_denylisted_commands_are_refused_with_a_reason(workspace):
    _, _, build = workspace
    result = build().run("run_shell", {"command": "sudo rm -rf /tmp/x"})
    assert "refused" in result


def test_backgrounded_commands_are_refused(workspace):
    _, _, build = workspace
    result = build().run("run_shell", {"command": "python -m http.server &"})
    assert "backgrounded" in result


def test_shell_timeout_keeps_partial_output(workspace):
    _, _, build = workspace
    result = build().run(
        "run_shell", {"command": "echo started; sleep 5", "timeout": 1}
    )
    assert "timed out after 1s" in result
    assert "started" in result


def test_shell_output_is_clipped_with_a_hint(workspace):
    _, _, build = workspace
    registry = build(max_noisy_output=400)
    result = registry.run(
        "run_shell", {"command": "python -c \"print('x' * 5000)\"", "timeout": 20}
    )
    assert "clipped" in result


# --- Loop breaking ---


def test_identical_repeat_is_blocked_on_the_second_call(workspace):
    _, history, build = workspace
    history.append(entry("read_file", path="app.py"))
    result = build().run("read_file", {"path": "app.py"})
    assert "cannot tell you anything new" in result


def test_alternating_calls_are_detected(workspace):
    _, history, build = workspace
    history.append(entry("read_file", path="app.py"))
    history.append(entry("list_files", path="."))
    history.append(entry("read_file", path="app.py"))
    result = build().run("list_files", {"path": "."})
    assert "alternating" in result


def test_repeating_a_test_run_after_an_edit_is_allowed(workspace):
    _, history, build = workspace
    # Re-running a command after changing something is the intended workflow,
    # so volatile tools are exempt from the alternation check.
    history.append(entry("run_shell", command="echo hi"))
    history.append(entry("write_file", path="app.py"))
    history.append(entry("run_shell", command="echo hi"))
    result = build().run("run_shell", {"command": "echo hi"})
    assert "alternating" not in result
    assert "hi" in result


# --- Profiles ---


def test_minimal_profile_hides_the_extra_tools(workspace):
    _, _, build = workspace
    names = {name for name, _ in build(profile="minimal").items()}
    assert "read_file" in names and "patch_file" in names
    assert "outline_file" not in names and "revert_file" not in names


def test_standard_profile_adds_navigation_and_verification(workspace):
    _, _, build = workspace
    names = {name for name, _ in build(profile="standard").items()}
    assert {"outline_file", "find_files", "git_status", "replace_lines"} <= names
    assert "revert_file" not in names


def test_full_profile_exposes_everything(workspace):
    _, _, build = workspace
    names = {name for name, _ in build(profile="full").items()}
    assert "revert_file" in names


# --- Navigation ---


def test_outline_file_lists_definitions_with_line_numbers(workspace):
    root, _, build = workspace
    (root / "big.py").write_text(
        "import os\n\n\nclass A:\n    def m(self):\n        pass\n", encoding="utf-8"
    )
    result = build().run("outline_file", {"path": "big.py"})
    assert "class A" in result and "def m" in result


def test_outline_file_reports_a_syntax_error_instead_of_failing(workspace):
    root, _, build = workspace
    (root / "bad.py").write_text("def (:\n", encoding="utf-8")
    result = build().run("outline_file", {"path": "bad.py"})
    assert "cannot outline" in result


def test_find_files_matches_a_bare_name(workspace):
    _, _, build = workspace
    result = build().run("find_files", {"pattern": "mod.py"})
    assert "pkg/mod.py" in result


def test_list_files_descends_when_asked(workspace):
    _, _, build = workspace
    shallow = build().run("list_files", {"path": ".", "depth": 1})
    deep = build().run("list_files", {"path": ".", "depth": 2})
    assert "pkg/mod.py" not in shallow
    assert "pkg/mod.py" in deep


def test_list_files_on_a_file_points_at_read_file(workspace):
    _, _, build = workspace
    result = build().run("list_files", {"path": "app.py"})
    assert "read_file" in result


# --- Schema generation ---


def test_optional_fields_get_a_default_of_their_declared_type(workspace):
    _, _, build = workspace
    schemas = {
        item["function"]["name"]: item["function"] for item in build().schemas()
    }
    params = schemas["read_file_range"]["parameters"]
    assert params["required"] == ["path"]
    assert isinstance(params["properties"]["start"]["default"], int)


def test_unparseable_numeric_default_does_not_become_a_string(workspace):
    assert ToolRegistry._parse_default("integer", "not-a-number") == 0


# --- Approval ---


def test_denied_approval_tells_the_model_not_to_retry(workspace):
    _, _, build = workspace
    registry = build(approval_policy="never")
    result = registry.run("write_file", {"path": "new.py", "content": "x = 1\n"})
    assert "approval denied" in result and "Do not retry" in result


def test_read_only_sessions_cannot_write(workspace):
    root, _, build = workspace
    registry = build(read_only=True)
    registry.run("write_file", {"path": "new.py", "content": "x = 1\n"})
    assert not (root / "new.py").exists()
