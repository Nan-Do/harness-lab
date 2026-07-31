"""Tests for the pure argument/output repair layer.

These cover the shapes small models actually emit -- alias keys, argument
envelopes, pasted line numbers, fenced content -- so a change to the repair
rules shows up here instead of only as a worse run against a model.
"""

import pytest

from tool_support import (
    apply_arg_aliases,
    clip,
    coerce_list_content,
    find_fuzzy_match,
    looks_line_numbered,
    match_lines,
    nearest_block,
    normalize_content,
    outline_generic,
    outline_python,
    resolve_tool_name,
    strip_code_fence,
    strip_line_numbers,
    unwrap_envelope,
    validate_syntax,
)

KNOWN = [
    "list_files",
    "read_file",
    "read_file_range",
    "search",
    "run_shell",
    "write_file",
    "patch_file",
    "replace_lines",
]


# --- Tool name recovery ---


def test_exact_name_passes_through():
    assert resolve_tool_name("read_file", KNOWN) == ("read_file", "")


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("cat", "read_file"),
        ("ls", "list_files"),
        ("grep", "search"),
        ("bash", "run_shell"),
        ("str_replace_editor", "patch_file"),
        ("Read_File", "read_file"),
        ("read-file", "read_file"),
    ],
)
def test_shell_and_other_harness_names_resolve(requested, expected):
    resolved, note = resolve_tool_name(requested, KNOWN)
    assert resolved == expected
    assert note  # the model is always told the canonical name


def test_unknown_name_suggests_close_match():
    resolved, note = resolve_tool_name("read_filee", KNOWN)
    assert resolved == ""
    assert "read_file" in note


def test_unresolvable_name_lists_available_tools():
    resolved, note = resolve_tool_name("teleport", KNOWN)
    assert resolved == ""
    assert "write_file" in note


def test_alias_absent_from_registry_is_not_resolved():
    # 'outline' aliases outline_file, which the minimal profile does not expose.
    resolved, _ = resolve_tool_name("outline", KNOWN)
    assert resolved == ""


# --- Argument normalization ---


def test_unwrap_envelope_handles_nested_dict():
    assert unwrap_envelope({"arguments": {"path": "a.py"}}) == {"path": "a.py"}


def test_unwrap_envelope_handles_json_string():
    assert unwrap_envelope({"input": '{"path": "a.py"}'}) == {"path": "a.py"}


def test_unwrap_envelope_leaves_real_args_alone():
    args = {"path": "a.py", "content": "x"}
    assert unwrap_envelope(args) == args


def test_unwrap_envelope_ignores_single_real_field():
    assert unwrap_envelope({"path": "a.py"}) == {"path": "a.py"}


def test_alias_renames_onto_declared_fields():
    args, renames = apply_arg_aliases(["path", "content"], {"file_path": "a.py", "text": "x"})
    assert args == {"path": "a.py", "content": "x"}
    assert sorted(renames) == [("file_path", "path"), ("text", "content")]


def test_alias_does_not_override_a_present_field():
    args, renames = apply_arg_aliases(["path"], {"path": "real.py", "file_path": "wrong.py"})
    assert args["path"] == "real.py"
    assert renames == []


def test_alias_respects_per_tool_meaning():
    # "search" means old_text for patch_file and pattern for search.
    patch_args, _ = apply_arg_aliases(
        ["path", "old_text", "new_text"], {"path": "a.py", "search": "x", "replace": "y"}
    )
    assert patch_args["old_text"] == "x" and patch_args["new_text"] == "y"

    search_args, _ = apply_arg_aliases(["pattern", "path"], {"search": "needle"})
    assert search_args["pattern"] == "needle"


def test_alias_never_steals_a_declared_field():
    # A tool declaring both `pattern` and `text` keeps them distinct.
    args, renames = apply_arg_aliases(["pattern", "text"], {"text": "x"})
    assert args == {"text": "x"}
    assert renames == []


def test_content_sent_as_list_is_joined():
    joined, note = coerce_list_content(["def f():", "    return 1"])
    assert joined == "def f():\n    return 1\n"
    assert note


def test_content_string_is_left_alone():
    value, note = coerce_list_content("already text")
    assert value == "already text" and note == ""


# --- Line numbers and fences ---


def test_strip_line_numbers_removes_read_file_prefixes():
    text = "  12: def f():\n  13:     return 1"
    assert strip_line_numbers(text) == "def f():\n    return 1"


def test_looks_line_numbered_requires_consecutive_numbers():
    assert looks_line_numbered("  1: a\n  2: b")
    # Real content that merely starts with digits must not be mangled.
    assert not looks_line_numbered("3: see appendix")
    assert not looks_line_numbered("10: alpha\n40: beta")
    assert not looks_line_numbered("plain text\nmore text")


def test_normalize_content_strips_fence_and_numbers():
    text, notes = normalize_content("```python\n  1: x = 1\n  2: y = 2\n```", ".py")
    # The fence goes; the content's own trailing newline stays.
    assert text == "x = 1\ny = 2\n"
    assert len(notes) == 2


def test_markdown_keeps_its_code_fence():
    body = "```python\nprint(1)\n```"
    assert strip_code_fence(body, ".md") == body
    assert strip_code_fence(body, ".py") == "print(1)\n"


def test_normalize_content_leaves_ordinary_content_untouched():
    text, notes = normalize_content("def f():\n    return 1\n", ".py")
    assert text == "def f():\n    return 1\n"
    assert notes == []


# --- Output shaping ---


def test_clip_keeps_head_and_tail_and_reports():
    body = "\n".join(f"line {index}" for index in range(2000))
    clipped = clip(body, 500, "narrow the pattern")
    assert len(clipped) < len(body)
    assert clipped.startswith("line 0")
    assert clipped.rstrip().endswith("line 1999")
    assert "clipped" in clipped and "narrow the pattern" in clipped


def test_clip_disabled_by_zero_limit():
    body = "x" * 10_000
    assert clip(body, 0) == body


def test_clip_leaves_short_output_alone():
    assert clip("short", 500) == "short"


# --- Post-write validation ---


def test_validate_syntax_accepts_valid_python():
    assert validate_syntax("a.py", "def f():\n    return 1\n") == "syntax OK"


def test_validate_syntax_catches_truncated_write():
    # The classic small-model failure: output cut off at the token limit.
    result = validate_syntax("a.py", "def f():\n    return (1,\n")
    assert result.startswith("WARNING invalid Python")


def test_validate_syntax_catches_bad_json():
    assert validate_syntax("a.json", '{"a": 1,}').startswith("WARNING invalid JSON")


def test_validate_syntax_ignores_other_suffixes():
    assert validate_syntax("notes.txt", "anything at all") == ""


# --- Outline ---


def test_outline_python_reports_classes_and_methods():
    source = "class A:\n    def m(self, x):\n        pass\n\n\ndef top():\n    pass\n"
    lines = outline_python(source)
    assert "class A" in lines[0]
    assert "def m(self, x)" in lines[1]
    assert "def top()" in lines[2]


def test_outline_generic_finds_declarations():
    lines = outline_generic("function go(a) {\n  return a;\n}\n")
    assert len(lines) == 1 and "function go" in lines[0]


# --- Patch diagnostics ---


def test_match_lines_reports_every_occurrence():
    text = "a\nreturn -1\nb\nreturn -1\n"
    assert match_lines(text, "return -1") == [2, 4]


def test_fuzzy_match_recovers_from_wrong_indentation():
    text = "def f():\n        return 1\n"
    span = find_fuzzy_match(text, "def f():\n    return 1")
    assert span is not None
    start, end = span
    assert text[start:end] == "def f():\n        return 1"


def test_fuzzy_match_refuses_ambiguous_text():
    text = "    return 1\n    return 1\n"
    assert find_fuzzy_match(text, "return 1") is None


def test_nearest_block_shows_the_real_text_with_line_numbers():
    text = "def f():\n    return -1\n"
    block = nearest_block(text, "    return -2")
    assert "return -1" in block
    assert "2:" in block


def test_nearest_block_is_empty_when_nothing_is_similar():
    assert nearest_block("alpha beta\n", "zzzzzzzz qqqqqqqq") == ""
