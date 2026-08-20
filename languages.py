"""Language packs: what a run gains when `--language` names the language the
workspace is worked in.

A pack is two things the harness does not otherwise know: the label that tells
the model which language it is writing, and the key that gates the tools able
to check that language's code. The tools themselves live in `tools.py` with
the rest of the catalog, declared with `language="python"` -- keeping every
callable the model can reach in one place -- so this module holds only what is
not code: the label the workspace line shows and the rules the system prompt
carries.

Adding a language is adding an entry here and its tools in `tools.py`.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Language:
    """One language the harness knows how to work in and verify."""

    # What --language accepts, and what a tool declares to be gated on it.
    key: str
    # How the workspace line and the front-end banners name it.
    label: str
    # Extra system-prompt rules, appended to the language-neutral ones.
    rules: Tuple[str, ...]

    def prompt_section(self) -> str:
        return f"{self.label}:\n" + "\n".join(f"- {rule}" for rule in self.rules)


# The rules are deliberately short and imperative. A small model given a long
# language essay stops reading it; what it needs is which tools exist, when to
# reach for them, and that a clean result is part of being finished.
PYTHON = Language(
    key="python",
    label="Python",
    rules=(
        "This workspace is written in Python; new files are Python unless the user asks otherwise.",
        "After writing or changing a .py file, check it: lint for style and obvious mistakes, type_check for type errors.",
        "Cover new code with tests: write_tests writes a pytest file and runs it in one step.",
        "Code is not finished until lint, type_check and its tests come back clean. Fix what they report, then run them again.",
        "lint and type_check only read the code, so run them before your final answer instead of claiming the code works.",
    ),
)

LANGUAGES: Dict[str, Language] = {PYTHON.key: PYTHON}
# What --language offers, smallest surprise first: the CLI validates against
# this, so an unknown key never reaches the agent.
LANGUAGE_CHOICES: Tuple[str, ...] = tuple(LANGUAGES)


def resolve_language(key: object) -> Language | None:
    """Look up a language pack by key; None (no language) is a valid answer."""
    if not key:
        return None
    if isinstance(key, Language):
        return key
    return LANGUAGES.get(str(key).strip().lower())
