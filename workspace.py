import subprocess
from pathlib import Path

from languages import Language, resolve_language
from utils import DOC_NAMES


class WorkspaceContext:
    def __init__(
        self,
        cwd,
        repo_root,
        branch,
        default_branch,
        status,
        recent_commits,
        project_docs,
        language=None,
    ):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs
        # The language the run was told it is working in (--language), or None
        # when nothing was named. Kept as the resolved pack rather than the raw
        # key so everything that displays it agrees on the label.
        self.language: Language | None = resolve_language(language)

    @classmethod
    def build(cls, cwd, language=None):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        docs = {}
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = path.read_text(encoding="utf-8", errors="replace")

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                git(
                    ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
                    "origin/main",
                )
                or "origin/main"
            ).removeprefix("origin/"),
            status=git(["status", "--short"], "clean") or "clean",
            recent_commits=[
                line for line in git(["log", "--oneline", "-5"]).splitlines() if line
            ],
            project_docs=docs,
            language=language,
        )

    def text(self):
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = (
            "\n".join(
                f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()
            )
            or "- none"
        )
        language = [f"- language: {self.language.label}"] if self.language else []
        return "\n".join(
            [
                "Workspace:",
                *language,
                f"- cwd: {self.cwd}",
                f"- repo_root: {self.repo_root}",
                f"- branch: {self.branch}",
                f"- default_branch: {self.default_branch}",
                "- status:",
                self.status,
                "- recent_commits:",
                commits,
                "- project_docs:",
                docs,
            ]
        )
