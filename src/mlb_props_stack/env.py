"""Local environment-file helpers for repo and worktree execution."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
import os
from pathlib import Path
import subprocess


def _parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[:1]
            closing_quote_index = value.find(quote, 1)
            if closing_quote_index != -1:
                value = value[1:closing_quote_index]
            else:
                value = value[1:]
        else:
            comment_start = value.find(" #")
            if comment_start != -1:
                value = value[:comment_start].rstrip()

        parsed[key] = value
    return parsed


def _load_env_file(
    path: Path,
    *,
    environ: MutableMapping[str, str],
) -> int:
    assignments = _parse_env_lines(path.read_text(encoding="utf-8").splitlines())
    loaded_count = 0
    for key, value in assignments.items():
        if key in environ:
            continue
        environ[key] = value
        loaded_count += 1
    return loaded_count


def _run_git(
    args: list[str],
    *,
    cwd: Path,
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_toplevel(cwd: Path) -> Path | None:
    output = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if not output:
        return None
    return Path(output).resolve()


def _list_git_worktrees(cwd: Path) -> list[tuple[Path, str | None]]:
    output = _run_git(["worktree", "list", "--porcelain"], cwd=cwd)
    if not output:
        return []

    worktrees: list[tuple[Path, str | None]] = []
    current_path: Path | None = None
    current_branch: str | None = None

    for line in output.splitlines():
        if not line.strip():
            if current_path is not None:
                worktrees.append((current_path.resolve(), current_branch))
            current_path = None
            current_branch = None
            continue
        if line.startswith("worktree "):
            current_path = Path(line.split(" ", 1)[1])
            current_branch = None
            continue
        if line.startswith("branch "):
            current_branch = line.split(" ", 1)[1]

    if current_path is not None:
        worktrees.append((current_path.resolve(), current_branch))
    return worktrees


def _candidate_env_paths(start_cwd: Path) -> list[Path]:
    git_root = _git_toplevel(start_cwd)
    if git_root is None:
        return [
            parent / ".env"
            for parent in (start_cwd, *start_cwd.parents)
        ]

    candidates: list[Path] = [git_root / ".env"]
    sibling_worktrees = [
        (path, branch)
        for path, branch in _list_git_worktrees(git_root)
        if path != git_root
    ]
    sibling_worktrees.sort(
        key=lambda item: (
            0 if item[1] == "refs/heads/main" else 1,
            0 if "/.codex/worktrees/" not in str(item[0]) else 1,
            str(item[0]),
        )
    )
    candidates.extend(path / ".env" for path, _branch in sibling_worktrees)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def load_repo_env(
    *,
    cwd: Path | str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Load a repo-local `.env` without overriding existing environment values."""
    resolved_cwd = Path(cwd or Path.cwd()).resolve()
    target_environ = os.environ if environ is None else environ

    first_loaded_path: Path | None = None
    for candidate in _candidate_env_paths(resolved_cwd):
        if not candidate.is_file():
            continue
        loaded_count = _load_env_file(candidate, environ=target_environ)
        if loaded_count > 0 and first_loaded_path is None:
            first_loaded_path = candidate
    return first_loaded_path
