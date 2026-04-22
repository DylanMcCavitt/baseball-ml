from pathlib import Path

from mlb_props_stack import env as env_module


def test_load_repo_env_reads_repo_root_dotenv_without_overriding_existing_values(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".env").write_text(
        "ODDS_API_KEY=from-dotenv\nOPENAI_API_KEY=secondary\n",
        encoding="utf-8",
    )

    environ = {"OPENAI_API_KEY": "already-set"}
    loaded_path = env_module.load_repo_env(cwd=repo_root, environ=environ)

    assert loaded_path == repo_root / ".env"
    assert environ["ODDS_API_KEY"] == "from-dotenv"
    assert environ["OPENAI_API_KEY"] == "already-set"


def test_load_repo_env_falls_back_to_main_worktree_dotenv(tmp_path, monkeypatch) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    (canonical_root / ".env").write_text(
        "ODDS_API_KEY=from-main-worktree\n",
        encoding="utf-8",
    )
    worktree_root = tmp_path / "feature-worktree"
    worktree_root.mkdir()

    monkeypatch.setattr(env_module, "_git_toplevel", lambda cwd: worktree_root)
    monkeypatch.setattr(
        env_module,
        "_list_git_worktrees",
        lambda cwd: [
            (worktree_root, "refs/heads/feat/age-188-fix-odds-api-joins"),
            (canonical_root, "refs/heads/main"),
        ],
    )

    environ: dict[str, str] = {}
    loaded_path = env_module.load_repo_env(cwd=worktree_root, environ=environ)

    assert loaded_path == canonical_root / ".env"
    assert environ["ODDS_API_KEY"] == "from-main-worktree"


def test_load_repo_env_merges_partial_worktree_env_with_main_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    (canonical_root / ".env").write_text(
        "ODDS_API_KEY=from-main-worktree\nSECONDARY_KEY=secondary\n",
        encoding="utf-8",
    )
    worktree_root = tmp_path / "feature-worktree"
    worktree_root.mkdir()
    (worktree_root / ".env").write_text(
        "OPENAI_API_KEY=worktree-openai\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(env_module, "_git_toplevel", lambda cwd: worktree_root)
    monkeypatch.setattr(
        env_module,
        "_list_git_worktrees",
        lambda cwd: [
            (worktree_root, "refs/heads/feat/age-188-fix-odds-api-joins"),
            (canonical_root, "refs/heads/main"),
        ],
    )

    environ: dict[str, str] = {}
    loaded_path = env_module.load_repo_env(cwd=worktree_root, environ=environ)

    assert loaded_path == worktree_root / ".env"
    assert environ == {
        "OPENAI_API_KEY": "worktree-openai",
        "ODDS_API_KEY": "from-main-worktree",
        "SECONDARY_KEY": "secondary",
    }


def test_parse_env_lines_supports_export_and_comments() -> None:
    parsed = env_module._parse_env_lines(
        [
            "# comment",
            "export ODDS_API_KEY=\"quoted-value\"",
            "OTHER=value # trailing comment",
            "ANOTHER=\"still-quoted\" # inline comment",
            "EMPTY=",
        ]
    )

    assert parsed == {
        "ODDS_API_KEY": "quoted-value",
        "OTHER": "value",
        "ANOTHER": "still-quoted",
        "EMPTY": "",
    }
