from __future__ import annotations

import json

import pytest

from mlb_props_stack.ingest import mlb_stats_api, odds_api, statcast_features


@pytest.fixture(
    params=[mlb_stats_api, odds_api, statcast_features],
    ids=["mlb_stats_api", "odds_api", "statcast_features"],
)
def write_jsonl(request):
    return request.param._write_jsonl


def test_write_jsonl_creates_target_and_removes_tmp(tmp_path, write_jsonl) -> None:
    path = tmp_path / "out.jsonl"
    records = [{"a": 1}, {"b": 2}]

    write_jsonl(path, records)

    assert path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records


def test_write_jsonl_creates_parent_directory(tmp_path, write_jsonl) -> None:
    path = tmp_path / "nested" / "run=20260422T000000Z" / "out.jsonl"

    write_jsonl(path, [{"x": 1}])

    assert path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_write_jsonl_failure_leaves_prior_file_intact_and_removes_tmp(
    tmp_path, write_jsonl
) -> None:
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [{"prior": True}])
    prior_content = path.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        write_jsonl(path, [object()])

    assert path.read_text(encoding="utf-8") == prior_content
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_write_jsonl_first_write_failure_does_not_create_target(
    tmp_path, write_jsonl
) -> None:
    path = tmp_path / "out.jsonl"

    with pytest.raises(TypeError):
        write_jsonl(path, [object()])

    assert not path.exists()
    assert not path.with_name(f"{path.name}.tmp").exists()
