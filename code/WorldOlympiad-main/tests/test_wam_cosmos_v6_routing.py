from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.robotics_views import sha256_file
from autoresearch.wam_cosmos import _runtime_profile_for_reference


def _reference(
    tmp_path: Path,
    predicates: list[dict[str, object]],
    *,
    recipe_id: str | None = None,
) -> tuple[Path, SimpleNamespace]:
    manifest = tmp_path / "robotics_reference_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    graph = tmp_path / "task_graph.json"
    graph.write_text(
        json.dumps(
            {
                "controller_program": [],
                "success_program": predicates,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reference = SimpleNamespace(
        schema_version="robotics_reference_bundle.v6",
        recipe_id=recipe_id,
        task_graph=graph.name,
        task_graph_sha256=sha256_file(graph),
    )
    return manifest, reference


@pytest.mark.parametrize(
    ("predicates", "profile_key"),
    [
        ([{"predicate": "in_container", "subject": "a", "object": "b"}], "in_container_single"),
        ([{"predicate": "stacked", "subject": "green", "object": "brown"}], "stacked"),
        ([{"predicate": "inserted", "subject": "peg", "object": "hole"}], "inserted"),
    ],
)
def test_cosmos_runtime_routes_only_on_typed_predicate_family(
    tmp_path: Path,
    predicates: list[dict[str, object]],
    profile_key: str,
) -> None:
    manifest, reference = _reference(tmp_path, predicates)

    profile = _runtime_profile_for_reference(manifest, reference)

    assert profile["profile_key"] == profile_key
    assert profile["selection_policy"] == "typed_terminal_predicate_family"
    assert len(profile["official_runtime_task_sha256"]) == 64


def test_cosmos_runtime_routes_multi_object_task_to_native_episode_runtime(
    tmp_path: Path,
) -> None:
    predicates = [
        {"predicate": "in_container", "subject": "red", "object": "left"},
        {"predicate": "inside", "subject": "blue", "object": "right"},
    ]
    manifest, reference = _reference(tmp_path, predicates)
    profile = _runtime_profile_for_reference(manifest, reference)

    assert profile["profile_key"] == "task_graph_native_episode"
    assert profile["termination_policy"] == "native_predicates_or_independent_episode_timeout"


@pytest.mark.parametrize(
    ("predicate", "operator"),
    [("hung", "hang"), ("contact", "strike"), ("inside", "pour")],
)
def test_cosmos_runtime_routes_unmodeled_operators_to_task_graph_runtime(
    tmp_path: Path,
    predicate: str,
    operator: str,
) -> None:
    manifest, reference = _reference(
        tmp_path,
        [{"predicate": predicate, "subject": "object", "object": "target"}],
    )
    graph = json.loads((tmp_path / "task_graph.json").read_text())
    graph["controller_program"] = [{"operator": operator, "subjects": ["object"]}]
    graph_path = tmp_path / "task_graph.json"
    graph_path.write_text(json.dumps(graph) + "\n", encoding="utf-8")
    reference.task_graph_sha256 = sha256_file(graph_path)

    profile = _runtime_profile_for_reference(manifest, reference)

    assert profile["profile_key"] == "task_graph_native_episode"
    assert profile["controller_operators"] == [operator]
    assert profile["official_runtime_task_sha256"] == sha256_file(graph_path)


def test_cosmos_runtime_rejects_recipe_backed_reference(tmp_path: Path) -> None:
    manifest, reference = _reference(
        tmp_path,
        [{"predicate": "in_container", "subject": "a", "object": "b"}],
        recipe_id="old_sample_recipe",
    )

    with pytest.raises(ValueError, match="must not name a sample recipe"):
        _runtime_profile_for_reference(manifest, reference)


@pytest.mark.parametrize("goal", [
    {"predicate": "on_region", "subject": "object", "object": "support"},
    {"predicate": "inside", "subject": "object", "object": "container", "terminal_upright": True},
])
def test_declared_region_and_orientation_do_not_inherit_an_unrelated_sample_goal(tmp_path, goal):
    manifest, reference = _reference(tmp_path, [goal])
    profile = _runtime_profile_for_reference(manifest, reference)
    assert profile["profile_key"] == "task_graph_native_episode"
    assert profile["terminal_predicates"] == [goal]
