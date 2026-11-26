import json
from types import SimpleNamespace

import pytest

from wo.eval import case


def test_build_case_accepts_native_nested_capture_with_exact_matching_view(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    (ref / "replay").mkdir(parents=True)
    (ref / "replay/exterior_left.mp4").write_bytes(b"reference video")
    task = tmp_path / "task.json"
    task.write_text("{}")
    (ref / "certificate.json").write_text(json.dumps({"task": str(task)}))

    episode = tmp_path / "episode"
    (episode / "capture").mkdir(parents=True)
    (episode / "capture/exterior_left.mp4").write_bytes(b"native exterior video")
    (episode / "capture/head.mp4").write_bytes(b"native head video")
    (episode / "episode.json").write_text(json.dumps({
        "model": "lingbot_va", "task_success": False,
        "capture": {"videos": {"exterior_left": "capture/exterior_left.mp4",
                                "head": "capture/head.mp4"}},
    }))

    monkeypatch.setattr(case.TaskSpec, "load", lambda _: SimpleNamespace(
        case_id="case_a", family="pick_place", instruction="move object"))
    monkeypatch.setattr("wo.eval.report.applicability_result", lambda _: None)
    monkeypatch.setattr(case, "_frames", lambda _: 1)
    monkeypatch.setattr(case, "write_binding", lambda *_: {})

    out = case.build_case(ref, episode, tmp_path / "score_case")
    record = json.loads((out / "worldolympiad_case.json").read_text())
    assert record["reference_view_id"] == "exterior_left"
    assert record["generated_view_id"] == "exterior_left"
    assert record["view_contract"] == "matched_single_physical_view_v2"
    assert record["task_success"] is False
    assert (out / "lingbot_va_gen_case_a.mp4").read_bytes() == b"native exterior video"


def test_lingbot_mismatched_native_camera_is_rejected(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "head.mp4").touch()
    with pytest.raises(FileNotFoundError, match="required matching view 'exterior_left'"):
        case._generated_video({"model": "lingbot_va", "capture": {"videos": {
            "head": "head.mp4", "left_wrist": "left.mp4"}}}, episode, "exterior_left")



def test_other_models_do_not_silently_substitute_a_native_camera(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "head.mp4").touch()
    with pytest.raises(FileNotFoundError, match="required matching view 'exterior_left'"):
        case._generated_video({"model": "other", "capture": {"videos": {"head": "head.mp4"}}},
                              episode, "exterior_left")


def test_build_case_prefers_exact_view_when_adapter_publishes_it(tmp_path):
    episode = tmp_path / "ep"
    episode.mkdir()
    path = episode / "view.mp4"
    path.touch()
    selected, view = case._generated_video(
        {"videos": {"exterior_left": str(path), "head": "other.mp4"}},
        episode, "exterior_left")
    assert selected == path
    assert view == "exterior_left"
