import json

from wo.eval.cae_episodes import _capture_metadata


def test_capture_metadata_reads_reference_bound_adapter_layout(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    expected = {"task_success": False, "completion_evidence": {"native_predicate_evidence": {"result": False}}}
    (capture / "capture_metadata.json").write_text(json.dumps(expected))
    assert _capture_metadata(tmp_path, {}) == expected


def test_capture_metadata_prefers_native_run_layout(tmp_path):
    native = tmp_path / "native_run/capture"
    native.mkdir(parents=True)
    (tmp_path / "capture").mkdir()
    expected = {"source": "native_run"}
    (native / "capture_metadata.json").write_text(json.dumps(expected))
    (tmp_path / "capture/capture_metadata.json").write_text(json.dumps({"source": "capture"}))
    assert _capture_metadata(tmp_path, {}) == expected


def test_capture_metadata_uses_embedded_receipt_when_files_are_absent(tmp_path):
    expected = {"task_success": True}
    assert _capture_metadata(tmp_path, {"native_result": expected}) == expected
