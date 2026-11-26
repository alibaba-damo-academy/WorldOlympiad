from types import SimpleNamespace


def test_shell_filters_api_credentials_and_preserves_runtime_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WO_EVAL_CONFIG", str(tmp_path / "missing-config.json"))
    from wo.agent import tools

    monkeypatch.setenv("WO_AGENT_API_KEY", "agent-fixture-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-fixture-secret")
    monkeypatch.setenv("PATH", "/fixture/bin:/usr/bin")
    monkeypatch.setenv("TMPDIR", "/fixture/tmp")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("WO_RUNTIME_SETTING", "preserved")
    monkeypatch.setattr(tools, "WORKSPACE", tmp_path)
    monkeypatch.setattr(tools, "ensure_worktree", lambda: None)

    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    result = tools.shell("echo ok")

    assert result.startswith("cwd=")
    child_env = captured["env"]
    assert "WO_AGENT_API_KEY" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert child_env["PATH"] == "/fixture/bin:/usr/bin"
    assert child_env["TMPDIR"] == "/fixture/tmp"
    assert child_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert child_env["WO_RUNTIME_SETTING"] == "preserved"
