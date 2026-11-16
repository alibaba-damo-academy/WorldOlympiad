from pathlib import Path
from wo.agent import tools as T


def setup(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    worktree = workspace / "code/agent"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(T, "WORKSPACE", workspace)
    monkeypatch.setattr(T, "WORKTREE", worktree)
    return workspace, worktree


def test_shell_and_workspace_tools_share_current_outputs_without_losing_old_data(tmp_path, monkeypatch):
    workspace, worktree = setup(tmp_path, monkeypatch)
    old = worktree / "outputs"
    old.mkdir()
    (old / "old_result.json").write_text("historical result")
    shared = workspace / "outputs"
    shared.mkdir()
    (shared / "current.txt").write_text("current shared artifact")
    result = T.shell("cat outputs/current.txt", timeout_s=10)
    assert "rc=0" in result and "current shared artifact" in result
    assert str(shared) in result
    assert old.is_symlink() and old.samefile(shared)
    archived = list((shared / "wo_agent/worktree_data_history").glob("*/old_result.json"))
    assert len(archived) == 1 and archived[0].read_text() == "historical result"
    T.ensure_worktree()
    assert len(list(archived[0].parent.parent.iterdir())) == 1
    assert T._inside("outputs/current.txt").read_text() == "current shared artifact"


def test_new_worktree_gets_shared_outputs_link(tmp_path, monkeypatch):
    workspace, worktree = setup(tmp_path, monkeypatch)
    T.ensure_worktree()
    assert (worktree / "outputs").samefile(workspace / "outputs")
    assert not (workspace / "outputs/wo_agent/worktree_data_history").exists()
