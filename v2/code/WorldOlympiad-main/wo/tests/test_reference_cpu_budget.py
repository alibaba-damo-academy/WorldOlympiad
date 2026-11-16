from wo.agent import stages


def test_reference_and_control_use_small_cpu_thread_pools(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "104")
    for kind in ("reference", "expert_check"):
        env = stages.STAGES[kind].env()
        assert env["OMP_NUM_THREADS"] == "1"
        assert env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["MKL_NUM_THREADS"] == "1"


def test_policy_and_scorer_thread_settings_are_preserved(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    for kind in ("episode", "score"):
        assert stages.STAGES[kind].env()["OMP_NUM_THREADS"] == "7"
