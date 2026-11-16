"""The episode stage model registry and the agent's view of it.

A bulk overwrite of wo/agent/*.py silently dropped two adapters from MODELS and
the agent then spent rounds believing lingbot_va and gr00t did not exist. These
tests pin the registry and the adapter modules behind it.
"""

import importlib

from wo.agent import stages, tools


def test_episode_stage_registers_every_available_adapter():
    assert set(stages.MODELS) >= {"cosmos3", "lingbot_va", "gr00t"}


def test_every_registered_adapter_module_imports():
    for name, module in stages.MODELS.items():
        mod = importlib.import_module(module)
        assert hasattr(mod, "main"), f"{module} for {name} has no CLI entry point"


def test_episode_stage_accepts_a_registered_model_and_rejects_an_unknown_one():
    stage = stages.STAGES["episode"]
    stage.validate({"run": "r", "case": "c", "model": "gr00t", "seed": 0})
    try:
        stage.validate({"run": "r", "case": "c", "model": "nope", "seed": 0})
    except ValueError as exc:
        assert "no adapter" in str(exc)
    else:
        raise AssertionError("an unregistered model must be refused")


def test_status_reports_adapter_names_from_a_fresh_reload():
    # The agent process outlives code changes; status must not serve a snapshot
    # of the registry taken when it started.
    names = tools.registered_models()
    assert set(names) >= {"cosmos3", "lingbot_va", "gr00t"}
    assert names == sorted(names)
