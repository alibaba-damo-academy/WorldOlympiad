"""The score stage passes exact per-episode inputs and preserves failed attempts."""

import shlex
from pathlib import Path

from wo.agent import stages


def _command(**over):
    args = {"run": "r", "case": "c", "model": "cosmos3", "seed": 0,
            "gpu": 0, "compute_gpu": 1, "score_mode": "split", **over}
    return stages.STAGES["score"].command(args, 1)


def test_stage_passes_both_gpus_and_exact_case_inputs():
    command = _command()
    pairs = list(zip(command, command[1:]))
    assert ("--gpu-id", "0") in pairs
    assert ("--compute-gpu-id", "1") in pairs
    assert ("--score-mode", "split") in pairs
    assert ("--workspace", str(stages.WORKSPACE)) in pairs
    assert command[command.index("--ref") + 1] == str(stages._ref({"run":"r", "case":"c"}))
    assert command[command.index("--episode") + 1] == str(stages._ep({"run":"r", "case":"c", "model":"cosmos3", "seed":0}))


def test_case_and_seed_are_scoped_to_this_episode():
    alpha = _command(case="alpha", seed=0)
    beta = _command(case="beta", seed=1)
    assert str(stages._ep({"run":"r", "case":"alpha", "model":"cosmos3", "seed":0}) / "case") in alpha
    assert str(stages._ep({"run":"r", "case":"beta", "model":"cosmos3", "seed":1}) / "case") in beta
    assert all("beta" not in arg for arg in alpha)
    assert all("alpha" not in arg for arg in beta)


def test_not_applicable_case_returns_before_robotics_scorer():
    source = Path(__file__).parents[1] / "agent/score_stage.py"
    text = source.read_text()
    assert 'if not (out / "worldolympiad_case.json").is_file()' in text
    assert 'return json.loads((out / "score.json").read_text())' in text
