import os
import shlex
import unittest

# The queue's deployment profile is not required to test command-line overrides.
os.environ["WO_EVAL_CONFIG"] = "/tmp/wo-test-profile-does-not-exist.json"
from wo.agent.stages import STAGES


class StageOverrideTests(unittest.TestCase):
    def test_lingbot_episode_uses_explicit_gpu(self):
        args = {
            "run": "test_run",
            "case": "test_case",
            "model": "lingbot_va",
            "seed": 0,
            "gpu": 0,
        }
        command = shlex.split(" ".join(STAGES["episode"].command(args, 123)))
        self.assertEqual(command[command.index("--gpu") + 1], "0")
        self.assertNotIn("--simulator-gpu", command)

    def test_score_uses_explicit_gpu_and_split_mode(self):
        args = {
            "run": "test_run",
            "case": "test_case",
            "model": "lingbot_va",
            "gpu": 1,
            "score_mode": "split",
        }
        stage = STAGES["score"]
        stage.validate(args)
        command = shlex.split(" ".join(stage.command(args, 124)))
        self.assertEqual(command[command.index("--gpu-id") + 1], "1")
        self.assertEqual(command[command.index("--score-mode") + 1], "split")

    def test_score_rejects_unknown_mode(self):
        args = {"run": "test_run", "case": "test_case", "model": "lingbot_va",
                "score_mode": "other"}
        with self.assertRaisesRegex(ValueError, "score_mode must be full or split"):
            STAGES["score"].validate(args)


if __name__ == "__main__":
    unittest.main()
