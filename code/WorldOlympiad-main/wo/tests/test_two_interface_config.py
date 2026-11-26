import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wo.agent import config, jobs, stages, tools


def profile_file(root: Path, *, base_url="https://agent.example/v1", model="agent-model",
                 adapter="lingbot_va", checkpoint="/weights/local-model", mode="auto", gpu=0,
                 simulator_gpu=1, policy_gpus="auto"):
    path = root / "two-interface.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "workspace_root": "/workspace",
        "agent_api": {"base_url": base_url, "model": model, "api_key_env": "WO_AGENT_API_KEY"},
        "tested_model": {"adapter": adapter, "checkpoint": checkpoint, "gpu": gpu,
                         "simulator_gpu": simulator_gpu, "policy_gpus": policy_gpus},
        "scoring": {"mode": mode, "gpu": gpu},
    }))
    return path


class TwoInterfaceConfigTests(unittest.TestCase):
    def test_maps_two_user_interfaces_without_printing_or_storing_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = profile_file(Path(tmp))
            profile = config.load_profile(path, required=True)
            inventory = [
                {"index": i, "total_mib": 32607, "free_mib": 32000, "utilization": 0}
                for i in range(3)
            ]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                env = config.apply_environment({"WO_AGENT_API_KEY": "test-secret"}, profile)
            self.assertEqual(env["WO_AGENT_API_BASE"], "https://agent.example/v1")
            self.assertEqual(env["OPENAI_API_BASE"], "https://agent.example/v1")
            self.assertEqual(env["OPENAI_BASE_URL"], "https://agent.example/v1")
            self.assertEqual(env["OPENAI_API_KEY"], "test-secret")
            self.assertEqual(env["WO_AGENT_MODEL"], "agent-model")
            self.assertEqual(env["WO_TESTED_MODEL"], "lingbot_va")
            self.assertEqual(env["WO_TESTED_MODEL_CHECKPOINT"], "/weights/local-model")
            self.assertNotIn("test-secret", path.read_text())

    def test_environment_endpoint_and_model_overrides_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp)))
            inventory = [
                {"index": i, "total_mib": 32607, "free_mib": 32000, "utilization": 0}
                for i in range(3)
            ]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                env = config.apply_environment({
                    "WO_AGENT_API_BASE": "https://override.example/v1",
                    "WO_AGENT_API_MODEL": "override-model",
                    "WO_AGENT_API_KEY": "test-secret",
                }, profile)
            self.assertEqual(env["WO_AGENT_API_BASE"], "https://override.example/v1")
            self.assertEqual(env["OPENAI_API_BASE"], "https://override.example/v1")
            self.assertEqual(env["OPENAI_BASE_URL"], "https://override.example/v1")
            self.assertEqual(env["WO_AGENT_MODEL"], "override-model")

    def test_rejects_secret_values_and_unconfigured_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = profile_file(Path(tmp))
            payload = json.loads(path.read_text())
            payload["agent_api"]["api_key"] = "must-not-be-saved"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "environment variable"):
                config.load_profile(path)
            path = profile_file(Path(tmp), base_url="https://YOUR_OPENAI_COMPATIBLE_ENDPOINT/v1")
            with self.assertRaisesRegex(ValueError, "base_url"):
                config.load_profile(path)

    def test_configured_model_is_the_only_episode_and_score_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp)))
            inventory = [
                {"index": i, "total_mib": 32607, "free_mib": 32000, "utilization": 0}
                for i in range(3)
            ]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                episode = config.job_defaults("episode", {"run": "r", "case": "c"}, profile)
                score = config.job_defaults("score", {"run": "r", "case": "c"}, profile)
            self.assertEqual(episode["model"], "lingbot_va")
            self.assertEqual(episode["gpu"], 0)
            self.assertEqual(episode["policy_gpus"], [0, 2])
            self.assertEqual(score["model"], "lingbot_va")
            with self.assertRaisesRegex(ValueError, "configured tested model"):
                config.job_defaults("episode", {"model": "cosmos3"}, profile)

    def test_automatic_score_mode_and_reservations_follow_device_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp)))
            self.assertEqual(config.score_mode(profile, info={"total_mib": 32607, "free_mib": 32000}), "split")
            self.assertEqual(config.score_mode(profile, info={"total_mib": 85651, "free_mib": 84000}), "full")
            low_info = {"total_mib": 32607, "free_mib": 32000}
            low_inventory = [{"index": 0, **low_info, "utilization": 0}]
            high_info = {"total_mib": 85651, "free_mib": 84000}
            high_inventory = [{"index": 0, **high_info, "utilization": 0}]
            low = config.resource_limits(profile, info=low_info, inventory=low_inventory)
            high = config.resource_limits(profile, info=high_info, inventory=high_inventory)
            self.assertLessEqual(low["score"], low["budget"])
            self.assertEqual(high["score"], 50)

    def test_auto_gpu_selects_a_device_with_memory_for_the_configured_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto"))
            inventory = [
                {"index": 0, "total_mib": 32607, "free_mib": 5000},
                {"index": 1, "total_mib": 32607, "free_mib": 31000},
                {"index": 2, "total_mib": 85651, "free_mib": 30000},
            ]
            self.assertEqual(config.resolve_gpu(profile, inventory=inventory), 1)
            self.assertEqual(config.resolve_gpu(profile, "scoring", inventory=inventory), 1)
            self.assertEqual(config.gpu_inventory("0, 32607, 30000\n1, 85651, 80000"), [
                {"index": 0, "total_mib": 32607, "free_mib": 30000},
                {"index": 1, "total_mib": 85651, "free_mib": 80000},
            ])

    def test_auto_simulator_gpu_splits_small_cards_and_keeps_large_cards_single_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            inventory = [
                {"index": 0, "total_mib": 32607, "free_mib": 32000, "utilization": 1},
                {"index": 1, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
                {"index": 2, "total_mib": 32607, "free_mib": 32000, "utilization": 45},
            ]
            self.assertEqual(config.resolve_simulator_gpu(profile, 0, inventory=inventory), 1)
            self.assertIsNone(config.resolve_simulator_gpu(profile, 0, inventory=inventory[:1]))
            large = [{"index": 0, "total_mib": 85651, "free_mib": 80000, "utilization": 0}]
            self.assertEqual(config.resolve_simulator_gpu(profile, 0, inventory=large), 0)
            self.assertEqual(config.gpu_inventory("0, 32607, 30000, 2"), [
                {"index": 0, "total_mib": 32607, "free_mib": 30000, "utilization": 2},
            ])

    def test_episode_defaults_include_a_separate_simulator_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            inventory = [
                {"index": 4, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
                {"index": 5, "total_mib": 32607, "free_mib": 32000, "utilization": 1},
                {"index": 6, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
            ]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                episode = config.job_defaults("episode", {"run": "r", "case": "c"}, profile)
            self.assertEqual(episode["gpu"], 4)
            self.assertEqual(episode["simulator_gpu"], 5)
            self.assertEqual(episode["policy_gpus"], [4, 6])

    def test_episode_is_not_queued_when_second_5090_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            only_one = [{"index": 5, "total_mib": 32607, "free_mib": 32000, "utilization": 0}]
            with patch.object(config, "all_gpu_info", return_value=only_one):
                with self.assertRaisesRegex(RuntimeError, "separate idle GPU"):
                    config.job_defaults("episode", {"run": "r", "case": "c"}, profile)

    def test_small_lingbot_requires_two_policy_gpus_plus_a_simulator_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            inventory = [
                {"index": 0, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
                {"index": 1, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
            ]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                with self.assertRaisesRegex(RuntimeError, "three idle GPUs"):
                    config.job_defaults("episode", {"run": "r", "case": "c"}, profile)

    def test_auto_gpu_does_not_claim_a_card_with_an_existing_compute_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            inventory = [
                {"index": 0, "total_mib": 32607, "free_mib": 31000, "utilization": 0,
                 "compute_process_count": 0, "process_query_ok": True},
                {"index": 2, "total_mib": 32607, "free_mib": 30900, "utilization": 0,
                 "compute_process_count": 0, "process_query_ok": True},
                {"index": 5, "total_mib": 32607, "free_mib": 32000, "utilization": 0,
                 "compute_process_count": 1, "process_query_ok": True},
            ]
            self.assertEqual(config.resolve_gpu(profile, inventory=inventory), 0)

    def test_preparation_phase_runs_with_two_cards_and_defers_episode_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            inventory = [
                {"index": 0, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
                {"index": 2, "total_mib": 32607, "free_mib": 31000, "utilization": 0},
            ]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                check = config.preflight(profile, {"WO_AGENT_API_KEY": "test-secret"}, phase="preparation")
                with self.assertRaisesRegex(RuntimeError, "three idle GPUs"):
                    config.preflight(profile, {"WO_AGENT_API_KEY": "test-secret"}, phase="evaluation")
            self.assertEqual(check["phase"], "preparation")
            self.assertTrue(check["tested_model"]["episode_resources_deferred"])
            self.assertEqual(check["resources"]["scoring"], {"deferred": True})

    def test_preparation_phase_cannot_submit_episode_jobs(self):
        with patch.dict("os.environ", {"WO_RESOURCE_PHASE": "preparation"}):
            result = tools.submit("episode", "{}")
        self.assertIn("preparation phase", result)
        self.assertIn("not queued", result)

    def test_worker_preserves_the_launchers_preparation_phase(self):
        with patch.dict("os.environ", {"WO_RESOURCE_PHASE": "preparation"}):
            with patch.object(jobs, "load_private_environment"):
                with patch.object(
                    jobs, "apply_environment",
                    side_effect=lambda env, phase: env.__setitem__("WO_RESOURCE_PHASE", phase),
                ) as apply:
                    jobs._prepare_worker_environment()
                    self.assertEqual(os.environ["WO_RESOURCE_PHASE"], "preparation")
        self.assertEqual(apply.call_args.kwargs["phase"], "worker")

    def test_episode_accepts_a_valid_queue_gpu_group_when_profile_uses_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = config.load_profile(profile_file(Path(tmp), gpu="auto", simulator_gpu="auto"))
            inventory = [
                {"index": i, "total_mib": 32607, "free_mib": 32000, "utilization": 0}
                for i in range(4)
            ]
            args = {"run": "r", "case": "c", "gpu": 0, "simulator_gpu": 2,
                    "policy_gpus": [0, 3]}
            with patch.object(config, "all_gpu_info", return_value=inventory):
                episode = config.job_defaults("episode", args, profile)
            self.assertEqual(episode["gpu"], 0)
            self.assertEqual(episode["simulator_gpu"], 2)
            self.assertEqual(episode["policy_gpus"], [0, 3])

    def test_episode_stage_passes_both_gpu_indices_to_lingbot(self):
        command = stages.STAGES["episode"].command({
            "run": "run0", "case": "case0", "model": "lingbot_va",
            "gpu": 4, "simulator_gpu": 5, "policy_gpus": [4, 6],
        }, 7)
        self.assertEqual(command[command.index("--gpu") + 1], "4")
        self.assertEqual(command[command.index("--simulator-gpu") + 1], "5")
        self.assertEqual(command[command.index("--policy-gpus") + 1:command.index("--policy-gpus") + 3],
                         ["4", "6"])

    def test_preflight_reports_missing_robotwin_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = json.loads(profile_file(root).read_text())
            payload["workspace_root"] = str(root)
            payload["tested_model"].update({
                "checkpoint": "weights/model",
                "simulator_gpu": 0,
                "robotwin_repo": "runtime/RoboTwin",
                "lingbot_repo": "runtime/lingbot-va",
                "robotwin_python": "envs/robotwin/bin/python",
                "lingbot_python": "envs/lingbot/bin/python",
                "scenecode_source": "code/SceneCode-main",
                "scenecode_runtime": "tmp/scenecode_runtime",
            })
            path = root / "two-interface.json"
            path.write_text(json.dumps(payload))
            for directory in [
                "weights/model", "runtime/RoboTwin/assets/embodiments/aloha-agilex",
                "runtime/RoboTwin/assets/objects/objaverse", "runtime/lingbot-va",
                "envs/robotwin/bin", "envs/lingbot/bin", "code/SceneCode-main",
                "tmp/scenecode_runtime",
            ]:
                (root / directory).mkdir(parents=True, exist_ok=True)
            (root / "runtime/RoboTwin/assets/objects/objaverse/list.json").write_text("[]")
            (root / "code/SceneCode-main/main.py").touch()
            (root / "tmp/scenecode_runtime/.worldolympiad_scenecode_runtime_v1").touch()
            profile = config.load_profile(path, required=True)
            inventory = [{"index": 0, "total_mib": 85651, "free_mib": 80000, "utilization": 0}]
            with patch.object(config, "all_gpu_info", return_value=inventory):
                check = config.preflight(profile, {"WO_AGENT_API_KEY": "test-secret"})
            self.assertFalse(check["tested_model"]["robotwin_runtime_ready"])
            self.assertFalse(check["tested_model"]["robotwin_required_files"][
                "assets/embodiments/aloha-agilex/config.yml"])


if __name__ == "__main__":
    unittest.main()
