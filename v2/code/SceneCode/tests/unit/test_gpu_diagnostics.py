import unittest

from unittest.mock import patch

from scenecode.agent_utils import gpu_diagnostics


class TestGpuDiagnostics(unittest.TestCase):
    def test_collect_gpu_snapshot_prefers_nvml(self):
        nvml_snapshot = {
            "backend": "nvml",
            "devices": [
                {
                    "index": 0,
                    "uuid": "GPU-123",
                    "name": "Test GPU",
                    "total_mib": 81920,
                    "used_mib": 1024,
                    "free_mib": 80896,
                }
            ],
            "processes": [
                {
                    "gpu_index": 0,
                    "gpu_uuid": "GPU-123",
                    "pid": 123,
                    "process_name": "python",
                    "used_mib": 512,
                }
            ],
        }
        with patch.object(
            gpu_diagnostics, "_collect_with_nvml", return_value=nvml_snapshot
        ), patch.object(gpu_diagnostics, "_collect_with_nvidia_smi") as mock_smi:
            snapshot = gpu_diagnostics.collect_gpu_snapshot("nvml test")

        self.assertEqual(snapshot["backend"], "nvml")
        mock_smi.assert_not_called()
        formatted = gpu_diagnostics.format_gpu_snapshot(snapshot)
        self.assertIn("GPU snapshot [nvml test]", formatted)
        self.assertIn("pid=123 name=python used=512 MiB", formatted)

    def test_collect_gpu_snapshot_falls_back_to_nvidia_smi(self):
        smi_snapshot = {
            "backend": "nvidia-smi",
            "devices": [
                {
                    "index": 1,
                    "uuid": "GPU-456",
                    "name": "Fallback GPU",
                    "total_mib": 40960,
                    "used_mib": 2048,
                    "free_mib": 38912,
                }
            ],
            "processes": [],
        }
        with patch.object(gpu_diagnostics, "_collect_with_nvml", return_value=None), patch.object(
            gpu_diagnostics,
            "_collect_with_nvidia_smi",
            return_value=smi_snapshot,
        ):
            snapshot = gpu_diagnostics.collect_gpu_snapshot("smi test")

        self.assertEqual(snapshot["backend"], "nvidia-smi")
        formatted = gpu_diagnostics.format_gpu_snapshot(snapshot)
        self.assertIn("GPU 1 (Fallback GPU)", formatted)
        self.assertIn("processes: none", formatted)

    def test_collect_gpu_snapshot_reports_unavailable(self):
        with patch.object(gpu_diagnostics, "_collect_with_nvml", return_value=None), patch.object(
            gpu_diagnostics, "_collect_with_nvidia_smi", return_value=None
        ):
            snapshot = gpu_diagnostics.collect_gpu_snapshot("missing test")

        self.assertEqual(snapshot["backend"], "unavailable")
        formatted = gpu_diagnostics.format_gpu_snapshot(snapshot)
        self.assertIn("Neither NVML nor nvidia-smi was available", formatted)
        self.assertIn("No GPU device information available.", formatted)
