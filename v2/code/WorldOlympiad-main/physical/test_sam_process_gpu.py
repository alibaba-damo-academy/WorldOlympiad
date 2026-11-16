"""Keep SAM3 on the scheduler-selected physical GPU."""
import os
import unittest
from unittest.mock import patch

from physical import sam_process


class SamProcessGpuTest(unittest.TestCase):
    def test_scorer_sam3_uses_physical_gpu(self):
        with patch.dict(os.environ, {"WOV2_SCORE_COMPUTE_GPU": "3", "CUDA_VISIBLE_DEVICES": "3"}):
            with patch.object(sam_process, "SAM3VideoSemanticPredictor", side_effect=lambda overrides: overrides):
                settings = sam_process.build_sam3_predictor(sam3_model_path="unused.pt", device="cuda:0")
        self.assertEqual(settings["device"], "3")

    def test_scorer_sam3_rejects_changed_gpu_mask(self):
        with patch.dict(os.environ, {"WOV2_SCORE_COMPUTE_GPU": "3", "CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "GPU mask changed"):
                sam_process.build_sam3_predictor(sam3_model_path="unused.pt", device="cuda:0")


if __name__ == "__main__":
    unittest.main()
