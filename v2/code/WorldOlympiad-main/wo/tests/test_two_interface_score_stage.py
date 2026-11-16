import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wo.agent.score_stage import archive_case, valid_receipt


class ScoreStageRetryTests(unittest.TestCase):
    def test_valid_receipt_binds_score_to_exact_video_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            files = {
                "generated.mp4": b"generated",
                "reference.mp4": b"reference",
                "prompt.json": b"[]",
                "generated_chunk_timestamps.json": b"{}",
                "score.json": b'{"combined_score":0.5}',
            }
            for name, payload in files.items():
                (case / name).write_bytes(payload)
            (case / "worldolympiad_case.json").write_text(json.dumps({
                "generated_video": "generated.mp4",
                "reference_video": "reference.mp4",
            }))
            names = ("generated.mp4", "reference.mp4", "prompt.json",
                     "generated_chunk_timestamps.json")
            receipt = {
                "inputs": {name: hashlib.sha256((case / name).read_bytes()).hexdigest()
                           for name in names},
                "score_sha256": hashlib.sha256((case / "score.json").read_bytes()).hexdigest(),
            }
            (case / "three_track_receipt.json").write_text(json.dumps(receipt))
            self.assertTrue(valid_receipt(case))
            (case / "generated.mp4").write_bytes(b"changed")
            self.assertFalse(valid_receipt(case))

    def test_incomplete_case_is_archived_with_all_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            case.mkdir()
            (case / "score.json").write_text('{"partial":true}')
            old = archive_case(case)
            self.assertFalse(case.exists())
            self.assertEqual((old / "score.json").read_text(), '{"partial":true}')


if __name__ == "__main__":
    unittest.main()
