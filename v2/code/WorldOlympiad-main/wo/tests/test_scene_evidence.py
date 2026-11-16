import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

module_path = Path(__file__).parents[1] / "agent/scene_evidence.py"
spec = importlib.util.spec_from_file_location("scene_evidence", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SceneEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        db = self.root / "outputs/wo_agent/jobs.sqlite3"
        db.parent.mkdir(parents=True)
        self.connection = sqlite3.connect(db)
        self.addCleanup(self.connection.close)
        self.connection.execute("create table jobs(id integer,kind text,status text,args text,started real,finished real,log text)")

    def job(self, number, *, status="running", case="box", run="nine", started=100, finished=None):
        self.connection.execute("insert into jobs values(?,?,?,?,?,?,?)", (number, "scene", status,
            json.dumps({"run": run, "case": case}), started, finished, None))
        self.connection.commit()

    def artifact(self, partial, relative, timestamp=110, origin=None):
        path = self.root / f"outputs/robotics_scenecode_generations/wo_box/{partial}/scene_run" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(time.strftime("[%Y-%m-%d %H:%M:%S] evidence", time.localtime(timestamp if origin is None else origin)))
        os.utime(path, (timestamp, timestamp))
        return path

    def test_queued_retry_does_not_present_old_partial_or_published_scene_as_current(self):
        self.job(1, status="failed", finished=120)
        self.job(2, status="queued", started=None)
        self.artifact(".box.partial.123", "main.log")
        task = self.root / "outputs/wo_runs/nine/tasks/box.json"
        task.parent.mkdir(parents=True)
        task.write_text(json.dumps({"generation": "/old/published/box"}))
        evidence = module.collect_scene_evidence(self.root, "nine", "box")
        self.assertEqual(evidence["latest_scene_job"]["id"], 2)
        self.assertEqual(evidence["current_attempts"], [])
        self.assertFalse(evidence["published_generation"]["belongs_to_latest_job"])
        self.assertEqual(evidence["acceptance"], "not_assessed")

    def test_current_physical_geometry_is_separate_from_concept_and_furniture(self):
        self.job(1)
        partial = ".box.partial.123"
        self.artifact(partial, "main.log")
        self.artifact(partial, "scene_000/room/generated_assets/manipuland/code_object/box/renders/full_object.png")
        self.artifact(partial, "scene_000/room/generated_assets/furniture/code_object/table/renders/full_object.png")
        self.artifact(partial, "scene_000/room/generated_assets/manipuland/code_object/box/image.png")
        self.artifact(".box.partial.older", "main.log", timestamp=50)
        evidence = module.collect_scene_evidence(self.root, "nine", "box")
        self.assertEqual(len(evidence["current_attempts"]), 1)
        attempt = evidence["current_attempts"][0]
        self.assertEqual({p["kind"] for p in attempt["geometry_renders"]}, {"manipuland_geometry", "furniture_geometry"})
        self.assertEqual(attempt["concept_images"][0]["kind"], "concept_only")
        self.assertEqual(attempt["scene_exports"], [])

    def test_other_runs_and_post_timeout_orphan_logs_are_not_current(self):
        self.job(1, status="failed", finished=120)
        self.job(2, run="different")
        self.artifact(".box.partial.orphan", "main.log", timestamp=130)
        evidence = module.collect_scene_evidence(self.root, "nine", "box")
        self.assertEqual(evidence["latest_scene_job"]["id"], 1)
        self.assertEqual(evidence["current_attempts"], [])

    def test_rejects_directory_traversal(self):
        with self.assertRaises(ValueError):
            module.collect_scene_evidence(self.root, "../nine", "box")

    def test_old_child_appending_during_active_retry_is_not_current(self):
        self.job(2, started=200)
        self.artifact(".box.partial.older", "main.log", timestamp=210, origin=100)
        self.artifact(".box.partial.current", "main.log", timestamp=210, origin=201)
        attempts = module.collect_scene_evidence(self.root, "nine", "box")["current_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertIn(".box.partial.current", attempts[0]["log"]["path"])


if __name__ == "__main__":
    unittest.main()
