"""Tests for fault-tolerant parallel execution utilities."""

import os
import signal
import unittest

from scenecode.utils.parallel import run_parallel_isolated


def _successful_task(value: int, name: str) -> dict:
    """Task that succeeds and returns a result."""
    return {"value": value * 2, "name": name}


def _failing_task(message: str) -> None:
    """Task that raises an exception."""
    raise ValueError(message)


def _void_task(value: int) -> None:
    """Task that succeeds without returning anything."""
    _ = value * 2


def _crashing_task() -> None:
    """Task that crashes the process (simulates OOM/SIGKILL)."""
    os.kill(os.getpid(), signal.SIGKILL)


class TestRunParallelIsolated(unittest.TestCase):
    """Tests for run_parallel_isolated function."""

    def test_all_tasks_succeed(self):
        """All tasks complete successfully."""
        tasks = [
            ("task_a", _successful_task, {"value": 1, "name": "A"}),
            ("task_b", _successful_task, {"value": 2, "name": "B"}),
            ("task_c", _successful_task, {"value": 3, "name": "C"}),
        ]

        results = run_parallel_isolated(tasks, max_workers=2, return_values=True)

        self.assertEqual(len(results), 3)
        for task_id, (success, result) in results.items():
            self.assertTrue(success, f"{task_id} should succeed")
            self.assertIsInstance(result, dict)

        self.assertEqual(results["task_a"][1]["value"], 2)
        self.assertEqual(results["task_b"][1]["value"], 4)
        self.assertEqual(results["task_c"][1]["value"], 6)

    def test_fault_isolation_exception(self):
        """One task raising exception doesn't affect others."""
        tasks = [
            ("ok_1", _successful_task, {"value": 1, "name": "OK1"}),
            ("fail", _failing_task, {"message": "intentional failure"}),
            ("ok_2", _successful_task, {"value": 2, "name": "OK2"}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=2, return_values=True)

        self.assertEqual(len(results), 3)

        # The failing task should fail.
        self.assertFalse(results["fail"][0])
        self.assertIn("intentional failure", results["fail"][1])

        # Other tasks should succeed.
        self.assertTrue(results["ok_1"][0])
        self.assertTrue(results["ok_2"][0])

    def test_fault_isolation_crash(self):
        """One task crashing (SIGKILL) doesn't affect others."""
        tasks = [
            ("ok_1", _successful_task, {"value": 1, "name": "OK1"}),
            ("crash", _crashing_task, {}),
            ("ok_2", _successful_task, {"value": 2, "name": "OK2"}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=2, return_values=True)

        self.assertEqual(len(results), 3)

        # The crashing task should fail with crash message.
        self.assertFalse(results["crash"][0])
        self.assertIn("crashed", results["crash"][1].lower())

        # Other tasks should succeed.
        self.assertTrue(results["ok_1"][0])
        self.assertTrue(results["ok_2"][0])

    def test_all_tasks_fail(self):
        """All tasks failing is handled gracefully."""
        tasks = [
            ("fail_1", _failing_task, {"message": "error 1"}),
            ("fail_2", _failing_task, {"message": "error 2"}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=2, return_values=True)

        self.assertEqual(len(results), 2)
        self.assertFalse(results["fail_1"][0])
        self.assertFalse(results["fail_2"][0])

    def test_return_values_false(self):
        """With return_values=False, successful tasks return None."""
        tasks = [
            ("task_a", _successful_task, {"value": 1, "name": "A"}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=1, return_values=False)

        self.assertEqual(len(results), 1)
        self.assertTrue(results["task_a"][0])
        self.assertIsNone(results["task_a"][1])

    def test_void_task(self):
        """Tasks that don't return values work correctly."""
        tasks = [
            ("void_1", _void_task, {"value": 42}),
            ("void_2", _void_task, {"value": 100}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=2, return_values=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(results["void_1"][0])
        self.assertTrue(results["void_2"][0])

    def test_single_worker(self):
        """Works correctly with max_workers=1 (sequential)."""
        tasks = [
            ("task_a", _successful_task, {"value": 1, "name": "A"}),
            ("task_b", _successful_task, {"value": 2, "name": "B"}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=1, return_values=True)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(success for success, _ in results.values()))

    def test_more_workers_than_tasks(self):
        """Works when max_workers exceeds task count."""
        tasks = [
            ("task_a", _successful_task, {"value": 1, "name": "A"}),
        ]

        results = run_parallel_isolated(tasks=tasks, max_workers=10, return_values=True)

        self.assertEqual(len(results), 1)
        self.assertTrue(results["task_a"][0])

    def test_empty_task_list(self):
        """Empty task list returns empty results."""
        results = run_parallel_isolated(tasks=[], max_workers=2)

        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
