import unittest

from scripts.select_setup_gpu import select_gpu


def card(index, free, *, utilization=0, pids=(), query_ok=True):
    return {"index": index, "uuid": f"GPU-{index}", "free_mib": free,
            "utilization": utilization, "compute_app_pids": list(pids),
            "compute_process_count": len(pids), "process_query_ok": query_ok}


class SelectSetupGpuTests(unittest.TestCase):
    def test_selects_most_free_eligible_card_and_breaks_ties_by_index(self):
        selected = select_gpu([card(4, 12000), card(2, 16000), card(1, 16000)])
        self.assertEqual(selected["index"], 1)

    def test_rejects_busy_or_under_threshold_cards(self):
        candidates = [card(0, 24000, pids=(123,)), card(1, 24000, utilization=15),
                      card(2, 7000), card(3, 24000, query_ok=False)]
        with self.assertRaisesRegex(RuntimeError, "idle allowlisted NVIDIA GPU"):
            select_gpu(candidates)


if __name__ == "__main__":
    unittest.main()
