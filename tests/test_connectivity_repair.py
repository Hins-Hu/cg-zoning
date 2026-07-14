"""Unit tests for the cost-preserving zone patching (cg/connectivity_repair)."""

import unittest

from cg.connectivity_repair import repair_zones


class ConnectivityRepairTests(unittest.TestCase):
    def test_repair_absorbs_cost_free_cells_from_full_candidate_set(self):
        distances = {
            "a": {"a": 0.0, "b": 1.0, "c": 1.0, "d": 2.0},
            "b": {"a": 1.0, "b": 0.0, "c": 1.0, "d": 2.0},
            "c": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 2.0},
            "d": {"a": 2.0, "b": 2.0, "c": 2.0, "d": 0.0},
        }

        repaired_zones, stats = repair_zones(
            zones=[{"a", "b"}],
            distances=distances,
        )

        self.assertEqual(repaired_zones, [{"a", "b", "c"}])
        self.assertEqual(stats[0].added_cells, ["c"])

    def test_repair_is_sequential_and_rejects_mutually_incompatible_additions(self):
        distances = {
            "a": {"a": 0.0, "b": 1.0, "c": 1.0, "d": 1.0},
            "b": {"a": 1.0, "b": 0.0, "c": 1.0, "d": 1.0},
            "c": {"a": 1.0, "b": 1.0, "c": 0.0, "d": 2.5},
            "d": {"a": 1.0, "b": 1.0, "c": 2.5, "d": 0.0},
        }

        repaired_zones, stats = repair_zones(
            zones=[{"a", "b"}],
            distances=distances,
            candidate_cells=["c", "d"],
        )

        self.assertEqual(repaired_zones, [{"a", "b", "c"}])
        self.assertEqual(stats[0].added_cells, ["c"])


if __name__ == "__main__":
    unittest.main()
