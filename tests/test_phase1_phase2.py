"""Small correctness checks for the competition metric and blocking invariants."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase1 import entity_f05, validation_member
from phase2 import blocking_keys, record_view


class MetricTests(unittest.TestCase):
    def test_singletons_and_false_merges(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05(set(), {"S2-1"}), 0.0)
        self.assertEqual(entity_f05({"S2-1"}, set()), 0.0)

    def test_precision_weighting(self):
        self.assertAlmostEqual(entity_f05({"S2-1"}, {"S2-1", "S3-1"}), 5 / 9)
        self.assertEqual(entity_f05({"S2-1", "S3-1"}, {"S2-1", "S3-1"}), 1.0)

    def test_split_is_repeatable(self):
        self.assertEqual(validation_member("S1-123", 2026), validation_member("S1-123", 2026))


class BlockingTests(unittest.TestCase):
    def test_local_variations_share_keys(self):
        left = {"business_name": "Orelee's Barbershop", "business_address": "1795 Westchester Drive, High Point, NC", "country": "US"}
        right = {"business_name": "Orelees Barber Shop", "business_address": "1795 Westchester Dr, High Point", "country": "US"}
        self.assertTrue(blocking_keys(record_view(left)) & blocking_keys(record_view(right)))

    def test_unseen_country_is_supported(self):
        french = {"business_name": "Maison Bleue", "business_address": "25 Rue Victor Hugo", "country": "France"}
        self.assertTrue(blocking_keys(record_view(french)))


if __name__ == "__main__":
    unittest.main()
