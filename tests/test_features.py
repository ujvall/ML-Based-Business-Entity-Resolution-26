"""Tests for Phase 3 feature extraction."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import EntityProfile, extract_pair_features, FEATURE_NAMES


class FeatureTests(unittest.TestCase):
    def test_feature_count(self):
        s1 = EntityProfile("Unified Green Target LLC", "1037 Cedar Grove Road, Halifax County, VA", "US")
        s2 = EntityProfile("Unified Green Target [LLC]", "1037. CEDAR GROVE ROAD, HALIFAX COUNTY, VA", "US")
        vec = extract_pair_features(s1, s2, rank=1, total_cands=10)
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_exact_match(self):
        s1 = EntityProfile("Acme Corp", "123 Main St", "US")
        s2 = EntityProfile("Acme Corp", "123 Main St", "US")
        vec = extract_pair_features(s1, s2, rank=1, total_cands=5)
        f_dict = dict(zip(FEATURE_NAMES, vec))
        self.assertEqual(f_dict["country_match"], 1.0)
        self.assertEqual(f_dict["name_exact"], 1.0)
        self.assertEqual(f_dict["addr_exact"], 1.0)
        self.assertEqual(f_dict["house_num_status"], 1.0)
        self.assertGreater(f_dict["sketch_score"], 5.0)

    def test_house_number_contradiction(self):
        s1 = EntityProfile("Acme Corp", "1037 Cedar Rd", "US")
        s2 = EntityProfile("Acme Corp", "1039 Cedar Rd", "US")
        vec = extract_pair_features(s1, s2, rank=2, total_cands=5)
        f_dict = dict(zip(FEATURE_NAMES, vec))
        self.assertEqual(f_dict["house_num_status"], -1.0)
        self.assertEqual(f_dict["numeric_conflict"], 1.0)

    def test_country_mismatch(self):
        s1 = EntityProfile("Acme Corp", "123 Main St", "US")
        s2 = EntityProfile("Acme Corp", "123 Main St", "India")
        vec = extract_pair_features(s1, s2, rank=1, total_cands=1)
        f_dict = dict(zip(FEATURE_NAMES, vec))
        self.assertEqual(f_dict["country_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
