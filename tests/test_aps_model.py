import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from aps_failure.modeling import aps_cost, select_threshold


class ApsModelTests(unittest.TestCase):
    def test_aps_cost_weights_false_negatives_more_than_false_positives(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])

        self.assertEqual(aps_cost(y_true, y_pred), 510)

    def test_select_threshold_minimizes_validation_cost(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.6, 0.7, 0.9])
        thresholds = np.array([0.5, 0.65, 0.95])

        threshold, cost = select_threshold(y_true, scores, thresholds)

        self.assertEqual(threshold, 0.65)
        self.assertEqual(cost, 0)


if __name__ == "__main__":
    unittest.main()
