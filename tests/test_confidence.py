import unittest

from fantasy_picker.confidence import (
    get_lineup_confidence,
    lineup_confidence_label,
)


class IncreasingConfidenceModel:
    def predict_proba(self, values):
        probabilities = []
        for (gap,) in values:
            positive = min(0.95, 0.50 + gap * 0.04)
            probabilities.append([1.0 - positive, positive])
        return probabilities


class ConfidenceTests(unittest.TestCase):
    def test_confidence_increases_with_lineup_gap(self):
        model = IncreasingConfidenceModel()

        low = get_lineup_confidence(0.5, model)
        medium = get_lineup_confidence(3.0, model)
        high = get_lineup_confidence(7.0, model)

        self.assertLess(low, medium)
        self.assertLess(medium, high)

    def test_confidence_labels_use_final_notebook_thresholds(self):
        cases = [
            (None, "Unknown"),
            (0.59, "Low"),
            (0.60, "Moderate"),
            (0.69, "Moderate"),
            (0.70, "High"),
            (0.79, "High"),
            (0.80, "Very High"),
        ]

        for probability, expected in cases:
            with self.subTest(probability=probability):
                self.assertEqual(lineup_confidence_label(probability), expected)


if __name__ == "__main__":
    unittest.main()
