import math
import unittest

from fantasy_picker.scoring import YAHOO_OFFENSIVE_SCORING, score_player


class YahooScoringTests(unittest.TestCase):
    def test_representative_quarterback_line(self):
        stats = {
            "completions": 25,
            "passing_yards": 305,
            "passing_tds": 2,
            "interceptions": 1,
            "rushing_yards": 20,
        }

        self.assertEqual(score_player(stats, YAHOO_OFFENSIVE_SCORING), 19.67)

    def test_representative_running_back_line(self):
        stats = {
            "rushing_yards": 150,
            "rushing_tds": 1,
            "receptions": 5,
            "receiving_yards": 40,
            "receiving_tds": 1,
            "rushing_2pt_conversions": 1,
            "rushing_fumbles_lost": 1,
        }

        self.assertEqual(score_player(stats, YAHOO_OFFENSIVE_SCORING), 35.5)

    def test_missing_and_nan_stats_match_notebook_behavior(self):
        stats = {
            "receptions": 10,
            "receiving_yards": 250,
            "receiving_tds": 2,
            "passing_yards": math.nan,
        }

        self.assertEqual(score_player(stats, YAHOO_OFFENSIVE_SCORING), 45.0)


if __name__ == "__main__":
    unittest.main()
