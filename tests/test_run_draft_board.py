import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.run_draft_board import main


PROJECTIONS = [
    {"player_id": "qb", "name": "Quarterback", "position": "QB", "team": "AAA", "projection": 22.0},
    {"player_id": "rb", "name": "Running Back", "position": "RB", "team": "BBB", "projection": 18.0},
    {"player_id": "wr", "name": "Receiver", "position": "WR", "team": "CCC", "projection": 17.0},
    {"player_id": "te", "name": "Tight End", "position": "TE", "team": "DDD", "projection": 12.0},
]
PRIOR_STATS = pd.DataFrame(
    [
        {"player_id": player["player_id"], "player_display_name": player["name"],
         "position": player["position"], "season": 2025, "week": week,
         "season_type": "REG", "my_fantasy_points": player["projection"] - 1}
        for player in PROJECTIONS for week in (1, 2)
    ]
)


class RunDraftBoardTests(unittest.TestCase):
    @patch("scripts.run_draft_board.nflverse.load_player_stats")
    @patch("scripts.run_draft_board.generate_current_week_projections")
    def test_cli_uses_mocked_current_and_prior_data(self, generate, load_stats):
        generate.return_value = (PROJECTIONS, ["mock warning"])
        load_stats.return_value = PRIOR_STATS
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(["--season", "2026", "--week", "1", "--top", "3"])
        self.assertEqual(status, 0)
        generate.assert_called_once_with(2026, 1, Path("artifacts/fantasy_models.pkl"))
        load_stats.assert_called_once_with(2025)
        rendered = output.getvalue()
        self.assertIn("Season Value", rendered)
        self.assertIn("Prior FPPG", rendered)
        self.assertIn("Top 3 season-long draft values", rendered)
        self.assertIn("SUCCESS: season-long draft board completed.", rendered)
        self.assertIn("mock warning", errors.getvalue())

    @patch("scripts.run_draft_board.generate_current_week_projections")
    def test_non_week_one_run_fails_before_projection(self, generate):
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = main(["--season", "2026", "--week", "2"])
        self.assertEqual(status, 1)
        generate.assert_not_called()
        self.assertIn("requires --week 1", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
