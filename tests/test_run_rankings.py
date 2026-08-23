import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fantasy_picker.league import MY_YAHOO_LEAGUE
from scripts.run_rankings import main


def player(name, position, projection):
    return {
        "name": name,
        "position": position,
        "team": "AAA",
        "opponent": "BBB",
        "projection": projection,
    }


PROJECTIONS = [
    player("QB One", "QB", 25.0),
    player("QB Two", "QB", 20.0),
    player("RB One", "RB", 18.0),
    player("RB Two", "RB", 15.0),
    player("WR One", "WR", 17.0),
    player("WR Two", "WR", 13.0),
    player("TE One", "TE", 12.0),
    player("TE Two", "TE", 8.0),
]


class RunRankingsTests(unittest.TestCase):
    @patch("scripts.run_rankings.generate_current_week_projections")
    def test_cli_uses_arguments_and_prints_draft_board(self, generate):
        generate.return_value = (PROJECTIONS, ["mock data warning"])
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(
                ["--season", "2026", "--week", "1", "--top", "5"]
            )

        self.assertEqual(status, 0)
        generate.assert_called_once_with(
            2026, 1, Path("artifacts/fantasy_models.pkl")
        )
        rendered = output.getvalue()
        self.assertIn("League teams: 12", rendered)
        self.assertIn("Starting roster:", rendered)
        self.assertIn("Replacement levels:", rendered)
        self.assertIn("Top 5 league-aware rankings:", rendered)
        self.assertIn("SUCCESS: league-aware rankings completed.", rendered)
        self.assertIn("mock data warning", errors.getvalue())

    @patch("scripts.run_rankings.generate_current_week_projections")
    def test_cli_loads_custom_league(self, generate):
        generate.return_value = (PROJECTIONS, [])
        custom = dict(MY_YAHOO_LEAGUE)
        custom["name"] = "Eight Team League"
        custom["team_count"] = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "league.json"
            path.write_text(json.dumps(custom), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "--season", "2026",
                        "--week", "1",
                        "--top", "2",
                        "--league", str(path),
                    ]
                )

        self.assertEqual(status, 0)
        self.assertIn("League: Eight Team League", output.getvalue())
        self.assertIn("League teams: 8", output.getvalue())

    @patch("scripts.run_rankings.generate_current_week_projections")
    def test_cli_fails_without_valid_projections(self, generate):
        generate.return_value = ([player("Missing", "RB", None)], [])
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = main(["--season", "2026", "--week", "1"])

        self.assertEqual(status, 1)
        self.assertIn("No valid QB/RB/WR/TE projections", errors.getvalue())
        self.assertNotIn("SUCCESS", output.getvalue())


if __name__ == "__main__":
    unittest.main()
