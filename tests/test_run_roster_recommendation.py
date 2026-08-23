import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_roster_recommendation import main


def projection(player_id, name, position, value):
    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": "AAA",
        "opponent": "BBB",
        "projection": value,
    }


PROJECTIONS = [
    projection("qb", "Quarterback", "QB", 20.0),
    projection("rb1", "Running Back One", "RB", 18.0),
    projection("rb2", "Running Back Two", "RB", 16.0),
    projection("rb3", "Running Back Three", "RB", 15.0),
    projection("wr1", "Receiver One", "WR", 17.0),
    projection("wr2", "Receiver Two", "WR", 14.0),
    projection("wr3", "Receiver Three", "WR", 13.0),
    projection("te1", "Tight End One", "TE", 10.0),
    projection("te2", "Tight End Two", "TE", 9.0),
]


def roster(include_unmatched=False):
    players = [
        {
            "player_id": item["player_id"],
            "name": item["name"],
            "position": item["position"],
            "team": item["team"],
        }
        for item in PROJECTIONS
    ]
    if include_unmatched:
        players.append(
            {
                "player_id": "missing",
                "name": "Unmatched Player",
                "position": "RB",
                "team": "ZZZ",
            }
        )
    return {"league_name": "CLI Test League", "players": players}


class RunRosterRecommendationTests(unittest.TestCase):
    def write_roster(self, directory, value):
        path = Path(directory) / "roster.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    @patch("scripts.run_roster_recommendation.generate_current_week_projections")
    def test_mocked_live_projections_produce_optimal_lineup(self, generate):
        generate.return_value = (PROJECTIONS, ["mock fresh-data warning"])
        with tempfile.TemporaryDirectory() as directory:
            roster_path = self.write_roster(directory, roster())
            output = io.StringIO()
            errors = io.StringIO()

            with redirect_stdout(output), redirect_stderr(errors):
                status = main(
                    [
                        "--season", "2026",
                        "--week", "1",
                        "--roster", str(roster_path),
                    ]
                )

        self.assertEqual(status, 0)
        generate.assert_called_once_with(
            2026, 1, Path("artifacts/fantasy_models.pkl")
        )
        rendered = output.getvalue()
        self.assertIn("Matched roster players: 9", rendered)
        self.assertIn("FLEX: Running Back Three", rendered)
        self.assertIn("Projected team total: 110.00", rendered)
        self.assertIn("Lineup advantages:", rendered)
        self.assertIn("mock fresh-data warning", errors.getvalue())

    @patch("scripts.run_roster_recommendation.generate_current_week_projections")
    def test_unmatched_player_is_reported_without_invented_projection(self, generate):
        generate.return_value = (PROJECTIONS, [])
        with tempfile.TemporaryDirectory() as directory:
            roster_path = self.write_roster(directory, roster(include_unmatched=True))
            output = io.StringIO()

            with redirect_stdout(output):
                status = main(
                    ["--season", "2026", "--week", "1", "--roster", str(roster_path)]
                )

        rendered = output.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("Unmatched roster players: 1", rendered)
        self.assertIn("Unmatched Player | RB | ZZZ | missing | no_projection_match", rendered)
        unmatched_line = next(
            line for line in rendered.splitlines() if "Unmatched Player" in line
        )
        self.assertNotIn("0.00", unmatched_line)

    @patch("scripts.run_roster_recommendation.generate_current_week_projections")
    def test_ambiguous_name_match_exits_clearly(self, generate):
        projections = PROJECTIONS + [projection("other", "Quarterback", "QB", 5.0)]
        generate.return_value = (projections, [])
        value = roster()
        value["players"][0].pop("player_id")
        with tempfile.TemporaryDirectory() as directory:
            roster_path = self.write_roster(directory, value)
            errors = io.StringIO()

            with redirect_stderr(errors):
                status = main(
                    ["--season", "2026", "--week", "1", "--roster", str(roster_path)]
                )

        self.assertEqual(status, 1)
        self.assertIn("roster matching is ambiguous", errors.getvalue())
        self.assertNotIn("SUCCESS", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
