import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from fantasy_picker.current_week import (
    build_current_week_lineup,
    build_current_week_projections,
)
from fantasy_picker.lineup import optimize_lineup_fast
from fantasy_picker.models import ModelArtifacts


POSITIONS = ("QB", "RB", "WR", "TE")
PLAYERS = (
    ("qb", "Quarterback A", "QB", 20.0, 22.0),
    ("rb1", "Running Back A", "RB", 14.0, 16.0),
    ("rb2", "Running Back B", "RB", 10.0, 12.0),
    ("wr1", "Receiver A", "WR", 13.0, 15.0),
    ("wr2", "Receiver B", "WR", 9.0, 11.0),
    ("te", "Tight End A", "TE", 8.0, 10.0),
)


class FirstFeatureModel:
    def predict(self, values):
        if hasattr(values, "iloc"):
            return values.iloc[:, 0].tolist()
        return [row[0] for row in values]


class FixedConfidenceModel:
    def predict_proba(self, values):
        return [[0.25, 0.75] for _ in values]


def artifacts():
    return ModelArtifacts(
        final_models={position: FirstFeatureModel() for position in POSITIONS},
        final_features={
            position: ["my_fantasy_points_last3"] for position in POSITIONS
        },
        final_medians={
            position: {"my_fantasy_points_last3": 0.0}
            for position in POSITIONS
        },
        metadata={"purpose": "deterministic test"},
    )


def player_stats(include_current=False, current_value=1000.0):
    rows = []
    for player_id, name, position, week_one, week_two in PLAYERS:
        for week, points in ((1, week_one), (2, week_two)):
            rows.append(
                {
                    "player_id": player_id,
                    "player_display_name": name,
                    "position": position,
                    "season": 2026,
                    "week": week,
                    "recent_team": "AAA",
                    "my_fantasy_points": points,
                    "carries": points,
                    "targets": points,
                    "receptions": points,
                    "rushing_yards": points,
                    "receiving_yards": points,
                    "target_share": points / 100,
                }
            )
        if include_current:
            rows.append(
                {
                    "player_id": player_id,
                    "player_display_name": name,
                    "position": position,
                    "season": 2026,
                    "week": 3,
                    "recent_team": "AAA",
                    "my_fantasy_points": current_value,
                    "carries": current_value,
                    "targets": current_value,
                    "receptions": current_value,
                    "rushing_yards": current_value,
                    "receiving_yards": current_value,
                    "target_share": current_value,
                }
            )
    return pd.DataFrame(rows)


def weekly_rosters():
    rows = [
        {
            "season": 2026,
            "week": 3,
            "gsis_id": player_id,
            "full_name": name,
            "team": "AAA",
            "position": position,
        }
        for player_id, name, position, _, _ in PLAYERS
    ]
    rows.append(
        {
            "season": 2026,
            "week": 3,
            "gsis_id": "k",
            "full_name": "Kicker A",
            "team": "AAA",
            "position": "K",
        }
    )
    return pd.DataFrame(rows)


def schedules():
    return pd.DataFrame(
        [
            {
                "game_id": f"2026_{week}_AAA_BBB",
                "season": 2026,
                "week": week,
                "gameday": f"2026-09-{week + 1:02d}",
                "spread_line": 3.0,
                "total_line": 45.0,
                "roof": "outdoors",
                "temp": 70.0,
                "wind": 5.0,
                "home_team": "AAA",
                "away_team": "BBB",
                "home_rest": 7,
                "away_rest": 7,
                "home_qb_name": "Quarterback A",
                "away_qb_name": "Quarterback B",
            }
            for week in (1, 2, 3)
        ]
    )


def raw_inputs(stats=None):
    return {
        "schedules": schedules(),
        "player_stats": player_stats() if stats is None else stats,
        "weekly_rosters": weekly_rosters(),
        "injuries": pd.DataFrame(),
        "snap_counts": pd.DataFrame(),
        "depth_charts": pd.DataFrame(),
        "ff_opportunity": pd.DataFrame(),
    }


LEAGUE = {
    "roster_slots": [
        {"slot": "QB", "eligible": ["QB"]},
        {"slot": "RB", "eligible": ["RB"]},
        {"slot": "WR", "eligible": ["WR"]},
        {"slot": "TE", "eligible": ["TE"]},
    ]
}


class CurrentWeekTests(unittest.TestCase):
    def test_selects_requested_week_and_projects_all_supported_positions(self):
        projected = build_current_week_projections(
            2026, 3, raw_inputs(), artifacts()
        )

        self.assertEqual(len(projected), 6)
        self.assertEqual({player["position"] for player in projected}, set(POSITIONS))
        self.assertTrue(all(player["season"] == 2026 for player in projected))
        self.assertTrue(all(player["week"] == 3 for player in projected))
        self.assertNotIn("Kicker A", {player["name"] for player in projected})

    def test_prior_games_drive_current_week_projection(self):
        projected = build_current_week_projections(
            2026, 3, raw_inputs(), artifacts()
        )
        by_name = {player["name"]: player["projection"] for player in projected}

        self.assertEqual(by_name["Quarterback A"], 21.0)
        self.assertEqual(by_name["Running Back A"], 15.0)
        self.assertEqual(by_name["Receiver A"], 14.0)
        self.assertEqual(by_name["Tight End A"], 9.0)

    def test_current_game_actual_stats_do_not_leak(self):
        low = build_current_week_projections(
            2026,
            3,
            raw_inputs(player_stats(include_current=True, current_value=-500.0)),
            artifacts(),
        )
        high = build_current_week_projections(
            2026,
            3,
            raw_inputs(player_stats(include_current=True, current_value=5000.0)),
            artifacts(),
        )

        low_values = {player["player_id"]: player["projection"] for player in low}
        high_values = {player["player_id"]: player["projection"] for player in high}
        self.assertEqual(low_values, high_values)

    def test_output_feeds_optimizer_and_roster_lineup_helper(self):
        projected = build_current_week_projections(
            2026, 3, raw_inputs(), artifacts()
        )
        direct = optimize_lineup_fast(LEAGUE, projected)
        result = build_current_week_lineup(
            projected,
            LEAGUE,
            roster_player_ids=[player[0] for player in PLAYERS],
        )

        self.assertEqual(direct["total_projection"], 59.0)
        self.assertEqual(result["total_projection"], 59.0)
        self.assertEqual(len(result["lineup"]), 4)
        self.assertTrue(result["decisions"])
        self.assertTrue(
            all(decision["confidence"] is None for decision in result["decisions"])
        )
        self.assertTrue(
            all(
                decision["confidence_label"] == "Unknown"
                for decision in result["decisions"]
            )
        )

    def test_optional_confidence_is_attached_when_supplied(self):
        projected = build_current_week_projections(
            2026, 3, raw_inputs(), artifacts()
        )
        result = build_current_week_lineup(
            projected, LEAGUE, confidence_model=FixedConfidenceModel()
        )

        measurable = [
            decision
            for decision in result["decisions"]
            if decision["lineup_value_gap"] is not None
        ]
        self.assertTrue(measurable)
        self.assertTrue(
            all(decision["confidence"] == 0.75 for decision in measurable)
        )
        self.assertTrue(
            all(decision["confidence_label"] == "High" for decision in measurable)
        )

    def test_source_dataframes_are_not_mutated(self):
        inputs = raw_inputs()
        originals = {name: frame.copy(deep=True) for name, frame in inputs.items()}

        build_current_week_projections(2026, 3, inputs, artifacts())

        for name, original in originals.items():
            with self.subTest(name=name):
                assert_frame_equal(inputs[name], original)


if __name__ == "__main__":
    unittest.main()
