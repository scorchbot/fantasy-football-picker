import math
import unittest

from fantasy_picker.lineup import optimize_lineup_fast
from fantasy_picker.projections import (
    build_projected_player,
    project_fantasy_points,
    project_players,
)


class RecordingModel:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def predict(self, model_input):
        if hasattr(model_input, "iloc"):
            values = model_input.iloc[0].tolist()
        else:
            values = list(model_input[0])
        self.calls.append(values)
        return [self.result if self.result is not None else values[0]]


class DataFrameLike:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return iter(enumerate(self.rows))


POSITIONS = ("QB", "RB", "WR", "TE")
FEATURES = {position: ["feature_a", "feature_b"] for position in POSITIONS}
MEDIANS = {position: {"feature_a": 1.0, "feature_b": 2.0} for position in POSITIONS}


def player_row(name, position, projection_value):
    return {
        "player_display_name": name,
        "position": position,
        "recent_team": "AAA",
        "opponent_team": "BBB",
        "my_fantasy_points": 9.5,
        "feature_a": projection_value,
        "feature_b": 0.0,
    }


class ProjectionTests(unittest.TestCase):
    def test_selects_the_correct_position_model(self):
        models = {
            position: RecordingModel(result=index + 10)
            for index, position in enumerate(POSITIONS)
        }

        for index, position in enumerate(POSITIONS):
            with self.subTest(position=position):
                projection = project_fantasy_points(
                    {"feature_a": 1.0, "feature_b": 2.0},
                    position,
                    FEATURES,
                    MEDIANS,
                    models,
                )
                self.assertEqual(projection, float(index + 10))
                self.assertEqual(len(models[position].calls), 1)

    def test_missing_features_are_filled_from_position_medians(self):
        model = RecordingModel(result=12.0)
        models = {position: RecordingModel() for position in POSITIONS}
        models["RB"] = model

        project_fantasy_points(
            {"feature_a": None, "feature_b": math.nan},
            "RB",
            FEATURES,
            MEDIANS,
            models,
        )

        self.assertEqual(model.calls[0], [1.0, 2.0])

    def test_projected_player_has_lineup_fields(self):
        models = {position: RecordingModel(result=14.25) for position in POSITIONS}
        row = player_row("Example Runner", "RB", 3.0)

        projected = build_projected_player(
            row, "RB", FEATURES, MEDIANS, models
        )

        self.assertEqual(
            projected,
            {
                "name": "Example Runner",
                "position": "RB",
                "projection": 14.25,
                "team": "AAA",
                "opponent": "BBB",
                "actual_points": 9.5,
            },
        )

    def test_unsupported_positions_are_clear_and_skipped_in_collections(self):
        models = {position: RecordingModel() for position in POSITIONS}

        with self.assertRaisesRegex(ValueError, "Unsupported position: K"):
            project_fantasy_points({}, "K", FEATURES, MEDIANS, models)

        projected = project_players(
            [player_row("Kicker", "K", 8.0)], FEATURES, MEDIANS, models
        )
        self.assertEqual(projected, [])

    def test_multiple_projected_players_feed_lineup_optimizer(self):
        models = {position: RecordingModel() for position in POSITIONS}
        rows = DataFrameLike(
            [
                player_row("QB A", "QB", 20.0),
                player_row("RB A", "RB", 15.0),
                player_row("WR A", "WR", 14.0),
                player_row("TE A", "TE", 10.0),
            ]
        )
        features = {position: ["feature_a"] for position in POSITIONS}
        medians = {position: {"feature_a": 0.0} for position in POSITIONS}
        league = {
            "roster_slots": [
                {"slot": "QB", "eligible": ["QB"]},
                {"slot": "RB", "eligible": ["RB"]},
                {"slot": "WR", "eligible": ["WR"]},
                {"slot": "TE", "eligible": ["TE"]},
            ]
        }

        projected = project_players(rows, features, medians, models)
        result = optimize_lineup_fast(league, projected)

        self.assertEqual(len(projected), 4)
        self.assertEqual(result["total_projection"], 59.0)

    def test_historical_style_roster_produces_expected_optimal_lineup(self):
        models = {position: RecordingModel() for position in POSITIONS}
        features = {position: ["feature_a"] for position in POSITIONS}
        medians = {position: {"feature_a": 0.0} for position in POSITIONS}
        rows = [
            player_row("Jalen Hurts", "QB", 21.37303),
            player_row("Saquon Barkley", "RB", 20.44516),
            player_row("Breece Hall", "RB", 13.81300),
            player_row("Bijan Robinson", "RB", 16.61194),
            player_row("A.J. Brown", "WR", 10.23843),
            player_row("Garrett Wilson", "WR", 12.43448),
            player_row("Ja'Marr Chase", "WR", 15.24662),
            player_row("Travis Kelce", "TE", 10.94840),
            player_row("George Kittle", "TE", 10.20601),
        ]
        league = {
            "roster_slots": [
                {"slot": "QB", "eligible": ["QB"]},
                {"slot": "RB1", "eligible": ["RB"]},
                {"slot": "RB2", "eligible": ["RB"]},
                {"slot": "WR1", "eligible": ["WR"]},
                {"slot": "WR2", "eligible": ["WR"]},
                {"slot": "TE", "eligible": ["TE"]},
                {"slot": "FLEX", "eligible": ["RB", "WR", "TE"]},
            ]
        }

        projected = project_players(rows, features, medians, models)
        result = optimize_lineup_fast(league, projected)
        lineup_by_slot = {
            player["slot"]: player["name"] for player in result["lineup"]
        }

        self.assertEqual(
            lineup_by_slot,
            {
                "QB": "Jalen Hurts",
                "RB1": "Saquon Barkley",
                "RB2": "Breece Hall",
                "WR1": "Garrett Wilson",
                "WR2": "Ja'Marr Chase",
                "TE": "Travis Kelce",
                "FLEX": "Bijan Robinson",
            },
        )
        self.assertEqual(round(result["total_projection"], 2), 110.87)


if __name__ == "__main__":
    unittest.main()
