import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fantasy_picker.models import ModelArtifacts
from scripts.run_current_week import (
    CurrentWeekRunError,
    generate_current_week_projections,
    main,
)


POSITIONS = ("QB", "RB", "WR", "TE")


class ConstantModel:
    def predict(self, values):
        return [10.0] * len(values)


def artifacts():
    return ModelArtifacts(
        final_models={position: ConstantModel() for position in POSITIONS},
        final_features={
            position: ["my_fantasy_points_last3"] for position in POSITIONS
        },
        final_medians={
            position: {"my_fantasy_points_last3": 8.0}
            for position in POSITIONS
        },
    )


def fresh_inputs(season=2026, week=1):
    generic = pd.DataFrame({"season": [season], "week": [week]})
    return {
        "schedules": generic.copy(),
        "player_stats": pd.DataFrame(
            {
                "season": [season],
                "week": [max(1, week - 1)],
                "player_id": ["player-1"],
                "my_fantasy_points": [10.0],
            }
        ),
        "weekly_rosters": generic.copy(),
        "injuries": generic.copy(),
        "snap_counts": generic.copy(),
        "depth_charts": generic.copy(),
        "ff_opportunity": generic.copy(),
    }


PROJECTIONS = [
    {
        "name": "Quarterback A",
        "position": "QB",
        "team": "AAA",
        "opponent": "BBB",
        "projection": 22.5,
    },
    {
        "name": "Running Back A",
        "position": "RB",
        "team": "CCC",
        "opponent": "DDD",
        "projection": 18.25,
    },
    {
        "name": "Receiver A",
        "position": "WR",
        "team": "EEE",
        "opponent": "FFF",
        "projection": 16.0,
    },
    {
        "name": "Tight End A",
        "position": "TE",
        "team": "GGG",
        "opponent": "HHH",
        "projection": 12.0,
    },
]


class RunCurrentWeekTests(unittest.TestCase):
    @patch("scripts.run_current_week.current_week.build_current_week_projections")
    @patch("scripts.run_current_week.nflverse.load_current_week_inputs")
    @patch("scripts.run_current_week.models.load_model_artifacts")
    def test_cli_respects_season_week_and_invokes_pipeline(
        self, load_artifacts, load_inputs, build_projections
    ):
        loaded_artifacts = artifacts()
        inputs = fresh_inputs(2026, 7)
        load_artifacts.return_value = loaded_artifacts
        load_inputs.return_value = inputs
        build_projections.return_value = PROJECTIONS

        with redirect_stdout(io.StringIO()):
            exit_code = main(["--season", "2026", "--week", "7"])

        self.assertEqual(exit_code, 0)
        load_artifacts.assert_called_once_with(Path("artifacts/fantasy_models.pkl"))
        load_inputs.assert_called_once_with(2026)
        build_projections.assert_called_once_with(
            2026, 7, inputs, loaded_artifacts
        )

    @patch("scripts.run_current_week.current_week.build_current_week_projections")
    @patch("scripts.run_current_week.nflverse.load_current_week_inputs")
    @patch("scripts.run_current_week.models.load_model_artifacts")
    def test_success_output_includes_counts_and_top_players(
        self, load_artifacts, load_inputs, build_projections
    ):
        load_artifacts.return_value = artifacts()
        load_inputs.return_value = fresh_inputs()
        build_projections.return_value = PROJECTIONS
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--season", "2026", "--week", "1"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Players projected: 4", rendered)
        self.assertIn("QB=1, RB=1, WR=1, TE=1", rendered)
        self.assertIn("Quarterback A | QB | AAA | BBB | 22.50", rendered)
        self.assertIn("Tight End A | TE | GGG | HHH | 12.00", rendered)

    @patch("scripts.run_current_week.current_week.build_current_week_projections")
    @patch("scripts.run_current_week.nflverse.load_current_week_inputs")
    @patch("scripts.run_current_week.models.load_model_artifacts")
    def test_missing_current_week_context_fails_without_fake_projections(
        self, load_artifacts, load_inputs, build_projections
    ):
        load_artifacts.return_value = artifacts()
        inputs = fresh_inputs()
        inputs["schedules"] = pd.DataFrame(
            {"season": [2026], "week": [0]}
        )
        inputs["weekly_rosters"] = pd.DataFrame()
        load_inputs.return_value = inputs

        with self.assertRaisesRegex(
            CurrentWeekRunError,
            "missing schedule rows.*missing current-week roster rows",
        ):
            generate_current_week_projections(2026, 1)

        build_projections.assert_not_called()

    @patch("scripts.run_current_week.nflverse.load_current_week_inputs")
    @patch("scripts.run_current_week.models.load_model_artifacts")
    def test_empty_required_feature_data_has_clear_cli_failure(
        self, load_artifacts, load_inputs
    ):
        required_artifacts = artifacts()
        required_artifacts = ModelArtifacts(
            final_models=required_artifacts.final_models,
            final_features={
                position: ["offense_pct_last3"] for position in POSITIONS
            },
            final_medians={
                position: {"offense_pct_last3": 0.6} for position in POSITIONS
            },
        )
        load_artifacts.return_value = required_artifacts
        inputs = fresh_inputs()
        inputs["snap_counts"] = pd.DataFrame()
        load_inputs.return_value = inputs
        error = io.StringIO()

        with redirect_stderr(error):
            exit_code = main(["--season", "2026", "--week", "1"])

        self.assertEqual(exit_code, 1)
        self.assertIn("snap_counts is empty", error.getvalue())
        self.assertIn("offense_pct_last3", error.getvalue())
        self.assertNotIn("Players projected:", error.getvalue())


if __name__ == "__main__":
    unittest.main()
