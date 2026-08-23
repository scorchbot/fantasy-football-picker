import io
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import evaluate_season_value as evaluation


def synthetic_inputs():
    predictions = []
    stats = []
    for season in (2022, 2023):
        for position in ("QB", "RB", "WR", "TE"):
            for index in range(4):
                player_id = f"{season}-{position}-{index}"
                predictions.append(
                    {
                        "season": season,
                        "position": position,
                        "player_id": player_id,
                        "player_name": player_id,
                        "prediction": 20.0 - index,
                    }
                )
                stats.extend(
                    [
                        {
                            "season": season - 1,
                            "week": 1,
                            "season_type": "REG",
                            "position": position,
                            "player_id": player_id,
                            "player_display_name": player_id,
                            "my_fantasy_points": 18.0 - index,
                        },
                        {
                            "season": season,
                            "week": 1,
                            "season_type": "REG",
                            "position": position,
                            "player_id": player_id,
                            "my_fantasy_points": 30.0 - index,
                        },
                    ]
                )
    return pd.DataFrame(predictions), pd.DataFrame(stats)


class EvaluateSeasonValueTests(unittest.TestCase):
    def parameter_grid(self):
        rows = []
        for season in (2022, 2023, 2024):
            for position, winner in (("QB", 0.0), ("RB", 1.0), ("WR", 0.3), ("TE", 0.7)):
                for weight in (0.0, 0.3, 0.7, 1.0):
                    quality = 1.0 - abs(weight - winner)
                    rows.append(
                        {
                            "season": season,
                            "position": position,
                            "week1_weight": weight,
                            "availability_strategy": "linear_17_games",
                            "rank_correlation": quality,
                            "top_n_overlap": quality,
                            "vor_rank_correlation": quality,
                        }
                    )
        return pd.DataFrame(rows)

    def test_build_rows_keeps_target_actuals_out_of_preseason_signals(self):
        predictions, stats = synthetic_inputs()
        original = evaluation.build_preseason_evaluation_rows(predictions, stats, [2022])
        changed_stats = stats.copy()
        changed_stats.loc[changed_stats["season"].eq(2022), "my_fantasy_points"] *= 100
        changed = evaluation.build_preseason_evaluation_rows(predictions, changed_stats, [2022])
        signal_columns = [
            "week1_projection_score",
            "prior_season_fppg_score",
            "hybrid_season_value_score",
        ]
        pd.testing.assert_frame_equal(original[signal_columns], changed[signal_columns])
        self.assertFalse(original["actual_season_points"].equals(changed["actual_season_points"]))

    def test_historical_evaluation_produces_every_season_and_approach(self):
        predictions, stats = synthetic_inputs()
        rows = evaluation.build_preseason_evaluation_rows(predictions, stats, [2022, 2023])
        results = evaluation.evaluate_preseason_rows(rows)
        aggregate = evaluation.summarize_results(results)
        self.assertEqual(set(results["season"]), {2022, 2023})
        self.assertEqual(set(results["approach"]), set(evaluation.APPROACHES))
        self.assertEqual(set(results["position"]), {"QB", "RB", "WR", "TE", "OVERALL"})
        self.assertEqual(set(aggregate["approach"]), set(evaluation.APPROACHES))

    def test_weight_selection_never_uses_target_season(self):
        selections = evaluation.select_walk_forward_parameters(
            self.parameter_grid(),
            [2022, 2023, 2024],
            parameter="week1_weight",
            default_value=0.4,
        )
        calibration = selections.loc[selections["season"].eq(2022)]
        self.assertTrue(calibration["calibration"].all())
        self.assertTrue((calibration["selected_week1_weight"] == 0.4).all())
        for _, row in selections.loc[~selections["calibration"]].iterrows():
            self.assertLess(row["selected_using_through_season"], row["season"])

    def test_future_seasons_cannot_change_earlier_selected_weights(self):
        grid = self.parameter_grid()
        original = evaluation.select_walk_forward_parameters(
            grid, [2022, 2023, 2024], parameter="week1_weight", default_value=0.4
        )
        future = grid.loc[grid["season"].eq(2024)].copy()
        future["season"] = 2025
        future["rank_correlation"] = future["week1_weight"] * 100
        future["top_n_overlap"] = future["week1_weight"] * 100
        future["vor_rank_correlation"] = future["week1_weight"] * 100
        changed = evaluation.select_walk_forward_parameters(
            pd.concat([grid, future], ignore_index=True),
            [2022, 2023, 2024],
            parameter="week1_weight",
            default_value=0.4,
        )
        pd.testing.assert_frame_equal(original, changed)

    def test_position_specific_weights_can_differ(self):
        selections = evaluation.select_walk_forward_parameters(
            self.parameter_grid(),
            [2022, 2023],
            parameter="week1_weight",
            default_value=0.4,
        )
        selected = selections.loc[selections["season"].eq(2023)].set_index("position")
        self.assertEqual(selected.loc["QB", "selected_week1_weight"], 0.0)
        self.assertEqual(selected.loc["RB", "selected_week1_weight"], 1.0)
        self.assertNotEqual(
            selected.loc["WR", "selected_week1_weight"],
            selected.loc["TE", "selected_week1_weight"],
        )

    def test_diagnostic_exactly_reconstructs_season_value(self):
        player = {
            "player_id": "p",
            "name": "Player",
            "position": "WR",
            "week1_projection": 12.0,
            "prior_season_fppg": 18.0,
            "prior_season_games": 10,
            "season_value_score": 0.4 * 12.0 + 0.6 * ((10 / 17) * 18 + (7 / 17) * 10),
            "replacement_level_score": 9.0,
            "season_value_vor": 4.2,
            "overall_rank": 12,
        }
        diagnostic = evaluation.build_player_diagnostic(player, 10.0)
        self.assertAlmostEqual(
            diagnostic["season_value_score"], player["season_value_score"]
        )
        self.assertAlmostEqual(diagnostic["reconstruction_difference"], 0.0)

    def test_rookie_rows_never_receive_fabricated_prior_history(self):
        predictions, stats = synthetic_inputs()
        rookie_prediction = pd.DataFrame(
            [{"season": 2022, "position": "WR", "player_id": "rookie", "player_name": "Rookie", "prediction": 11.0}]
        )
        rookie_actual = pd.DataFrame(
            [{"season": 2022, "week": 1, "season_type": "REG", "position": "WR", "player_id": "rookie", "my_fantasy_points": 20.0}]
        )
        rows = evaluation.build_preseason_evaluation_rows(
            pd.concat([predictions, rookie_prediction], ignore_index=True),
            pd.concat([stats, rookie_actual], ignore_index=True),
            [2022],
        )
        rookie = rows.loc[rows["player_id"].eq("rookie")].iloc[0]
        self.assertTrue(pd.isna(rookie["prior_season_fppg"]))
        self.assertEqual(rookie["prior_season_games"], 0)
        self.assertTrue(rookie["used_prior_median_fallback"])

    def test_rookie_bias_is_measured_against_full_position_pool(self):
        rows = pd.DataFrame(
            [
                {"season": 2022, "position": "WR", "player_id": "v1", "prior_season_fppg": 10.0,
                 "hybrid_season_value_score": 10.0, "actual_season_points": 100.0},
                {"season": 2022, "position": "WR", "player_id": "v2", "prior_season_fppg": 9.0,
                 "hybrid_season_value_score": 9.0, "actual_season_points": 90.0},
                {"season": 2022, "position": "WR", "player_id": "rookie", "prior_season_fppg": None,
                 "hybrid_season_value_score": 20.0, "actual_season_points": 1.0},
            ]
        )
        result = evaluation.evaluate_rookie_fallback(rows).iloc[0]
        self.assertGreater(result["mean_percentile_bias"], 0)

    def test_recommendation_selects_stronger_synthetic_approach(self):
        aggregate = pd.DataFrame(
            [
                {"approach": approach, "position": "OVERALL", "rank_correlation": score,
                 "top12_hit_rate": score, "top24_hit_rate": score,
                 "top50_overlap": score, "vor_rank_correlation": score}
                for approach, score in [
                    ("week1_projection", 0.3),
                    ("prior_season_fppg", 0.4),
                    ("hybrid_season_value", 0.7),
                ]
            ]
        )
        self.assertEqual(evaluation.recommend_approach(aggregate)["approach"], "hybrid_season_value")

    @patch.object(evaluation, "run_evaluation")
    def test_cli_prints_mocked_results_without_network(self, run):
        frame = pd.DataFrame(
            [{"season": 2022, "approach": "hybrid_season_value", "position": "QB"}]
        )
        run.return_value = {
            "season_results": frame,
            "aggregate_results": frame,
            "coverage": pd.DataFrame([{"season": 2022, "position": "QB"}]),
            "recommendation": {"approach": "hybrid_season_value", "formula": "test"},
        }
        with redirect_stdout(io.StringIO()):
            status = evaluation.main(["--evaluation-seasons", "2022", "2023"])
        self.assertEqual(status, 0)
        run.assert_called_once_with([2022, 2023])


if __name__ == "__main__":
    unittest.main()
