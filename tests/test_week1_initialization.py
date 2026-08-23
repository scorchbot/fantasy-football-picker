import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import evaluate_week1_initialization as experiment


def rolling_rows(current_points=99.0, current_carries=99.0):
    rows = []
    for season, week, points, carries, defense in [
        (2021, 16, 10.0, 5.0, "BBB"),
        (2021, 17, 20.0, 10.0, "BBB"),
        (2021, 18, 30.0, 15.0, "BBB"),
        (2022, 1, current_points, current_carries, "BBB"),
    ]:
        row = {
            "player_id": "rb1",
            "player_display_name": "Returning RB",
            "position": "RB",
            "season": season,
            "week": week,
            "opponent_team": defense,
            "my_fantasy_points": points,
            "carries": carries,
            "targets": carries / 5,
            "receptions": carries / 10,
            "rushing_yards": carries * 4,
            "receiving_yards": carries * 2,
            "target_share": carries / 100,
            "offense_pct": carries / 20,
        }
        for column in experiment.QB_ROLLING_COLUMNS:
            row[column] = 0.0
        for column in experiment.OPPORTUNITY_COLUMNS:
            row[column] = carries
        rows.append(row)
    reset = pd.DataFrame(rows)
    for column in (
        *experiment.PLAYER_ROLLING_COLUMNS,
        *experiment.QB_ROLLING_COLUMNS,
        *experiment.OPPORTUNITY_COLUMNS,
    ):
        reset[f"{column}_last3"] = np.nan
    reset["offense_pct_last3"] = np.nan
    reset["offense_pct_last5"] = np.nan
    reset["snap_trend"] = np.nan
    reset["defense_fp_allowed_last3"] = np.nan
    return reset


def complete_feature_frame(approach, offset=0.0):
    rows = []
    all_features = sorted({name for values in experiment.NOTEBOOK_FEATURES.values() for name in values})
    for season in (2021, 2022):
        for position_index, position in enumerate(experiment.POSITIONS):
            weeks = (2, 3) if season == 2021 else (1,)
            for player_index in range(2):
                for week in weeks:
                    base = float(position_index + player_index + week + offset)
                    row = {
                        "approach": approach,
                        "season": season,
                        "week": week,
                        "position": position,
                        "player_id": f"{position}-{player_index}",
                        "player_display_name": f"{position} {player_index}",
                        "my_fantasy_points": base + player_index,
                    }
                    row.update({feature: base for feature in all_features})
                    rows.append(row)
    return pd.DataFrame(rows)


class FirstColumnModel:
    def fit(self, features, target):
        self.was_fit = True
        return self

    def predict(self, features):
        return features.iloc[:, 0].to_numpy(dtype=float)


class WeekOneInitializationTests(unittest.TestCase):
    def test_carryover_uses_prior_season_regular_games(self):
        carryover = experiment.add_prior_season_carryover_features(rolling_rows())
        week_one = carryover.loc[(carryover["season"] == 2022) & (carryover["week"] == 1)].iloc[0]

        self.assertEqual(week_one["my_fantasy_points_last3"], 20.0)
        self.assertEqual(week_one["carries_last3"], 10.0)
        self.assertAlmostEqual(week_one["offense_pct_last3"], 0.5)
        self.assertEqual(week_one["my_expected_opportunity_points_last3"], 10.0)

    def test_week_one_actuals_cannot_change_carryover_predictors(self):
        original = experiment.add_prior_season_carryover_features(rolling_rows())
        changed = experiment.add_prior_season_carryover_features(
            rolling_rows(current_points=9999.0, current_carries=8888.0)
        )
        columns = [
            "my_fantasy_points_last3",
            "carries_last3",
            "offense_pct_last3",
            "defense_fp_allowed_last3",
            "my_expected_opportunity_points_last3",
        ]
        original_row = original.loc[(original["season"] == 2022) & (original["week"] == 1)].iloc[0]
        changed_row = changed.loc[(changed["season"] == 2022) & (changed["week"] == 1)].iloc[0]
        pd.testing.assert_series_equal(original_row[columns], changed_row[columns])

    def test_carryover_does_not_use_history_older_than_prior_season(self):
        rows = rolling_rows().loc[lambda frame: frame["season"] == 2021].copy()
        week_one = rows.iloc[[-1]].copy()
        week_one["season"] = 2023
        week_one["week"] = 1
        combined = pd.concat([rows, week_one], ignore_index=True)

        carryover = experiment.add_prior_season_carryover_features(combined)
        row = carryover.loc[(carryover["season"] == 2023) & (carryover["week"] == 1)].iloc[0]

        self.assertTrue(pd.isna(row["my_fantasy_points_last3"]))
        self.assertTrue(pd.isna(row["offense_pct_last3"]))
        self.assertTrue(pd.isna(row["defense_fp_allowed_last3"]))

    def test_prepare_player_stats_filters_postseason_and_scores_yahoo_points(self):
        base = {
            "player_id": "qb1",
            "player_display_name": "QB",
            "position": "QB",
            "season": 2022,
            "week": 1,
            "team": "AAA",
            "completions": 20,
            "passing_yards": 300,
            "passing_tds": 2,
            "passing_interceptions": 1,
        }
        stats = pd.DataFrame([{**base, "season_type": "REG"}, {**base, "season_type": "POST"}])

        prepared = experiment.prepare_player_stats(stats)

        self.assertEqual(len(prepared), 1)
        self.assertIn("recent_team", prepared)
        self.assertIn("interceptions", prepared)
        self.assertEqual(prepared.iloc[0]["my_fantasy_points"], 18.0)

    def test_each_approach_trains_compatible_temporary_models(self):
        feature_sets = {
            "season_reset": complete_feature_frame("season_reset"),
            "prior_season_carryover": complete_feature_frame(
                "prior_season_carryover", offset=0.5
            ),
        }

        metrics, predictions = experiment.evaluate_week1_initialization(
            feature_sets, [2022], model_factory=FirstColumnModel
        )

        self.assertEqual(len(metrics), 8)
        self.assertEqual(set(predictions["approach"]), set(experiment.APPROACHES))
        self.assertEqual(set(predictions["position"]), set(experiment.POSITIONS))
        self.assertEqual(len(predictions), 16)

    def test_metrics_include_mae_rank_and_pairwise_accuracy(self):
        rows = pd.DataFrame(
            {
                "season": [2022, 2022, 2022],
                "actual": [1.0, 2.0, 3.0],
                "prediction": [1.0, 3.0, 2.0],
            }
        )

        metrics = experiment.calculate_metrics(rows)

        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rank_correlation"], 0.5)
        self.assertAlmostEqual(metrics["pairwise_accuracy"], 2 / 3)
        self.assertEqual(metrics["sample_count"], 3)

    def test_coverage_distinguishes_returners_and_new_players(self):
        history = pd.DataFrame(
            [
                {"season": 2020, "week": 4, "position": "RB", "player_id": "older", "my_fantasy_points_last3": 5.0},
                {"season": 2021, "week": 4, "position": "RB", "player_id": "returner", "my_fantasy_points_last3": 5.0},
                {"season": 2022, "week": 1, "position": "RB", "player_id": "returner", "my_fantasy_points_last3": 6.0},
                {"season": 2022, "week": 1, "position": "RB", "player_id": "older", "my_fantasy_points_last3": np.nan},
                {"season": 2022, "week": 1, "position": "RB", "player_id": "rookie", "my_fantasy_points_last3": np.nan},
            ]
        )

        coverage = experiment.calculate_prior_history_coverage(history, [2022])
        rb = coverage.loc[coverage["position"] == "RB"].iloc[0]

        self.assertEqual(rb["week1_players"], 3)
        self.assertEqual(rb["prior_season_history"], 1)
        self.assertEqual(rb["older_history_only"], 1)
        self.assertEqual(rb["no_prior_nfl_history"], 1)
        self.assertEqual(rb["usable_carryover"], 1)
        self.assertEqual(rb["fallback_median"], 2)

    def test_filter_keeps_depth_relevant_rookies_and_used_veterans(self):
        rosters = pd.DataFrame(
            [
                {"season": 2026, "team": "AAA", "position": "RB", "gsis_id": "rookie"},
                {"season": 2026, "team": "AAA", "position": "RB", "gsis_id": "veteran"},
                {"season": 2026, "team": "AAA", "position": "RB", "gsis_id": "camp"},
            ]
        )
        depth = pd.DataFrame(
            [
                {"dt": "new", "team": "AAA", "gsis_id": "rookie", "pos_abb": "RB", "pos_rank": 2},
                {"dt": "new", "team": "AAA", "gsis_id": "veteran", "pos_abb": "RB", "pos_rank": 6},
                {"dt": "new", "team": "AAA", "gsis_id": "camp", "pos_abb": "RB", "pos_rank": 7},
            ]
        )
        prior = pd.DataFrame(
            [
                {"season": 2025, "player_id": "veteran", "attempts": 0, "carries": 20, "targets": 10},
                {"season": 2025, "player_id": "camp", "attempts": 0, "carries": 1, "targets": 0},
            ]
        )

        analysis = experiment.analyze_fantasy_relevance(
            rosters, depth, prior, season=2026
        )
        rb = analysis.loc[analysis["position"] == "RB"].iloc[0]

        self.assertEqual(rb["roster_players"], 3)
        self.assertEqual(rb["depth_rule"], 1)
        self.assertEqual(rb["prior_usage_rule"], 1)
        self.assertEqual(rb["depth_or_usage"], 2)

    @patch.object(experiment, "run_experiment")
    def test_cli_prints_report_without_network(self, run_experiment):
        run_experiment.return_value = (
            pd.DataFrame([{"approach": "season_reset", "position": "QB", "mae": 1.0}]),
            pd.DataFrame([{"season": 2022, "position": "QB", "week1_players": 1}]),
            pd.DataFrame([{"position": "QB", "depth_or_usage": 2}]),
        )

        with patch("builtins.print") as output:
            status = experiment.main(["--evaluation-seasons", "2022", "2023"])

        self.assertEqual(status, 0)
        run_experiment.assert_called_once_with([2022, 2023], current_season=2026)
        self.assertTrue(output.called)


if __name__ == "__main__":
    unittest.main()
