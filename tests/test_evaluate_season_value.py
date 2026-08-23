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
