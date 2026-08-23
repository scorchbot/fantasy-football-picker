import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from fantasy_picker.features import (
    INJURY_FEATURES,
    PREGAME_FEATURE_NAMES,
    QB_OPPORTUNITY_FEATURES,
    SCHEDULE_CONTEXT_FEATURES,
    add_defense_matchup_features,
    add_player_rolling_features,
    add_snap_rolling_features,
    build_pregame_features,
    build_team_game_context,
    select_current_week_feature_row,
)


def make_player_games():
    rows = []
    for week, points, carries, targets, share in zip(
        [1, 2, 3, 4],
        [10.0, 20.0, 30.0, 999.0],
        [5.0, 10.0, 15.0, 999.0],
        [2.0, 4.0, 6.0, 999.0],
        [0.1, 0.2, 0.3, 0.99],
    ):
        rows.append(
            {
                "player_id": "rb1",
                "player_display_name": "Prior Games RB",
                "position": "RB",
                "season": 2025,
                "week": week,
                "recent_team": "AAA",
                "my_fantasy_points": points,
                "carries": carries,
                "targets": targets,
                "receptions": targets / 2,
                "rushing_yards": carries * 4,
                "receiving_yards": targets * 5,
                "target_share": share,
            }
        )

    for week in [1, 2, 3, 4]:
        rows.append(
            {
                "player_id": "qb1",
                "player_display_name": "Prior Games QB",
                "position": "QB",
                "season": 2025,
                "week": week,
                "recent_team": "AAA",
                "my_fantasy_points": float(week * 5),
                "carries": float(week),
                "targets": 0.0,
                "receptions": 0.0,
                "rushing_yards": float(week * 3),
                "receiving_yards": 0.0,
                "target_share": 0.0,
                "completions": float(week * 10),
                "attempts": float(week * 12),
                "passing_yards": float(week * 100),
                "passing_tds": float(week),
                "interceptions": 0.0,
                "passing_air_yards": float(week * 120),
                "passing_epa": float(week),
                "rushing_tds": 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_schedules():
    return pd.DataFrame(
        [
            {
                "game_id": f"2025_{week}_AAA_BBB",
                "season": 2025,
                "week": week,
                "gameday": f"2025-09-{week:02d}",
                "home_team": "AAA",
                "away_team": "BBB",
                "spread_line": 3.0,
                "total_line": 47.0,
                "roof": "dome",
                "temp": 70.0,
                "wind": 5.0,
                "home_rest": 7,
                "away_rest": 6,
                "home_qb_name": "Home QB",
                "away_qb_name": "Away QB",
            }
            for week in [1, 2, 3, 4]
        ]
    )


def make_snap_counts():
    rows = []
    for player_id, values in {
        "rb1": [0.4, 0.5, 0.6, 0.99],
        "qb1": [0.9, 0.8, 0.7, 0.1],
    }.items():
        for week, value in enumerate(values, start=1):
            rows.append(
                {
                    "season": 2025,
                    "week": week,
                    "player_id": player_id,
                    "offense_pct": value,
                }
            )
    return pd.DataFrame(rows)


def make_injuries():
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 4,
                "team": "AAA",
                "gsis_id": "rb1",
                "report_status": "Questionable",
                "practice_status": "Limited Participation in Practice",
            }
        ]
    )


def make_opportunity():
    rows = []
    for week, pass_yards, pass_tds in zip(
        [1, 2, 3, 4], [30.0, 60.0, 90.0, 999.0], [1.0, 2.0, 3.0, 99.0]
    ):
        row = {
            "season": 2025,
            "week": week,
            "posteam": "AAA",
            "player_id": "qb1",
            "pass_completions_exp": 0.0,
            "pass_yards_gained_exp": pass_yards,
            "pass_touchdown_exp": pass_tds,
            "pass_interception_exp": 0.0,
            "rush_yards_gained_exp": 0.0,
            "rush_touchdown_exp": 0.0,
            "receptions_exp": 0.0,
            "rec_yards_gained_exp": 0.0,
            "rec_touchdown_exp": 0.0,
            "pass_two_point_conv_exp": 0.0,
            "rush_two_point_conv_exp": 0.0,
            "rec_two_point_conv_exp": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = build_pregame_features(
            make_player_games(),
            schedules=make_schedules(),
            snap_counts=make_snap_counts(),
            injuries=make_injuries(),
            ff_opportunity=make_opportunity(),
        )

    def current_row(self, player_id="rb1"):
        return select_current_week_feature_row(
            self.features, 2025, 4, player_id=player_id
        )

    def test_core_last3_features_exclude_current_game(self):
        row = self.current_row()

        self.assertEqual(row["my_fantasy_points_last3"], 20.0)
        self.assertEqual(row["carries_last3"], 10.0)
        self.assertAlmostEqual(row["target_share_last3"], 0.2)

    def test_snap_features_exclude_current_game(self):
        row = self.current_row()

        self.assertAlmostEqual(row["offense_pct_last3"], 0.5)
        self.assertAlmostEqual(row["offense_pct_last5"], 0.5)
        self.assertAlmostEqual(row["snap_trend"], 0.0)

    def test_defense_feature_excludes_current_game(self):
        row = self.current_row()

        self.assertEqual(row["defense_fp_allowed_last3"], 20.0)

    def test_injury_flags_are_attached(self):
        row = self.current_row()

        self.assertEqual(row["on_injury_report"], 1)
        self.assertEqual(row["questionable"], 1)
        self.assertEqual(row["practice_limited"], 1)
        for column in ("doubtful", "out", "practice_full", "practice_dnp"):
            self.assertEqual(row[column], 0)

    def test_schedule_context_and_spread_convention(self):
        row = self.current_row()

        self.assertEqual(row["opponent_team"], "BBB")
        self.assertTrue(row["is_home"])
        self.assertEqual(row["team_spread"], -3.0)
        self.assertEqual(row["total_line"], 47.0)
        self.assertEqual(row["roof"], "dome")
        self.assertEqual(row["rest_days"], 7)
        self.assertEqual(row["starting_qb"], "Home QB")

        context = build_team_game_context(make_schedules())
        away = context[(context["week"] == 4) & (context["team"] == "BBB")].iloc[0]
        self.assertFalse(away["is_home"])
        self.assertEqual(away["team_spread"], 3.0)

    def test_qb_opportunity_features_exclude_current_game(self):
        row = self.current_row("qb1")

        self.assertEqual(row["my_expected_opportunity_points_last3"], 12.0)
        self.assertEqual(row["pass_yards_gained_exp_last3"], 60.0)
        self.assertEqual(row["pass_touchdown_exp_last3"], 2.0)

    def test_current_week_row_can_use_blank_current_results(self):
        games = make_player_games()
        current = (games["player_id"] == "rb1") & (games["week"] == 4)
        result_columns = [
            "my_fantasy_points",
            "carries",
            "targets",
            "receptions",
            "rushing_yards",
            "receiving_yards",
            "target_share",
        ]
        games.loc[current, result_columns] = float("nan")

        features = build_pregame_features(
            games,
            schedules=make_schedules(),
            snap_counts=make_snap_counts(),
            injuries=make_injuries(),
            ff_opportunity=make_opportunity(),
        )
        row = select_current_week_feature_row(features, 2025, 4, player_id="rb1")

        self.assertEqual(row["my_fantasy_points_last3"], 20.0)
        self.assertEqual(row["carries_last3"], 10.0)
        self.assertEqual(row["opponent_team"], "BBB")
        self.assertEqual(row["team_spread"], -3.0)

    def test_changing_current_results_does_not_change_pregame_features(self):
        changed_games = make_player_games()
        current = changed_games["week"] == 4
        changed_games.loc[current, "my_fantasy_points"] = -5000.0
        changed_games.loc[current, "carries"] = -4000.0
        changed_games.loc[current, "targets"] = -3000.0
        changed_games.loc[current, "receptions"] = -2500.0
        changed_games.loc[current, "rushing_yards"] = -2400.0
        changed_games.loc[current, "receiving_yards"] = -2300.0
        changed_games.loc[current, "target_share"] = -2000.0

        changed_snaps = make_snap_counts()
        changed_snaps.loc[changed_snaps["week"] == 4, "offense_pct"] = -1000.0

        changed_opportunity = make_opportunity()
        changed_opportunity.loc[
            changed_opportunity["week"] == 4, "pass_yards_gained_exp"
        ] = -9000.0

        changed = build_pregame_features(
            changed_games,
            schedules=make_schedules(),
            snap_counts=changed_snaps,
            injuries=make_injuries(),
            ff_opportunity=changed_opportunity,
        )

        columns = [
            "my_fantasy_points_last3",
            "carries_last3",
            "targets_last3",
            "receptions_last3",
            "rushing_yards_last3",
            "receiving_yards_last3",
            "target_share_last3",
            "offense_pct_last3",
            "offense_pct_last5",
            "snap_trend",
            "defense_fp_allowed_last3",
            *QB_OPPORTUNITY_FEATURES,
        ]
        original_rows = self.features[self.features["week"] == 4].set_index("player_id")
        changed_rows = changed[changed["week"] == 4].set_index("player_id")

        for column in columns:
            assert_series_equal(
                original_rows[column], changed_rows[column], check_names=False
            )

    def test_output_feature_names_match_notebook_names(self):
        expected = {
            "my_fantasy_points_last3",
            "carries_last3",
            "targets_last3",
            "receptions_last3",
            "rushing_yards_last3",
            "receiving_yards_last3",
            "target_share_last3",
            "offense_pct_last3",
            "offense_pct_last5",
            "snap_trend",
            "defense_fp_allowed_last3",
            *INJURY_FEATURES,
            *QB_OPPORTUNITY_FEATURES,
            *SCHEDULE_CONTEXT_FEATURES,
        }

        self.assertTrue(expected.issubset(self.features.columns))
        self.assertTrue(expected.issubset(PREGAME_FEATURE_NAMES))
        self.assertTrue(set(PREGAME_FEATURE_NAMES).issubset(self.features.columns))


if __name__ == "__main__":
    unittest.main()
