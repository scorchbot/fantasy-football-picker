import copy
import unittest

import pandas as pd

from fantasy_picker.league import MY_YAHOO_LEAGUE, create_league_config
from fantasy_picker.scoring import MY_YAHOO_SCORING
from fantasy_picker.season_value import (
    assign_season_value_tiers,
    build_season_value_rankings,
    calculate_season_value_score,
    summarize_prior_season,
)


def projection(player_id, name, position, value):
    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": "AAA",
        "opponent": "BBB",
        "projection": value,
    }


def stat(player_id, name, position, season, week, points):
    return {
        "player_id": player_id,
        "player_display_name": name,
        "position": position,
        "season": season,
        "week": week,
        "season_type": "REG",
        "my_fantasy_points": points,
    }


def player_pool():
    rows = []
    for position, start in (("QB", 25), ("RB", 20), ("WR", 19), ("TE", 14)):
        for index in range(16):
            rows.append(projection(f"{position}{index}", f"{position} {index}", position, start - index * 0.5))
    return rows


def prior_pool():
    return pd.DataFrame(
        [
            stat(item["player_id"], item["name"], item["position"], 2025, week, item["projection"] - 1)
            for item in player_pool()
            for week in (1, 2, 3)
        ]
    )


class SeasonValueTests(unittest.TestCase):
    def test_season_tiers_use_deterministic_value_bands(self):
        players = [
            {"value_over_replacement": 8.0},
            {"value_over_replacement": 7.4},
            {"value_over_replacement": 6.9},
            {"value_over_replacement": 5.8},
        ]
        self.assertEqual(assign_season_value_tiers(players, 1.0), [1, 1, 2, 3])

    def test_prior_summary_uses_only_immediately_prior_season(self):
        stats = pd.DataFrame(
            [
                stat("p", "Player", "RB", 2024, 1, 99.0),
                stat("p", "Player", "RB", 2025, 1, 10.0),
                stat("p", "Player", "RB", 2025, 2, 20.0),
                stat("p", "Player", "RB", 2026, 1, 999.0),
            ]
        )
        summary = summarize_prior_season(stats, 2026, MY_YAHOO_SCORING)
        self.assertEqual(summary.iloc[0]["prior_season_fppg"], 15.0)
        self.assertEqual(summary.iloc[0]["prior_season_games"], 2)

    def test_target_season_results_cannot_change_preseason_value(self):
        projections = [projection("p", "Player", "RB", 15.0)]
        base = [stat("p", "Player", "RB", 2025, 1, 10.0)]
        original = build_season_value_rankings(
            projections, pd.DataFrame(base + [stat("p", "Player", "RB", 2026, 1, 1.0)]), MY_YAHOO_LEAGUE, 2026
        )
        changed = build_season_value_rankings(
            projections, pd.DataFrame(base + [stat("p", "Player", "RB", 2026, 1, 9999.0)]), MY_YAHOO_LEAGUE, 2026
        )
        self.assertEqual(original[0]["season_value_score"], changed[0]["season_value_score"])

    def test_rookie_has_empty_prior_fields_and_documented_median_fallback(self):
        projections = [
            projection("v", "Veteran", "RB", 10.0),
            projection("r", "Rookie", "RB", 12.0),
        ]
        stats = pd.DataFrame([stat("v", "Veteran", "RB", 2025, 1, 8.0)])
        board = build_season_value_rankings(projections, stats, MY_YAHOO_LEAGUE, 2026)
        rookie = next(row for row in board if row["player_id"] == "r")
        self.assertIsNone(rookie["prior_season_fppg"])
        self.assertEqual(rookie["prior_season_games"], 0)
        self.assertTrue(rookie["used_prior_median_fallback"])
        self.assertEqual(rookie["adjusted_prior_fppg"], 8.0)

    def test_stale_two_year_old_history_is_not_used(self):
        projections = [
            projection("current", "Current", "WR", 12.0),
            projection("stale", "Stale", "WR", 12.0),
        ]
        stats = pd.DataFrame(
            [
                stat("current", "Current", "WR", 2025, 1, 10.0),
                stat("stale", "Stale", "WR", 2024, 1, 50.0),
            ]
        )
        board = build_season_value_rankings(projections, stats, MY_YAHOO_LEAGUE, 2026)
        stale = next(row for row in board if row["player_id"] == "stale")
        self.assertIsNone(stale["prior_season_fppg"])
        self.assertEqual(stale["adjusted_prior_fppg"], 10.0)

    def test_availability_shrinks_small_samples_toward_position_median(self):
        score, adjusted, fallback = calculate_season_value_score(
            20.0, 30.0, 1, 10.0
        )
        self.assertAlmostEqual(adjusted, 10.0 + 20.0 / 17.0)
        self.assertFalse(fallback)
        self.assertAlmostEqual(score, 0.4 * 20.0 + 0.6 * adjusted)

    def test_superflex_and_position_demand_change_league_value(self):
        projections = player_pool()
        stats = prior_pool()
        one_qb = create_league_config(
            "One QB", MY_YAHOO_SCORING,
            [{"slot": "QB", "eligible": ["QB"]}], team_count=4,
        )
        superflex = create_league_config(
            "Superflex", MY_YAHOO_SCORING,
            [
                {"slot": "QB", "eligible": ["QB"]},
                {"slot": "SUPERFLEX", "eligible": ["QB", "RB", "WR", "TE"]},
            ], team_count=4,
        )
        one = build_season_value_rankings(projections, stats, one_qb, 2026)
        two = build_season_value_rankings(projections, stats, superflex, 2026)
        one_qb_vor = next(row["season_value_vor"] for row in one if row["position"] == "QB")
        super_qb_vor = next(row["season_value_vor"] for row in two if row["position"] == "QB")
        self.assertGreater(super_qb_vor, one_qb_vor)

        shallow = create_league_config(
            "Shallow", MY_YAHOO_SCORING,
            [{"slot": "RB", "eligible": ["RB"]}, {"slot": "WR", "eligible": ["WR"]}], team_count=4,
        )
        deep = create_league_config(
            "Deep", MY_YAHOO_SCORING,
            [
                {"slot": "RB1", "eligible": ["RB"]}, {"slot": "RB2", "eligible": ["RB"]},
                {"slot": "WR1", "eligible": ["WR"]}, {"slot": "WR2", "eligible": ["WR"]},
            ], team_count=4,
        )
        shallow_board = build_season_value_rankings(projections, stats, shallow, 2026)
        deep_board = build_season_value_rankings(projections, stats, deep, 2026)
        for position in ("RB", "WR"):
            shallow_level = next(row["replacement_level_score"] for row in shallow_board if row["position"] == position)
            deep_level = next(row["replacement_level_score"] for row in deep_board if row["position"] == position)
            self.assertLess(deep_level, shallow_level)

    def test_rankings_are_deterministic_and_do_not_fabricate_missing_projection(self):
        projections = player_pool() + [projection("bad", "No Projection", "RB", None)]
        first = build_season_value_rankings(projections, prior_pool(), MY_YAHOO_LEAGUE, 2026)
        second = build_season_value_rankings(list(reversed(projections)), prior_pool(), MY_YAHOO_LEAGUE, 2026)
        self.assertEqual(first, second)
        self.assertNotIn("bad", {row["player_id"] for row in first})


if __name__ == "__main__":
    unittest.main()
