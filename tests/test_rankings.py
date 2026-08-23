import copy
import unittest

from fantasy_picker.league import MY_YAHOO_LEAGUE, create_league_config
from fantasy_picker.rankings import (
    assign_tiers,
    build_league_rankings,
    calculate_position_demand,
    calculate_replacement_levels,
)
from fantasy_picker.scoring import MY_YAHOO_SCORING


def player(name, position, projection):
    return {
        "name": name,
        "position": position,
        "team": "AAA",
        "opponent": "BBB",
        "projection": projection,
    }


def position_pool(position, values):
    return [player(f"{position} Player {index}", position, value) for index, value in enumerate(values, 1)]


PROJECTIONS = (
    position_pool("QB", range(30, 10, -1))
    + position_pool("RB", range(25, 0, -1))
    + position_pool("WR", range(24, -1, -1))
    + position_pool("TE", range(18, -2, -1))
)


def league(slots, team_count=4):
    return create_league_config(
        "Ranking Test League",
        MY_YAHOO_SCORING,
        slots,
        team_count=team_count,
    )


class RankingTests(unittest.TestCase):
    def test_higher_projection_has_higher_same_position_value(self):
        rankings = build_league_rankings(PROJECTIONS, MY_YAHOO_LEAGUE)
        running_backs = [row for row in rankings if row["position"] == "RB"]
        self.assertGreater(running_backs[0]["projection"], running_backs[1]["projection"])
        self.assertGreater(
            running_backs[0]["value_over_replacement"],
            running_backs[1]["value_over_replacement"],
        )

    def test_replacement_levels_differ_by_league_structure(self):
        one_qb = league([{"slot": "QB", "eligible": ["QB"]}])
        two_qb = league(
            [
                {"slot": "QB1", "eligible": ["QB"]},
                {"slot": "QB2", "eligible": ["QB"]},
            ]
        )
        one_level = calculate_replacement_levels(PROJECTIONS, one_qb)["QB"]
        two_level = calculate_replacement_levels(PROJECTIONS, two_qb)["QB"]
        self.assertLess(two_level, one_level)

    def test_two_qb_and_superflex_increase_qb_scarcity_and_value(self):
        one_qb = league([{"slot": "QB", "eligible": ["QB"]}])
        superflex = league(
            [
                {"slot": "QB", "eligible": ["QB"]},
                {"slot": "SUPERFLEX", "eligible": ["QB", "RB", "WR", "TE"]},
            ]
        )
        two_qb = league(
            [
                {"slot": "QB1", "eligible": ["QB"]},
                {"slot": "QB2", "eligible": ["QB"]},
            ]
        )

        def top_qb_value(config):
            return next(
                row["value_over_replacement"]
                for row in build_league_rankings(PROJECTIONS, config)
                if row["position"] == "QB"
            )

        self.assertGreater(top_qb_value(superflex), top_qb_value(one_qb))
        self.assertGreater(top_qb_value(two_qb), top_qb_value(superflex))

    def test_more_required_rb_or_wr_starters_increase_scarcity(self):
        one_each = league(
            [
                {"slot": "RB", "eligible": ["RB"]},
                {"slot": "WR", "eligible": ["WR"]},
            ]
        )
        deeper = league(
            [
                {"slot": "RB1", "eligible": ["RB"]},
                {"slot": "RB2", "eligible": ["RB"]},
                {"slot": "WR1", "eligible": ["WR"]},
                {"slot": "WR2", "eligible": ["WR"]},
            ]
        )
        shallow_levels = calculate_replacement_levels(PROJECTIONS, one_each)
        deep_levels = calculate_replacement_levels(PROJECTIONS, deeper)
        self.assertLess(deep_levels["RB"], shallow_levels["RB"])
        self.assertLess(deep_levels["WR"], shallow_levels["WR"])

    def test_flex_demand_is_split_without_double_counting(self):
        config = league(
            [
                {"slot": "RB", "eligible": ["RB"]},
                {"slot": "FLEX", "eligible": ["RB", "WR", "TE"]},
            ],
            team_count=12,
        )
        demand = calculate_position_demand(config)
        self.assertEqual(demand["RB"], 16.0)
        self.assertEqual(demand["WR"], 4.0)
        self.assertEqual(demand["TE"], 4.0)
        self.assertEqual(sum(demand.values()), 24.0)

    def test_rankings_and_tiers_are_deterministic(self):
        first = build_league_rankings(PROJECTIONS, MY_YAHOO_LEAGUE, tier_drop=2)
        second = build_league_rankings(list(reversed(PROJECTIONS)), MY_YAHOO_LEAGUE, tier_drop=2)
        self.assertEqual(first, second)
        self.assertEqual(
            assign_tiers(
                [
                    {"value_over_replacement": 10.0},
                    {"value_over_replacement": 8.5},
                    {"value_over_replacement": 6.0},
                ],
                2.0,
            ),
            [1, 1, 2],
        )

    def test_positional_ranks_are_correct(self):
        rankings = build_league_rankings(PROJECTIONS, MY_YAHOO_LEAGUE)
        for position in ("QB", "RB", "WR", "TE"):
            rows = [row for row in rankings if row["position"] == position]
            self.assertEqual(
                [row["positional_rank"] for row in rows],
                list(range(1, len(rows) + 1)),
            )
            self.assertEqual(
                [row["projection"] for row in rows],
                sorted((row["projection"] for row in rows), reverse=True),
            )

    def test_team_count_deepens_replacement_level(self):
        slots = [{"slot": "QB", "eligible": ["QB"]}]
        four_team = league(slots, team_count=4)
        twelve_team = league(slots, team_count=12)
        self.assertLess(
            calculate_replacement_levels(PROJECTIONS, twelve_team)["QB"],
            calculate_replacement_levels(PROJECTIONS, four_team)["QB"],
        )

    def test_missing_projections_and_unsupported_positions_are_not_ranked(self):
        values = PROJECTIONS + [
            player("Missing", "RB", None),
            player("Not finite", "WR", float("nan")),
            player("Defense", "DST", 50.0),
        ]
        names = {row["name"] for row in build_league_rankings(values, MY_YAHOO_LEAGUE)}
        self.assertNotIn("Missing", names)
        self.assertNotIn("Not finite", names)
        self.assertNotIn("Defense", names)

    def test_current_yahoo_league_generates_valid_rankings(self):
        rankings = build_league_rankings(PROJECTIONS, MY_YAHOO_LEAGUE)
        levels = calculate_replacement_levels(PROJECTIONS, MY_YAHOO_LEAGUE)
        self.assertTrue(rankings)
        self.assertEqual(set(levels), {"QB", "RB", "WR", "TE"})
        self.assertEqual(MY_YAHOO_LEAGUE["team_count"], 12)
        self.assertTrue(all(row["overall_rank"] > 0 for row in rankings))

    def test_legacy_league_without_team_count_remains_supported(self):
        legacy = copy.deepcopy(MY_YAHOO_LEAGUE)
        legacy.pop("team_count")
        self.assertTrue(build_league_rankings(PROJECTIONS, legacy))


if __name__ == "__main__":
    unittest.main()
