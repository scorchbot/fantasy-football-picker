import unittest

from fantasy_picker.lineup import is_player_eligible_for_slot, optimize_lineup


NORMAL_FLEX_LEAGUE = {
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

SUPERFLEX_LEAGUE = {
    "roster_slots": [
        *NORMAL_FLEX_LEAGUE["roster_slots"][:-1],
        {"slot": "SUPERFLEX", "eligible": ["QB", "RB", "WR", "TE"]},
    ]
}

NORMAL_ROSTER = [
    {"name": "QB A", "position": "QB", "projection": 22.0},
    {"name": "RB A", "position": "RB", "projection": 17.0},
    {"name": "RB B", "position": "RB", "projection": 14.0},
    {"name": "RB C", "position": "RB", "projection": 12.5},
    {"name": "WR A", "position": "WR", "projection": 18.0},
    {"name": "WR B", "position": "WR", "projection": 15.5},
    {"name": "WR C", "position": "WR", "projection": 13.0},
    {"name": "TE A", "position": "TE", "projection": 11.0},
    {"name": "TE B", "position": "TE", "projection": 8.0},
]


class LineupTests(unittest.TestCase):
    def test_rb_is_eligible_for_rb_and_flex(self):
        self.assertTrue(is_player_eligible_for_slot(NORMAL_FLEX_LEAGUE, "RB", "RB1"))
        self.assertTrue(is_player_eligible_for_slot(NORMAL_FLEX_LEAGUE, "RB", "FLEX"))

    def test_qb_is_not_eligible_for_normal_flex(self):
        self.assertFalse(is_player_eligible_for_slot(NORMAL_FLEX_LEAGUE, "QB", "FLEX"))

    def test_qb_is_eligible_for_superflex(self):
        self.assertTrue(is_player_eligible_for_slot(SUPERFLEX_LEAGUE, "QB", "SUPERFLEX"))

    def test_optimizer_does_not_use_a_player_twice(self):
        result = optimize_lineup(NORMAL_FLEX_LEAGUE, NORMAL_ROSTER)
        names = [player["name"] for player in result["lineup"]]

        self.assertEqual(len(names), len(set(names)))

    def test_normal_flex_roster_total(self):
        result = optimize_lineup(NORMAL_FLEX_LEAGUE, NORMAL_ROSTER)

        self.assertEqual(result["total_projection"], 110.5)

    def test_superflex_roster_total(self):
        roster = [
            NORMAL_ROSTER[0],
            {"name": "QB B", "position": "QB", "projection": 19.0},
            *NORMAL_ROSTER[1:],
        ]
        result = optimize_lineup(SUPERFLEX_LEAGUE, roster)

        self.assertEqual(result["total_projection"], 116.5)


if __name__ == "__main__":
    unittest.main()
