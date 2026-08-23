import unittest

from fantasy_picker.decisions import (
    build_lineup_decision_report,
    summarize_lineup_changes,
)


LEAGUE = {
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

PLAYERS = [
    {"name": "Jalen Hurts", "position": "QB", "projection": 21.37},
    {"name": "Saquon Barkley", "position": "RB", "projection": 20.45},
    {"name": "Breece Hall", "position": "RB", "projection": 13.81},
    {"name": "Bijan Robinson", "position": "RB", "projection": 16.61},
    {"name": "A.J. Brown", "position": "WR", "projection": 10.24},
    {"name": "Ja'Marr Chase", "position": "WR", "projection": 15.25},
    {"name": "Garrett Wilson", "position": "WR", "projection": 12.43},
    {"name": "Travis Kelce", "position": "TE", "projection": 10.947},
    {"name": "George Kittle", "position": "TE", "projection": 10.214},
]

BASELINE_LINEUP = [
    {"slot": "QB", **PLAYERS[0]},
    {"slot": "RB1", **PLAYERS[1]},
    {"slot": "RB2", **PLAYERS[2]},
    {"slot": "WR1", **PLAYERS[6]},
    {"slot": "WR2", **PLAYERS[5]},
    {"slot": "TE", **PLAYERS[7]},
    {"slot": "FLEX", **PLAYERS[3]},
]

LINEUP_RESULT = {
    "players": PLAYERS,
    "lineup": BASELINE_LINEUP,
    "total_projection": sum(player["projection"] for player in BASELINE_LINEUP),
}


class DecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_lineup_decision_report(LEAGUE, LINEUP_RESULT)

    def decision_for(self, starter):
        return next(
            decision for decision in self.report if decision["starter"] == starter
        )

    def test_breece_removal_moves_bijan_and_adds_aj_brown(self):
        decision = self.decision_for("Breece Hall")

        self.assertIn("A.J. Brown", decision["summary"]["in"])
        self.assertIn(
            {"player": "Bijan Robinson", "from": "FLEX", "to": "RB2"},
            decision["summary"]["moves"],
        )
        self.assertAlmostEqual(decision["lineup_value_gap"], 3.57, places=2)

    def test_cosmetic_receiver_swap_is_not_meaningful(self):
        baseline = [
            {"slot": "WR1", "name": "Receiver A"},
            {"slot": "WR2", "name": "Receiver B"},
        ]
        alternate = [
            {"slot": "WR1", "name": "Receiver B"},
            {"slot": "WR2", "name": "Receiver A"},
        ]

        summary = summarize_lineup_changes(baseline, alternate)

        self.assertEqual(summary["moves"], [])
        self.assertEqual(summary["out"], [])
        self.assertEqual(summary["in"], [])

    def test_kelce_removal_adds_kittle_with_expected_gap(self):
        decision = self.decision_for("Travis Kelce")

        self.assertEqual(decision["summary"]["in"], ["George Kittle"])
        self.assertEqual(decision["summary"]["out"], ["Travis Kelce"])
        self.assertAlmostEqual(decision["lineup_value_gap"], 0.73, places=2)


if __name__ == "__main__":
    unittest.main()
