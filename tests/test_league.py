import json
import unittest

from fantasy_picker.league import (
    MY_YAHOO_LEAGUE,
    create_league_config,
    get_roster_slots,
    get_scoring_settings,
    has_required_fields,
    validate_league_config,
)
from fantasy_picker.lineup import get_slot_config, is_player_eligible_for_slot
from fantasy_picker.scoring import MY_YAHOO_SCORING


class LeagueTests(unittest.TestCase):
    def test_yahoo_config_has_expected_roster_slots(self):
        slot_names = [slot["slot"] for slot in get_roster_slots(MY_YAHOO_LEAGUE)]

        self.assertEqual(
            slot_names, ["QB", "RB1", "RB2", "WR1", "WR2", "TE", "FLEX"]
        )

    def test_yahoo_flex_eligibility_matches_notebook(self):
        for position in ("RB", "WR", "TE"):
            with self.subTest(position=position):
                self.assertTrue(
                    is_player_eligible_for_slot(MY_YAHOO_LEAGUE, position, "FLEX")
                )

        self.assertFalse(
            is_player_eligible_for_slot(MY_YAHOO_LEAGUE, "QB", "FLEX")
        )

    def test_superflex_can_allow_all_offensive_positions(self):
        league = create_league_config(
            "Superflex League",
            MY_YAHOO_SCORING,
            [{"slot": "SUPERFLEX", "eligible": ["QB", "RB", "WR", "TE"]}],
        )

        for position in ("QB", "RB", "WR", "TE"):
            with self.subTest(position=position):
                self.assertTrue(
                    is_player_eligible_for_slot(league, position, "SUPERFLEX")
                )

    def test_multiple_flexible_slots_are_supported(self):
        league = create_league_config(
            "Multiple Flex League",
            MY_YAHOO_SCORING,
            [
                {"slot": "FLEX1", "eligible": ["RB", "WR", "TE"]},
                {"slot": "FLEX2", "eligible": ["RB", "WR", "TE"]},
                {
                    "slot": "SUPERFLEX1",
                    "eligible": ["QB", "RB", "WR", "TE"],
                },
                {
                    "slot": "SUPERFLEX2",
                    "eligible": ["QB", "RB", "WR", "TE"],
                },
            ],
        )

        self.assertEqual(len(get_roster_slots(league)), 4)
        self.assertEqual(
            get_slot_config(league, "SUPERFLEX2")["eligible"],
            ["QB", "RB", "WR", "TE"],
        )

    def test_scoring_and_roster_settings_are_retrieved_independently(self):
        scoring = get_scoring_settings(MY_YAHOO_LEAGUE)
        roster_slots = get_roster_slots(MY_YAHOO_LEAGUE)

        self.assertEqual(scoring, MY_YAHOO_SCORING)
        self.assertEqual(roster_slots[-1]["slot"], "FLEX")
        self.assertNotIn("roster_slots", scoring)

    def test_invalid_configs_fail_clearly(self):
        self.assertFalse(has_required_fields({"name": "Incomplete"}))
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_league_config({"name": "Incomplete"})

        duplicate_slots = {
            "name": "Duplicates",
            "scoring": MY_YAHOO_SCORING,
            "roster_slots": [
                {"slot": "FLEX", "eligible": ["RB"]},
                {"slot": "FLEX", "eligible": ["WR"]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_league_config(duplicate_slots)

    def test_json_round_trip_preserves_behavior(self):
        restored = json.loads(json.dumps(MY_YAHOO_LEAGUE))

        validate_league_config(restored)
        self.assertEqual(restored, MY_YAHOO_LEAGUE)
        self.assertEqual(get_scoring_settings(restored), MY_YAHOO_SCORING)
        self.assertTrue(is_player_eligible_for_slot(restored, "RB", "FLEX"))
        self.assertFalse(is_player_eligible_for_slot(restored, "QB", "FLEX"))


if __name__ == "__main__":
    unittest.main()
