import json
from pathlib import Path
import tempfile
import unittest

from fantasy_picker.league import MY_YAHOO_LEAGUE, create_league_config
from fantasy_picker.roster import (
    RosterMatchError,
    RosterValidationError,
    build_roster_recommendation,
    create_roster_config,
    load_roster_json,
    match_roster_to_projections,
    validate_roster_config,
)
from fantasy_picker.scoring import MY_YAHOO_SCORING


def player(player_id, name, position, projection, team="AAA"):
    return {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": team,
        "opponent": "BBB",
        "projection": projection,
    }


PROJECTIONS = [
    player("qb1", "Quarterback One", "QB", 20.0),
    player("qb2", "Quarterback Two", "QB", 19.0),
    player("rb1", "Running Back One", "RB", 18.0),
    player("rb2", "Running Back Two", "RB", 16.0),
    player("rb3", "Running Back Three", "RB", 15.0),
    player("wr1", "Receiver One", "WR", 17.0),
    player("wr2", "Receiver Two", "WR", 14.0),
    player("wr3", "Receiver Three", "WR", 13.0),
    player("te1", "Tight End One", "TE", 10.0),
    player("te2", "Tight End Two", "TE", 9.0),
]


def roster_for(projections=PROJECTIONS):
    return create_roster_config(
        "Test League",
        [
            {
                "player_id": projection["player_id"],
                "name": projection["name"],
                "position": projection["position"],
                "team": projection["team"],
            }
            for projection in projections
        ],
    )


class RosterTests(unittest.TestCase):
    def test_id_matching_is_preferred_over_name_matching(self):
        roster = create_roster_config(
            "Test",
            [{"player_id": "right", "name": "Shared Name", "position": "QB", "team": "AAA"}],
        )
        projections = [
            player("right", "Different Projection Name", "QB", 21.0),
            player("wrong", "Shared Name", "QB", 5.0),
        ]

        result = match_roster_to_projections(roster, projections)

        self.assertEqual(result["matched_players"][0]["projection"], 21.0)
        self.assertEqual(result["matched_players"][0]["match_method"], "player_id")

    def test_normalized_exact_name_fallback_works_without_id(self):
        roster = create_roster_config(
            "Test",
            [{"name": "  Quarterback   One ", "position": "QB", "team": "AAA"}],
        )

        result = match_roster_to_projections(roster, PROJECTIONS)

        self.assertEqual(result["matched_players"][0]["player_id"], "qb1")
        self.assertEqual(result["matched_players"][0]["match_method"], "exact_name")

    def test_ambiguous_duplicate_projection_names_fail_clearly(self):
        roster = create_roster_config(
            "Test",
            [{"name": "Alex Smith", "position": "QB", "team": "AAA"}],
        )
        projections = [
            player("a", "Alex Smith", "QB", 10.0),
            player("b", "alex  smith", "QB", 9.0),
        ]
        diagnostics = match_roster_to_projections(roster, projections)

        self.assertEqual(len(diagnostics["ambiguous_matches"]), 1)
        with self.assertRaisesRegex(RosterMatchError, "Ambiguous exact-name"):
            build_roster_recommendation(MY_YAHOO_LEAGUE, roster, projections)

    def test_unmatched_players_are_preserved_without_projection(self):
        roster = create_roster_config(
            "Test",
            [{"player_id": "missing", "name": "Missing Player", "position": "RB", "team": "AAA"}],
        )

        result = match_roster_to_projections(roster, PROJECTIONS)

        self.assertFalse(result["matched_players"])
        self.assertEqual(result["unmatched_players"][0]["name"], "Missing Player")
        self.assertNotIn("projection", result["unmatched_players"][0])

    def test_duplicate_roster_player_ids_fail_validation(self):
        roster = {
            "league_name": "Test",
            "players": [
                {"player_id": "same", "name": "One", "position": "RB", "team": "AAA"},
                {"player_id": "same", "name": "Two", "position": "WR", "team": "BBB"},
            ],
        }

        with self.assertRaisesRegex(RosterValidationError, "Duplicate roster player_id"):
            validate_roster_config(roster)

    def test_duplicate_normalized_roster_names_fail_validation(self):
        roster = {
            "league_name": "Test",
            "players": [
                {"player_id": "one", "name": "Same Name", "position": "RB", "team": "AAA"},
                {"player_id": "two", "name": "same  name", "position": "WR", "team": "BBB"},
            ],
        }

        with self.assertRaisesRegex(RosterValidationError, "Duplicate normalized roster names"):
            validate_roster_config(roster)

    def test_matched_roster_optimizes_yahoo_flex_and_decisions(self):
        result = build_roster_recommendation(
            MY_YAHOO_LEAGUE, roster_for(), PROJECTIONS
        )
        lineup = {starter["slot"]: starter for starter in result["optimal_lineup"]}

        self.assertEqual(len(result["matched_players"]), len(PROJECTIONS))
        self.assertEqual(lineup["FLEX"]["name"], "Running Back Three")
        self.assertEqual(result["projected_total"], 110.0)
        self.assertEqual(len(result["decisions"]), 7)

    def test_superflex_league_uses_second_quarterback(self):
        superflex = create_league_config(
            "Superflex",
            MY_YAHOO_SCORING,
            [
                {"slot": "QB", "eligible": ["QB"]},
                {"slot": "SUPERFLEX", "eligible": ["QB", "RB", "WR", "TE"]},
            ],
        )

        result = build_roster_recommendation(superflex, roster_for(), PROJECTIONS)
        lineup = {starter["slot"]: starter for starter in result["optimal_lineup"]}

        self.assertEqual(lineup["QB"]["name"], "Quarterback One")
        self.assertEqual(lineup["SUPERFLEX"]["name"], "Quarterback Two")
        self.assertEqual(result["projected_total"], 39.0)

    def test_json_roster_round_trips_cleanly(self):
        original = roster_for(PROJECTIONS[:2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roster.json"
            path.write_text(json.dumps(original), encoding="utf-8")

            loaded = load_roster_json(path)

        self.assertEqual(loaded, original)


if __name__ == "__main__":
    unittest.main()
