"""Serializable league configuration helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

from .scoring import MY_YAHOO_SCORING


REQUIRED_LEAGUE_FIELDS = ("name", "scoring", "roster_slots")
REQUIRED_SCORING_FIELDS = tuple(MY_YAHOO_SCORING)


def has_required_fields(league: Any) -> bool:
    """Return whether a value contains the top-level league fields."""

    return isinstance(league, Mapping) and all(
        field in league for field in REQUIRED_LEAGUE_FIELDS
    )


def validate_league_config(league: Any) -> Mapping[str, Any]:
    """Validate the serializable fields used by scoring and lineup helpers."""

    if not isinstance(league, Mapping):
        raise ValueError("League configuration must be a mapping.")

    missing_fields = [
        field for field in REQUIRED_LEAGUE_FIELDS if field not in league
    ]
    if missing_fields:
        raise ValueError(
            "League configuration is missing required fields: "
            + ", ".join(missing_fields)
        )

    if not isinstance(league["name"], str) or not league["name"].strip():
        raise ValueError("League name must be a nonempty string.")

    scoring = league["scoring"]
    if not isinstance(scoring, Mapping):
        raise ValueError("League scoring settings must be a mapping.")

    missing_scoring = [
        field for field in REQUIRED_SCORING_FIELDS if field not in scoring
    ]
    if missing_scoring:
        raise ValueError(
            "League scoring settings are missing required fields: "
            + ", ".join(missing_scoring)
        )

    roster_slots = league["roster_slots"]
    if (
        not isinstance(roster_slots, Sequence)
        or isinstance(roster_slots, (str, bytes))
        or not roster_slots
    ):
        raise ValueError("League roster_slots must be a nonempty sequence.")

    slot_names = set()
    for index, slot in enumerate(roster_slots):
        if not isinstance(slot, Mapping):
            raise ValueError(f"Roster slot at index {index} must be a mapping.")
        if "slot" not in slot or "eligible" not in slot:
            raise ValueError(
                f"Roster slot at index {index} requires slot and eligible fields."
            )

        slot_name = slot["slot"]
        if not isinstance(slot_name, str) or not slot_name.strip():
            raise ValueError(f"Roster slot at index {index} needs a valid name.")
        if slot_name in slot_names:
            raise ValueError(f"Roster slot names must be unique: {slot_name}")
        slot_names.add(slot_name)

        eligible = slot["eligible"]
        if (
            not isinstance(eligible, Sequence)
            or isinstance(eligible, (str, bytes))
            or not eligible
            or any(not isinstance(position, str) or not position for position in eligible)
        ):
            raise ValueError(
                f"Roster slot '{slot_name}' needs eligible position names."
            )

    bench_slots = league.get("bench_slots", 0)
    if not isinstance(bench_slots, int) or isinstance(bench_slots, bool):
        raise ValueError("League bench_slots must be a nonnegative integer.")
    if bench_slots < 0:
        raise ValueError("League bench_slots must be a nonnegative integer.")

    return league


def create_league_config(
    name: str,
    scoring: Mapping[str, Any],
    roster_slots: Sequence[Mapping[str, Any]],
    bench_slots: int = 0,
) -> dict[str, Any]:
    """Create and validate a JSON-serializable league configuration."""

    league = {
        "name": name,
        "scoring": dict(scoring),
        "roster_slots": [
            {
                **dict(slot),
                "eligible": [
                    position.upper() for position in slot.get("eligible", [])
                ],
            }
            for slot in roster_slots
        ],
        "bench_slots": bench_slots,
    }
    validate_league_config(league)
    return league


def get_roster_slots(league: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Return roster-slot eligibility independently from scoring settings."""

    validate_league_config(league)
    return league["roster_slots"]


def get_scoring_settings(league: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return scoring rules independently from roster-slot settings."""

    validate_league_config(league)
    return league["scoring"]


MY_YAHOO_LEAGUE = create_league_config(
    name="My Yahoo League",
    scoring=MY_YAHOO_SCORING,
    roster_slots=[
        {"slot": "QB", "eligible": ["QB"]},
        {"slot": "RB1", "eligible": ["RB"]},
        {"slot": "RB2", "eligible": ["RB"]},
        {"slot": "WR1", "eligible": ["WR"]},
        {"slot": "WR2", "eligible": ["WR"]},
        {"slot": "TE", "eligible": ["TE"]},
        {"slot": "FLEX", "eligible": ["RB", "WR", "TE"]},
    ],
    bench_slots=6,
)

DEFAULT_LEAGUE = MY_YAHOO_LEAGUE
