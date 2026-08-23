"""Data construction for lineup-level start and bench decisions."""

from typing import Any, Mapping, Optional, Sequence

from .lineup import optimize_lineup_fast


Player = Mapping[str, Any]
Lineup = Sequence[Player]


def remove_player(players: Sequence[Player], player_name: str) -> list[Player]:
    """Return a roster without the named player."""

    return [player for player in players if player["name"] != player_name]


def measure_lineup_value_gap(
    baseline_total: float,
    alternate_lineup: Optional[Lineup],
    alternate_total: float,
) -> Optional[float]:
    """Measure the projected loss, or return ``None`` if no lineup is valid."""

    if alternate_lineup is None:
        return None
    return baseline_total - alternate_total


def _base_slot(slot_name: str) -> str:
    return "".join(character for character in slot_name if not character.isdigit())


def summarize_lineup_changes(
    baseline_lineup: Lineup, alternate_lineup: Optional[Lineup]
) -> dict[str, list[Any]]:
    """Summarize players leaving, entering, and making meaningful slot moves."""

    if alternate_lineup is None:
        return {"out": [], "in": [], "moves": []}

    baseline_names = {player["name"] for player in baseline_lineup}
    alternate_names = {player["name"] for player in alternate_lineup}

    baseline_slots = {
        player["name"]: player["slot"] for player in baseline_lineup
    }
    alternate_slots = {
        player["name"]: player["slot"] for player in alternate_lineup
    }

    moves = []
    for name in sorted(baseline_names & alternate_names):
        old_slot = baseline_slots[name]
        new_slot = alternate_slots[name]
        if _base_slot(old_slot) != _base_slot(new_slot):
            moves.append({"player": name, "from": old_slot, "to": new_slot})

    return {
        "out": sorted(baseline_names - alternate_names),
        "in": sorted(alternate_names - baseline_names),
        "moves": moves,
    }


def _slot_changes(
    baseline_lineup: Lineup, alternate_lineup: Optional[Lineup]
) -> list[dict[str, Any]]:
    baseline_by_slot = {
        player["slot"]: player["name"] for player in baseline_lineup
    }
    alternate_by_slot = {
        player["slot"]: player["name"] for player in alternate_lineup or []
    }

    return [
        {
            "slot": slot,
            "old_player": old_player,
            "new_player": alternate_by_slot.get(slot),
        }
        for slot, old_player in baseline_by_slot.items()
        if old_player != alternate_by_slot.get(slot)
    ]


def _evaluate_starter(
    league: Mapping[str, Any], lineup_result: Mapping[str, Any], starter: Player
) -> dict[str, Any]:
    baseline_lineup = lineup_result["lineup"]
    roster_without_starter = remove_player(
        lineup_result["players"], starter["name"]
    )
    alternate_result = optimize_lineup_fast(league, roster_without_starter)
    alternate_lineup = alternate_result["lineup"]
    alternate_total = alternate_result["total_projection"]

    return {
        "slot": starter["slot"],
        "starter": starter["name"],
        "position": starter["position"],
        "projection": starter["projection"],
        "lineup_value_gap": measure_lineup_value_gap(
            lineup_result["total_projection"], alternate_lineup, alternate_total
        ),
        "alternate_total": alternate_total,
        "changes": _slot_changes(baseline_lineup, alternate_lineup),
        "summary": summarize_lineup_changes(baseline_lineup, alternate_lineup),
    }


def build_lineup_decision_report(
    league: Mapping[str, Any], lineup_result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build the notebook's clean, whole-lineup decision report data."""

    return [
        _evaluate_starter(league, lineup_result, starter)
        for starter in lineup_result["lineup"]
    ]


def build_global_lineup_decision_report(
    league: Mapping[str, Any], lineup_result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build the earlier global report shape without clean summaries."""

    report = build_lineup_decision_report(league, lineup_result)
    return [
        {key: value for key, value in decision.items() if key != "summary"}
        for decision in report
    ]
