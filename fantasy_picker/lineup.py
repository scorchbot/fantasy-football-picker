"""Roster-slot eligibility and full-lineup optimization."""

from typing import Any, Mapping, Sequence


def get_slot_config(
    league: Mapping[str, Any], slot_name: str
) -> Mapping[str, Any]:
    """Return the configuration for a named starting roster slot."""

    for slot in league["roster_slots"]:
        if slot["slot"] == slot_name:
            return slot

    raise ValueError(f"Slot '{slot_name}' not found in league configuration.")


def is_player_eligible_for_slot(
    league: Mapping[str, Any], player_position: str, slot_name: str
) -> bool:
    """Return whether a position is allowed in a configured roster slot."""

    slot = get_slot_config(league, slot_name)
    return player_position.upper() in slot["eligible"]


def optimize_lineup_fast(
    league: Mapping[str, Any], players: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Find the highest-projected complete lineup with no repeated player."""

    roster_slots = league["roster_slots"]
    ordered_slots = sorted(roster_slots, key=lambda slot: len(slot["eligible"]))

    best_lineup = None
    best_total = float("-inf")

    def search(
        slot_index: int,
        used_players: set[int],
        current_lineup: list[dict[str, Any]],
        current_total: float,
    ) -> None:
        nonlocal best_lineup, best_total

        if slot_index == len(ordered_slots):
            if current_total > best_total:
                best_total = current_total
                best_lineup = current_lineup.copy()
            return

        slot = ordered_slots[slot_index]

        for player_index, player in enumerate(players):
            if player_index in used_players:
                continue
            if not is_player_eligible_for_slot(
                league, player["position"], slot["slot"]
            ):
                continue

            used_players.add(player_index)
            current_lineup.append(
                {
                    "slot": slot["slot"],
                    "name": player["name"],
                    "position": player["position"],
                    "projection": player["projection"],
                }
            )
            search(
                slot_index + 1,
                used_players,
                current_lineup,
                current_total + player["projection"],
            )
            current_lineup.pop()
            used_players.remove(player_index)

    search(0, set(), [], 0)

    slot_order = {
        slot["slot"]: index for index, slot in enumerate(roster_slots)
    }
    if best_lineup is not None:
        best_lineup = sorted(
            best_lineup, key=lambda player: slot_order[player["slot"]]
        )

    return {"lineup": best_lineup, "total_projection": best_total}


# A concise public name while retaining the notebook function name above.
optimize_lineup = optimize_lineup_fast
