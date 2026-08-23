"""Explainable league-aware fantasy player rankings.

Replacement demand starts with each position-only lineup slot. Demand from a
flexible slot is divided evenly among its eligible QB/RB/WR/TE positions. This
simple approximation counts every starting slot exactly once without assuming
a particular provider or attempting to simulate an entire draft.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

from fantasy_picker.league import get_roster_slots, get_team_count


SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_TIER_DROP = 3.0


def calculate_position_demand(
    league: Mapping[str, Any],
) -> dict[str, float]:
    """Estimate league-wide starter demand by position.

    A position-only slot contributes one starter per team. A FLEX/SUPERFLEX
    style slot contributes an equal fraction to each supported eligible
    position, so flexible demand is never counted more than once.
    """

    team_count = get_team_count(league)
    demand = {position: 0.0 for position in SUPPORTED_POSITIONS}
    for slot in get_roster_slots(league):
        eligible = list(
            dict.fromkeys(
                position.upper()
                for position in slot["eligible"]
                if position.upper() in SUPPORTED_POSITIONS
            )
        )
        if not eligible:
            continue
        share = team_count / len(eligible)
        for position in eligible:
            demand[position] += share
    return demand


def calculate_replacement_levels(
    projections: Sequence[Mapping[str, Any]],
    league: Mapping[str, Any],
) -> dict[str, float | None]:
    """Return the projection at each position's estimated replacement rank.

    Missing and non-finite projections are ignored. If fewer projected players
    exist than the estimated demand, the last available player is the observable
    replacement level. Positions with no valid projections return ``None``.
    """

    demand = calculate_position_demand(league)
    by_position: dict[str, list[float]] = defaultdict(list)
    for player in projections:
        position = str(player.get("position", "")).upper()
        projection = player.get("projection")
        if (
            position not in SUPPORTED_POSITIONS
            or isinstance(projection, bool)
            or not isinstance(projection, Real)
            or not math.isfinite(float(projection))
        ):
            continue
        by_position[position].append(float(projection))

    levels: dict[str, float | None] = {}
    for position in SUPPORTED_POSITIONS:
        values = sorted(by_position[position], reverse=True)
        if not values:
            levels[position] = None
            continue
        replacement_rank = max(1, math.ceil(demand[position]))
        levels[position] = values[min(replacement_rank, len(values)) - 1]
    return levels


def assign_tiers(
    ranked_players: Sequence[Mapping[str, Any]],
    drop_threshold: float = DEFAULT_TIER_DROP,
) -> list[int]:
    """Assign a new tier whenever adjacent VOR drops by the threshold."""

    if isinstance(drop_threshold, bool) or not isinstance(drop_threshold, Real):
        raise ValueError("drop_threshold must be a positive number")
    if not math.isfinite(float(drop_threshold)) or drop_threshold <= 0:
        raise ValueError("drop_threshold must be a positive number")

    tiers: list[int] = []
    tier = 1
    previous: float | None = None
    for player in ranked_players:
        value = float(player["value_over_replacement"])
        if previous is not None and previous - value >= drop_threshold:
            tier += 1
        tiers.append(tier)
        previous = value
    return tiers


def build_league_rankings(
    projections: Sequence[Mapping[str, Any]],
    league: Mapping[str, Any],
    *,
    tier_drop: float = DEFAULT_TIER_DROP,
) -> list[dict[str, Any]]:
    """Build deterministic overall and positional rankings primarily by VOR."""

    levels = calculate_replacement_levels(projections, league)
    players: list[dict[str, Any]] = []
    for player in projections:
        position = str(player.get("position", "")).upper()
        projection = player.get("projection")
        if (
            position not in SUPPORTED_POSITIONS
            or levels[position] is None
            or isinstance(projection, bool)
            or not isinstance(projection, Real)
            or not math.isfinite(float(projection))
        ):
            continue
        name = player.get("name", player.get("player_display_name"))
        if not isinstance(name, str) or not name.strip():
            continue
        projection_value = float(projection)
        replacement = float(levels[position])
        players.append(
            {
                **dict(player),
                "name": name,
                "position": position,
                "projection": projection_value,
                "replacement_projection": replacement,
                "value_over_replacement": projection_value - replacement,
            }
        )

    position_order = {position: index for index, position in enumerate(SUPPORTED_POSITIONS)}
    players.sort(
        key=lambda player: (
            -player["value_over_replacement"],
            -player["projection"],
            position_order[player["position"]],
            player["name"].casefold(),
            str(player.get("player_id", "")),
        )
    )

    positional_counts = defaultdict(int)
    for overall_rank, player in enumerate(players, start=1):
        positional_counts[player["position"]] += 1
        player["overall_rank"] = overall_rank
        player["positional_rank"] = positional_counts[player["position"]]

    for player, tier in zip(players, assign_tiers(players, tier_drop)):
        player["tier"] = tier
    return players


# Short public alias for callers that prefer a verb phrase.
rank_players = build_league_rankings
