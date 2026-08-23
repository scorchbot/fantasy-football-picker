"""Provider-neutral fantasy roster validation, matching, and recommendations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from .confidence import get_lineup_confidence, lineup_confidence_label
from .decisions import build_lineup_decision_report
from .league import validate_league_config
from .lineup import optimize_lineup_fast


SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")


class RosterValidationError(ValueError):
    """Raised when a provider-neutral roster is malformed or unsafe to use."""


class RosterMatchError(ValueError):
    """Raised when ambiguous projection matches prevent safe optimization."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def normalize_player_name(name: str) -> str:
    """Normalize case and whitespace while retaining exact spelling semantics."""

    return " ".join(name.split()).casefold()


def validate_roster_config(roster: Any) -> dict[str, Any]:
    """Validate and return a plain JSON-serializable roster dictionary."""

    if not isinstance(roster, Mapping):
        raise RosterValidationError("Roster configuration must be a mapping.")
    league_name = roster.get("league_name")
    if not isinstance(league_name, str) or not league_name.strip():
        raise RosterValidationError("Roster league_name must be a nonempty string.")
    players = roster.get("players")
    if (
        not isinstance(players, Sequence)
        or isinstance(players, (str, bytes))
        or not players
    ):
        raise RosterValidationError("Roster must contain a nonempty players list.")

    normalized_players = []
    player_ids = set()
    normalized_names = []
    for index, player in enumerate(players):
        if not isinstance(player, Mapping):
            raise RosterValidationError(
                f"Roster player at index {index} must be a mapping."
            )
        missing = [field for field in ("name", "position", "team") if field not in player]
        if missing:
            raise RosterValidationError(
                f"Roster player at index {index} is missing: {', '.join(missing)}."
            )
        name = player["name"]
        position = player["position"]
        team = player["team"]
        if not isinstance(name, str) or not name.strip():
            raise RosterValidationError(
                f"Roster player at index {index} needs a nonempty name."
            )
        if not isinstance(position, str) or position.upper() not in SUPPORTED_POSITIONS:
            raise RosterValidationError(
                f"Unsupported roster position at index {index}: {position!r}."
            )
        if not isinstance(team, str) or not team.strip():
            raise RosterValidationError(
                f"Roster player at index {index} needs a nonempty team."
            )
        player_id = player.get("player_id")
        if player_id is not None:
            if not isinstance(player_id, str) or not player_id.strip():
                raise RosterValidationError(
                    f"player_id at index {index} must be a nonempty string when supplied."
                )
            if player_id in player_ids:
                raise RosterValidationError(
                    f"Duplicate roster player_id: {player_id}"
                )
            player_ids.add(player_id)

        normalized_name = normalize_player_name(name)
        normalized_names.append(normalized_name)
        normalized = {
            "name": " ".join(name.split()),
            "position": position.upper(),
            "team": team.upper(),
        }
        if player_id is not None:
            normalized["player_id"] = player_id
        normalized_players.append(normalized)

    duplicate_names = sorted(
        name for name, count in Counter(normalized_names).items() if count > 1
    )
    if duplicate_names:
        raise RosterValidationError(
            "Duplicate normalized roster names are unsafe for lineup decisions: "
            + ", ".join(duplicate_names)
        )

    result = {"league_name": league_name.strip(), "players": normalized_players}
    json.dumps(result)
    return result


def create_roster_config(
    league_name: str, players: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Create a validated provider-neutral roster configuration."""

    return validate_roster_config(
        {"league_name": league_name, "players": [dict(player) for player in players]}
    )


def load_roster_json(path: str | Path) -> dict[str, Any]:
    """Load and validate a provider-neutral roster JSON file."""

    try:
        with Path(path).open(encoding="utf-8") as roster_file:
            value = json.load(roster_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RosterValidationError(f"Could not load roster JSON: {exc}") from exc
    return validate_roster_config(value)


def _projection_indices(
    projections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_id: dict[str, list[int]] = defaultdict(list)
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, projection in enumerate(projections):
        name = projection.get("name") or projection.get("player_display_name")
        if isinstance(name, str) and name.strip():
            by_name[normalize_player_name(name)].append(index)
        player_id = projection.get("player_id")
        if isinstance(player_id, str) and player_id:
            by_id[player_id].append(index)
    duplicate_ids = sorted(player_id for player_id, rows in by_id.items() if len(rows) > 1)
    if duplicate_ids:
        raise RosterValidationError(
            "Current-week projections contain duplicate player IDs: "
            + ", ".join(duplicate_ids)
        )
    return by_id, by_name


def match_roster_to_projections(
    roster: Mapping[str, Any],
    projections: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Match every roster player by ID first, then normalized exact name."""

    validated = validate_roster_config(roster)
    if not isinstance(projections, Sequence) or isinstance(projections, (str, bytes)):
        raise RosterValidationError("Current-week projections must be a sequence.")
    by_id, by_name = _projection_indices(projections)
    matched = []
    unmatched = []
    ambiguous = []
    used_projection_rows = set()

    for roster_player in validated["players"]:
        match_index = None
        match_method = None
        player_id = roster_player.get("player_id")
        if player_id is not None and player_id in by_id:
            match_index = by_id[player_id][0]
            match_method = "player_id"
        else:
            candidates = by_name.get(normalize_player_name(roster_player["name"]), [])
            if len(candidates) > 1:
                ambiguous.append(
                    {
                        "roster_player": dict(roster_player),
                        "candidate_projections": [dict(projections[index]) for index in candidates],
                    }
                )
                continue
            if len(candidates) == 1:
                match_index = candidates[0]
                match_method = "exact_name"

        if match_index is None:
            unmatched.append({**dict(roster_player), "reason": "no_projection_match"})
            continue
        if match_index in used_projection_rows:
            ambiguous.append(
                {
                    "roster_player": dict(roster_player),
                    "candidate_projections": [dict(projections[match_index])],
                    "reason": "projection_already_matched",
                }
            )
            continue

        projection = projections[match_index]
        if projection.get("position", "").upper() != roster_player["position"]:
            unmatched.append(
                {
                    **dict(roster_player),
                    "reason": "position_mismatch",
                    "projection_position": projection.get("position"),
                }
            )
            continue
        used_projection_rows.add(match_index)
        matched_player = dict(projection)
        matched_player.update(
            {
                "player_id": roster_player.get("player_id", projection.get("player_id")),
                "name": roster_player["name"],
                "position": roster_player["position"],
                "team": projection.get("team") or roster_player["team"],
                "match_method": match_method,
            }
        )
        matched.append(matched_player)

    return {
        "matched_players": matched,
        "unmatched_players": unmatched,
        "ambiguous_matches": ambiguous,
    }


def build_roster_recommendation(
    league: Mapping[str, Any],
    roster: Mapping[str, Any],
    current_week_projections: Sequence[Mapping[str, Any]],
    confidence_model: Any | None = None,
) -> dict[str, Any]:
    """Match a roster, optimize its lineup, and build clean decision data."""

    validate_league_config(league)
    validated_roster = validate_roster_config(roster)
    diagnostics = match_roster_to_projections(validated_roster, current_week_projections)
    if diagnostics["ambiguous_matches"]:
        names = [
            match["roster_player"]["name"]
            for match in diagnostics["ambiguous_matches"]
        ]
        raise RosterMatchError(
            "Ambiguous exact-name projection matches: " + ", ".join(names),
            diagnostics,
        )

    players = diagnostics["matched_players"]
    optimized = optimize_lineup_fast(league, players)
    if optimized["lineup"] is None:
        return {
            "league_name": validated_roster["league_name"],
            **diagnostics,
            "optimal_lineup": None,
            "projected_total": None,
            "decisions": [],
        }

    lineup_result = {
        "players": players,
        "lineup": optimized["lineup"],
        "total_projection": optimized["total_projection"],
    }
    decisions = build_lineup_decision_report(league, lineup_result)
    for decision in decisions:
        gap = decision["lineup_value_gap"]
        if confidence_model is None or gap is None:
            decision["confidence"] = None
            decision["confidence_label"] = "Unknown"
        else:
            confidence = get_lineup_confidence(gap, confidence_model)
            decision["confidence"] = confidence
            decision["confidence_label"] = lineup_confidence_label(confidence)

    return {
        "league_name": validated_roster["league_name"],
        **diagnostics,
        "optimal_lineup": optimized["lineup"],
        "projected_total": optimized["total_projection"],
        "decisions": decisions,
    }
