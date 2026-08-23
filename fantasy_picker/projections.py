"""Reusable direct-model inference helpers for fantasy projections."""

import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")


def _validated_position(position: str) -> str:
    normalized = position.upper()
    if normalized not in SUPPORTED_POSITIONS:
        raise ValueError(f"Unsupported position: {position}")
    return normalized


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, Real) and math.isnan(value))


def _feature_input(
    player_row: Mapping[str, Any],
    feature_names: Sequence[str],
    medians: Mapping[str, Any],
) -> Any:
    """Build the notebook's one-row feature frame, with a dependency-free fallback."""

    feature_values = {name: player_row[name] for name in feature_names}

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return [
            [
                medians[name] if _is_missing(feature_values[name]) else feature_values[name]
                for name in feature_names
            ]
        ]

    return pd.DataFrame([feature_values], columns=feature_names).fillna(medians)


def project_fantasy_points(
    player_row: Mapping[str, Any],
    position: str,
    final_features: Mapping[str, Sequence[str]],
    final_medians: Mapping[str, Mapping[str, Any]],
    final_models: Mapping[str, Any],
) -> float:
    """Project one player with the supplied fitted position model."""

    position = _validated_position(position)
    features = final_features[position]
    model_input = _feature_input(player_row, features, final_medians[position])
    projection = final_models[position].predict(model_input)[0]
    return float(projection)


def build_projected_player(
    player_row: Mapping[str, Any],
    position: str,
    final_features: Mapping[str, Sequence[str]],
    final_medians: Mapping[str, Mapping[str, Any]],
    final_models: Mapping[str, Any],
    *,
    name_field: str = "player_display_name",
    team_field: str = "recent_team",
    opponent_field: str = "opponent_team",
    actual_points_field: str = "my_fantasy_points",
) -> dict[str, Any]:
    """Build a projected-player record accepted by the lineup optimizer."""

    position = _validated_position(position)
    projected_player = {
        "name": player_row[name_field],
        "position": position,
        "projection": project_fantasy_points(
            player_row,
            position,
            final_features,
            final_medians,
            final_models,
        ),
        "team": player_row[team_field],
        "opponent": player_row[opponent_field],
    }

    if actual_points_field in player_row:
        projected_player["actual_points"] = player_row[actual_points_field]

    return projected_player


def _iter_rows(selected_players: Any) -> Iterable[Mapping[str, Any]]:
    if hasattr(selected_players, "iterrows"):
        return (row for _, row in selected_players.iterrows())
    return iter(selected_players)


def project_players(
    selected_players: Any,
    final_features: Mapping[str, Sequence[str]],
    final_medians: Mapping[str, Mapping[str, Any]],
    final_models: Mapping[str, Any],
    *,
    position_field: str = "position",
    name_field: str = "player_display_name",
    team_field: str = "recent_team",
    opponent_field: str = "opponent_team",
    actual_points_field: str = "my_fantasy_points",
) -> list[dict[str, Any]]:
    """Project selected player rows, skipping positions outside QB/RB/WR/TE."""

    projected_players = []

    for row in _iter_rows(selected_players):
        position = str(row[position_field]).upper()
        if position not in SUPPORTED_POSITIONS:
            continue

        projected_players.append(
            build_projected_player(
                row,
                position,
                final_features,
                final_medians,
                final_models,
                name_field=name_field,
                team_field=team_field,
                opponent_field=opponent_field,
                actual_points_field=actual_points_field,
            )
        )

    return projected_players
