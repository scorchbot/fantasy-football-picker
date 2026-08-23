"""Orchestration helpers for current-week fantasy projections and lineups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fantasy_picker.confidence import (
    get_lineup_confidence,
    lineup_confidence_label,
)
from fantasy_picker.decisions import build_lineup_decision_report
from fantasy_picker.features import build_pregame_features
from fantasy_picker.lineup import optimize_lineup_fast
from fantasy_picker.models import ModelArtifacts, coerce_model_artifacts
from fantasy_picker.projections import project_players


SUPPORTED_POSITIONS = ("QB", "RB", "WR", "TE")


def _validate_season_week(season: int, week: int) -> None:
    if isinstance(season, bool) or not isinstance(season, int):
        raise ValueError("season must be an integer")
    if isinstance(week, bool) or not isinstance(week, int) or week < 1:
        raise ValueError("week must be a positive integer")


def _copy_frame(raw_inputs: Mapping[str, Any], name: str) -> pd.DataFrame:
    value = raw_inputs.get(name)
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"raw_inputs[{name!r}] must be a pandas DataFrame")
    return value.copy(deep=True)


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def _current_roster_rows(
    weekly_rosters: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """Normalize current weekly roster identifiers into player-game placeholders."""

    if weekly_rosters.empty or not {"season", "week", "position"}.issubset(
        weekly_rosters.columns
    ):
        return pd.DataFrame()

    roster = weekly_rosters.loc[
        (weekly_rosters["season"] == season) & (weekly_rosters["week"] == week)
    ].copy()
    if roster.empty:
        return roster

    player_id = _first_present(roster.columns, ("player_id", "gsis_id"))
    player_name = _first_present(
        roster.columns,
        ("player_display_name", "full_name", "player_name", "football_name"),
    )
    team = _first_present(roster.columns, ("recent_team", "team"))
    if player_id is None:
        raise ValueError("weekly_rosters must include player_id or gsis_id")

    placeholders = pd.DataFrame(
        {
            "player_id": roster[player_id],
            "player_display_name": (
                roster[player_name] if player_name else roster[player_id]
            ),
            "position": roster["position"],
            "season": season,
            "week": week,
        }
    )
    if team:
        placeholders["recent_team"] = roster[team].values

    return placeholders.loc[
        placeholders["player_id"].notna()
        & placeholders["position"].isin(SUPPORTED_POSITIONS)
    ].drop_duplicates(subset=["season", "week", "player_id"], keep="last")


def _add_missing_current_rows(
    player_stats: pd.DataFrame, weekly_rosters: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    placeholders = _current_roster_rows(weekly_rosters, season, week)
    if placeholders.empty:
        return player_stats

    if {"season", "week", "player_id"}.issubset(player_stats.columns):
        existing = set(
            player_stats.loc[
                (player_stats["season"] == season) & (player_stats["week"] == week),
                "player_id",
            ].dropna()
        )
        placeholders = placeholders.loc[~placeholders["player_id"].isin(existing)]

    if placeholders.empty:
        return player_stats
    return pd.concat([player_stats, placeholders], ignore_index=True, sort=False)


def build_current_week_projections(
    season: int,
    week: int,
    raw_inputs: Mapping[str, pd.DataFrame],
    model_artifacts: ModelArtifacts | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build leak-free features and project eligible players for a requested week."""

    _validate_season_week(season, week)
    if not isinstance(raw_inputs, Mapping):
        raise TypeError("raw_inputs must be a mapping of pandas DataFrames")

    artifacts = coerce_model_artifacts(model_artifacts)
    player_stats = _copy_frame(raw_inputs, "player_stats")
    weekly_rosters = _copy_frame(raw_inputs, "weekly_rosters")
    player_games = _add_missing_current_rows(
        player_stats, weekly_rosters, season, week
    )
    if player_games.empty:
        return []

    feature_rows = build_pregame_features(
        player_games=player_games,
        schedules=_copy_frame(raw_inputs, "schedules"),
        snap_counts=_copy_frame(raw_inputs, "snap_counts"),
        injuries=_copy_frame(raw_inputs, "injuries"),
        ff_opportunity=_copy_frame(raw_inputs, "ff_opportunity"),
    )
    current_rows = feature_rows.loc[
        (feature_rows["season"] == season)
        & (feature_rows["week"] == week)
        & feature_rows["position"].isin(SUPPORTED_POSITIONS)
    ].copy()
    if current_rows.empty:
        return []

    projected = project_players(
        current_rows,
        final_features=artifacts.final_features,
        final_medians=artifacts.final_medians,
        final_models=artifacts.final_models,
    )
    for (_, row), player in zip(current_rows.iterrows(), projected):
        player["player_id"] = row.get("player_id")
        player["player_display_name"] = row.get(
            "player_display_name", player["name"]
        )
        player["season"] = season
        player["week"] = week

    return projected


def build_current_week_lineup(
    projections: Sequence[Mapping[str, Any]],
    league: Mapping[str, Any],
    roster_player_names: Sequence[str] | None = None,
    roster_player_ids: Sequence[str] | None = None,
    confidence_model: Any | None = None,
) -> dict[str, Any]:
    """Optimize a selected roster and construct clean lineup decision records."""

    names = set(roster_player_names or ())
    player_ids = set(roster_player_ids or ())
    players = [dict(player) for player in projections]
    if names or player_ids:
        players = [
            player
            for player in players
            if player.get("name") in names or player.get("player_id") in player_ids
        ]

    optimized = optimize_lineup_fast(league, players)
    if optimized["lineup"] is None:
        return {
            "players": players,
            "lineup": None,
            "total_projection": None,
            "decisions": [],
        }

    lineup_result = {
        "players": players,
        "lineup": optimized["lineup"],
        "total_projection": optimized["total_projection"],
    }
    decisions = build_lineup_decision_report(league, lineup_result)
    for decision in decisions:
        gap = decision.get("lineup_value_gap")
        if confidence_model is None or gap is None:
            decision["confidence"] = None
            decision["confidence_label"] = "Unknown"
        else:
            confidence = get_lineup_confidence(gap, confidence_model)
            decision["confidence"] = confidence
            decision["confidence_label"] = lineup_confidence_label(confidence)

    return {**lineup_result, "decisions": decisions}
