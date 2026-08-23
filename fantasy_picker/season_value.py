"""Season-long fantasy draft values from leak-free preseason information.

The first production formula is deliberately small and explainable::

    adjusted_prior = availability * prior_fppg
                     + (1 - availability) * position_prior_median
    season_value = 0.40 * week1_projection + 0.60 * adjusted_prior

``availability`` is prior-season games divided by 17, capped at one. Players
without immediately prior-season history keep their prior fields empty and use
the position median only as the score fallback. If a position has no prior data
at all, its Week 1 projection is the neutral fallback. No older season is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import pandas as pd

from fantasy_picker.rankings import SUPPORTED_POSITIONS, build_league_rankings
from fantasy_picker.scoring import score_player


WEEK1_WEIGHT = 0.40
PRIOR_WEIGHT = 0.60
EXPECTED_REGULAR_SEASON_GAMES = 17
DEFAULT_SEASON_TIER_DROP = 1.0


def _column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def summarize_prior_season(
    player_stats: pd.DataFrame,
    target_season: int,
    scoring: Mapping[str, Real],
) -> pd.DataFrame:
    """Summarize only ``target_season - 1`` regular-season NFL production."""

    if isinstance(target_season, bool) or not isinstance(target_season, int):
        raise ValueError("target_season must be an integer")
    if not isinstance(player_stats, pd.DataFrame):
        raise TypeError("player_stats must be a pandas DataFrame")
    columns = [
        "player_id",
        "name",
        "position",
        "prior_season_fppg",
        "prior_season_games",
    ]
    if player_stats.empty:
        return pd.DataFrame(columns=columns)
    required = {"season", "position"}
    missing = sorted(required - set(player_stats.columns))
    if missing:
        raise ValueError(
            "player_stats is missing required columns: " + ", ".join(missing)
        )

    result = player_stats.loc[player_stats["season"] == target_season - 1].copy()
    if "interceptions" not in result and "passing_interceptions" in result:
        result["interceptions"] = result["passing_interceptions"]
    if "season_type" in result:
        result = result.loc[result["season_type"] == "REG"].copy()
    result["position"] = result["position"].astype(str).str.upper()
    result = result.loc[result["position"].isin(SUPPORTED_POSITIONS)].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)

    id_column = _column(result, ("player_id", "gsis_id"))
    if id_column is None:
        raise ValueError("player_stats must include player_id or gsis_id")
    name_column = _column(
        result,
        ("player_display_name", "player_name", "full_name", "name"),
    )
    result["_player_id"] = result[id_column]
    result["_name"] = result[name_column] if name_column else result[id_column]
    if "my_fantasy_points" in result:
        result["_fantasy_points"] = pd.to_numeric(
            result["my_fantasy_points"], errors="coerce"
        )
    else:
        result["_fantasy_points"] = result.apply(
            lambda row: score_player(row, scoring), axis=1
        )
    result = result.loc[
        result["_player_id"].notna() & result["_fantasy_points"].notna()
    ].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)

    group_keys = ["_player_id", "_name", "position"]
    summary = (
        result.groupby(group_keys, as_index=False, dropna=False)["_fantasy_points"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "_player_id": "player_id",
                "_name": "name",
                "mean": "prior_season_fppg",
                "count": "prior_season_games",
            }
        )
    )
    return summary[columns]


def _valid_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def calculate_season_value_score(
    week1_projection: float,
    prior_season_fppg: float | None,
    prior_season_games: int,
    position_median: float | None,
    *,
    week1_weight: float = WEEK1_WEIGHT,
    expected_games: int = EXPECTED_REGULAR_SEASON_GAMES,
) -> tuple[float, float, bool]:
    """Return score, adjusted prior component, and median-fallback indicator."""

    if not _valid_number(week1_projection):
        raise ValueError("week1_projection must be a finite number")
    if not 0 <= week1_weight <= 1:
        raise ValueError("week1_weight must be between zero and one")
    if isinstance(expected_games, bool) or not isinstance(expected_games, int) or expected_games < 1:
        raise ValueError("expected_games must be a positive integer")
    median = (
        float(position_median)
        if _valid_number(position_median)
        else float(week1_projection)
    )
    has_history = _valid_number(prior_season_fppg) and prior_season_games > 0
    if has_history:
        availability = min(float(prior_season_games) / expected_games, 1.0)
        adjusted_prior = (
            availability * float(prior_season_fppg)
            + (1.0 - availability) * median
        )
    else:
        adjusted_prior = median
    score = week1_weight * float(week1_projection) + (1.0 - week1_weight) * adjusted_prior
    return score, adjusted_prior, not has_history


def assign_season_value_tiers(
    ranked_players: Sequence[Mapping[str, Any]],
    drop_threshold: float = DEFAULT_SEASON_TIER_DROP,
) -> list[int]:
    """Create deterministic VOR bands measured from each tier's first player."""

    if not _valid_number(drop_threshold) or float(drop_threshold) <= 0:
        raise ValueError("drop_threshold must be a positive finite number")
    tiers: list[int] = []
    tier = 1
    tier_anchor: float | None = None
    for player in ranked_players:
        value = float(player["value_over_replacement"])
        if tier_anchor is not None and tier_anchor - value >= drop_threshold:
            tier += 1
            tier_anchor = value
        elif tier_anchor is None:
            tier_anchor = value
        tiers.append(tier)
    return tiers


def build_season_value_rankings(
    week1_projections: Sequence[Mapping[str, Any]],
    prior_player_stats: pd.DataFrame,
    league: Mapping[str, Any],
    target_season: int,
    *,
    week1_weight: float = WEEK1_WEIGHT,
    tier_drop: float = DEFAULT_SEASON_TIER_DROP,
) -> list[dict[str, Any]]:
    """Build league-aware draft rankings from Week 1 and prior-season inputs."""

    prior = summarize_prior_season(
        prior_player_stats, target_season, league["scoring"]
    )
    medians = prior.groupby("position")["prior_season_fppg"].median().to_dict()
    by_id = {
        str(row["player_id"]): row
        for _, row in prior.iterrows()
        if pd.notna(row["player_id"])
    }

    candidates: list[dict[str, Any]] = []
    for source in week1_projections:
        position = str(source.get("position", "")).upper()
        week1_projection = source.get("projection")
        if position not in SUPPORTED_POSITIONS or not _valid_number(week1_projection):
            continue
        name = source.get("name", source.get("player_display_name"))
        if not isinstance(name, str) or not name.strip():
            continue
        player_id = source.get("player_id")
        history = by_id.get(str(player_id)) if player_id is not None else None
        if history is not None and history["position"] != position:
            history = None
        prior_fppg = (
            float(history["prior_season_fppg"]) if history is not None else None
        )
        prior_games = int(history["prior_season_games"]) if history is not None else 0
        score, adjusted_prior, used_fallback = calculate_season_value_score(
            float(week1_projection),
            prior_fppg,
            prior_games,
            medians.get(position),
            week1_weight=week1_weight,
        )
        candidates.append(
            {
                **dict(source),
                "name": name,
                "position": position,
                "week1_projection": float(week1_projection),
                "prior_season_fppg": prior_fppg,
                "prior_season_games": prior_games,
                "adjusted_prior_fppg": adjusted_prior,
                "used_prior_median_fallback": used_fallback,
                "season_value_score": score,
                "projection": score,
            }
        )

    ranked = build_league_rankings(candidates, league, tier_drop=tier_drop)
    season_tiers = assign_season_value_tiers(ranked, tier_drop)
    for player, tier in zip(ranked, season_tiers):
        player["tier"] = tier
        player["replacement_level_score"] = player["replacement_projection"]
        player["season_value_vor"] = player["value_over_replacement"]
        player["season_value_score"] = player["projection"]
    return ranked


build_draft_board = build_season_value_rankings
