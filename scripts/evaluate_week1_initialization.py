"""Research historical Week 1 feature initialization strategies.

This is an offline experiment, not production inference.  It trains temporary
walk-forward models whose feature definitions match their respective strategy:

* ``season_reset`` restarts player rolling features each season, so Week 1 uses
  medians learned from prior training seasons.
* ``prior_season_carryover`` lets prior-game rolling windows cross the season
  boundary, so returning players begin Week 1 with their latest prior-season
  regular-season information.

The default RandomForest settings are the settings in the authoritative
``Fantasy_Football_Picker.ipynb`` notebook.  No trained artifact is read or
modified by this script.

Run from the repository root::

    python -m scripts.evaluate_week1_initialization
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fantasy_picker.data import nflverse
from fantasy_picker.features import (
    OPPORTUNITY_COLUMNS,
    PLAYER_ROLLING_COLUMNS,
    QB_ROLLING_COLUMNS,
    build_pregame_features,
)
from fantasy_picker.scoring import MY_YAHOO_SCORING, score_player


POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_EVALUATION_SEASONS = (2022, 2023, 2024, 2025)
APPROACHES = ("season_reset", "prior_season_carryover")
SHADOW_APPROACH = "hybrid_shadow"

# Exact final feature sets in the newest Fantasy_Football_Picker.ipynb.
SKILL_FEATURES = [
    "my_fantasy_points_last3",
    "carries_last3",
    "targets_last3",
    "rushing_yards_last3",
    "receiving_yards_last3",
    "target_share_last3",
    "defense_fp_allowed_last3",
    "team_spread",
    "total_line",
    "offense_pct_last3",
    "offense_pct_last5",
    "snap_trend",
    "on_injury_report",
    "questionable",
    "practice_full",
    "practice_limited",
    "practice_dnp",
]
NOTEBOOK_FEATURES = {
    "QB": [
        "my_fantasy_points_last3",
        "attempts_last3",
        "completions_last3",
        "passing_yards_last3",
        "passing_tds_last3",
        "interceptions_last3",
        "passing_air_yards_last3",
        "passing_epa_last3",
        "carries_last3",
        "rushing_yards_last3",
        "rushing_tds_last3",
        "defense_fp_allowed_last3",
        "team_spread",
        "total_line",
        "is_home",
        "temp",
        "wind",
        "rest_days",
        "my_expected_opportunity_points_last3",
        "pass_yards_gained_exp_last3",
        "pass_touchdown_exp_last3",
    ],
    "RB": list(SKILL_FEATURES),
    "WR": list(SKILL_FEATURES),
    "TE": list(SKILL_FEATURES),
}

RANDOM_FOREST_SETTINGS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "random_state": 42,
    "n_jobs": -1,
}

# Intentionally conservative candidates, to be evaluated before production use.
DEPTH_LIMITS = {"QB": 2, "RB": 4, "WR": 6, "TE": 3}
PRIOR_USAGE_LIMITS = {
    "QB": ("attempts", 50.0),
    "RB": ("opportunities", 25.0),
    "WR": ("targets", 25.0),
    "TE": ("targets", 15.0),
}
WEEK1_CONTRIBUTOR_POINTS = {"QB": 10.0, "RB": 5.0, "WR": 5.0, "TE": 5.0}
TEAM_CODE_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
}


def _normalize_team_codes(values: pd.Series) -> pd.Series:
    return values.replace(TEAM_CODE_ALIASES)


def prepare_player_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize only names the notebook feature code expects and score rows."""

    required = {"player_id", "position", "season", "week"}
    missing = sorted(required - set(player_stats.columns))
    if missing:
        raise ValueError(f"player_stats is missing required columns: {', '.join(missing)}")

    result = player_stats.copy(deep=True)
    if "season_type" in result:
        result = result.loc[result["season_type"] == "REG"].copy()
    result = result.loc[result["position"].isin(POSITIONS)].copy()
    result = result.rename(
        columns={"team": "recent_team", "passing_interceptions": "interceptions"}
    )
    if "player_display_name" not in result:
        result["player_display_name"] = result.get("player_name", result["player_id"])
    result["my_fantasy_points"] = result.apply(
        lambda row: score_player(row, MY_YAHOO_SCORING), axis=1
    )
    return result.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _regular_season_rows(frame: pd.DataFrame, type_column: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy(deep=True)
    result = frame.copy(deep=True)
    if type_column in result:
        result = result.loc[result[type_column] == "REG"].copy()
    return result


def prepare_historical_inputs(raw_inputs: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Copy and normalize raw historical inputs without merging or fetching."""

    stats = prepare_player_stats(raw_inputs["player_stats"])
    schedules = _regular_season_rows(raw_inputs.get("schedules", pd.DataFrame()), "game_type")
    snaps = _regular_season_rows(raw_inputs.get("snap_counts", pd.DataFrame()), "game_type")
    injuries = _regular_season_rows(raw_inputs.get("injuries", pd.DataFrame()), "game_type")
    opportunity = raw_inputs.get("ff_opportunity", pd.DataFrame()).copy(deep=True)
    if not opportunity.empty:
        opportunity["season"] = pd.to_numeric(opportunity["season"], errors="raise").astype(int)
        opportunity["week"] = pd.to_numeric(opportunity["week"], errors="raise").astype(int)
    return {
        "player_stats": stats,
        "schedules": schedules,
        "snap_counts": snaps,
        "injuries": injuries,
        "ff_opportunity": opportunity,
    }


def _cross_season_rolling(
    frame: pd.DataFrame, source_columns: Sequence[str], window: int
) -> pd.DataFrame:
    result = frame.sort_values(["player_id", "season", "week"]).copy()
    previous_season = result.groupby("player_id", sort=False)["season"].shift(1)
    stale_history = previous_season.lt(result["season"] - 1)
    for column in source_columns:
        if column not in result:
            result[column] = np.nan
        result[f"{column}_last{window}"] = (
            result.groupby("player_id", sort=False)[column]
            .transform(lambda values: values.shift(1).rolling(window, min_periods=1).mean())
        )
        result.loc[stale_history, f"{column}_last{window}"] = np.nan
    return result


def _add_cross_season_defense_feature(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["defense_fp_allowed_last3"], errors="ignore").copy()
    defense_games = (
        result.groupby(["season", "week", "opponent_team", "position"], dropna=False)[
            "my_fantasy_points"
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "opponent_team": "defense_team",
                "my_fantasy_points": "fantasy_points_allowed",
            }
        )
        .sort_values(["defense_team", "position", "season", "week"])
    )
    defense_games["defense_fp_allowed_last3"] = (
        defense_games.groupby(["defense_team", "position"], sort=False)[
            "fantasy_points_allowed"
        ]
        .transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    )
    previous_season = defense_games.groupby(
        ["defense_team", "position"], sort=False
    )["season"].shift(1)
    defense_games.loc[
        previous_season.lt(defense_games["season"] - 1),
        "defense_fp_allowed_last3",
    ] = np.nan
    return result.merge(
        defense_games[
            ["season", "week", "defense_team", "position", "defense_fp_allowed_last3"]
        ],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense_team", "position"],
        how="left",
    )


def add_prior_season_carryover_features(reset_features: pd.DataFrame) -> pd.DataFrame:
    """Rebuild rolling predictors across regular-season boundaries using shift(1)."""

    rolling_sources = PLAYER_ROLLING_COLUMNS + QB_ROLLING_COLUMNS
    result = _cross_season_rolling(reset_features, rolling_sources, 3)
    result = _cross_season_rolling(result, ("offense_pct",), 3)
    result = _cross_season_rolling(result, ("offense_pct",), 5)
    result["snap_trend"] = result["offense_pct_last3"] - result["offense_pct_last5"]
    result = _cross_season_rolling(result, OPPORTUNITY_COLUMNS, 3)
    result = _add_cross_season_defense_feature(result)
    return result.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def build_initialization_feature_sets(
    raw_inputs: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build season-reset and prior-season-carryover feature definitions."""

    prepared = prepare_historical_inputs(raw_inputs)
    reset = build_pregame_features(
        prepared["player_stats"],
        schedules=prepared["schedules"],
        snap_counts=prepared["snap_counts"],
        injuries=prepared["injuries"],
        ff_opportunity=prepared["ff_opportunity"],
    )
    feature_sets = {
        "season_reset": reset,
        "prior_season_carryover": add_prior_season_carryover_features(reset),
    }
    feature_sets[SHADOW_APPROACH] = build_hybrid_shadow_features(
        feature_sets["season_reset"], feature_sets["prior_season_carryover"]
    )
    return feature_sets


def build_hybrid_shadow_features(
    reset_features: pd.DataFrame, carryover_features: pd.DataFrame
) -> pd.DataFrame:
    """Use carryover only for players present in the immediately prior season.

    New players and players with only older history retain season-reset rolling
    values, which are missing at Week 1 and therefore receive training medians
    during model inference.  The choice depends only on prior-season presence.
    """

    keys = ["player_id", "season", "week"]
    if len(reset_features) != len(carryover_features):
        raise ValueError("reset and carryover feature rows must have equal length")
    reset = reset_features.sort_values(keys).reset_index(drop=True).copy()
    carryover = carryover_features.sort_values(keys).reset_index(drop=True).copy()
    if not reset[keys].equals(carryover[keys]):
        raise ValueError("reset and carryover feature rows must identify the same games")

    season_players = {
        season: set(group["player_id"])
        for season, group in carryover.groupby("season", sort=False)
    }
    returning = pd.Series(
        [
            player_id in season_players.get(season - 1, set())
            for player_id, season in zip(carryover["player_id"], carryover["season"])
        ],
        index=carryover.index,
    )
    result = reset.copy()
    common_columns = [column for column in carryover.columns if column in result.columns]
    result.loc[returning, common_columns] = carryover.loc[returning, common_columns]
    result["has_immediate_prior_season"] = returning.astype(bool)
    return result


def _default_model_factory() -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**RANDOM_FOREST_SETTINGS)


def pairwise_ranking_accuracy(actual: Sequence[float], predicted: Sequence[float]) -> tuple[float, int]:
    """Return ranking accuracy across all non-tied player pairs."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    correct = 0
    comparisons = 0
    for left in range(len(actual_values)):
        for right in range(left + 1, len(actual_values)):
            actual_delta = actual_values[left] - actual_values[right]
            if actual_delta == 0:
                continue
            predicted_delta = predicted_values[left] - predicted_values[right]
            correct += int(np.sign(predicted_delta) == np.sign(actual_delta))
            comparisons += 1
    return (correct / comparisons if comparisons else float("nan"), comparisons)


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate Week 1 MAE, Spearman rank correlation, and pairwise accuracy."""

    if predictions.empty:
        return {"mae": float("nan"), "rank_correlation": float("nan"),
                "pairwise_accuracy": float("nan"), "pairwise_comparisons": 0,
                "sample_count": 0}
    actual = predictions["actual"].astype(float)
    predicted = predictions["prediction"].astype(float)
    pair_correct = 0.0
    pair_total = 0
    for _, group in predictions.groupby("season"):
        accuracy, comparisons = pairwise_ranking_accuracy(group["actual"], group["prediction"])
        if comparisons:
            pair_correct += accuracy * comparisons
            pair_total += comparisons
    return {
        "mae": float((actual - predicted).abs().mean()),
        "rank_correlation": float(actual.rank().corr(predicted.rank(), method="pearson")),
        "pairwise_accuracy": pair_correct / pair_total if pair_total else float("nan"),
        "pairwise_comparisons": pair_total,
        "sample_count": len(predictions),
    }


def evaluate_week1_initialization(
    feature_sets: Mapping[str, pd.DataFrame],
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
    *,
    model_factory: Callable[[], Any] = _default_model_factory,
    approaches: Sequence[str] = APPROACHES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run separate walk-forward models for each initialization definition."""

    prediction_rows: list[dict[str, Any]] = []
    for approach in approaches:
        data = feature_sets[approach]
        for season in evaluation_seasons:
            for position in POSITIONS:
                columns = NOTEBOOK_FEATURES[position]
                train = data.loc[
                    (data["season"] < season)
                    & (data["position"] == position)
                    & data["my_fantasy_points_last3"].notna()
                ].copy()
                test = data.loc[
                    (data["season"] == season)
                    & (data["week"] == 1)
                    & (data["position"] == position)
                ].copy()
                if train.empty or test.empty:
                    continue
                medians = train[columns].median(numeric_only=True)
                missing_medians = [name for name in columns if name not in medians or pd.isna(medians[name])]
                if missing_medians:
                    raise ValueError(
                        f"{approach} {season} {position} has no training median for: "
                        + ", ".join(missing_medians)
                    )
                model = model_factory()
                model.fit(train[columns].fillna(medians), train["my_fantasy_points"])
                predictions = model.predict(test[columns].fillna(medians))
                for (_, row), prediction in zip(test.iterrows(), predictions):
                    prediction_rows.append(
                        {
                            "approach": approach,
                            "season": season,
                            "position": position,
                            "player_id": row["player_id"],
                            "player_name": row.get("player_display_name", row["player_id"]),
                            "actual": float(row["my_fantasy_points"]),
                            "prediction": float(prediction),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for (approach, position), group in predictions.groupby(["approach", "position"]):
        metric_rows.append({"approach": approach, "position": position, **calculate_metrics(group)})
    return pd.DataFrame(metric_rows), predictions


def summarize_season_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pair reset and carryover Week 1 metrics for every season and position."""

    rows = []
    core = predictions.loc[predictions["approach"].isin(APPROACHES)]
    for (season, position), group in core.groupby(["season", "position"]):
        metrics = {
            approach: calculate_metrics(approach_rows)
            for approach, approach_rows in group.groupby("approach")
        }
        if set(metrics) != set(APPROACHES):
            continue
        reset = metrics["season_reset"]
        carryover = metrics["prior_season_carryover"]
        rows.append(
            {
                "season": int(season),
                "position": position,
                "sample_count": reset["sample_count"],
                "reset_mae": reset["mae"],
                "carryover_mae": carryover["mae"],
                "reset_rank_correlation": reset["rank_correlation"],
                "carryover_rank_correlation": carryover["rank_correlation"],
                "reset_pairwise_accuracy": reset["pairwise_accuracy"],
                "carryover_pairwise_accuracy": carryover["pairwise_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "position"]).reset_index(drop=True)


def summarize_aggregate_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Preserve the existing aggregate result layout across all seasons."""

    rows = []
    for (approach, position), group in predictions.groupby(["approach", "position"]):
        rows.append(
            {"approach": approach, "position": position, **calculate_metrics(group)}
        )
    return pd.DataFrame(rows).sort_values(["approach", "position"]).reset_index(drop=True)


def calculate_prior_history_coverage(
    carryover_features: pd.DataFrame,
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
) -> pd.DataFrame:
    """Count Week 1 players with history, usable carryover, or median fallback."""

    rows = []
    for season in evaluation_seasons:
        prior_season_ids = set(
            carryover_features.loc[carryover_features["season"] == season - 1, "player_id"]
        )
        any_prior_ids = set(
            carryover_features.loc[carryover_features["season"] < season, "player_id"]
        )
        week_one = carryover_features.loc[
            (carryover_features["season"] == season) & (carryover_features["week"] == 1)
        ]
        for position in POSITIONS:
            players = week_one.loc[week_one["position"] == position].copy()
            has_prior_season = players["player_id"].isin(prior_season_ids)
            has_any_history = players["player_id"].isin(any_prior_ids)
            usable = players["my_fantasy_points_last3"].notna()
            rows.append(
                {
                    "season": season,
                    "position": position,
                    "week1_players": len(players),
                    "prior_season_history": int(has_prior_season.sum()),
                    "older_history_only": int((has_any_history & ~has_prior_season).sum()),
                    "no_prior_nfl_history": int((~has_any_history).sum()),
                    "usable_carryover": int(usable.sum()),
                    "fallback_median": int((~usable).sum()),
                }
            )
    return pd.DataFrame(rows)


def _latest_depth_chart(depth_charts: pd.DataFrame) -> pd.DataFrame:
    if depth_charts.empty:
        return depth_charts.copy()
    required = {"team", "gsis_id", "pos_abb", "pos_rank"}
    missing = sorted(required - set(depth_charts.columns))
    if missing:
        raise ValueError(f"depth_charts is missing required columns: {', '.join(missing)}")
    latest = depth_charts.copy(deep=True)
    if "dt" in latest:
        latest = latest.loc[latest["dt"] == latest["dt"].max()].copy()
    return latest.loc[latest["pos_abb"].isin(POSITIONS)].drop_duplicates(
        subset=["team", "gsis_id", "pos_abb"], keep="last"
    )


def analyze_fantasy_relevance(
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    prior_player_stats: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Count conservative depth-chart/prior-usage candidate filters by position."""

    roster = rosters.copy(deep=True)
    if "season" in roster:
        roster = roster.loc[roster["season"] == season].copy()
    required = {"team", "position", "gsis_id"}
    missing = sorted(required - set(roster.columns))
    if missing:
        raise ValueError(f"rosters is missing required columns: {', '.join(missing)}")
    roster = roster.loc[roster["position"].isin(POSITIONS) & roster["gsis_id"].notna()].copy()
    roster = roster.drop_duplicates(subset=["team", "gsis_id"], keep="last")
    roster["team"] = _normalize_team_codes(roster["team"])

    depth = _latest_depth_chart(depth_charts).rename(
        columns={"pos_abb": "position", "pos_rank": "depth_rank"}
    )
    depth["team"] = _normalize_team_codes(depth["team"])
    roster = roster.merge(
        depth[["team", "gsis_id", "position", "depth_rank"]],
        on=["team", "gsis_id", "position"],
        how="left",
    )
    roster["depth_rank"] = pd.to_numeric(roster["depth_rank"], errors="coerce")

    prior = prior_player_stats.copy(deep=True)
    if "season" in prior and not prior.empty:
        prior = prior.loc[prior["season"] == season - 1].copy()
    prior = prior.rename(columns={"player_id": "gsis_id"})
    for column in ("attempts", "carries", "targets"):
        if column not in prior:
            prior[column] = 0.0
    prior["opportunities"] = prior["carries"].fillna(0) + prior["targets"].fillna(0)
    usage = prior.groupby("gsis_id", as_index=False)[["attempts", "targets", "opportunities"]].sum()
    roster = roster.merge(usage, on="gsis_id", how="left")
    roster[["attempts", "targets", "opportunities"]] = roster[
        ["attempts", "targets", "opportunities"]
    ].fillna(0)

    rows = []
    for position in POSITIONS:
        players = roster.loc[roster["position"] == position]
        depth_selected = players["depth_rank"].le(DEPTH_LIMITS[position])
        usage_column, usage_limit = PRIOR_USAGE_LIMITS[position]
        usage_selected = players[usage_column].ge(usage_limit)
        rows.append(
            {
                "position": position,
                "roster_players": len(players),
                "depth_rule": int(depth_selected.sum()),
                "prior_usage_rule": int(usage_selected.sum()),
                "depth_or_usage": int((depth_selected | usage_selected).sum()),
                "no_prior_usage": int((players[usage_column] == 0).sum()),
                "depth_limit": DEPTH_LIMITS[position],
                "usage_threshold": f"{usage_column}>={usage_limit:g}",
            }
        )
    return pd.DataFrame(rows)


def _historical_depth_snapshot(
    depth_charts: pd.DataFrame, schedules: pd.DataFrame, season: int
) -> tuple[pd.DataFrame, str | None]:
    """Select official Week 1 rows or a dated snapshot published before Week 1."""

    if depth_charts.empty:
        return pd.DataFrame(), "no depth-chart data"
    weekly_columns = {
        "season",
        "week",
        "game_type",
        "club_code",
        "gsis_id",
        "position",
        "depth_team",
    }
    if weekly_columns.issubset(depth_charts.columns):
        weekly = depth_charts.loc[
            (depth_charts["season"] == season)
            & (depth_charts["week"] == 1)
            & (depth_charts["game_type"] == "REG")
        ]
        if not weekly.empty:
            return weekly[
                ["club_code", "gsis_id", "position", "depth_team"]
            ].rename(
                columns={
                    "club_code": "team",
                    "position": "pos_abb",
                    "depth_team": "pos_rank",
                }
            ), None

    required = {"dt", "team", "gsis_id", "pos_abb", "pos_rank"}
    if not required.issubset(depth_charts.columns):
        return pd.DataFrame(), "depth-chart data lacks dated player ranks"
    games = schedules.loc[
        (schedules["season"] == season) & (schedules["week"] == 1)
    ].copy()
    if "game_type" in games:
        games = games.loc[games["game_type"] == "REG"]
    if games.empty or "gameday" not in games:
        return pd.DataFrame(), "Week 1 schedule date is unavailable"
    cutoff = pd.to_datetime(games["gameday"], utc=True, errors="coerce").min()
    dated = depth_charts.copy(deep=True)
    dated["_snapshot_time"] = pd.to_datetime(dated["dt"], utc=True, errors="coerce")
    dated = dated.loc[
        dated["_snapshot_time"].notna()
        & (dated["_snapshot_time"] < cutoff)
    ]
    if dated.empty:
        return pd.DataFrame(), "no depth-chart snapshot predates Week 1"
    snapshot_time = dated["_snapshot_time"].max()
    snapshot = dated.loc[
        dated["_snapshot_time"] == snapshot_time,
        ["team", "gsis_id", "pos_abb", "pos_rank"],
    ]
    return snapshot, None


def _filter_player_rows(
    rosters: pd.DataFrame,
    depth_snapshot: pd.DataFrame,
    prior_player_stats: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    roster = rosters.loc[rosters["season"] == season].copy()
    if "week" in roster:
        roster = roster.loc[roster["week"] == 1].copy()
    roster = roster.loc[
        roster["position"].isin(POSITIONS) & roster["gsis_id"].notna()
    ].drop_duplicates(subset=["team", "gsis_id"], keep="last")
    roster["team"] = _normalize_team_codes(roster["team"])
    depth = _latest_depth_chart(depth_snapshot).rename(
        columns={"pos_abb": "position", "pos_rank": "depth_rank"}
    )
    depth["team"] = _normalize_team_codes(depth["team"])
    roster = roster.merge(
        depth[["team", "gsis_id", "position", "depth_rank"]],
        on=["team", "gsis_id", "position"],
        how="left",
    )
    roster["depth_rank"] = pd.to_numeric(roster["depth_rank"], errors="coerce")

    prior = prior_player_stats.loc[prior_player_stats["season"] == season - 1].copy()
    prior = prior.rename(columns={"player_id": "gsis_id"})
    for column in ("attempts", "carries", "targets"):
        if column not in prior:
            prior[column] = 0.0
    prior["opportunities"] = prior["carries"].fillna(0) + prior["targets"].fillna(0)
    usage = prior.groupby("gsis_id", as_index=False)[
        ["attempts", "targets", "opportunities"]
    ].sum()
    roster = roster.merge(usage, on="gsis_id", how="left")
    roster[["attempts", "targets", "opportunities"]] = roster[
        ["attempts", "targets", "opportunities"]
    ].fillna(0)
    roster["passes_depth_rule"] = False
    roster["passes_prior_usage_rule"] = False
    for position in POSITIONS:
        selected = roster["position"] == position
        usage_column, usage_limit = PRIOR_USAGE_LIMITS[position]
        roster.loc[selected, "passes_depth_rule"] = roster.loc[
            selected, "depth_rank"
        ].le(DEPTH_LIMITS[position])
        roster.loc[selected, "passes_prior_usage_rule"] = roster.loc[
            selected, usage_column
        ].ge(usage_limit)
    roster["retained"] = (
        roster["passes_depth_rule"] | roster["passes_prior_usage_rule"]
    )
    return roster


def evaluate_historical_filtering(
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame,
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the candidate filter only where dated pre-Week-1 depth data exists."""

    summaries: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    actual = prepare_player_stats(player_stats).rename(
        columns={"player_id": "gsis_id", "recent_team": "team"}
    )
    for season in evaluation_seasons:
        snapshot, limitation = _historical_depth_snapshot(depth_charts, schedules, season)
        season_rosters = rosters.loc[rosters["season"] == season]
        if "week" in season_rosters:
            season_rosters = season_rosters.loc[season_rosters["week"] == 1]
        if season_rosters.empty:
            limitation = "Week 1 weekly roster data is unavailable"
        if limitation:
            for position in POSITIONS:
                summaries.append(
                    {
                        "season": season,
                        "position": position,
                        "available": False,
                        "limitation": limitation,
                    }
                )
            continue

        represented_teams = snapshot["team"].nunique()
        represented_positions = set(snapshot["pos_abb"].dropna()) & set(POSITIONS)
        if represented_teams < 30 or represented_positions != set(POSITIONS):
            limitation = (
                "pregame depth chart is incomplete: "
                f"{represented_teams} teams and {sorted(represented_positions)} positions"
            )
            for position in POSITIONS:
                summaries.append(
                    {
                        "season": season,
                        "position": position,
                        "available": False,
                        "limitation": limitation,
                    }
                )
            continue

        candidates = _filter_player_rows(
            rosters, snapshot, player_stats, season=season
        )

        week_one = actual.loc[(actual["season"] == season) & (actual["week"] == 1)]
        actual_columns = [
            "gsis_id",
            "my_fantasy_points",
            "attempts",
            "carries",
            "targets",
        ]
        week_one = week_one.reindex(columns=actual_columns).groupby(
            "gsis_id", as_index=False
        ).sum(numeric_only=True)
        candidates = candidates.merge(week_one, on="gsis_id", how="left")
        candidates[["my_fantasy_points", "attempts_y", "carries", "targets_y"]] = (
            candidates[["my_fantasy_points", "attempts_y", "carries", "targets_y"]]
            .fillna(0)
        )
        candidates["week1_opportunities"] = (
            candidates["carries"] + candidates["targets_y"]
        )

        for position in POSITIONS:
            players = candidates.loc[candidates["position"] == position].copy()
            contributor = players["my_fantasy_points"].ge(
                WEEK1_CONTRIBUTOR_POINTS[position]
            )
            if position == "QB":
                low_usage = players["attempts_y"].eq(0)
            else:
                low_usage = players["week1_opportunities"].eq(0)
            false_excluded = players.loc[contributor & ~players["retained"]]
            for _, player in false_excluded.iterrows():
                exclusions.append(
                    {
                        "season": season,
                        "position": position,
                        "player_id": player["gsis_id"],
                        "player_name": player.get("full_name", player["gsis_id"]),
                        "team": player["team"],
                        "week1_points": player["my_fantasy_points"],
                        "depth_rank": player["depth_rank"],
                    }
                )
            contributor_count = int(contributor.sum())
            low_usage_count = int(low_usage.sum())
            summaries.append(
                {
                    "season": season,
                    "position": position,
                    "available": True,
                    "limitation": "",
                    "roster_players": len(players),
                    "depth_listed_players": int(players["depth_rank"].notna().sum()),
                    "retained_players": int(players["retained"].sum()),
                    "player_count_reduction_pct": float((~players["retained"]).mean()),
                    "contributors": contributor_count,
                    "contributor_retention_pct": (
                        float(players.loc[contributor, "retained"].mean())
                        if contributor_count else float("nan")
                    ),
                    "low_no_usage_players": low_usage_count,
                    "low_no_usage_removal_pct": (
                        float((~players.loc[low_usage, "retained"]).mean())
                        if low_usage_count else float("nan")
                    ),
                    "false_exclusions": len(false_excluded),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(exclusions)


def assess_production_readiness(
    season_results: pd.DataFrame,
    filtering_results: pd.DataFrame,
    false_exclusions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Apply explicit promotion criteria to initialization and filtering evidence."""

    consistency = {}
    catastrophic_rows = []
    for position, rows in season_results.groupby("position"):
        rank_wins = int(
            (rows["carryover_rank_correlation"] > rows["reset_rank_correlation"]).sum()
        )
        pairwise_wins = int(
            (rows["carryover_pairwise_accuracy"] > rows["reset_pairwise_accuracy"]).sum()
        )
        required_wins = len(rows) // 2 + 1
        consistency[position] = {
            "seasons": len(rows),
            "rank_wins": rank_wins,
            "pairwise_wins": pairwise_wins,
            "passes": rank_wins >= required_wins and pairwise_wins >= required_wins,
        }
        catastrophic = rows.loc[
            (rows["carryover_mae"] - rows["reset_mae"] > 2.0)
            | (
                rows["carryover_rank_correlation"]
                - rows["reset_rank_correlation"]
                < -0.20
            )
            | (
                rows["carryover_pairwise_accuracy"]
                - rows["reset_pairwise_accuracy"]
                < -0.10
            )
        ]
        catastrophic_rows.extend(
            (int(row["season"]), position) for _, row in catastrophic.iterrows()
        )

    initialization_ready = bool(consistency) and all(
        result["passes"] for result in consistency.values()
    ) and not catastrophic_rows
    available_filtering = filtering_results.loc[filtering_results["available"] == True]
    expected_filter_rows = (
        season_results["season"].nunique() * season_results["position"].nunique()
    )
    filtering_complete = len(available_filtering) == expected_filter_rows
    total_contributors = available_filtering.get("contributors", pd.Series(dtype=float)).sum()
    retained_contributors = (
        available_filtering.get("contributors", pd.Series(dtype=float))
        * available_filtering.get("contributor_retention_pct", pd.Series(dtype=float))
    ).sum()
    contributor_retention = (
        float(retained_contributors / total_contributors)
        if total_contributors else float("nan")
    )
    minimum_position_season_retention = (
        float(available_filtering["contributor_retention_pct"].min())
        if not available_filtering.empty else float("nan")
    )
    false_exclusions = (
        pd.DataFrame() if false_exclusions is None else false_exclusions
    )
    high_impact_false_exclusions = (
        false_exclusions.loc[false_exclusions["week1_points"] >= 10].to_dict("records")
        if not false_exclusions.empty and "week1_points" in false_exclusions
        else []
    )
    filtering_ready = bool(
        filtering_complete
        and total_contributors
        and contributor_retention >= 0.99
        and minimum_position_season_retention >= 0.95
        and not high_impact_false_exclusions
    )
    ready = initialization_ready and filtering_ready
    reasons = []
    if not initialization_ready:
        reasons.append("carryover does not improve rank and pairwise accuracy in most seasons for every position")
    if catastrophic_rows:
        reasons.append(f"catastrophic regressions: {catastrophic_rows}")
    if not filtering_complete:
        reasons.append("historical filtering evidence is incomplete")
    elif not filtering_ready:
        reasons.append(
            "filtering fails the 99% aggregate/95% position-season retention rule "
            "or has a 10+ point false exclusion"
        )
    return {
        "ready": ready,
        "initialization_ready": initialization_ready,
        "filtering_ready": filtering_ready,
        "consistency": consistency,
        "catastrophic_regressions": catastrophic_rows,
        "historical_filtering_complete": filtering_complete,
        "contributor_retention": contributor_retention,
        "minimum_position_season_retention": minimum_position_season_retention,
        "high_impact_false_exclusions": high_impact_false_exclusions,
        "recommendation": (
            "Promote the hybrid Week 1 strategy and candidate filter."
            if ready
            else (
                "The hybrid Week 1 initialization is ready for a separate production "
                "change, but the candidate player filter is not. "
                if initialization_ready and not filtering_ready
                else "Keep production unchanged; continue shadow-mode evaluation. "
            ) + "; ".join(reasons)
        ),
    }


def load_historical_inputs(seasons: Sequence[int]) -> dict[str, pd.DataFrame]:
    """Load the raw data used by the experiment through the existing adapter."""

    requested = list(seasons)
    return {
        "player_stats": nflverse.load_player_stats(requested),
        "schedules": nflverse.load_schedules(requested),
        "snap_counts": nflverse.load_snap_counts(requested),
        "injuries": nflverse.load_injuries(requested),
        "ff_opportunity": nflverse.load_ff_opportunity(requested),
        "weekly_rosters": nflverse.load_weekly_rosters(requested),
        "depth_charts": nflverse.load_depth_charts(requested),
    }


def print_report(results: Mapping[str, Any]) -> None:
    print("Historical Week 1 results by season")
    print(
        results["season_results"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print("\nAggregate historical Week 1 model performance")
    print(
        results["aggregate_results"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print("\nHybrid shadow-mode performance")
    print(
        results["shadow_results"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print("\nPrior-season history coverage")
    print(results["coverage"].to_string(index=False))
    print("\nHistorical fantasy-relevance filtering")
    print(
        results["historical_filtering"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    if not results["false_exclusions"].empty:
        print("\nHistorical notable false exclusions")
        print(results["false_exclusions"].to_string(index=False))
    print("\n2026 conservative fantasy-relevance filter candidates")
    print(results["current_filtering"].to_string(index=False))
    print("\nProduction-readiness recommendation")
    print(results["readiness"]["recommendation"])


def run_experiment(
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
    *,
    current_season: int = 2026,
) -> dict[str, Any]:
    history_seasons = list(range(min(evaluation_seasons) - 1, max(evaluation_seasons) + 1))
    raw = load_historical_inputs(history_seasons)
    feature_sets = build_initialization_feature_sets(raw)
    _, predictions = evaluate_week1_initialization(
        feature_sets,
        evaluation_seasons,
        approaches=(*APPROACHES, SHADOW_APPROACH),
    )
    aggregate = summarize_aggregate_results(
        predictions.loc[predictions["approach"].isin(APPROACHES)]
    )
    season_results = summarize_season_results(predictions)
    shadow_results = summarize_aggregate_results(
        predictions.loc[predictions["approach"] == SHADOW_APPROACH]
    )
    coverage = calculate_prior_history_coverage(
        feature_sets["prior_season_carryover"], evaluation_seasons
    )
    rosters = nflverse.load_weekly_rosters(current_season, allow_preseason_fallback=True)
    depth = nflverse.load_depth_charts(current_season)
    prior_stats = nflverse.load_player_stats(current_season - 1)
    current_filtering = analyze_fantasy_relevance(
        rosters, depth, prior_stats, season=current_season
    )
    historical_filtering, false_exclusions = evaluate_historical_filtering(
        raw["weekly_rosters"],
        raw["depth_charts"],
        raw["player_stats"],
        raw["schedules"],
        evaluation_seasons,
    )
    readiness = assess_production_readiness(
        season_results, historical_filtering, false_exclusions
    )
    return {
        "season_results": season_results,
        "aggregate_results": aggregate,
        "shadow_results": shadow_results,
        "coverage": coverage,
        "historical_filtering": historical_filtering,
        "false_exclusions": false_exclusions,
        "current_filtering": current_filtering,
        "readiness": readiness,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVALUATION_SEASONS),
    )
    parser.add_argument("--current-season", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        results = run_experiment(
            args.evaluation_seasons, current_season=args.current_season
        )
        print_report(results)
    except Exception as exc:
        print(f"Week 1 initialization experiment failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
