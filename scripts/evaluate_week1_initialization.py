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
    return {
        "season_reset": reset,
        "prior_season_carryover": add_prior_season_carryover_features(reset),
    }


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run separate walk-forward models for each initialization definition."""

    prediction_rows: list[dict[str, Any]] = []
    for approach in APPROACHES:
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

    depth = _latest_depth_chart(depth_charts).rename(
        columns={"pos_abb": "position", "pos_rank": "depth_rank"}
    )
    roster = roster.merge(
        depth[["team", "gsis_id", "position", "depth_rank"]],
        on=["team", "gsis_id", "position"],
        how="left",
    )

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


def load_historical_inputs(seasons: Sequence[int]) -> dict[str, pd.DataFrame]:
    """Load the raw data used by the experiment through the existing adapter."""

    requested = list(seasons)
    return {
        "player_stats": nflverse.load_player_stats(requested),
        "schedules": nflverse.load_schedules(requested),
        "snap_counts": nflverse.load_snap_counts(requested),
        "injuries": nflverse.load_injuries(requested),
        "ff_opportunity": nflverse.load_ff_opportunity(requested),
    }


def print_report(metrics: pd.DataFrame, coverage: pd.DataFrame, filtering: pd.DataFrame) -> None:
    print("Historical Week 1 model performance")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nPrior-season history coverage")
    print(coverage.to_string(index=False))
    print("\n2026 conservative fantasy-relevance filter candidates")
    print(filtering.to_string(index=False))


def run_experiment(
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
    *,
    current_season: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history_seasons = list(range(min(evaluation_seasons) - 1, max(evaluation_seasons) + 1))
    raw = load_historical_inputs(history_seasons)
    feature_sets = build_initialization_feature_sets(raw)
    metrics, _ = evaluate_week1_initialization(feature_sets, evaluation_seasons)
    coverage = calculate_prior_history_coverage(
        feature_sets["prior_season_carryover"], evaluation_seasons
    )
    rosters = nflverse.load_weekly_rosters(current_season, allow_preseason_fallback=True)
    depth = nflverse.load_depth_charts(current_season)
    prior_stats = nflverse.load_player_stats(current_season - 1)
    filtering = analyze_fantasy_relevance(
        rosters, depth, prior_stats, season=current_season
    )
    return metrics, coverage, filtering


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
        metrics, coverage, filtering = run_experiment(
            args.evaluation_seasons, current_season=args.current_season
        )
        print_report(metrics, coverage, filtering)
    except Exception as exc:
        print(f"Week 1 initialization experiment failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
