"""Evaluate preseason season-value signals against historical season outcomes.

Temporary Week 1 models are trained walk-forward through the existing Week 1
research helpers. Target-season statistics are joined only after preseason
signals have been constructed and are used solely as evaluation labels.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fantasy_picker.league import MY_YAHOO_LEAGUE
from fantasy_picker.rankings import build_league_rankings
from fantasy_picker.scoring import score_player
from fantasy_picker.season_value import calculate_season_value_score, summarize_prior_season
from scripts import evaluate_week1_initialization as week1_research


DEFAULT_EVALUATION_SEASONS = (2022, 2023, 2024, 2025)
APPROACHES = ("week1_projection", "prior_season_fppg", "hybrid_season_value")
WEIGHT_GRID = tuple(round(step / 10, 1) for step in range(11))
AVAILABILITY_OPTIONS = {
    "no_shrink": None,
    "cap_8_games": 8,
    "cap_10_games": 10,
    "cap_12_games": 12,
    "linear_17_games": 17,
}
DEFAULT_WEEK1_WEIGHT = 0.40
DEFAULT_AVAILABILITY = "linear_17_games"


def _prepare_actual_stats(
    player_stats: pd.DataFrame, league: Mapping[str, Any]
) -> pd.DataFrame:
    result = player_stats.copy(deep=True)
    if "season_type" in result:
        result = result.loc[result["season_type"] == "REG"].copy()
    if "player_id" not in result and "gsis_id" in result:
        result = result.rename(columns={"gsis_id": "player_id"})
    if "interceptions" not in result and "passing_interceptions" in result:
        result["interceptions"] = result["passing_interceptions"]
    if "my_fantasy_points" not in result:
        result["my_fantasy_points"] = result.apply(
            lambda row: score_player(row, league["scoring"]), axis=1
        )
    return result


def build_preseason_evaluation_rows(
    week1_predictions: pd.DataFrame,
    player_stats: pd.DataFrame,
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
    league: Mapping[str, Any] = MY_YAHOO_LEAGUE,
) -> pd.DataFrame:
    """Construct all draft signals before joining target-season outcomes."""

    required = {"season", "position", "player_id", "prediction"}
    missing = sorted(required - set(week1_predictions.columns))
    if missing:
        raise ValueError(
            "week1_predictions is missing required columns: " + ", ".join(missing)
        )
    stats = _prepare_actual_stats(player_stats, league)
    rows: list[dict[str, Any]] = []
    for season in evaluation_seasons:
        preseason = week1_predictions.loc[
            week1_predictions["season"].eq(season)
        ].copy()
        if "approach" in preseason:
            preferred = preseason.loc[
                preseason["approach"].eq(week1_research.SHADOW_APPROACH)
            ]
            if not preferred.empty:
                preseason = preferred
        preseason = preseason.drop_duplicates("player_id", keep="last")

        prior = summarize_prior_season(stats, season, league["scoring"])
        prior_by_id = prior.set_index(prior["player_id"].astype(str), drop=False)
        medians = prior.groupby("position")["prior_season_fppg"].median().to_dict()
        for _, player in preseason.iterrows():
            position = str(player["position"]).upper()
            player_id = str(player["player_id"])
            history = prior_by_id.loc[player_id] if player_id in prior_by_id.index else None
            if isinstance(history, pd.DataFrame):
                history = history.iloc[-1]
            if history is not None and history["position"] != position:
                history = None
            prior_fppg = float(history["prior_season_fppg"]) if history is not None else None
            prior_games = int(history["prior_season_games"]) if history is not None else 0
            position_median = medians.get(position)
            hybrid, adjusted_prior, fallback = calculate_season_value_score(
                float(player["prediction"]), prior_fppg, prior_games, position_median
            )
            prior_signal = (
                prior_fppg
                if prior_fppg is not None
                else (
                    float(position_median)
                    if pd.notna(position_median)
                    else float(player["prediction"])
                )
            )
            rows.append(
                {
                    "season": season,
                    "player_id": player["player_id"],
                    "name": player.get("player_name", player.get("name", player["player_id"])),
                    "position": position,
                    "team": player.get("team"),
                    "week1_projection": float(player["prediction"]),
                    "prior_season_fppg": prior_fppg,
                    "prior_season_games": prior_games,
                    "position_median_fppg": (
                        float(position_median) if pd.notna(position_median) else None
                    ),
                    "adjusted_prior_fppg": adjusted_prior,
                    "used_prior_median_fallback": fallback,
                    "week1_projection_score": float(player["prediction"]),
                    "prior_season_fppg_score": prior_signal,
                    "hybrid_season_value_score": hybrid,
                }
            )

    preseason_rows = pd.DataFrame(rows)
    actual = stats.loc[stats["season"].isin(evaluation_seasons)].copy()
    actual = (
        actual.groupby(["season", "player_id"], as_index=False)["my_fantasy_points"]
        .sum()
        .rename(columns={"my_fantasy_points": "actual_season_points"})
    )
    return preseason_rows.merge(actual, on=["season", "player_id"], how="inner")


def adjusted_prior_for_strategy(
    week1_projection: float,
    prior_season_fppg: float | None,
    prior_season_games: int,
    position_median: float | None,
    availability_strategy: str = DEFAULT_AVAILABILITY,
) -> float:
    """Apply one explicit prior-season credibility strategy."""

    if availability_strategy not in AVAILABILITY_OPTIONS:
        raise ValueError(f"Unknown availability strategy: {availability_strategy}")
    median = (
        float(position_median)
        if position_median is not None and pd.notna(position_median)
        else float(week1_projection)
    )
    if prior_season_fppg is None or pd.isna(prior_season_fppg) or prior_season_games <= 0:
        return median
    credibility_games = AVAILABILITY_OPTIONS[availability_strategy]
    if credibility_games is None:
        return float(prior_season_fppg)
    credibility = min(float(prior_season_games) / credibility_games, 1.0)
    return credibility * float(prior_season_fppg) + (1.0 - credibility) * median


def candidate_scores(
    rows: pd.DataFrame,
    week1_weight: float,
    availability_strategy: str = DEFAULT_AVAILABILITY,
) -> pd.Series:
    """Score preseason rows without consulting any target-season outcome."""

    if not 0 <= week1_weight <= 1:
        raise ValueError("week1_weight must be between zero and one")
    values = []
    for _, row in rows.iterrows():
        adjusted = adjusted_prior_for_strategy(
            row["week1_projection"],
            row["prior_season_fppg"],
            int(row["prior_season_games"]),
            row["position_median_fppg"],
            availability_strategy,
        )
        values.append(
            week1_weight * float(row["week1_projection"])
            + (1.0 - week1_weight) * adjusted
        )
    return pd.Series(values, index=rows.index, dtype=float)


def _position_candidate_metrics(
    rows: pd.DataFrame, score_column: str
) -> dict[str, float | int]:
    actual = rows.set_index(rows["player_id"].astype(str))["actual_season_points"]
    predicted = rows.set_index(rows["player_id"].astype(str))[score_column]
    top_n = 24 if rows["position"].iloc[0] in {"RB", "WR"} else 12
    rank_correlation = float(actual.rank().corr(predicted.rank()))
    return {
        "sample_count": len(rows),
        "rank_correlation": rank_correlation,
        "top_n": top_n,
        "top_n_overlap": _overlap(actual, predicted, top_n),
        # Within one position, subtracting one replacement value preserves rank.
        "vor_rank_correlation": rank_correlation,
    }


def evaluate_parameter_grid(
    rows: pd.DataFrame,
    *,
    weights: Sequence[float] = WEIGHT_GRID,
    availability_strategies: Sequence[str] = (DEFAULT_AVAILABILITY,),
) -> pd.DataFrame:
    """Evaluate candidate weights and shrinkage rules by season and position."""

    results = []
    for weight in weights:
        for availability in availability_strategies:
            scored = rows.copy()
            scored["_candidate_score"] = candidate_scores(
                scored, float(weight), availability
            )
            for (season, position), group in scored.groupby(["season", "position"]):
                results.append(
                    {
                        "season": int(season),
                        "position": position,
                        "week1_weight": float(weight),
                        "availability_strategy": availability,
                        **_position_candidate_metrics(group, "_candidate_score"),
                    }
                )
    return pd.DataFrame(results).sort_values(
        ["season", "position", "availability_strategy", "week1_weight"]
    ).reset_index(drop=True)


def _selection_objective(rows: pd.DataFrame) -> float:
    """Combine ranking and top-N evidence without using raw point accuracy."""

    return float(
        0.70 * rows["rank_correlation"].mean()
        + 0.20 * rows["top_n_overlap"].mean()
        + 0.10 * rows["vor_rank_correlation"].mean()
    )


def select_walk_forward_parameters(
    grid_results: pd.DataFrame,
    evaluation_seasons: Sequence[int],
    *,
    parameter: str,
    default_value: Any,
) -> pd.DataFrame:
    """Select each target parameter using strictly earlier evaluation seasons."""

    if parameter not in {"week1_weight", "availability_strategy"}:
        raise ValueError("parameter must be week1_weight or availability_strategy")
    selections = []
    for season in sorted(evaluation_seasons):
        for position in week1_research.POSITIONS:
            earlier = grid_results.loc[
                grid_results["season"].lt(season)
                & grid_results["position"].eq(position)
            ]
            if earlier.empty:
                selected = default_value
                score = float("nan")
                calibration = True
                through = None
            else:
                candidates = []
                for value, candidate_rows in earlier.groupby(parameter, dropna=False):
                    candidates.append((value, _selection_objective(candidate_rows)))
                # Prefer the default on an exact evidence tie, then stable text order.
                selected, score = max(
                    candidates,
                    key=lambda item: (
                        item[1],
                        item[0] == default_value,
                        str(item[0]),
                    ),
                )
                calibration = False
                through = int(earlier["season"].max())
            selections.append(
                {
                    "season": int(season),
                    "position": position,
                    f"selected_{parameter}": selected,
                    "selection_score": score,
                    "selected_using_through_season": through,
                    "calibration": calibration,
                }
            )
    return pd.DataFrame(selections)


def evaluate_walk_forward_selection(
    rows: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    parameter: str,
    fixed_week1_weight: float = DEFAULT_WEEK1_WEIGHT,
    fixed_availability: str = DEFAULT_AVAILABILITY,
) -> pd.DataFrame:
    """Score target seasons with parameters selected only from earlier seasons."""

    selected_column = f"selected_{parameter}"
    merged = rows.merge(
        selections[
            [
                "season",
                "position",
                selected_column,
                "selected_using_through_season",
                "calibration",
            ]
        ],
        on=["season", "position"],
        how="left",
        validate="many_to_one",
    )
    scored_groups = []
    result_rows = []
    for (season, position), group in merged.groupby(["season", "position"]):
        selected = group[selected_column].iloc[0]
        weight = float(selected) if parameter == "week1_weight" else fixed_week1_weight
        availability = str(selected) if parameter == "availability_strategy" else fixed_availability
        group = group.copy()
        group["_walk_forward_score"] = candidate_scores(group, weight, availability)
        scored_groups.append(group)
        result_rows.append(
            {
                "season": int(season),
                "position": position,
                "approach": (
                    "walk_forward_position_weights"
                    if parameter == "week1_weight"
                    else "walk_forward_availability"
                ),
                "selected_week1_weight": weight,
                "selected_availability_strategy": availability,
                "selected_using_through_season": group[
                    "selected_using_through_season"
                ].iloc[0],
                "calibration": bool(group["calibration"].iloc[0]),
                **_position_candidate_metrics(group, "_walk_forward_score"),
            }
        )
    return pd.DataFrame(result_rows).sort_values(
        ["season", "position"]
    ).reset_index(drop=True)


def fixed_vs_walk_forward_comparison(
    rows: pd.DataFrame, walk_forward_results: pd.DataFrame
) -> pd.DataFrame:
    """Put fixed 40/60 and selected position-specific results side by side."""

    fixed_grid = evaluate_parameter_grid(
        rows, weights=(DEFAULT_WEEK1_WEIGHT,), availability_strategies=(DEFAULT_AVAILABILITY,)
    )
    fixed = fixed_grid.rename(
        columns={
            "rank_correlation": "fixed_rank_correlation",
            "top_n_overlap": "fixed_top_n_overlap",
            "vor_rank_correlation": "fixed_vor_rank_correlation",
        }
    )
    selected = walk_forward_results.rename(
        columns={
            "rank_correlation": "walk_forward_rank_correlation",
            "top_n_overlap": "walk_forward_top_n_overlap",
            "vor_rank_correlation": "walk_forward_vor_rank_correlation",
        }
    )
    columns = [
        "season",
        "position",
        "sample_count",
        "top_n",
        "fixed_rank_correlation",
        "fixed_top_n_overlap",
        "fixed_vor_rank_correlation",
    ]
    selected_columns = [
        "season",
        "position",
        "selected_week1_weight",
        "selected_using_through_season",
        "calibration",
        "walk_forward_rank_correlation",
        "walk_forward_top_n_overlap",
        "walk_forward_vor_rank_correlation",
    ]
    if "selected_availability_strategy" in selected:
        selected_columns.insert(3, "selected_availability_strategy")
    return fixed[columns].merge(
        selected[selected_columns], on=["season", "position"], validate="one_to_one"
    )


def evaluate_rookie_fallback(rows: pd.DataFrame) -> pd.DataFrame:
    """Measure rank behavior and percentile bias for no-prior-season players."""

    results = []
    for (season, position), full_group in rows.groupby(["season", "position"]):
        rookie_mask = full_group["prior_season_fppg"].isna()
        group = full_group.loc[rookie_mask]
        if group.empty:
            continue
        actual_percentile = full_group["actual_season_points"].rank(pct=True).loc[
            group.index
        ]
        predicted_percentile = full_group["hybrid_season_value_score"].rank(
            pct=True
        ).loc[group.index]
        results.append(
            {
                "season": int(season),
                "position": position,
                "new_player_count": len(group),
                "rank_correlation": (
                    float(actual_percentile.corr(predicted_percentile))
                    if len(group) >= 2 else float("nan")
                ),
                "mean_percentile_bias": float(
                    (predicted_percentile - actual_percentile).mean()
                ),
                "median_actual_season_points": float(
                    group["actual_season_points"].median()
                ),
            }
        )
    return pd.DataFrame(results).sort_values(["season", "position"]).reset_index(drop=True)


def summarize_walk_forward_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixed and strict walk-forward metrics after calibration."""

    evaluated = comparison.loc[~comparison["calibration"]]
    metric_columns = [
        "fixed_rank_correlation",
        "walk_forward_rank_correlation",
        "fixed_top_n_overlap",
        "walk_forward_top_n_overlap",
        "fixed_vor_rank_correlation",
        "walk_forward_vor_rank_correlation",
    ]
    by_position = evaluated.groupby("position", as_index=False)[metric_columns].mean()
    overall = {"position": "ALL_POSITIONS"}
    overall.update({column: float(evaluated[column].mean()) for column in metric_columns})
    return pd.concat([by_position, pd.DataFrame([overall])], ignore_index=True)


def build_player_diagnostic(
    player: Mapping[str, Any],
    position_median: float,
    *,
    week1_weight: float = DEFAULT_WEEK1_WEIGHT,
    availability_strategy: str = DEFAULT_AVAILABILITY,
) -> dict[str, Any]:
    """Explain and exactly reconstruct one ranked player's season value."""

    adjusted = adjusted_prior_for_strategy(
        player["week1_projection"],
        player.get("prior_season_fppg"),
        int(player.get("prior_season_games", 0)),
        position_median,
        availability_strategy,
    )
    value = week1_weight * float(player["week1_projection"]) + (1 - week1_weight) * adjusted
    return {
        "player_id": player.get("player_id"),
        "name": player["name"],
        "position": player["position"],
        "week1_projection": float(player["week1_projection"]),
        "prior_season_fppg": player.get("prior_season_fppg"),
        "prior_season_games": int(player.get("prior_season_games", 0)),
        "position_median_fppg": float(position_median),
        "adjusted_prior_fppg": adjusted,
        "week1_weight": week1_weight,
        "prior_weight": 1.0 - week1_weight,
        "season_value_score": value,
        "positional_replacement_level": float(player["replacement_level_score"]),
        "value_over_replacement": float(player["season_value_vor"]),
        "overall_rank": int(player["overall_rank"]),
        "reconstruction_difference": value - float(player["season_value_score"]),
    }


def diagnose_players(
    board: Sequence[Mapping[str, Any]],
    prior_player_stats: pd.DataFrame,
    target_season: int,
    names: Sequence[str],
    league: Mapping[str, Any] = MY_YAHOO_LEAGUE,
) -> pd.DataFrame:
    """Build diagnostics for requested exact player names from a real board."""

    prior = summarize_prior_season(prior_player_stats, target_season, league["scoring"])
    medians = prior.groupby("position")["prior_season_fppg"].median().to_dict()
    by_name = {player["name"]: player for player in board}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError("Players not found on draft board: " + ", ".join(missing))
    return pd.DataFrame(
        [
            build_player_diagnostic(
                by_name[name], medians[by_name[name]["position"]]
            )
            for name in names
        ]
    )


def summarize_best_grid_parameters(
    grid_results: pd.DataFrame, parameter: str
) -> pd.DataFrame:
    """Report retrospective best candidates for interpretation, not deployment."""

    rows = []
    for position, position_rows in grid_results.groupby("position"):
        candidates = []
        for value, candidate_rows in position_rows.groupby(parameter, dropna=False):
            candidates.append((value, _selection_objective(candidate_rows)))
        value, score = max(candidates, key=lambda item: (item[1], str(item[0])))
        rows.append({"position": position, f"best_{parameter}": value, "objective": score})
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def assess_production_recommendation(
    weight_comparison: pd.DataFrame,
    availability_comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Require multi-season, multi-position consistency before promotion."""

    def evidence(comparison: pd.DataFrame) -> dict[str, Any]:
        evaluated = comparison.loc[~comparison["calibration"]].copy()
        evaluated["rank_gain"] = (
            evaluated["walk_forward_rank_correlation"]
            - evaluated["fixed_rank_correlation"]
        )
        evaluated["overlap_gain"] = (
            evaluated["walk_forward_top_n_overlap"]
            - evaluated["fixed_top_n_overlap"]
        )
        by_position = []
        for position, rows in evaluated.groupby("position"):
            needed = len(rows) // 2 + 1
            by_position.append(
                {
                    "position": position,
                    "seasons": len(rows),
                    "rank_wins": int(rows["rank_gain"].gt(0).sum()),
                    "mean_rank_gain": float(rows["rank_gain"].mean()),
                    "mean_overlap_gain": float(rows["overlap_gain"].mean()),
                    "consistent": bool(
                        rows["rank_gain"].gt(0).sum() >= needed
                        and rows["rank_gain"].mean() > 0.01
                        and rows["overlap_gain"].mean() >= 0
                    ),
                }
            )
        details = pd.DataFrame(by_position)
        consistent_positions = int(details["consistent"].sum()) if not details.empty else 0
        catastrophic = bool(
            (evaluated["rank_gain"] < -0.10).any()
            or (evaluated["overlap_gain"] < -0.20).any()
        )
        return {
            "ready": consistent_positions >= 3 and not catastrophic,
            "positions": details,
            "mean_rank_gain": float(evaluated["rank_gain"].mean()),
            "mean_overlap_gain": float(evaluated["overlap_gain"].mean()),
            "catastrophic_regression": catastrophic,
        }

    weight_evidence = evidence(weight_comparison)
    availability_evidence = evidence(availability_comparison)
    if weight_evidence["ready"] and availability_evidence["ready"]:
        choice = 4
        recommendation = "Promote position-specific weights and availability changes."
    elif weight_evidence["ready"]:
        choice = 2
        recommendation = "Promote position-specific weights; keep current availability shrinkage."
    elif availability_evidence["ready"]:
        choice = 3
        recommendation = "Change availability shrinkage; keep universal 40/60 weights."
    elif (
        abs(weight_evidence["mean_rank_gain"]) <= 0.01
        and abs(availability_evidence["mean_rank_gain"]) <= 0.01
    ):
        choice = 1
        recommendation = "Keep universal 40/60 with linear games/17 shrinkage."
    else:
        choice = 5
        recommendation = "More research required; gains are not sufficiently consistent."
    return {
        "choice": choice,
        "recommendation": recommendation,
        "weight_evidence": weight_evidence,
        "availability_evidence": availability_evidence,
    }


def _overlap(actual: pd.Series, predicted: pd.Series, count: int) -> float:
    size = min(count, len(actual))
    if size == 0:
        return float("nan")
    actual_top = set(actual.nlargest(size).index)
    predicted_top = set(predicted.nlargest(size).index)
    return len(actual_top & predicted_top) / size


def _vor_map(
    frame: pd.DataFrame, score_column: str, league: Mapping[str, Any]
) -> dict[str, float]:
    players = [
        {
            "player_id": row["player_id"],
            "name": str(row["name"]),
            "position": row["position"],
            "projection": float(row[score_column]),
        }
        for _, row in frame.iterrows()
    ]
    return {
        str(player["player_id"]): player["value_over_replacement"]
        for player in build_league_rankings(players, league)
    }


def evaluate_preseason_rows(
    rows: pd.DataFrame,
    league: Mapping[str, Any] = MY_YAHOO_LEAGUE,
) -> pd.DataFrame:
    """Report season/position ranking, hit-rate, overlap, and VOR metrics."""

    results: list[dict[str, Any]] = []
    for season, season_rows in rows.groupby("season"):
        actual_vor = _vor_map(season_rows, "actual_season_points", league)
        for approach in APPROACHES:
            score_column = f"{approach}_score"
            predicted_vor = _vor_map(season_rows, score_column, league)
            for position in (*week1_research.POSITIONS, "OVERALL"):
                group = (
                    season_rows
                    if position == "OVERALL"
                    else season_rows.loc[season_rows["position"].eq(position)]
                ).copy()
                if group.empty:
                    continue
                actual = group.set_index(group["player_id"].astype(str))["actual_season_points"]
                predicted = group.set_index(group["player_id"].astype(str))[score_column]
                common = actual.index.intersection(predicted.index)
                actual = actual.loc[common]
                predicted = predicted.loc[common]
                actual_values = pd.Series(
                    [actual_vor[player_id] for player_id in common], index=common
                )
                predicted_values = pd.Series(
                    [predicted_vor[player_id] for player_id in common], index=common
                )
                results.append(
                    {
                        "season": int(season),
                        "approach": approach,
                        "position": position,
                        "sample_count": len(group),
                        "rank_correlation": float(actual.rank().corr(predicted.rank())),
                        "top12_hit_rate": (
                            _overlap(actual, predicted, 12)
                            if position != "OVERALL" else float("nan")
                        ),
                        "top24_hit_rate": (
                            _overlap(actual, predicted, 24)
                            if position in {"RB", "WR"} else float("nan")
                        ),
                        "top50_overlap": (
                            _overlap(actual, predicted, 50)
                            if position == "OVERALL" else float("nan")
                        ),
                        "vor_rank_correlation": float(
                            actual_values.rank().corr(predicted_values.rank())
                        ),
                    }
                )
    return pd.DataFrame(results).sort_values(
        ["season", "approach", "position"]
    ).reset_index(drop=True)


def summarize_results(season_results: pd.DataFrame) -> pd.DataFrame:
    """Average available metrics across evaluated seasons by approach/position."""

    metric_columns = [
        "sample_count",
        "rank_correlation",
        "top12_hit_rate",
        "top24_hit_rate",
        "top50_overlap",
        "vor_rank_correlation",
    ]
    return (
        season_results.groupby(["approach", "position"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values(["approach", "position"])
        .reset_index(drop=True)
    )


def recommend_approach(aggregate_results: pd.DataFrame) -> dict[str, Any]:
    """Recommend from the league-wide draft-board objectives.

    Overall rank correlation receives half the weight; top-50 overlap and
    league-aware VOR rank correlation receive one quarter each. Position-level
    hit rates remain diagnostic and are not duplicated in this selection score.
    """

    scores = {}
    overall = aggregate_results.loc[aggregate_results["position"].eq("OVERALL")]
    for _, row in overall.iterrows():
        scores[row["approach"]] = float(
            0.50 * row["rank_correlation"]
            + 0.25 * row["top50_overlap"]
            + 0.25 * row["vor_rank_correlation"]
        )
    if not scores:
        raise ValueError("aggregate_results has no OVERALL rows to recommend from")
    recommended = max(scores, key=lambda name: (scores[name], name))
    return {
        "approach": recommended,
        "evidence_score": scores[recommended],
        "scores": scores,
        "formula": (
            "40% Week 1 projection + 60% availability-adjusted prior-season FPPG"
            if recommended == "hybrid_season_value"
            else recommended.replace("_", " ")
        ),
    }


def load_historical_inputs(seasons: Sequence[int]) -> dict[str, pd.DataFrame]:
    """Load only the datasets required to reconstruct historical Week 1 signals."""

    requested = list(seasons)
    return {
        "player_stats": week1_research.nflverse.load_player_stats(requested),
        "schedules": week1_research.nflverse.load_schedules(requested),
        "snap_counts": week1_research.nflverse.load_snap_counts(requested),
        "injuries": week1_research.nflverse.load_injuries(requested),
        "ff_opportunity": week1_research.nflverse.load_ff_opportunity(requested),
    }


def run_evaluation(
    evaluation_seasons: Sequence[int] = DEFAULT_EVALUATION_SEASONS,
) -> dict[str, Any]:
    """Run walk-forward Week 1 inference and compare season-value approaches."""

    history_seasons = list(
        range(min(evaluation_seasons) - 1, max(evaluation_seasons) + 1)
    )
    raw = load_historical_inputs(history_seasons)
    feature_sets = week1_research.build_initialization_feature_sets(raw)
    _, predictions = week1_research.evaluate_week1_initialization(
        feature_sets,
        evaluation_seasons,
        approaches=(week1_research.SHADOW_APPROACH,),
    )
    rows = build_preseason_evaluation_rows(
        predictions, raw["player_stats"], evaluation_seasons
    )
    season_results = evaluate_preseason_rows(rows)
    aggregate = summarize_results(season_results)
    weight_grid = evaluate_parameter_grid(rows)
    weight_selections = select_walk_forward_parameters(
        weight_grid,
        evaluation_seasons,
        parameter="week1_weight",
        default_value=DEFAULT_WEEK1_WEIGHT,
    )
    walk_forward_weights = evaluate_walk_forward_selection(
        rows, weight_selections, parameter="week1_weight"
    )
    weight_comparison = fixed_vs_walk_forward_comparison(
        rows, walk_forward_weights
    )
    availability_grid = evaluate_parameter_grid(
        rows,
        weights=(DEFAULT_WEEK1_WEIGHT,),
        availability_strategies=tuple(AVAILABILITY_OPTIONS),
    )
    availability_selections = select_walk_forward_parameters(
        availability_grid,
        evaluation_seasons,
        parameter="availability_strategy",
        default_value=DEFAULT_AVAILABILITY,
    )
    walk_forward_availability = evaluate_walk_forward_selection(
        rows, availability_selections, parameter="availability_strategy"
    )
    availability_comparison = fixed_vs_walk_forward_comparison(
        rows, walk_forward_availability
    )
    coverage = (
        rows.groupby(["season", "position"], as_index=False)
        .agg(
            players=("player_id", "count"),
            returning_players=("prior_season_fppg", "count"),
            median_fallbacks=("used_prior_median_fallback", "sum"),
        )
    )
    return {
        "season_results": season_results,
        "aggregate_results": aggregate,
        "coverage": coverage,
        "weight_grid": weight_grid,
        "best_weights": summarize_best_grid_parameters(
            weight_grid, "week1_weight"
        ),
        "weight_selections": weight_selections,
        "walk_forward_weight_results": walk_forward_weights,
        "weight_comparison": weight_comparison,
        "weight_aggregate_comparison": summarize_walk_forward_comparison(
            weight_comparison
        ),
        "availability_grid": availability_grid,
        "best_availability": summarize_best_grid_parameters(
            availability_grid, "availability_strategy"
        ),
        "availability_selections": availability_selections,
        "walk_forward_availability_results": walk_forward_availability,
        "availability_comparison": availability_comparison,
        "availability_aggregate_comparison": summarize_walk_forward_comparison(
            availability_comparison
        ),
        "rookie_results": evaluate_rookie_fallback(rows),
        "production_recommendation": assess_production_recommendation(
            weight_comparison, availability_comparison
        ),
        "recommendation": recommend_approach(aggregate),
    }


def print_report(results: Mapping[str, Any]) -> None:
    print("Season-by-season draft signal evaluation")
    print(results["season_results"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nAggregate results")
    print(results["aggregate_results"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPrior-season coverage")
    print(results["coverage"].to_string(index=False))
    if "best_weights" in results:
        print("\nRetrospective weight-grid winners (diagnostic only)")
        print(results["best_weights"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nStrict walk-forward selected weights and results")
        print(results["weight_comparison"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nAggregate fixed 40/60 vs walk-forward weights (excluding calibration)")
        print(results["weight_aggregate_comparison"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nRetrospective availability winners (diagnostic only)")
        print(results["best_availability"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nStrict walk-forward availability results")
        print(results["availability_comparison"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nAggregate fixed vs walk-forward availability (excluding calibration)")
        print(results["availability_aggregate_comparison"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nRookie/new-player fallback results")
        print(results["rookie_results"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nProduction recommendation")
        print(results["production_recommendation"]["recommendation"])
    recommendation = results["recommendation"]
    print("\nRecommendation")
    print(f"{recommendation['approach']}: {recommendation['formula']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-seasons",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVALUATION_SEASONS),
    )
    args = parser.parse_args(argv)
    try:
        print_report(run_evaluation(args.evaluation_seasons))
    except Exception as exc:
        print(f"Season-value evaluation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
