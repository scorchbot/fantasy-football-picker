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
            hybrid, _, fallback = calculate_season_value_score(
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
        "recommendation": recommend_approach(aggregate),
    }


def print_report(results: Mapping[str, Any]) -> None:
    print("Season-by-season draft signal evaluation")
    print(results["season_results"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nAggregate results")
    print(results["aggregate_results"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPrior-season coverage")
    print(results["coverage"].to_string(index=False))
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
