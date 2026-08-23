"""Pregame feature construction from already-loaded pandas DataFrames."""

from typing import Iterable, Optional, Sequence

import pandas as pd

from .scoring import MY_YAHOO_SCORING


PLAYER_ROLLING_COLUMNS = (
    "my_fantasy_points",
    "carries",
    "targets",
    "receptions",
    "rushing_yards",
    "receiving_yards",
    "target_share",
)

QB_ROLLING_COLUMNS = (
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "passing_air_yards",
    "passing_epa",
    "rushing_tds",
)

OPPORTUNITY_COLUMNS = (
    "my_expected_opportunity_points",
    "rush_yards_gained_exp",
    "rec_yards_gained_exp",
    "receptions_exp",
    "rush_touchdown_exp",
    "rec_touchdown_exp",
    "pass_yards_gained_exp",
    "pass_touchdown_exp",
)

QB_OPPORTUNITY_FEATURES = (
    "my_expected_opportunity_points_last3",
    "pass_yards_gained_exp_last3",
    "pass_touchdown_exp_last3",
)

INJURY_FEATURES = (
    "on_injury_report",
    "questionable",
    "doubtful",
    "out",
    "practice_full",
    "practice_limited",
    "practice_dnp",
)

SCHEDULE_CONTEXT_FEATURES = (
    "opponent_team",
    "is_home",
    "team_spread",
    "total_line",
    "roof",
    "temp",
    "wind",
    "rest_days",
    "starting_qb",
)

PREGAME_FEATURE_NAMES = (
    *(f"{column}_last3" for column in PLAYER_ROLLING_COLUMNS),
    *(f"{column}_last3" for column in QB_ROLLING_COLUMNS),
    "offense_pct_last3",
    "offense_pct_last5",
    "snap_trend",
    "defense_fp_allowed_last3",
    *INJURY_FEATURES,
    *(f"{column}_last3" for column in OPPORTUNITY_COLUMNS),
    *SCHEDULE_CONTEXT_FEATURES,
)

WEEK1_CARRYOVER_FEATURES = (
    *(f"{column}_last3" for column in PLAYER_ROLLING_COLUMNS),
    *(f"{column}_last3" for column in QB_ROLLING_COLUMNS),
    "offense_pct_last3",
    "offense_pct_last5",
    "snap_trend",
    *(f"{column}_last3" for column in OPPORTUNITY_COLUMNS),
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = float("nan")
    return result


def add_prior_game_rolling_features(
    player_games: pd.DataFrame,
    source_columns: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Add player-season rolling means based strictly on prior games."""

    _require_columns(player_games, ("player_id", "season", "week"), "player_games")
    result = _ensure_columns(player_games, source_columns)
    result = result.sort_values(["player_id", "season", "week"]).copy()

    for column in source_columns:
        result[f"{column}_last{window}"] = (
            result.groupby(["player_id", "season"])[column]
            .transform(
                lambda values: values.shift(1).rolling(
                    window, min_periods=1
                ).mean()
            )
        )

    return result


def add_player_rolling_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's core and QB-specific prior-three-game features."""

    columns = PLAYER_ROLLING_COLUMNS + QB_ROLLING_COLUMNS
    return add_prior_game_rolling_features(player_games, columns, window=3)


def attach_snap_counts(
    player_games: pd.DataFrame, snap_counts: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Attach raw offensive snap percentage using identifiers when available."""

    result = player_games.drop(
        columns=[column for column in ("offense_pct", "offense_snaps") if column in player_games]
    ).copy()

    if snap_counts is None or snap_counts.empty:
        return _ensure_columns(result, ("offense_pct", "offense_snaps"))

    if "player_id" in snap_counts.columns:
        keys = ["season", "week", "player_id"]
        _require_columns(result, keys, "player_games")
        _require_columns(snap_counts, keys + ["offense_pct"], "snap_counts")
        value_columns = [
            column for column in ("offense_snaps", "offense_pct") if column in snap_counts
        ]
        return result.merge(snap_counts[keys + value_columns], on=keys, how="left")

    left_keys = [
        "season",
        "week",
        "player_display_name",
        "recent_team",
        "opponent_team",
    ]
    right_keys = ["season", "week", "player", "team", "opponent"]
    _require_columns(result, left_keys, "player_games")
    _require_columns(snap_counts, right_keys + ["offense_pct"], "snap_counts")

    snap = snap_counts.rename(
        columns={
            "player": "player_display_name",
            "team": "recent_team",
            "opponent": "opponent_team",
        }
    )
    value_columns = [
        column for column in ("offense_snaps", "offense_pct") if column in snap
    ]
    return result.merge(snap[left_keys + value_columns], on=left_keys, how="left")


def add_snap_rolling_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """Add prior-three, prior-five, and trend snap-share features."""

    result = add_prior_game_rolling_features(
        player_games, ("offense_pct",), window=3
    )
    result = add_prior_game_rolling_features(result, ("offense_pct",), window=5)
    result["snap_trend"] = (
        result["offense_pct_last3"] - result["offense_pct_last5"]
    )
    return result


def add_defense_matchup_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """Add prior-three fantasy points allowed by defense and player position."""

    required = (
        "season",
        "week",
        "opponent_team",
        "position",
        "my_fantasy_points",
    )
    _require_columns(player_games, required, "player_games")

    result = player_games.drop(
        columns=[
            column
            for column in ("defense_team", "defense_fp_allowed_last3")
            if column in player_games
        ]
    ).copy()
    defense_games = (
        result.groupby(["season", "week", "opponent_team", "position"])[
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
        defense_games.groupby(["defense_team", "position", "season"])[
            "fantasy_points_allowed"
        ]
        .transform(
            lambda values: values.shift(1).rolling(3, min_periods=1).mean()
        )
    )

    return result.merge(
        defense_games[
            [
                "season",
                "week",
                "defense_team",
                "position",
                "defense_fp_allowed_last3",
            ]
        ],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense_team", "position"],
        how="left",
    )


def prepare_injury_features(injuries: pd.DataFrame) -> pd.DataFrame:
    """Convert notebook injury and practice statuses into binary flags."""

    columns = (
        "season",
        "week",
        "team",
        "gsis_id",
        "report_status",
        "practice_status",
    )
    _require_columns(injuries, columns, "injuries")
    result = injuries[list(columns)].copy()
    result["on_injury_report"] = 1
    result["questionable"] = (result["report_status"] == "Questionable").astype(int)
    result["doubtful"] = (result["report_status"] == "Doubtful").astype(int)
    result["out"] = (result["report_status"] == "Out").astype(int)
    result["practice_full"] = (
        result["practice_status"] == "Full Participation in Practice"
    ).astype(int)
    result["practice_limited"] = (
        result["practice_status"] == "Limited Participation in Practice"
    ).astype(int)
    result["practice_dnp"] = (
        result["practice_status"] == "Did Not Participate In Practice"
    ).astype(int)
    return result.drop_duplicates(subset=["season", "week", "team", "gsis_id"])


def attach_injury_features(
    player_games: pd.DataFrame, injuries: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Attach current pregame injury flags by season, week, team, and player ID."""

    result = player_games.drop(
        columns=[column for column in INJURY_FEATURES if column in player_games]
    ).copy()
    if injuries is None or injuries.empty:
        for column in INJURY_FEATURES:
            result[column] = 0
        return result

    prepared = prepare_injury_features(injuries).rename(
        columns={"team": "recent_team", "gsis_id": "player_id"}
    )
    keys = ["season", "week", "recent_team", "player_id"]
    _require_columns(result, keys, "player_games")
    result = result.merge(prepared[keys + list(INJURY_FEATURES)], on=keys, how="left")
    result[list(INJURY_FEATURES)] = result[list(INJURY_FEATURES)].fillna(0)
    return result


def calculate_expected_opportunity_points(
    opportunity: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the notebook's Yahoo-weighted opportunity formula without bonuses."""

    source_columns = (
        "pass_completions_exp",
        "pass_yards_gained_exp",
        "pass_touchdown_exp",
        "pass_interception_exp",
        "rush_yards_gained_exp",
        "rush_touchdown_exp",
        "receptions_exp",
        "rec_yards_gained_exp",
        "rec_touchdown_exp",
        "pass_two_point_conv_exp",
        "rush_two_point_conv_exp",
        "rec_two_point_conv_exp",
    )
    _require_columns(opportunity, source_columns, "ff_opportunity")
    result = opportunity.copy()
    scoring = MY_YAHOO_SCORING
    result["my_expected_opportunity_points"] = (
        result["pass_completions_exp"] * scoring["completion_points"]
        + result["pass_yards_gained_exp"] / scoring["passing_yards_per_point"]
        + result["pass_touchdown_exp"] * scoring["passing_td_points"]
        + result["pass_interception_exp"] * scoring["interception_points"]
        + result["rush_yards_gained_exp"] / scoring["rushing_yards_per_point"]
        + result["rush_touchdown_exp"] * scoring["rushing_td_points"]
        + result["receptions_exp"] * scoring["reception_points"]
        + result["rec_yards_gained_exp"] / scoring["receiving_yards_per_point"]
        + result["rec_touchdown_exp"] * scoring["receiving_td_points"]
        + (
            result["pass_two_point_conv_exp"]
            + result["rush_two_point_conv_exp"]
            + result["rec_two_point_conv_exp"]
        )
        * scoring["two_point_conversion_points"]
    )
    return result


def attach_opportunity_features(
    player_games: pd.DataFrame, ff_opportunity: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Attach raw opportunity values using the notebook's source identifiers."""

    result = player_games.drop(
        columns=[column for column in OPPORTUNITY_COLUMNS if column in player_games]
    ).copy()
    if ff_opportunity is None or ff_opportunity.empty:
        return _ensure_columns(result, OPPORTUNITY_COLUMNS)

    opportunity = calculate_expected_opportunity_points(ff_opportunity).rename(
        columns={"posteam": "recent_team"}
    )
    keys = ["season", "week", "recent_team", "player_id"]
    _require_columns(result, keys, "player_games")
    _require_columns(opportunity, keys + list(OPPORTUNITY_COLUMNS), "ff_opportunity")
    return result.merge(opportunity[keys + list(OPPORTUNITY_COLUMNS)], on=keys, how="left")


def add_opportunity_rolling_features(player_games: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's prior-three expected-opportunity features."""

    return add_prior_game_rolling_features(
        player_games, OPPORTUNITY_COLUMNS, window=3
    )


def build_team_game_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Create one pregame context row per team, with negative spread for favorites."""

    common = [
        "game_id",
        "season",
        "week",
        "gameday",
        "spread_line",
        "total_line",
        "roof",
        "temp",
        "wind",
    ]
    required = common + [
        "home_team",
        "away_team",
        "home_rest",
        "away_rest",
        "home_qb_name",
        "away_qb_name",
    ]
    _require_columns(schedules, required, "schedules")

    home = schedules[
        common + ["home_team", "away_team", "home_rest", "home_qb_name"]
    ].copy()
    home = home.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent_team",
            "home_rest": "rest_days",
            "home_qb_name": "starting_qb",
        }
    )
    home["is_home"] = True
    home["team_spread"] = -home["spread_line"]

    away = schedules[
        common + ["away_team", "home_team", "away_rest", "away_qb_name"]
    ].copy()
    away = away.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent_team",
            "away_rest": "rest_days",
            "away_qb_name": "starting_qb",
        }
    )
    away["is_home"] = False
    away["team_spread"] = away["spread_line"]

    return pd.concat([home, away], ignore_index=True)


def attach_schedule_context(
    player_games: pd.DataFrame, schedules: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Attach stable team-game pregame context to player-game rows."""

    result = player_games.copy()
    if schedules is None or schedules.empty:
        return _ensure_columns(result, SCHEDULE_CONTEXT_FEATURES)

    context = build_team_game_context(schedules)
    keys = ["season", "week", "recent_team"]
    _require_columns(result, keys, "player_games")
    context = context.rename(columns={"team": "recent_team"})

    context_columns = [
        "opponent_team",
        "is_home",
        "team_spread",
        "total_line",
        "roof",
        "temp",
        "wind",
        "rest_days",
        "starting_qb",
    ]
    result = result.drop(
        columns=[column for column in context_columns if column in result]
    )
    return result.merge(context[keys + context_columns], on=keys, how="left")


def build_pregame_features(
    player_games: pd.DataFrame,
    *,
    schedules: Optional[pd.DataFrame] = None,
    snap_counts: Optional[pd.DataFrame] = None,
    injuries: Optional[pd.DataFrame] = None,
    ff_opportunity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build reusable pregame features without loading data or training models."""

    result = attach_schedule_context(player_games, schedules)
    result = attach_snap_counts(result, snap_counts)
    result = attach_injury_features(result, injuries)
    result = attach_opportunity_features(result, ff_opportunity)
    result = add_player_rolling_features(result)
    result = add_snap_rolling_features(result)
    result = add_defense_matchup_features(result)
    result = add_opportunity_rolling_features(result)
    return result.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _last_completed_game_means(
    player_games: pd.DataFrame,
    source_columns: Sequence[str],
    window: int,
) -> pd.DataFrame:
    """Summarize each player's latest completed rows without crossing seasons."""

    result = _ensure_columns(player_games, source_columns)
    result = result.sort_values(["player_id", "week"])
    summaries = result[["player_id"]].drop_duplicates().set_index("player_id")
    for column in source_columns:
        summaries[f"{column}_last{window}"] = result.groupby("player_id")[
            column
        ].apply(lambda values: values.tail(window).mean())
    return summaries.reset_index()


def build_prior_season_carryover_summary(
    prior_player_games: pd.DataFrame,
    season: int,
    *,
    prior_snap_counts: Optional[pd.DataFrame] = None,
    prior_ff_opportunity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build Week 1 rolling values from the immediately prior regular season."""

    if prior_player_games is None or prior_player_games.empty:
        return pd.DataFrame(columns=["player_id", *WEEK1_CARRYOVER_FEATURES])
    _require_columns(
        prior_player_games,
        ("player_id", "season", "week"),
        "prior_player_games",
    )
    prior = prior_player_games.loc[prior_player_games["season"] == season - 1].copy()
    if "season_type" in prior:
        prior = prior.loc[prior["season_type"] == "REG"].copy()
    if prior.empty:
        return pd.DataFrame(columns=["player_id", *WEEK1_CARRYOVER_FEATURES])

    prior = attach_snap_counts(prior, prior_snap_counts)
    prior = attach_opportunity_features(prior, prior_ff_opportunity)
    rolling_sources = PLAYER_ROLLING_COLUMNS + QB_ROLLING_COLUMNS
    last3 = _last_completed_game_means(prior, rolling_sources, 3)
    snap3 = _last_completed_game_means(prior, ("offense_pct",), 3)
    snap5 = _last_completed_game_means(prior, ("offense_pct",), 5)
    opportunity = _last_completed_game_means(prior, OPPORTUNITY_COLUMNS, 3)
    summary = last3.merge(snap3, on="player_id", how="outer")
    summary = summary.merge(snap5, on="player_id", how="outer")
    summary = summary.merge(opportunity, on="player_id", how="outer")
    summary["snap_trend"] = (
        summary["offense_pct_last3"] - summary["offense_pct_last5"]
    )
    return summary[["player_id", *WEEK1_CARRYOVER_FEATURES]]


def apply_week1_prior_season_carryover(
    features: pd.DataFrame,
    prior_player_games: pd.DataFrame,
    season: int,
    week: int,
    *,
    prior_snap_counts: Optional[pd.DataFrame] = None,
    prior_ff_opportunity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Override Week 1 rolling values for immediate prior-season returners only."""

    result = features.copy(deep=True)
    result["uses_prior_season_history"] = False
    if week != 1:
        return result

    summary = build_prior_season_carryover_summary(
        prior_player_games,
        season,
        prior_snap_counts=prior_snap_counts,
        prior_ff_opportunity=prior_ff_opportunity,
    )
    if summary.empty:
        return result

    summary = summary.set_index("player_id")
    selected = (result["season"] == season) & (result["week"] == 1)
    player_ids = result.loc[selected, "player_id"]
    returning = player_ids.isin(summary.index)
    result.loc[selected, "uses_prior_season_history"] = returning.to_numpy()
    for feature in WEEK1_CARRYOVER_FEATURES:
        result.loc[selected, feature] = player_ids.map(summary[feature]).to_numpy()
    return result


def select_current_week_feature_row(
    features: pd.DataFrame,
    season: int,
    week: int,
    *,
    player_id: Optional[str] = None,
    player_name: Optional[str] = None,
    player_name_column: str = "player_display_name",
) -> pd.Series:
    """Select one already-constructed player feature row for a requested week."""

    if player_id is None and player_name is None:
        raise ValueError("Provide player_id or player_name.")

    selected = features[(features["season"] == season) & (features["week"] == week)]
    if player_id is not None:
        selected = selected[selected["player_id"] == player_id]
    if player_name is not None:
        selected = selected[selected[player_name_column] == player_name]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one current-week feature row, found {len(selected)}."
        )
    return selected.iloc[0].copy()
