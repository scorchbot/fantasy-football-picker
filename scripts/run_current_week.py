"""Run the local current-week projection pipeline with fresh nflverse data.

Example, from the repository root::

    python3 -m scripts.run_current_week --season 2026 --week 1
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from fantasy_picker import current_week, models
from fantasy_picker.data import nflverse
from fantasy_picker.features import (
    INJURY_FEATURES,
    OPPORTUNITY_COLUMNS,
    PREGAME_FEATURE_NAMES,
)


DEFAULT_ARTIFACT_PATH = Path("artifacts/fantasy_models.pkl")

SNAP_FEATURES = {"offense_pct_last3", "offense_pct_last5", "snap_trend"}
OPPORTUNITY_FEATURES = {
    f"{column}_last3" for column in OPPORTUNITY_COLUMNS
}
FEATURE_SOURCE_REQUIREMENTS = {
    "snap_counts": SNAP_FEATURES,
    "injuries": set(INJURY_FEATURES),
    "ff_opportunity": OPPORTUNITY_FEATURES,
}


class CurrentWeekRunError(RuntimeError):
    """Raised when fresh inputs cannot support a truthful projection run."""


def _target_week_count(frame: Any, season: int, week: int) -> int:
    if frame is None or frame.empty or not {"season", "week"}.issubset(frame.columns):
        return 0
    return int(((frame["season"] == season) & (frame["week"] == week)).sum())


def _current_roster_count(frame: Any, season: int, week: int) -> int:
    """Count weekly rows or legitimate preseason seasonal-roster fallback rows."""

    if frame is None or frame.empty or "season" not in frame.columns:
        return 0
    selected = frame.loc[frame["season"] == season]
    if "week" in selected.columns:
        selected = selected.loc[selected["week"] == week]
    return len(selected)


def _required_model_features(artifacts: models.ModelArtifacts) -> set[str]:
    return {
        feature
        for position in models.REQUIRED_POSITIONS
        for feature in artifacts.final_features[position]
    }


def validate_fresh_inputs(
    raw_inputs: Mapping[str, Any],
    artifacts: models.ModelArtifacts,
    season: int,
    week: int,
) -> list[str]:
    """Validate current-week context and return non-blocking data warnings."""

    expected_datasets = (
        "schedules",
        "player_stats",
        "weekly_rosters",
        "injuries",
        "snap_counts",
        "depth_charts",
        "ff_opportunity",
    )
    missing = [name for name in expected_datasets if name not in raw_inputs]
    if missing:
        raise CurrentWeekRunError(
            f"Missing nflverse datasets: {', '.join(missing)}"
        )

    empty = [name for name in expected_datasets if raw_inputs[name].empty]
    errors = []
    warnings = []
    pre_week_one_without_stats = week == 1 and "player_stats" in empty
    if pre_week_one_without_stats:
        warnings.append(
            f"{season} player stats are not published yet; continuing with "
            "pregame roster/context data."
        )
    elif "player_stats" in empty:
        errors.append("player_stats is empty; prior-game features are unavailable")

    schedules = raw_inputs["schedules"]
    schedule_count = _target_week_count(schedules, season, week)
    if schedule_count == 0:
        available_week = (
            int(schedules.loc[schedules["season"] == season, "week"].max())
            if not schedules.empty
            and {"season", "week"}.issubset(schedules.columns)
            and (schedules["season"] == season).any()
            else None
        )
        detail = (
            f"; latest available week is {available_week}"
            if available_week is not None
            else "; no schedule data is available for that season"
        )
        errors.append(
            f"missing schedule rows for season {season}, week {week}{detail}"
        )

    rosters = raw_inputs["weekly_rosters"]
    roster_count = _current_roster_count(rosters, season, week)
    if roster_count == 0:
        errors.append(
            f"missing current-week roster rows for season {season}, week {week}"
        )

    required_features = _required_model_features(artifacts)
    constructible = set(PREGAME_FEATURE_NAMES)
    if not raw_inputs["player_stats"].empty:
        constructible.update(raw_inputs["player_stats"].columns)
    unavailable_features = sorted(required_features - constructible)
    if unavailable_features:
        errors.append(
            "required model features cannot be constructed: "
            + ", ".join(unavailable_features)
        )

    for dataset, dataset_features in FEATURE_SOURCE_REQUIREMENTS.items():
        affected = sorted(required_features & dataset_features)
        if dataset in empty and affected:
            if pre_week_one_without_stats and dataset in {
                "snap_counts",
                "injuries",
                "ff_opportunity",
            }:
                warnings.append(
                    f"{dataset} is not published yet; using stored model medians "
                    "for unavailable features."
                )
            else:
                errors.append(
                    f"{dataset} is empty; required model features are unavailable: "
                    + ", ".join(affected)
                )

    if errors:
        empty_detail = f" Empty nflverse datasets: {', '.join(empty)}." if empty else ""
        raise CurrentWeekRunError("; ".join(errors) + empty_detail)

    diagnosed = {
        "player_stats",
        *(
            dataset
            for dataset, dataset_features in FEATURE_SOURCE_REQUIREMENTS.items()
            if required_features & dataset_features
        ),
    }
    optional_empty = [name for name in empty if name not in diagnosed]
    if optional_empty:
        warnings.append(
            f"Empty optional nflverse datasets: {', '.join(optional_empty)}"
        )
    return warnings


def generate_current_week_projections(
    season: int,
    week: int,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load fresh data and artifacts, validate them, and run existing inference."""

    artifacts = models.load_model_artifacts(artifact_path)
    raw_inputs = nflverse.load_current_week_inputs(season)
    warnings = validate_fresh_inputs(raw_inputs, artifacts, season, week)
    projections = current_week.build_current_week_projections(
        season,
        week,
        raw_inputs,
        artifacts,
    )
    if not projections:
        raise CurrentWeekRunError(
            f"No QB/RB/WR/TE projections were produced for season {season}, week {week}."
        )
    return projections, warnings


def print_projection_summary(projections: list[dict[str, Any]]) -> None:
    """Print projection totals, position counts, and the ten highest projections."""

    counts = Counter(player["position"] for player in projections)
    print(f"Players projected: {len(projections)}")
    print(
        "Counts by position: "
        + ", ".join(
            f"{position}={counts.get(position, 0)}"
            for position in models.REQUIRED_POSITIONS
        )
    )
    print("Top 10 projected players:")
    for player in sorted(
        projections, key=lambda item: item["projection"], reverse=True
    )[:10]:
        print(
            f"{player['name']} | {player['position']} | "
            f"{player.get('team', '-')} | {player.get('opponent', '-')} | "
            f"{player['projection']:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project a requested NFL week using fresh nflverse inputs."
    )
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    arguments = parser.parse_args(argv)

    try:
        projections, warnings = generate_current_week_projections(
            arguments.season,
            arguments.week,
        )
    except Exception as exc:
        print(f"FAILURE: current-week projection run failed: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print_projection_summary(projections)
    print("SUCCESS: fresh-data current-week projections completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
