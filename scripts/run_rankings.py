"""Generate a league-aware current-week fantasy draft board.

Example::

    python -m scripts.run_rankings --season 2026 --week 1 --top 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from fantasy_picker.league import (
    MY_YAHOO_LEAGUE,
    get_roster_slots,
    get_team_count,
    validate_league_config,
)
from fantasy_picker.rankings import (
    SUPPORTED_POSITIONS,
    build_league_rankings,
    calculate_replacement_levels,
)
from scripts.run_current_week import (
    DEFAULT_ARTIFACT_PATH,
    generate_current_week_projections,
)


def load_league_json(path: str | Path) -> Mapping[str, Any]:
    """Load and validate an optional JSON league configuration."""

    try:
        with Path(path).open(encoding="utf-8") as league_file:
            league = json.load(league_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load league JSON: {exc}") from exc
    return validate_league_config(league)


def generate_rankings(
    season: int,
    week: int,
    *,
    league_path: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], dict[str, float | None], list[str]]:
    """Generate projections and league-aware rankings for one requested week."""

    league = load_league_json(league_path) if league_path else MY_YAHOO_LEAGUE
    projections, warnings = generate_current_week_projections(
        season, week, artifact_path
    )
    replacement_levels = calculate_replacement_levels(projections, league)
    rankings = build_league_rankings(projections, league)
    if not rankings:
        raise ValueError("No valid QB/RB/WR/TE projections are available to rank.")
    return rankings, league, replacement_levels, warnings


def print_rankings_report(
    rankings: list[Mapping[str, Any]],
    league: Mapping[str, Any],
    replacement_levels: Mapping[str, float | None],
    top: int,
) -> None:
    """Print league assumptions, replacement levels, and a concise draft board."""

    print(f"League: {league['name']}")
    print(f"League teams: {get_team_count(league)}")
    print(
        "Starting roster: "
        + ", ".join(
            f"{slot['slot']}={'/'.join(slot['eligible'])}"
            for slot in get_roster_slots(league)
        )
    )
    print(
        "Replacement levels: "
        + ", ".join(
            f"{position}="
            + (
                f"{replacement_levels[position]:.2f}"
                if replacement_levels[position] is not None
                else "unavailable"
            )
            for position in SUPPORTED_POSITIONS
        )
    )
    print(f"Top {min(top, len(rankings))} league-aware rankings:")
    print("Rank | Tier | Pos Rank | Player | Pos | Team | Opp | Proj | VOR")
    for player in rankings[:top]:
        print(
            f"{player['overall_rank']:>4} | {player['tier']:>4} | "
            f"{player['position']}{player['positional_rank']:<3} | "
            f"{player['name']} | {player['position']} | "
            f"{player.get('team', '-')} | {player.get('opponent', '-')} | "
            f"{player['projection']:.2f} | {player['value_over_replacement']:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate league-aware rankings from current-week projections."
    )
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--league", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be a positive integer")

    try:
        rankings, league, replacement_levels, warnings = generate_rankings(
            args.season,
            args.week,
            league_path=args.league,
            artifact_path=args.artifact,
        )
    except Exception as exc:
        print(f"FAILURE: rankings run failed: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print_rankings_report(rankings, league, replacement_levels, args.top)
    print("SUCCESS: league-aware rankings completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
