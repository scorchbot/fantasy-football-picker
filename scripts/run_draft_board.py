"""Generate a league-aware season-long draft board.

Example::

    python -m scripts.run_draft_board --season 2026 --week 1 --top 100
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

from fantasy_picker.data import nflverse
from fantasy_picker.league import MY_YAHOO_LEAGUE, get_team_count
from fantasy_picker.season_value import build_season_value_rankings
from scripts.run_current_week import DEFAULT_ARTIFACT_PATH, generate_current_week_projections
from scripts.run_rankings import load_league_json


def generate_draft_board(
    season: int,
    week: int,
    *,
    league_path: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], list[str]]:
    """Load current projections and immediate prior-season stats, then rank."""

    if week != 1:
        raise ValueError("This first preseason draft-board version requires --week 1.")
    league = load_league_json(league_path) if league_path else MY_YAHOO_LEAGUE
    projections, warnings = generate_current_week_projections(
        season, week, artifact_path
    )
    prior_stats = nflverse.load_player_stats(season - 1)
    board = build_season_value_rankings(
        projections, prior_stats, league, season
    )
    if not board:
        raise ValueError("No valid QB/RB/WR/TE season values were produced.")
    return board, league, warnings


def print_draft_board(
    board: list[Mapping[str, Any]], league: Mapping[str, Any], top: int
) -> None:
    print(f"League: {league['name']} ({get_team_count(league)} teams)")
    print("Formula: 40% Week 1 projection + 60% availability-adjusted prior-season FPPG")
    print(f"Top {min(top, len(board))} season-long draft values:")
    print("Rank | Tier | Player | Pos | Team | Season Value | Week 1 | Prior FPPG | VOR")
    for player in board[:top]:
        prior = (
            f"{player['prior_season_fppg']:.2f}"
            if player["prior_season_fppg"] is not None
            else "new/fallback"
        )
        print(
            f"{player['overall_rank']:>4} | {player['tier']:>4} | "
            f"{player['name']} | {player['position']} | {player.get('team', '-')} | "
            f"{player['season_value_score']:.2f} | {player['week1_projection']:.2f} | "
            f"{prior} | {player['season_value_vor']:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a season-long fantasy draft board.")
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--league", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be a positive integer")
    try:
        board, league, warnings = generate_draft_board(
            args.season,
            args.week,
            league_path=args.league,
            artifact_path=args.artifact,
        )
    except Exception as exc:
        print(f"FAILURE: draft-board run failed: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print_draft_board(board, league, args.top)
    print("SUCCESS: season-long draft board completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
