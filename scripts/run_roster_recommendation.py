"""Generate an optimal lineup for a provider-neutral roster JSON file.

Example::

    python -m scripts.run_roster_recommendation \
        --season 2026 --week 1 --roster examples/sample_roster.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from fantasy_picker.league import MY_YAHOO_LEAGUE, validate_league_config
from fantasy_picker.roster import (
    RosterMatchError,
    build_roster_recommendation,
    load_roster_json,
)
from scripts.run_current_week import (
    DEFAULT_ARTIFACT_PATH,
    generate_current_week_projections,
)


def load_league_json(path: str | Path) -> Mapping[str, Any]:
    """Load an optional serializable league configuration."""

    try:
        with Path(path).open(encoding="utf-8") as league_file:
            league = json.load(league_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load league JSON: {exc}") from exc
    return validate_league_config(league)


def generate_roster_recommendation(
    season: int,
    week: int,
    roster_path: str | Path,
    *,
    league_path: str | Path | None = None,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> tuple[dict[str, Any], list[str]]:
    """Run current-week inference and optimize one provider-neutral roster."""

    roster = load_roster_json(roster_path)
    league = load_league_json(league_path) if league_path else MY_YAHOO_LEAGUE
    projections, warnings = generate_current_week_projections(
        season, week, artifact_path
    )
    recommendation = build_roster_recommendation(league, roster, projections)
    return recommendation, warnings


def print_recommendation_report(recommendation: Mapping[str, Any]) -> None:
    """Print matched diagnostics, starters, total, and lineup advantages."""

    matched = recommendation["matched_players"]
    unmatched = recommendation["unmatched_players"]
    print(f"League: {recommendation['league_name']}")
    print(f"Matched roster players: {len(matched)}")
    print(f"Unmatched roster players: {len(unmatched)}")
    if unmatched:
        print("Unmatched players:")
        for player in unmatched:
            identifier = player.get("player_id", "no ID")
            print(
                f"- {player['name']} | {player['position']} | "
                f"{player['team']} | {identifier} | {player['reason']}"
            )

    lineup = recommendation["optimal_lineup"]
    if lineup is None:
        print("No complete lineup can be formed from matched projections.")
        return
    print("Optimal starters:")
    for starter in lineup:
        print(
            f"- {starter['slot']}: {starter['name']} | "
            f"{starter['position']} | {starter['projection']:.2f}"
        )
    print(f"Projected team total: {recommendation['projected_total']:.2f}")
    print("Lineup advantages:")
    for decision in recommendation["decisions"]:
        gap = decision["lineup_value_gap"]
        if gap is None:
            detail = "no complete alternate lineup"
        else:
            entering = decision["summary"]["in"]
            replacement = f"; next in: {', '.join(entering)}" if entering else ""
            detail = f"{gap:.2f} projected points{replacement}"
        confidence = (
            f"; confidence: {decision['confidence_label']}"
            if decision["confidence"] is not None else ""
        )
        print(f"- {decision['slot']} {decision['starter']}: {detail}{confidence}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a current-week lineup recommendation from roster JSON."
    )
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--week", required=True, type=int)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--league", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args(argv)

    try:
        recommendation, warnings = generate_roster_recommendation(
            args.season,
            args.week,
            args.roster,
            league_path=args.league,
            artifact_path=args.artifact,
        )
    except RosterMatchError as exc:
        print(f"FAILURE: roster matching is ambiguous: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILURE: roster recommendation failed: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print_recommendation_report(recommendation)
    if recommendation["optimal_lineup"] is None:
        return 1
    print("SUCCESS: roster recommendation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
