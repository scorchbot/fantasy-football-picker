"""Data access adapters for the fantasy football picker."""

from .nflverse import (
    NFLVerseDataError,
    load_current_week_inputs,
    load_depth_charts,
    load_ff_opportunity,
    load_injuries,
    load_player_stats,
    load_schedules,
    load_snap_counts,
    load_weekly_rosters,
)

__all__ = [
    "NFLVerseDataError",
    "load_current_week_inputs",
    "load_depth_charts",
    "load_ff_opportunity",
    "load_injuries",
    "load_player_stats",
    "load_schedules",
    "load_snap_counts",
    "load_weekly_rosters",
]
