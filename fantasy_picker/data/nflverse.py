"""Pandas-facing adapter for raw nflverse data loaded by nflreadpy."""

from importlib import import_module
from typing import Any, Callable, Union


Seasons = Union[int, list[int]]


class NFLVerseDataError(RuntimeError):
    """Raised when nflreadpy cannot be imported or load a dataset."""


def _validate_seasons(seasons: Seasons) -> Seasons:
    if isinstance(seasons, bool):
        raise ValueError("seasons must be an integer or a nonempty list of integers.")
    if isinstance(seasons, int):
        return seasons
    if (
        isinstance(seasons, list)
        and seasons
        and all(isinstance(season, int) and not isinstance(season, bool) for season in seasons)
    ):
        return seasons
    raise ValueError("seasons must be an integer or a nonempty list of integers.")


def _get_nflreadpy() -> Any:
    try:
        return import_module("nflreadpy")
    except ImportError as exc:
        raise NFLVerseDataError(
            "nflreadpy is required to load nflverse data. Install nflreadpy first."
        ) from exc


def _to_pandas(polars_frame: Any) -> Any:
    """Convert one nflreadpy Polars result without changing its columns."""

    try:
        return polars_frame.to_pandas()
    except Exception as exc:
        raise NFLVerseDataError(
            "nflreadpy returned a result that could not be converted to pandas."
        ) from exc


def _load_dataset(loader_name: str, seasons: Seasons) -> Any:
    seasons = _validate_seasons(seasons)
    nflreadpy = _get_nflreadpy()

    try:
        loader: Callable[..., Any] = getattr(nflreadpy, loader_name)
        result = loader(seasons)
    except Exception as exc:
        raise NFLVerseDataError(
            f"nflreadpy.{loader_name} failed for seasons {seasons}: {exc}"
        ) from exc

    return _to_pandas(result)


def load_schedules(seasons: Seasons) -> Any:
    """Load raw schedules as a pandas DataFrame."""

    return _load_dataset("load_schedules", seasons)


def load_player_stats(seasons: Seasons) -> Any:
    """Load raw weekly player game stats as a pandas DataFrame."""

    return _load_dataset("load_player_stats", seasons)


def load_weekly_rosters(seasons: Seasons) -> Any:
    """Load raw weekly rosters as a pandas DataFrame."""

    return _load_dataset("load_rosters_weekly", seasons)


def load_injuries(seasons: Seasons) -> Any:
    """Load raw injury reports as a pandas DataFrame."""

    return _load_dataset("load_injuries", seasons)


def load_snap_counts(seasons: Seasons) -> Any:
    """Load raw snap counts as a pandas DataFrame."""

    return _load_dataset("load_snap_counts", seasons)


def load_depth_charts(seasons: Seasons) -> Any:
    """Load raw depth charts as a pandas DataFrame."""

    return _load_dataset("load_depth_charts", seasons)


def load_ff_opportunity(seasons: Seasons) -> Any:
    """Load raw weekly fantasy opportunity data as a pandas DataFrame."""

    return _load_dataset("load_ff_opportunity", seasons)


def load_current_week_inputs(season: int) -> dict[str, Any]:
    """Load unfiltered raw inputs for future current-week feature construction."""

    if isinstance(season, bool) or not isinstance(season, int):
        raise ValueError("season must be an integer.")

    return {
        "schedules": load_schedules(season),
        "player_stats": load_player_stats(season),
        "weekly_rosters": load_weekly_rosters(season),
        "injuries": load_injuries(season),
        "snap_counts": load_snap_counts(season),
        "depth_charts": load_depth_charts(season),
        "ff_opportunity": load_ff_opportunity(season),
    }
