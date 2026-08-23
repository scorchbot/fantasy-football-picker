"""Pandas-facing adapter for raw nflverse data loaded by nflreadpy."""

from importlib import import_module
import re
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


def _is_http_not_found(error: Exception) -> bool:
    """Return whether an exception chain represents an explicit HTTP 404."""

    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_codes = (
            getattr(current, "status_code", None),
            getattr(current, "code", None),
            getattr(response, "status_code", None),
        )
        if 404 in status_codes:
            return True

        message = str(current)
        if re.search(
            r"(?:HTTP(?: Error)? 404|404 Not Found|response status 404)",
            message,
            flags=re.IGNORECASE,
        ):
            return True
        current = current.__cause__ or current.__context__

    return False


def _is_next_season_unavailable(error: Exception, seasons: Seasons) -> bool:
    """Recognize nflreadpy's narrow preseason current-season validation error."""

    requested = [seasons] if isinstance(seasons, int) else seasons
    if len(requested) != 1:
        return False

    match = re.search(
        r"Season must be between \d+ and (\d+)",
        str(error),
        flags=re.IGNORECASE,
    )
    return bool(match and requested[0] == int(match.group(1)) + 1)


def _empty_pandas_frame() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise NFLVerseDataError(
            "pandas is required to represent an unpublished nflverse dataset."
        ) from exc
    return pd.DataFrame()


def _load_dataset(
    loader_name: str,
    seasons: Seasons,
    *,
    allow_missing: bool = False,
) -> Any:
    seasons = _validate_seasons(seasons)
    nflreadpy = _get_nflreadpy()

    try:
        loader: Callable[..., Any] = getattr(nflreadpy, loader_name)
        result = loader(seasons)
    except Exception as exc:
        if allow_missing and (
            _is_http_not_found(exc)
            or _is_next_season_unavailable(exc, seasons)
        ):
            return _empty_pandas_frame()
        raise NFLVerseDataError(
            f"nflreadpy.{loader_name} failed for seasons {seasons}: {exc}"
        ) from exc

    return _to_pandas(result)


def load_schedules(seasons: Seasons) -> Any:
    """Load raw schedules as a pandas DataFrame."""

    return _load_dataset("load_schedules", seasons)


def load_player_stats(seasons: Seasons, *, allow_missing: bool = False) -> Any:
    """Load raw weekly player game stats as a pandas DataFrame."""

    return _load_dataset(
        "load_player_stats", seasons, allow_missing=allow_missing
    )


def load_weekly_rosters(
    seasons: Seasons,
    *,
    allow_preseason_fallback: bool = False,
) -> Any:
    """Load raw weekly rosters as a pandas DataFrame."""

    seasons = _validate_seasons(seasons)
    nflreadpy = _get_nflreadpy()
    try:
        result = nflreadpy.load_rosters_weekly(seasons)
    except Exception as exc:
        if not (
            allow_preseason_fallback
            and _is_next_season_unavailable(exc, seasons)
        ):
            raise NFLVerseDataError(
                f"nflreadpy.load_rosters_weekly failed for seasons {seasons}: {exc}"
            ) from exc
        try:
            result = nflreadpy.load_rosters(seasons)
        except Exception as fallback_exc:
            raise NFLVerseDataError(
                "nflreadpy preseason roster fallback failed for seasons "
                f"{seasons}: {fallback_exc}"
            ) from fallback_exc
    return _to_pandas(result)


def load_injuries(seasons: Seasons, *, allow_missing: bool = False) -> Any:
    """Load raw injury reports as a pandas DataFrame."""

    return _load_dataset(
        "load_injuries", seasons, allow_missing=allow_missing
    )


def load_snap_counts(seasons: Seasons, *, allow_missing: bool = False) -> Any:
    """Load raw snap counts as a pandas DataFrame."""

    return _load_dataset(
        "load_snap_counts", seasons, allow_missing=allow_missing
    )


def load_depth_charts(seasons: Seasons, *, allow_missing: bool = False) -> Any:
    """Load raw depth charts as a pandas DataFrame."""

    return _load_dataset(
        "load_depth_charts", seasons, allow_missing=allow_missing
    )


def load_ff_opportunity(seasons: Seasons, *, allow_missing: bool = False) -> Any:
    """Load raw weekly fantasy opportunity data as a pandas DataFrame."""

    return _load_dataset(
        "load_ff_opportunity", seasons, allow_missing=allow_missing
    )


def load_current_week_inputs(
    season: int, *, include_prior_season_history: bool = False
) -> dict[str, Any]:
    """Load unfiltered raw inputs for future current-week feature construction."""

    if isinstance(season, bool) or not isinstance(season, int):
        raise ValueError("season must be an integer.")

    inputs = {
        "schedules": load_schedules(season),
        "player_stats": load_player_stats(season, allow_missing=True),
        "weekly_rosters": load_weekly_rosters(
            season, allow_preseason_fallback=True
        ),
        "injuries": load_injuries(season, allow_missing=True),
        "snap_counts": load_snap_counts(season, allow_missing=True),
        "depth_charts": load_depth_charts(season, allow_missing=True),
        "ff_opportunity": load_ff_opportunity(season, allow_missing=True),
    }
    if include_prior_season_history:
        prior_season = season - 1
        inputs.update(
            {
                "prior_player_stats": load_player_stats(prior_season),
                "prior_snap_counts": load_snap_counts(
                    prior_season, allow_missing=True
                ),
                "prior_ff_opportunity": load_ff_opportunity(
                    prior_season, allow_missing=True
                ),
            }
        )
    return inputs
