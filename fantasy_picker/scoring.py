"""Configuration-driven offensive fantasy scoring."""

import math
from numbers import Real
from typing import Any, Mapping


YAHOO_OFFENSIVE_SCORING = {
    # Passing
    "completion_points": -0.1,
    "passing_yards_per_point": 30,
    "passing_td_points": 5,
    "interception_points": -2,
    "passing_bonus_300": 2,
    "passing_bonus_400": 3,
    "passing_bonus_500": 4,
    # Rushing
    "rushing_yards_per_point": 10,
    "rushing_td_points": 6,
    "rushing_bonus_150": 2,
    "rushing_bonus_200": 3,
    "rushing_bonus_300": 5,
    # Receiving
    "reception_points": 0.5,
    "receiving_yards_per_point": 10,
    "receiving_td_points": 6,
    "receiving_bonus_150": 1,
    "receiving_bonus_200": 2,
    "receiving_bonus_250": 3,
    # Miscellaneous offense
    "two_point_conversion_points": 2,
    "fumble_lost_points": -2,
}

# Retain the names used by the current and earlier generic scoring cells.
MY_YAHOO_SCORING = YAHOO_OFFENSIVE_SCORING
DEFAULT_LEAGUE_SETTINGS = YAHOO_OFFENSIVE_SCORING


def get_stat(stats: Mapping[str, Any], stat_name: str) -> Real:
    """Return a stat value, treating missing and null values as zero."""

    value = stats.get(stat_name, 0)
    if value is None or (isinstance(value, Real) and math.isnan(value)):
        return 0
    return value


def score_player(stats: Mapping[str, Any], settings: Mapping[str, Real]) -> float:
    """Score one projected or actual offensive stat line using ``settings``."""

    points = 0.0

    completions = get_stat(stats, "completions")
    passing_yards = get_stat(stats, "passing_yards")
    passing_tds = get_stat(stats, "passing_tds")
    interceptions = get_stat(stats, "interceptions")

    points += completions * settings["completion_points"]
    points += passing_yards / settings["passing_yards_per_point"]
    points += passing_tds * settings["passing_td_points"]
    points += interceptions * settings["interception_points"]

    if passing_yards >= 500:
        points += settings["passing_bonus_500"]
    elif passing_yards >= 400:
        points += settings["passing_bonus_400"]
    elif passing_yards >= 300:
        points += settings["passing_bonus_300"]

    rushing_yards = get_stat(stats, "rushing_yards")
    rushing_tds = get_stat(stats, "rushing_tds")

    points += rushing_yards / settings["rushing_yards_per_point"]
    points += rushing_tds * settings["rushing_td_points"]

    if rushing_yards >= 300:
        points += settings["rushing_bonus_300"]
    elif rushing_yards >= 200:
        points += settings["rushing_bonus_200"]
    elif rushing_yards >= 150:
        points += settings["rushing_bonus_150"]

    receptions = get_stat(stats, "receptions")
    receiving_yards = get_stat(stats, "receiving_yards")
    receiving_tds = get_stat(stats, "receiving_tds")

    points += receptions * settings["reception_points"]
    points += receiving_yards / settings["receiving_yards_per_point"]
    points += receiving_tds * settings["receiving_td_points"]

    if receiving_yards >= 250:
        points += settings["receiving_bonus_250"]
    elif receiving_yards >= 200:
        points += settings["receiving_bonus_200"]
    elif receiving_yards >= 150:
        points += settings["receiving_bonus_150"]

    two_point_conversions = (
        get_stat(stats, "passing_2pt_conversions")
        + get_stat(stats, "rushing_2pt_conversions")
        + get_stat(stats, "receiving_2pt_conversions")
    )
    points += two_point_conversions * settings["two_point_conversion_points"]

    fumbles_lost = (
        get_stat(stats, "rushing_fumbles_lost")
        + get_stat(stats, "receiving_fumbles_lost")
        + get_stat(stats, "sack_fumbles_lost")
    )
    points += fumbles_lost * settings["fumble_lost_points"]

    return round(points, 2)
