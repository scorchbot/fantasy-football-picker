"""Reusable scoring and lineup tools for the fantasy football picker."""

from .lineup import (
    get_slot_config,
    is_player_eligible_for_slot,
    optimize_lineup,
    optimize_lineup_fast,
)
from .scoring import (
    DEFAULT_LEAGUE_SETTINGS,
    MY_YAHOO_SCORING,
    YAHOO_OFFENSIVE_SCORING,
    score_player,
)

__all__ = [
    "DEFAULT_LEAGUE_SETTINGS",
    "MY_YAHOO_SCORING",
    "YAHOO_OFFENSIVE_SCORING",
    "get_slot_config",
    "is_player_eligible_for_slot",
    "optimize_lineup",
    "optimize_lineup_fast",
    "score_player",
]
