#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.generator.week_posters import (
    fetch_url,
    scoreboard_url,
    extract_game_ids_from_scoreboard_html,
)


def now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def preferred_regular_season() -> int:
    now = now_et()
    return now.year if now.month >= 9 else now.year - 1


def regular_week_limit(season: int) -> int:
    return 18 if season >= 2021 else 17


def season_candidates() -> list[int]:
    base = preferred_regular_season()
    out = []
    for s in [base, base - 1, base + 1]:
        if s not in out:
            out.append(s)
    return out


def week_has_games(season: int, week: int) -> bool:
    html = fetch_url(scoreboard_url(season, week, 2))
    game_ids = extract_game_ids_from_scoreboard_html(html)
    return bool(game_ids)


def season_has_regular_games(season: int) -> bool:
    for week in range(1, regular_week_limit(season) + 1):
        try:
            if week_has_games(season, week):
                return True
        except Exception:
            continue
    return False


def resolve_current_regular_season() -> int:
    for season in season_candidates():
        try:
            if season_has_regular_games(season):
                return season
        except Exception:
            continue
    raise RuntimeError("Could not resolve a valid regular season.")


def resolve_latest_regular_week(season: int) -> int:
    latest = None
    for week in range(1, regular_week_limit(season) + 1):
        try:
            if week_has_games(season, week):
                latest = week
        except Exception:
            continue
    if latest is None:
        raise RuntimeError(f"Could not resolve a regular-season week for {season}.")
    return latest


def resolve_current_regular_context() -> tuple[int, int, int]:
    season = resolve_current_regular_season()
    week = resolve_latest_regular_week(season)
    seasontype = 2
    return season, week, seasontype
