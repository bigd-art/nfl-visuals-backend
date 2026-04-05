#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo


def now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def preferred_regular_season() -> int:
    """
    NFL regular season usually begins in September.
    Before September, default to previous season.
    From September onward, prefer current calendar year.
    """
    now = now_et()
    return now.year if now.month >= 9 else now.year - 1


def candidate_regular_seasons() -> list[int]:
    """
    Ordered from most likely to least likely.
    """
    base = preferred_regular_season()
    candidates = [base, base - 1, base + 1]
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_first_valid(candidates: Iterable[int], validator: Callable[[int], bool]) -> int:
    for season in candidates:
        try:
            if validator(season):
                return season
        except Exception:
            continue
    raise RuntimeError("Could not resolve a valid season from candidate list.")


def current_draft_cycle_candidates() -> list[int]:
    """
    For draft content:
    - Jan-Apr: current calendar year draft is usually the active one
    - May-Dec: next calendar year draft is usually the active one
    """
    now = now_et()
    preferred = now.year if now.month <= 4 else now.year + 1
    candidates = [preferred, preferred - 1, preferred + 1]
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def maybe_resolve_week(weeks: Iterable[int], validator: Callable[[int], bool]) -> Optional[int]:
    last_good = None
    for week in weeks:
        try:
            if validator(week):
                last_good = week
        except Exception:
            continue
    return last_good
