#!/usr/bin/env python3
"""
Cache warm script (PLAYOFFS ONLY) for Favorite Team Posters.

Playoff week mapping (your internal week numbers):
  1 = Wild Card
  2 = Divisional
  3 = Conference Championship
  4 = Pro Bowl
  5 = Super Bowl

This script assumes your backend already has a favorite-team poster endpoint
that:
  - generates the PNG (or returns its URL)
  - uploads to Supabase storage
  - writes/returns cached URL so future requests are instant

YOU MUST SET:
  API_BASE_URL  (e.g. https://your-render-service.onrender.com)

OPTIONAL:
  API_KEY       (if you protect the endpoint)

YOU MAY NEED TO EDIT:
  ENDPOINT_PATH (if your endpoint is not /generate-favorite-team)
  TEAM_INPUT_MODE / team field name (team, team_abbr, favorite_team, etc.)
"""

from __future__ import annotations

import os
import sys
import time
import json
from typing import Optional, Dict, Any, List, Tuple

import requests


# -------------------------
# CONFIG (edit if needed)
# -------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

# Favorite team endpoint (you mentioned you already have /generate-favorite-team)
ENDPOINT_PATH = os.getenv("FAVORITE_TEAM_ENDPOINT", "/generate-favorite-team")

# Seasons to cache
START_SEASON = int(os.getenv("START_SEASON", "2020"))
END_SEASON = int(os.getenv("END_SEASON", "2025"))  # inclusive

# Postseason-only weeks (your mapping)
PLAYOFF_WEEKS = [1, 2, 3, 4, 5]

# Teams to generate for:
# Provide comma-separated team abbreviations via env TEAMS="KC,BUF,PHI,..."
# If not provided, this default list will be used (standard 32-team abbreviations).
DEFAULT_TEAMS = [
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
]

TEAMS_ENV = os.getenv("TEAMS", "").strip()
TEAMS: List[str] = [t.strip().upper() for t in TEAMS_ENV.split(",") if t.strip()] if TEAMS_ENV else DEFAULT_TEAMS

# How the endpoint expects the team field.
# Most common: "team" or "team_abbr" or "favorite_team".
TEAM_FIELD_NAME = os.getenv("TEAM_FIELD_NAME", "team")

# Network behavior
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))
RETRIES = int(os.getenv("RETRIES", "3"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.35"))

# Extra payload if your endpoint requires it (leave empty if not)
EXTRA_PAYLOAD: Dict[str, Any] = {}


# -------------------------
# Helpers
# -------------------------

def require_env() -> None:
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL is not set.")
        print("Example: export API_BASE_URL='https://your-service.onrender.com'")
        sys.exit(1)


def build_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        # change if your backend uses a different header name
        headers["X-API-Key"] = API_KEY
    return headers


def post_json_with_retries(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
            try:
                data = r.json()
            except Exception:
                data = {"raw_text": r.text}

            if 200 <= r.status_code < 300:
                return True, data, ""
            else:
                last_err = f"HTTP {r.status_code}: {json.dumps(data)[:600]}"
        except Exception as e:
            last_err = f"Request error: {repr(e)}"

        if attempt < RETRIES:
            time.sleep(1.5 * attempt)

    return False, None, last_err


def pretty_label(week: int) -> str:
    return {
        1: "Wild Card",
        2: "Divisional",
        3: "Conference Championship",
        4: "Pro Bowl",
        5: "Super Bowl",
    }.get(week, f"Week {week}")


def extract_url(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for k in ["url", "public_url", "image_url", "poster_url", "cached_url"]:
        v = data.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    return None


# -------------------------
# Main
# -------------------------

def main() -> None:
    require_env()

    url = f"{API_BASE_URL}{ENDPOINT_PATH}"
    headers = build_headers()

    total = 0
    ok = 0
    fail = 0

    print("=== Cache Warm: PLAYOFF Favorite Team Posters ===")
    print(f"API: {url}")
    print(f"Seasons: {START_SEASON}..{END_SEASON} (inclusive)")
    print(f"Playoff weeks: {PLAYOFF_WEEKS} (1=WC,2=DIV,3=CONF,4=PB,5=SB)")
    print(f"Teams ({len(TEAMS)}): {', '.join(TEAMS)}")
    print(f"Team field name: {TEAM_FIELD_NAME}")
    print("")

    for season in range(START_SEASON, END_SEASON + 1):
        for week in PLAYOFF_WEEKS:
            label = pretty_label(week)

            for team in TEAMS:
                total += 1

                payload = {
                    "season": season,
                    "week": week,
                    "season_type": 3,  # postseason
                    TEAM_FIELD_NAME: team,
                    **EXTRA_PAYLOAD,
                }

                print(f"[{season} | Playoffs Week {week} - {label} | {team}] Generating + caching...")

                success, data, err = post_json_with_retries(url, payload, headers)

                if success:
                    ok += 1
                    returned_url = extract_url(data)
                    if returned_url:
                        print(f"  ✅ OK: {returned_url}")
                    else:
                        keys = list(data.keys()) if isinstance(data, dict) else []
                        print(f"  ✅ OK (no url field found; response keys={keys})")
                else:
                    fail += 1
                    print(f"  ❌ FAIL: {err}")

                time.sleep(SLEEP_BETWEEN_CALLS)

    print("")
    print("=== Done ===")
    print(f"Total: {total} | Success: {ok} | Failed: {fail}")

    if fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()

