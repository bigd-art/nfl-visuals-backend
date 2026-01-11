#!/usr/bin/env python3
"""
Cache warm script (PLAYOFFS ONLY) for Week Posters.

Mapping you requested (your internal week numbers):
  1 = Wild Card
  2 = Divisional
  3 = Conference Championship
  4 = Pro Bowl
  5 = Super Bowl

This script assumes your backend already has a "week poster" generation endpoint
that:
  - generates the PNG (or returns its URL)
  - uploads to Supabase storage
  - writes/returns cached URL so future requests are instant

YOU MUST SET:
  API_BASE_URL  (e.g. https://your-render-service.onrender.com)
OPTIONAL:
  API_KEY       (if you protect the endpoint)
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

# Your backend base URL (Render)
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")

# If your backend requires an API key header
API_KEY = os.getenv("API_KEY", "")

# Endpoint path for week poster generation (EDIT THIS if yours differs)
# Common patterns:
#   /generate-week-poster
#   /generate_week_poster
#   /poster/week
ENDPOINT_PATH = os.getenv("WEEK_POSTER_ENDPOINT", "/generate-week-poster")

# Seasons to cache
START_SEASON = int(os.getenv("START_SEASON", "2020"))
END_SEASON = int(os.getenv("END_SEASON", "2025"))  # inclusive

# Playoff weeks to cache (your mapping)
PLAYOFF_WEEKS = [1, 2, 3, 4, 5]

# Network behavior
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))
RETRIES = int(os.getenv("RETRIES", "3"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.5"))

# If your endpoint expects additional fields, add them here (kept empty by default)
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
        # Change header name if your backend uses something else
        headers["X-API-Key"] = API_KEY
    return headers


def post_json_with_retries(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)
            # Try to parse json even on non-200 (helpful error detail)
            try:
                data = r.json()
            except Exception:
                data = {"raw_text": r.text}

            if 200 <= r.status_code < 300:
                return True, data, ""
            else:
                last_err = f"HTTP {r.status_code}: {json.dumps(data)[:500]}"
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

    print("=== Cache Warm: PLAYOFF Week Posters ===")
    print(f"API: {url}")
    print(f"Seasons: {START_SEASON}..{END_SEASON} (inclusive)")
    print(f"Playoff weeks: {PLAYOFF_WEEKS} (1=WC,2=DIV,3=CONF,4=PB,5=SB)")
    print("")

    for season in range(START_SEASON, END_SEASON + 1):
        for week in PLAYOFF_WEEKS:
            total += 1
            label = pretty_label(week)

            # Payload: keep it aligned with your existing week-poster endpoint contract.
            # If your backend expects `season_type` for playoffs, set it here.
            # Common values: 2=regular, 3=postseason in many APIs.
            payload = {
                "season": season,
                "week": week,
                "season_type": 3,  # postseason
                **EXTRA_PAYLOAD,
            }

            print(f"[{season} | Playoffs Week {week} - {label}] Generating + caching...")

            success, data, err = post_json_with_retries(url, payload, headers)

            if success:
                ok += 1
                # Try to print whatever URL field your endpoint returns
                returned_url = None
                if isinstance(data, dict):
                    for k in ["url", "public_url", "image_url", "poster_url", "cached_url"]:
                        if k in data and isinstance(data[k], str) and data[k].startswith("http"):
                            returned_url = data[k]
                            break

                if returned_url:
                    print(f"  ✅ OK: {returned_url}")
                else:
                    print(f"  ✅ OK (no url field found in response keys={list(data.keys()) if isinstance(data, dict) else 'n/a'})")
            else:
                fail += 1
                print(f"  ❌ FAIL: {err}")

            time.sleep(SLEEP_BETWEEN_CALLS)

    print("")
    print("=== Done ===")
    print(f"Total: {total} | Success: {ok} | Failed: {fail}")

    # Exit non-zero if anything failed (useful in CI)
    if fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()

