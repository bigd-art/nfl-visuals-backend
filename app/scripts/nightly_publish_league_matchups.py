#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.storage_supabase import upload_file_return_url
from app.scripts.league_matchups_poster import (
    SCOREBOARD_URL,
    get_json,
    parse_week_games,
    make_poster,
)


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def resolve_current_nfl_season() -> int:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return now_et.year if now_et.month >= 3 else now_et.year - 1


def resolve_current_regular_week(season: int) -> int:
    latest_found = None
    for week in range(1, 19):
        try:
            url = SCOREBOARD_URL.format(year=season, week=week)
            data = get_json(url)
            games = parse_week_games(data)
            if games:
                latest_found = week
        except Exception:
            continue

    if latest_found is None:
        raise RuntimeError(f"Could not resolve a regular-season week for {season}.")
    return latest_found


def publish_league_matchups(season: int = None, week: int = None, keep_versioned: bool = False):
    if season is None:
        season = resolve_current_nfl_season()
    if week is None:
        week = resolve_current_regular_week(season)

    url = SCOREBOARD_URL.format(year=season, week=week)
    data = get_json(url)
    games = parse_week_games(data)

    local_png = f"/tmp/league_matchups_{season}_week_{week}.png"
    make_poster(season, week, games, local_png)

    current_key = "league_matchups/current.png"
    image_url = upload_file_return_url(local_png, current_key)

    payload = {
        "season": season,
        "week": week,
        "image_url": image_url,
        "storage_key": current_key,
        "game_count": len(games),
    }

    local_json = f"/tmp/league_matchups_{season}_week_{week}.json"
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    meta_key = "league_matchups/current.json"
    metadata_url = upload_file_return_url(local_json, meta_key)
    payload["metadata_url"] = metadata_url

    if keep_versioned:
        prefix = f"league_matchups/history/{season}/week{week:02d}"
        payload["versioned_image_url"] = upload_file_return_url(local_png, f"{prefix}/poster.png")
        payload["versioned_metadata_url"] = upload_file_return_url(local_json, f"{prefix}/metadata.json")

    return payload


def get_current_league_matchups_payload():
    return {
        "image_url": public_storage_url("league_matchups/current.png"),
        "metadata_url": public_storage_url("league_matchups/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_league_matchups(
        season=args.season,
        week=args.week,
        keep_versioned=args.keep_versioned,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
