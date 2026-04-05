#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from app.services.storage_supabase import upload_file_return_url
from app.scripts.league_matchups_poster import SCOREBOARD_URL, get_json, parse_week_games, make_poster
from app.scripts.season_auto import candidate_regular_seasons, resolve_first_valid, maybe_resolve_week


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def season_has_regular_data(season: int) -> bool:
    for week in range(1, 19):
        try:
            data = get_json(SCOREBOARD_URL.format(year=season, week=week))
            games = parse_week_games(data)
            if games:
                return True
        except Exception:
            continue
    return False


def latest_regular_week(season: int) -> int:
    week = maybe_resolve_week(
        range(1, 19),
        lambda w: bool(parse_week_games(get_json(SCOREBOARD_URL.format(year=season, week=w))))
    )
    if week is None:
        raise RuntimeError(f"Could not resolve a regular-season week for {season}.")
    return week


def publish_league_matchups(keep_versioned: bool = False) -> dict:
    season = resolve_first_valid(candidate_regular_seasons(), season_has_regular_data)
    week = latest_regular_week(season)

    data = get_json(SCOREBOARD_URL.format(year=season, week=week))
    games = parse_week_games(data)

    local_png = f"/tmp/league_matchups_{season}_week_{week}.png"
    make_poster(season, week, games, local_png)

    current_key = "league_matchups/current.png"
    image_url = upload_file_return_url(local_png, current_key)

    payload = {
        "season": season,
        "week": week,
        "game_count": len(games),
        "image_url": image_url,
        "storage_key": current_key,
    }

    local_json = f"/tmp/league_matchups_{season}_week_{week}.json"
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    payload["metadata_url"] = upload_file_return_url(local_json, "league_matchups/current.json")

    if keep_versioned:
        prefix = f"league_matchups/history/{season}/week{week:02d}"
        payload["versioned_image_url"] = upload_file_return_url(local_png, f"{prefix}/poster.png")
        payload["versioned_metadata_url"] = upload_file_return_url(local_json, f"{prefix}/metadata.json")

    return payload


def get_current_league_matchups_payload() -> dict:
    return {
        "image_url": public_storage_url("league_matchups/current.png"),
        "metadata_url": public_storage_url("league_matchups/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(json.dumps(publish_league_matchups(keep_versioned=args.keep_versioned), indent=2))


if __name__ == "__main__":
    main()
