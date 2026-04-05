#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from app.generator.week_posters import fetch_url, fetch_summary, scoreboard_url, extract_game_ids_from_scoreboard_html
from app.services.storage_supabase import upload_file_return_url
from app.scripts.current_week_context import resolve_current_regular_context
from app.scripts.favorite_team_posters import generate_favorite_team_poster


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def teams_in_week(season: int, week: int, seasontype: int) -> list[str]:
    html = fetch_url(scoreboard_url(season, week, seasontype))
    game_ids = extract_game_ids_from_scoreboard_html(html)

    teams = set()
    for gid in game_ids:
        try:
            summary = fetch_summary(gid)
            comp = summary["header"]["competitions"][0]
            for c in comp.get("competitors", []):
                abbr = c["team"]["abbreviation"]
                if abbr:
                    teams.add(abbr.upper())
        except Exception:
            continue

    return sorted(teams)


def publish_current_favorite_team_posters(keep_versioned: bool = False) -> dict:
    season, week, seasontype = resolve_current_regular_context()
    teams = teams_in_week(season, week, seasontype)

    if not teams:
        raise RuntimeError("No teams found for current regular-season week.")

    posters = {}
    for team in teams:
        local_path = generate_favorite_team_poster(
            year=season,
            week=week,
            seasontype=seasontype,
            team=team,
        )
        posters[team] = upload_file_return_url(
            local_path,
            f"favorite_team/current/{team}.png"
        )

    payload = {
        "season": season,
        "week": week,
        "seasontype": seasontype,
        "count": len(posters),
        "posters": posters,
    }

    local_json = f"/tmp/favorite_team_{season}_week_{week}.json"
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    payload["metadata_url"] = upload_file_return_url(local_json, "favorite_team/current.json")

    if keep_versioned:
        payload["versioned_metadata_url"] = upload_file_return_url(
            local_json,
            f"favorite_team/history/{season}/week{week:02d}/metadata.json"
        )

    return payload


def get_current_favorite_team_payload() -> dict:
    return {
        "metadata_url": public_storage_url("favorite_team/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_current_favorite_team_posters(keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
