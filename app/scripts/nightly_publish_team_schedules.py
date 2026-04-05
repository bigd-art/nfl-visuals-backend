#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.storage_supabase import upload_file_return_url
from app.scripts.team_schedule_espn_poster import (
    build_team_map,
    SCHEDULE_URL,
    get_json,
    build_full_18_week_schedule,
    make_poster,
)


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def resolve_current_nfl_season() -> int:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return now_et.year if now_et.month >= 3 else now_et.year - 1


def publish_team_schedules(season: int = None, keep_versioned: bool = False):
    if season is None:
        season = resolve_current_nfl_season()

    team_map = build_team_map()

    with tempfile.TemporaryDirectory() as tmpdir:
        posters = {}

        for team_abbr in sorted(team_map.keys()):
            team_id = team_map[team_abbr]["id"]
            team_name = team_map[team_abbr]["display_name"]

            data = get_json(SCHEDULE_URL.format(team_id=team_id, year=season))
            games = build_full_18_week_schedule(team_abbr, data, team_map)

            local_path = os.path.join(tmpdir, f"{team_abbr.lower()}_{season}_schedule_poster.png")
            make_poster(team_abbr, team_name, season, games, local_path)

            storage_key = f"team_schedules/current/{team_abbr}.png"
            posters[team_abbr] = upload_file_return_url(local_path, storage_key)

        payload = {
            "season": season,
            "count": len(posters),
            "posters": posters,
        }

        local_json = os.path.join(tmpdir, "current.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        payload["metadata_url"] = upload_file_return_url(local_json, "team_schedules/current.json")

        if keep_versioned:
            payload["versioned_metadata_url"] = upload_file_return_url(
                local_json,
                f"team_schedules/history/{season}/metadata.json"
            )

        return payload


def get_current_team_schedules_payload():
    return {
        "metadata_url": public_storage_url("team_schedules/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_team_schedules(season=args.season, keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
