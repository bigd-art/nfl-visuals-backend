#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
from app.scripts.team_schedule_espn_poster import (
    build_team_map,
    SCHEDULE_URL,
    get_json,
    build_full_18_week_schedule,
    make_poster,
)
from app.scripts.season_auto import candidate_regular_seasons, resolve_first_valid


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def season_has_schedule_data(season: int) -> bool:
    team_map = build_team_map()
    sample_team = "PHI" if "PHI" in team_map else next(iter(team_map.keys()))
    team_id = team_map[sample_team]["id"]
    data = get_json(SCHEDULE_URL.format(team_id=team_id, year=season))
    events = data.get("events", [])
    return bool(events)


def publish_team_schedules(keep_versioned: bool = False) -> dict:
    season = resolve_first_valid(candidate_regular_seasons(), season_has_schedule_data)
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

            posters[team_abbr] = upload_file_return_url(
                local_path,
                f"team_schedules/current/{team_abbr}.png"
            )

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


def get_current_team_schedules_payload() -> dict:
    return {
        "metadata_url": public_storage_url("team_schedules/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(json.dumps(publish_team_schedules(keep_versioned=args.keep_versioned), indent=2))


if __name__ == "__main__":
    main()
