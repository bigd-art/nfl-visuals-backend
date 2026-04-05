#!/usr/bin/env python3
import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
import app.scripts.nfl_rosters_generate as rosters


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def publish_team_rosters(keep_versioned: bool = False):
    with tempfile.TemporaryDirectory() as tmpdir:
        posters = {}

        for team_code in sorted(rosters.TEAM_INFO.keys()):
            sections = rosters.parse_team_roster(team_code)
            posters[team_code.upper()] = {}

            for unit in ["offense", "defense", "special_teams"]:
                local_path = rosters.create_single_poster(team_code, unit, sections[unit], tmpdir)
                storage_key = f"team_rosters/current/{team_code.upper()}_{unit}.png"
                posters[team_code.upper()][unit] = upload_file_return_url(local_path, storage_key)

        payload = {
            "count": len(posters),
            "posters": posters,
        }

        local_json = os.path.join(tmpdir, "current.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        payload["metadata_url"] = upload_file_return_url(local_json, "team_rosters/current.json")

        if keep_versioned:
            payload["versioned_metadata_url"] = upload_file_return_url(
                local_json,
                "team_rosters/history/metadata.json"
            )

        return payload


def get_current_team_rosters_payload():
    return {
        "metadata_url": public_storage_url("team_rosters/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_team_rosters(keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
