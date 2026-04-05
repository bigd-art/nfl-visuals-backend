#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
from app.scripts.season_auto import candidate_regular_seasons, resolve_first_valid

from app.scripts.nfl_standings_conference_generate import generate_standings_conference_png
from app.scripts.nfl_stat_leaders_generate import generate_all_stat_leader_posters, STAT_CONFIG


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def standings_has_data(season: int) -> bool:
    from app.scripts.nfl_standings_conference_generate import get_json
    data = get_json(season)
    children = data.get("children", [])
    return bool(children)


def leaders_has_data(season: int, seasontype: int) -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = generate_all_stat_leader_posters(season=season, seasontype=seasontype, outdir=tmpdir)
            return bool(outputs)
    except Exception:
        return False


def publish_posters(keep_versioned: bool = False) -> dict:
    season = resolve_first_valid(candidate_regular_seasons(), standings_has_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "season": season,
            "standings": {},
            "stat_leaders": {},
        }

        # Standings
        standings_png = os.path.join(tmpdir, f"standings_conference_{season}.png")
        generate_standings_conference_png(season, standings_png)
        standings_url = upload_file_return_url(standings_png, "standings/current.png")
        payload["standings"] = {
            "season": season,
            "image_url": standings_url,
        }

        # Stat leaders regular season only if valid
        regular_posters = {}
        if leaders_has_data(season, 2):
            regular_dir = os.path.join(tmpdir, "leaders_regular")
            os.makedirs(regular_dir, exist_ok=True)
            outputs = generate_all_stat_leader_posters(season=season, seasontype=2, outdir=regular_dir)

            for slug, _full_title, _short_title in STAT_CONFIG:
                local_path = outputs[slug]
                regular_posters[slug] = upload_file_return_url(
                    local_path,
                    f"stat_leaders/current/regular/{slug}.png"
                )

        payload["stat_leaders"]["regular"] = regular_posters

        # Postseason only if data exists
        postseason_posters = {}
        if leaders_has_data(season, 3):
            post_dir = os.path.join(tmpdir, "leaders_post")
            os.makedirs(post_dir, exist_ok=True)
            outputs = generate_all_stat_leader_posters(season=season, seasontype=3, outdir=post_dir)

            for slug, _full_title, _short_title in STAT_CONFIG:
                local_path = outputs[slug]
                postseason_posters[slug] = upload_file_return_url(
                    local_path,
                    f"stat_leaders/current/postseason/{slug}.png"
                )

        payload["stat_leaders"]["postseason"] = postseason_posters

        local_json = os.path.join(tmpdir, "nightly_posters_current.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        payload["metadata_url"] = upload_file_return_url(local_json, "nightly_posters/current.json")

        if keep_versioned:
            payload["versioned_metadata_url"] = upload_file_return_url(
                local_json,
                f"nightly_posters/history/{season}/metadata.json"
            )

        return payload


def get_current_posters_payload() -> dict:
    return {
        "metadata_url": public_storage_url("nightly_posters/current.json"),
        "standings_url": public_storage_url("standings/current.png"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(json.dumps(publish_posters(keep_versioned=args.keep_versioned), indent=2))


if __name__ == "__main__":
    main()
