#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

from app.services.storage_supabase import upload_file_return_url
import app.scripts.pff_big_board_posters as bigboard


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def resolve_draft_season() -> int:
    now_et = datetime.now(ZoneInfo("America/New_York"))
    return now_et.year + 1 if now_et.month >= 5 else now_et.year


def publish_pff_big_board(season: int = None, keep_versioned: bool = False):
    if season is None:
        season = resolve_draft_season()

    with tempfile.TemporaryDirectory() as tmpdir:
        bigboard.OUTPUT_DIR = tmpdir
        bigboard.ensure_output_dir()

        data = bigboard.fetch_big_board(season)
        players = bigboard.get_player_list(data)
        grouped = bigboard.group_top_players(players)

        uploaded = {}
        for pos, plist in grouped.items():
            bigboard.create_poster(pos, plist, season)
            filename = f"{bigboard.safe_filename(pos)}_top_5.png"
            local_path = os.path.join(tmpdir, filename)
            storage_key = f"pff_big_board/current/{filename}"
            uploaded[pos] = upload_file_return_url(local_path, storage_key)

        payload = {
            "season": season,
            "count": len(uploaded),
            "posters": uploaded,
        }

        local_json = os.path.join(tmpdir, "current.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        payload["metadata_url"] = upload_file_return_url(local_json, "pff_big_board/current.json")

        if keep_versioned:
            versioned = {}
            for pos, url in uploaded.items():
                filename = f"{bigboard.safe_filename(pos)}_top_5.png"
                local_path = os.path.join(tmpdir, filename)
                versioned[pos] = upload_file_return_url(
                    local_path,
                    f"pff_big_board/history/{season}/{filename}"
                )
            payload["versioned_posters"] = versioned
            payload["versioned_metadata_url"] = upload_file_return_url(
                local_json,
                f"pff_big_board/history/{season}/metadata.json"
            )

        return payload


def get_current_pff_big_board_payload():
    return {
        "metadata_url": public_storage_url("pff_big_board/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_pff_big_board(season=args.season, keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
