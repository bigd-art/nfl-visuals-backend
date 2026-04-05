#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
import app.scripts.pff_big_board_posters as bigboard
from app.scripts.season_auto import current_draft_cycle_candidates, resolve_first_valid


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def draft_cycle_has_big_board(season: int) -> bool:
    data = bigboard.fetch_big_board(season)
    players = bigboard.get_player_list(data)
    return bool(players)


def publish_pff_big_board(keep_versioned: bool = False) -> dict:
    season = resolve_first_valid(current_draft_cycle_candidates(), draft_cycle_has_big_board)

    with tempfile.TemporaryDirectory() as tmpdir:
        bigboard.OUTPUT_DIR = tmpdir
        bigboard.ensure_output_dir()

        data = bigboard.fetch_big_board(season)
        players = bigboard.get_player_list(data)
        grouped = bigboard.group_top_players(players)

        posters = {}
        for pos, plist in grouped.items():
            bigboard.create_poster(pos, plist, season)
            filename = f"{bigboard.safe_filename(pos)}_top_5.png"
            local_path = os.path.join(tmpdir, filename)
            posters[pos] = upload_file_return_url(
                local_path,
                f"pff_big_board/current/{filename}"
            )

        payload = {
            "season": season,
            "count": len(posters),
            "posters": posters,
        }

        local_json = os.path.join(tmpdir, "current.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        payload["metadata_url"] = upload_file_return_url(local_json, "pff_big_board/current.json")

        if keep_versioned:
            payload["versioned_metadata_url"] = upload_file_return_url(
                local_json,
                f"pff_big_board/history/{season}/metadata.json"
            )

        return payload


def get_current_pff_big_board_payload() -> dict:
    return {
        "metadata_url": public_storage_url("pff_big_board/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(json.dumps(publish_pff_big_board(keep_versioned=args.keep_versioned), indent=2))


if __name__ == "__main__":
    main()
