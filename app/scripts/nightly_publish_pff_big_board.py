#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
import app.scripts.pff_big_board_posters as bigboard


PFF_BIG_BOARD_SEASON = 2026


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get(
        "SUPABASE_BUCKET",
        "nfl-posters",
    )

    return (
        f"{base}/storage/v1/object/public/"
        f"{bucket}/{storage_key}"
    )


def draft_cycle_has_big_board(
    season: int,
) -> bool:
    try:
        data = bigboard.fetch_big_board(
            season,
        )

        players = bigboard.get_player_list(
            data,
        )

        return bool(players)

    except Exception as error:
        print(
            f"WARNING: PFF Big Board unavailable "
            f"for season {season}: {error}"
        )

        return False


def publish_pff_big_board(
    keep_versioned: bool = False,
) -> dict:
    season = PFF_BIG_BOARD_SEASON

    print(
        f"Publishing PFF Big Board for {season}..."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bigboard.OUTPUT_DIR = tmpdir
        bigboard.ensure_output_dir()

        data = bigboard.fetch_big_board(
            season,
        )

        players = bigboard.get_player_list(
            data,
        )

        if not players:
            raise RuntimeError(
                f"PFF returned no Big Board players "
                f"for season {season}."
            )

        grouped = bigboard.group_top_players(
            players,
        )

        if not grouped:
            raise RuntimeError(
                f"No PFF position groups were created "
                f"for season {season}."
            )

        posters = {}

        for position, player_list in grouped.items():
            if position == "UNK":
                print(
                    "WARNING: Skipping invalid "
                    "UNK position group."
                )
                continue

            bigboard.create_poster(
                position,
                player_list,
                season,
            )

            filename = (
                f"{bigboard.safe_filename(position)}"
                "_top_5.png"
            )

            local_path = os.path.join(
                tmpdir,
                filename,
            )

            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    f"Expected poster was not created: "
                    f"{local_path}"
                )

            storage_key = (
                "pff_big_board/current/"
                f"{filename}"
            )

            posters[position] = (
                upload_file_return_url(
                    local_path,
                    storage_key,
                )
            )

        if not posters:
            raise RuntimeError(
                f"No valid PFF Big Board posters "
                f"were generated for season {season}."
            )

        payload = {
            "season": season,
            "count": len(posters),
            "posters": posters,
        }

        local_json = os.path.join(
            tmpdir,
            "current.json",
        )

        with open(
            local_json,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        payload["metadata_url"] = (
            upload_file_return_url(
                local_json,
                "pff_big_board/current.json",
            )
        )

        if keep_versioned:
            payload["versioned_metadata_url"] = (
                upload_file_return_url(
                    local_json,
                    (
                        "pff_big_board/history/"
                        f"{season}/metadata.json"
                    ),
                )
            )

        return payload


def get_current_pff_big_board_payload() -> dict:
    return {
        "metadata_url": public_storage_url(
            "pff_big_board/current.json"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keep_versioned",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = publish_pff_big_board(
        keep_versioned=args.keep_versioned,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
