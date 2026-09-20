#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
import app.scripts.pff_big_board_posters as bigboard


# ============================================================
# DRAFT CLASS / BIG BOARD SEASON
# ============================================================

PFF_BIG_BOARD_SEASON = 2027


# ============================================================
# STORAGE HELPERS
# ============================================================

def public_storage_url(
    storage_key: str,
) -> str:

    base = (
        os.environ[
            "SUPABASE_URL"
        ].rstrip("/")
    )

    bucket = os.environ.get(
        "SUPABASE_BUCKET",
        "nfl-posters",
    )

    return (
        f"{base}/storage/v1/object/public/"
        f"{bucket}/{storage_key}"
    )


# ============================================================
# BIG BOARD AVAILABILITY CHECK
# ============================================================

def draft_cycle_has_big_board(
    season: int,
) -> bool:

    try:

        data = bigboard.fetch_big_board(
            season
        )

        players = bigboard.get_player_list(
            data
        )

        return bool(
            players
        )

    except Exception as error:

        print(
            f"WARNING: PFF Big Board "
            f"unavailable for season "
            f"{season}: {error}"
        )

        return False


# ============================================================
# DEBUG HELPERS
# ============================================================

def print_player_debug(
    players,
    limit: int = 3,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "PFF BIG BOARD RAW PLAYER DEBUG"
    )

    print(
        "=" * 80
    )

    print(
        f"Total raw players returned: "
        f"{len(players)}"
    )

    print()

    for index, player in enumerate(
        players[:limit],
        start=1,
    ):

        print(
            "-" * 80
        )

        print(
            f"RAW PLAYER #{index}"
        )

        print(
            "-" * 80
        )

        print(
            json.dumps(
                player,
                indent=2,
                default=str,
            )
        )

        if isinstance(
            player,
            dict,
        ):

            print()

            print(
                "TOP-LEVEL KEYS:"
            )

            print(
                sorted(
                    player.keys()
                )
            )

        print()

    print(
        "=" * 80
    )


# ============================================================
# PUBLISH BIG BOARD
# ============================================================

def publish_pff_big_board(
    keep_versioned: bool = False,
) -> dict:

    season = (
        PFF_BIG_BOARD_SEASON
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"Publishing PFF Big Board "
        f"for {season}..."
    )

    print(
        "=" * 80
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        # ----------------------------------------------------
        # SEND GENERATOR OUTPUT TO TEMP DIRECTORY
        # ----------------------------------------------------

        bigboard.OUTPUT_DIR = (
            tmpdir
        )

        bigboard.ensure_output_dir()

        # ----------------------------------------------------
        # FETCH BIG BOARD
        # ----------------------------------------------------

        data = bigboard.fetch_big_board(
            season
        )

        players = bigboard.get_player_list(
            data
        )

        if not players:

            raise RuntimeError(
                f"PFF returned no Big Board "
                f"players for season "
                f"{season}."
            )

        # ----------------------------------------------------
        # DEBUG RAW 2027 PLAYER DATA
        #
        # This lets us see exactly which keys PFF uses
        # for position, player name, school, etc.
        # ----------------------------------------------------

        print_player_debug(
            players,
            limit=3,
        )

        # ----------------------------------------------------
        # GROUP BY POSITION
        # ----------------------------------------------------

        grouped = bigboard.group_top_players(
            players
        )

        print()
        print(
            "=" * 80
        )

        print(
            "GROUPED POSITION SUMMARY"
        )

        print(
            "=" * 80
        )

        for (
            position,
            player_list,
        ) in grouped.items():

            print(
                f"{position}: "
                f"{len(player_list)} players"
            )

        print(
            "=" * 80
        )

        if not grouped:

            raise RuntimeError(
                f"No PFF position groups "
                f"were created for season "
                f"{season}."
            )

        posters = {}

        # ----------------------------------------------------
        # GENERATE + UPLOAD POSTERS
        # ----------------------------------------------------

        for (
            position,
            player_list,
        ) in grouped.items():

            if (
                position
                == "UNK"
            ):

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

            if not os.path.exists(
                local_path
            ):

                raise FileNotFoundError(
                    f"Expected poster was "
                    f"not created: "
                    f"{local_path}"
                )

            storage_key = (
                "pff_big_board/current/"
                f"{filename}"
            )

            posters[
                position
            ] = (
                upload_file_return_url(
                    local_path,
                    storage_key,
                )
            )

            print(
                f"Uploaded "
                f"{position} "
                f"-> {storage_key}"
            )

        # ----------------------------------------------------
        # ENSURE WE ACTUALLY CREATED SOMETHING
        # ----------------------------------------------------

        if not posters:

            raise RuntimeError(
                f"No valid PFF Big Board "
                f"posters were generated "
                f"for season {season}."
            )

        # ----------------------------------------------------
        # METADATA PAYLOAD
        # ----------------------------------------------------

        payload = {
            "season": season,
            "count": len(
                posters
            ),
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

        # ----------------------------------------------------
        # CURRENT METADATA
        # ----------------------------------------------------

        payload[
            "metadata_url"
        ] = (
            upload_file_return_url(
                local_json,
                "pff_big_board/current.json",
            )
        )

        # ----------------------------------------------------
        # VERSIONED METADATA
        # ----------------------------------------------------

        if keep_versioned:

            payload[
                "versioned_metadata_url"
            ] = (
                upload_file_return_url(
                    local_json,
                    (
                        "pff_big_board/history/"
                        f"{season}/"
                        "metadata.json"
                    ),
                )
            )

        return payload


# ============================================================
# CURRENT PUBLIC PAYLOAD
# ============================================================

def get_current_pff_big_board_payload() -> dict:

    return {
        "metadata_url": (
            public_storage_url(
                "pff_big_board/current.json"
            )
        ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keep_versioned",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    result = publish_pff_big_board(
        keep_versioned=(
            args.keep_versioned
        ),
    )

    print()

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
