#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile

from app.services.storage_supabase import (
    upload_file_return_url,
)

import app.scripts.pff_big_board_posters as bigboard


# ============================================================
# CONFIG
# ============================================================

BIG_BOARD_SEASON = 2027

SOURCE_NAME = "DraftTek"


# ============================================================
# STORAGE
#
# Keep the existing pff_big_board storage path so we do not
# break the mobile app or backend routes that already expect it.
# The data source itself is now DraftTek.
# ============================================================

CURRENT_STORAGE_PREFIX = (
    "pff_big_board/current"
)

HISTORY_STORAGE_PREFIX = (
    "pff_big_board/history"
)


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
# CURRENT PAYLOAD
# ============================================================

def get_current_pff_big_board_payload() -> dict:

    return {
        "metadata_url": (
            public_storage_url(
                (
                    f"{CURRENT_STORAGE_PREFIX}/"
                    "current.json"
                )
            )
        ),
    }


# ============================================================
# AVAILABILITY CHECK
# ============================================================

def draft_cycle_has_big_board(
    season: int,
) -> bool:

    try:

        data = (
            bigboard.fetch_big_board(
                season
            )
        )

        players = (
            bigboard.get_player_list(
                data
            )
        )

        grouped = (
            bigboard.group_top_players(
                players
            )
        )

        bigboard.validate_position_groups(
            grouped
        )

        return True

    except Exception as error:

        print(
            f"WARNING: {SOURCE_NAME} "
            f"Big Board unavailable "
            f"for season {season}: "
            f"{error}"
        )

        return False


# ============================================================
# SKIP WITHOUT DESTROYING CURRENT POSTERS
# ============================================================

def skipped_payload(
    season: int,
    reason: str,
) -> dict:

    metadata_url = (
        public_storage_url(
            (
                f"{CURRENT_STORAGE_PREFIX}/"
                "current.json"
            )
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "BIG BOARD REFRESH SKIPPED"
    )

    print(
        "=" * 80
    )

    print(
        f"Source: {SOURCE_NAME}"
    )

    print(
        f"Requested season: "
        f"{season}"
    )

    print(
        f"Reason: {reason}"
    )

    print(
        "Existing published Big Board "
        "posters remain untouched."
    )

    print(
        "=" * 80
    )

    return {
        "status": "skipped",
        "source": SOURCE_NAME,
        "requested_season": season,
        "reason": reason,
        "preserved_existing_posters": True,
        "metadata_url": metadata_url,
    }


# ============================================================
# PUBLISH
# ============================================================

def publish_pff_big_board(
    keep_versioned: bool = False,
) -> dict:

    season = (
        BIG_BOARD_SEASON
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"Publishing {SOURCE_NAME} "
        f"Big Board for {season}..."
    )

    print(
        "=" * 80
    )

    # ========================================================
    # FETCH
    # ========================================================

    try:

        data = (
            bigboard.fetch_big_board(
                season
            )
        )

        players = (
            bigboard.get_player_list(
                data
            )
        )

    except Exception as error:

        return skipped_payload(
            season=season,
            reason=(
                f"{SOURCE_NAME} fetch/"
                f"parse failed: {error}"
            ),
        )

    print()
    print(
        f"{SOURCE_NAME} returned "
        f"{len(players)} prospects."
    )

    # ========================================================
    # GROUP
    # ========================================================

    try:

        grouped = (
            bigboard.group_top_players(
                players
            )
        )

        bigboard.validate_position_groups(
            grouped
        )

    except Exception as error:

        return skipped_payload(
            season=season,
            reason=(
                f"{SOURCE_NAME} position "
                f"grouping failed: {error}"
            ),
        )

    # ========================================================
    # GENERATE EVERYTHING LOCALLY FIRST
    #
    # This ensures a parsing/generation problem does not
    # partially replace the currently published board.
    # ========================================================

    with tempfile.TemporaryDirectory() as tmpdir:

        bigboard.OUTPUT_DIR = (
            tmpdir
        )

        bigboard.ensure_output_dir()

        generated_files = {}

        try:

            for position in (
                bigboard.TARGET_POSITIONS
            ):

                player_list = (
                    grouped[
                        position
                    ]
                )

                bigboard.create_poster(
                    position,
                    player_list,
                    season,
                )

                filename = (
                    f"{bigboard.safe_filename(position)}"
                    "_top_5.png"
                )

                local_path = (
                    os.path.join(
                        tmpdir,
                        filename,
                    )
                )

                if not os.path.exists(
                    local_path
                ):

                    raise FileNotFoundError(
                        "Expected Big Board "
                        "poster was not "
                        f"created: {local_path}"
                    )

                generated_files[
                    position
                ] = {
                    "filename": filename,
                    "local_path": local_path,
                }

        except Exception as error:

            return skipped_payload(
                season=season,
                reason=(
                    "Poster generation "
                    f"failed: {error}"
                ),
            )

        if (
            len(
                generated_files
            )
            != len(
                bigboard.TARGET_POSITIONS
            )
        ):

            return skipped_payload(
                season=season,
                reason=(
                    "Not all required "
                    "position posters "
                    "were generated."
                ),
            )

        print()
        print(
            f"Generated all "
            f"{len(generated_files)} "
            f"required Big Board posters."
        )

        # ====================================================
        # UPLOAD CURRENT POSTERS
        # ====================================================

        posters = {}

        for position in (
            bigboard.TARGET_POSITIONS
        ):

            file_info = (
                generated_files[
                    position
                ]
            )

            filename = (
                file_info[
                    "filename"
                ]
            )

            local_path = (
                file_info[
                    "local_path"
                ]
            )

            storage_key = (
                f"{CURRENT_STORAGE_PREFIX}/"
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

        # ====================================================
        # METADATA
        # ====================================================

        payload = {
            "status": "published",
            "source": SOURCE_NAME,
            "season": season,
            "count": len(
                posters
            ),
            "positions": list(
                bigboard.TARGET_POSITIONS
            ),
            "posters": posters,
        }

        local_json = (
            os.path.join(
                tmpdir,
                "current.json",
            )
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

        payload[
            "metadata_url"
        ] = (
            upload_file_return_url(
                local_json,
                (
                    f"{CURRENT_STORAGE_PREFIX}/"
                    "current.json"
                ),
            )
        )

        # ====================================================
        # VERSIONED METADATA
        # ====================================================

        if keep_versioned:

            payload[
                "versioned_metadata_url"
            ] = (
                upload_file_return_url(
                    local_json,
                    (
                        f"{HISTORY_STORAGE_PREFIX}/"
                        f"{season}/metadata.json"
                    ),
                )
            )

        print()
        print(
            "=" * 80
        )

        print(
            f"{SOURCE_NAME.upper()} "
            f"BIG BOARD {season} "
            f"PUBLISHED SUCCESSFULLY"
        )

        print(
            "=" * 80
        )

        return payload


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--keep_versioned",
        action="store_true",
    )

    return (
        parser.parse_args()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = (
        parse_args()
    )

    result = (
        publish_pff_big_board(
            keep_versioned=(
                args.keep_versioned
            ),
        )
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
