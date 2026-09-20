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

    base = os.environ["SUPABASE_URL"].rstrip("/")

    bucket = os.environ.get(
        "SUPABASE_BUCKET",
        "nfl-posters",
    )

    return (
        f"{base}/storage/v1/object/public/"
        f"{bucket}/{storage_key}"
    )


# ============================================================
# CURRENT PUBLISHED BIG BOARD
# ============================================================

def get_current_pff_big_board_payload() -> dict:

    return {
        "metadata_url": public_storage_url(
            "pff_big_board/current.json"
        ),
    }


# ============================================================
# PLAYER EXTRACTION
# ============================================================

def get_available_players(
    data,
):

    # --------------------------------------------------------
    # PFF 2027 CURRENT RESPONSE
    #
    # The response currently contains a top-level "players"
    # array, but PFF is returning it empty.
    #
    # Handle that explicitly instead of treating conference/
    # team metadata as player records.
    # --------------------------------------------------------

    if isinstance(
        data,
        dict,
    ):

        players = data.get(
            "players"
        )

        if isinstance(
            players,
            list,
        ):

            return players

    # --------------------------------------------------------
    # FALLBACK
    #
    # If PFF changes the response again in the future,
    # allow the Big Board parser to look for player records.
    # --------------------------------------------------------

    try:

        return bigboard.get_player_list(
            data
        )

    except Exception as error:

        print(
            f"WARNING: Could not locate "
            f"PFF player records: {error}"
        )

        return []


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

        players = get_available_players(
            data
        )

        return bool(
            players
        )

    except Exception as error:

        print(
            f"WARNING: PFF Big Board unavailable "
            f"for season {season}: {error}"
        )

        return False


# ============================================================
# SKIP PAYLOAD
# ============================================================

def skipped_payload(
    season: int,
    reason: str,
) -> dict:

    current_metadata_url = (
        public_storage_url(
            "pff_big_board/current.json"
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "PFF BIG BOARD REFRESH SKIPPED"
    )

    print(
        "=" * 80
    )

    print(
        f"Requested season: {season}"
    )

    print(
        f"Reason: {reason}"
    )

    print(
        "Existing published Big Board "
        "files will remain untouched."
    )

    print(
        f"Current metadata remains at: "
        f"{current_metadata_url}"
    )

    print(
        "=" * 80
    )

    return {
        "status": "skipped",
        "requested_season": season,
        "reason": reason,
        "preserved_existing_posters": True,
        "metadata_url": current_metadata_url,
    }


# ============================================================
# PUBLISH BIG BOARD
# ============================================================

def publish_pff_big_board(
    keep_versioned: bool = False,
) -> dict:

    season = PFF_BIG_BOARD_SEASON

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

    # ========================================================
    # FETCH PFF DATA
    # ========================================================

    try:

        data = bigboard.fetch_big_board(
            season
        )

    except Exception as error:

        return skipped_payload(
            season=season,
            reason=(
                "PFF Big Board request failed: "
                f"{error}"
            ),
        )

    # ========================================================
    # READ PLAYER ARRAY
    # ========================================================

    players = get_available_players(
        data
    )

    # ========================================================
    # NO 2027 PLAYERS YET
    #
    # This is expected while PFF returns:
    #
    #     "players": []
    #
    # Do NOT overwrite current posters.
    # Do NOT fail the GitHub Action.
    # ========================================================

    if not players:

        return skipped_payload(
            season=season,
            reason=(
                f"PFF returned no player records "
                f"for season {season}."
            ),
        )

    print(
        f"PFF returned "
        f"{len(players)} player records "
        f"for season {season}."
    )

    # ========================================================
    # GROUP PLAYERS
    # ========================================================

    try:

        grouped = (
            bigboard.group_top_players(
                players
            )
        )

    except Exception as error:

        return skipped_payload(
            season=season,
            reason=(
                "PFF player grouping failed: "
                f"{error}"
            ),
        )

    # ========================================================
    # FILTER INVALID GROUPS
    # ========================================================

    valid_grouped = {}

    for (
        position,
        player_list,
    ) in grouped.items():

        if (
            not position
            or position == "UNK"
        ):

            print(
                "WARNING: Skipping invalid "
                f"position group: {position}"
            )

            continue

        if not player_list:

            print(
                "WARNING: Skipping empty "
                f"position group: {position}"
            )

            continue

        valid_grouped[
            position
        ] = player_list

    if not valid_grouped:

        return skipped_payload(
            season=season,
            reason=(
                f"PFF returned player data for "
                f"season {season}, but no valid "
                f"position groups could be created."
            ),
        )

    # ========================================================
    # GENERATE INTO TEMP DIRECTORY
    #
    # Nothing touches the current published posters until
    # valid posters have actually been generated.
    # ========================================================

    with tempfile.TemporaryDirectory() as tmpdir:

        bigboard.OUTPUT_DIR = tmpdir

        bigboard.ensure_output_dir()

        generated_files = {}

        # ----------------------------------------------------
        # GENERATE ALL POSTERS FIRST
        # ----------------------------------------------------

        try:

            for (
                position,
                player_list,
            ) in valid_grouped.items():

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
                        f"Expected poster was not created: "
                        f"{local_path}"
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
                    "PFF poster generation failed: "
                    f"{error}"
                ),
            )

        if not generated_files:

            return skipped_payload(
                season=season,
                reason=(
                    f"No PFF Big Board posters "
                    f"were generated for season "
                    f"{season}."
                ),
            )

        print()
        print(
            f"Successfully generated "
            f"{len(generated_files)} "
            f"Big Board posters."
        )

        # ====================================================
        # UPLOAD CURRENT POSTERS
        # ====================================================

        posters = {}

        for (
            position,
            file_info,
        ) in generated_files.items():

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

        # ====================================================
        # METADATA PAYLOAD
        # ====================================================

        payload = {
            "status": "published",
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

        # ====================================================
        # CURRENT METADATA
        # ====================================================

        payload[
            "metadata_url"
        ] = (
            upload_file_return_url(
                local_json,
                "pff_big_board/current.json",
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
                        "pff_big_board/history/"
                        f"{season}/"
                        "metadata.json"
                    ),
                )
            )

        print()
        print(
            "=" * 80
        )

        print(
            f"PFF BIG BOARD {season} "
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
