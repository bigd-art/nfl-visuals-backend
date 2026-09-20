#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Dict

from app.services.storage_supabase import upload_file_return_url
from app.scripts.nfl_standings_conference_generate import (
    generate_standings_conference_png,
)
from app.scripts.nfl_stat_leaders_generate import (
    generate_all_stat_leader_posters,
    STAT_CONFIG,
)


# ============================================================
# SEASON CONFIG
# ============================================================

STANDINGS_SEASON = 2026

STAT_LEADERS_REGULAR_SEASON = 2026

STAT_LEADERS_POSTSEASON_SEASON = 2025


# ============================================================
# STORAGE HELPERS
# ============================================================

def public_storage_url(
    storage_key: str,
) -> str:

    base = (
        os.environ[
            "SUPABASE_URL"
        ].rstrip(
            "/"
        )
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
# STANDINGS DATA CHECK
# ============================================================

def standings_has_data(
    season: int,
) -> bool:

    from app.scripts.nfl_standings_conference_generate import (
        get_json,
    )

    data = get_json(
        season
    )

    return bool(
        data.get(
            "AFC"
        )
        or data.get(
            "NFC"
        )
    )


# ============================================================
# STAT LEADER UPLOAD
# ============================================================

def upload_stat_leaders(
    tmpdir: str,
    season: int,
    seasontype: int,
    phase: str,
) -> Dict[str, str]:

    posters: Dict[
        str,
        str,
    ] = {}

    try:

        outdir = os.path.join(
            tmpdir,
            f"stat_leaders_{phase}",
        )

        os.makedirs(
            outdir,
            exist_ok=True,
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"GENERATING STAT LEADERS "
            f"FOR SEASON {season} "
            f"TYPE {seasontype} "
            f"({phase.upper()})"
        )

        print(
            "=" * 80
        )

        outputs = (
            generate_all_stat_leader_posters(
                season=season,
                seasontype=seasontype,
                outdir=outdir,
            )
        )

        for (
            slug,
            *_rest,
        ) in STAT_CONFIG:

            local_path = (
                outputs.get(
                    slug
                )
            )

            if (
                not local_path
                or not os.path.exists(
                    local_path
                )
            ):

                print(
                    f"WARNING: Missing generated "
                    f"stat leader poster for "
                    f"{phase}/{slug}"
                )

                continue

            storage_key = (
                f"stat_leaders/current/"
                f"{phase}/{slug}.png"
            )

            posters[
                slug
            ] = (
                upload_file_return_url(
                    local_path,
                    storage_key,
                )
            )

            print(
                f"Uploaded "
                f"{phase}/{slug} "
                f"-> {storage_key}"
            )

    except Exception as error:

        print(
            f"WARNING: stat leaders "
            f"{phase} upload failed: "
            f"{error}"
        )

    return posters


# ============================================================
# PUBLISH ALL NIGHTLY POSTERS
# ============================================================

def publish_posters(
    keep_versioned: bool = False,
) -> dict:

    print()
    print(
        "=" * 80
    )

    print(
        "NIGHTLY POSTER SEASON CONFIG"
    )

    print(
        "=" * 80
    )

    print(
        f"Standings: "
        f"{STANDINGS_SEASON}"
    )

    print(
        f"Regular stat leaders: "
        f"{STAT_LEADERS_REGULAR_SEASON}"
    )

    print(
        f"Postseason stat leaders: "
        f"{STAT_LEADERS_POSTSEASON_SEASON}"
    )

    print(
        "=" * 80
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        payload = {
            "season": STANDINGS_SEASON,

            "standings": {},

            "stat_leaders": {
                "regular": {},
                "postseason": {},
            },

            "seasons": {
                "standings": (
                    STANDINGS_SEASON
                ),
                "stat_leaders_regular": (
                    STAT_LEADERS_REGULAR_SEASON
                ),
                "stat_leaders_postseason": (
                    STAT_LEADERS_POSTSEASON_SEASON
                ),
            },
        }

        # ====================================================
        # STANDINGS
        #
        # 2026 REGULAR-SEASON STANDINGS
        # ====================================================

        standings_png = os.path.join(
            tmpdir,
            (
                "standings_conference_"
                f"{STANDINGS_SEASON}.png"
            ),
        )

        generate_standings_conference_png(
            STANDINGS_SEASON,
            standings_png,
        )

        standings_url = (
            upload_file_return_url(
                standings_png,
                "standings/current.png",
            )
        )

        payload[
            "standings"
        ] = {
            "season": (
                STANDINGS_SEASON
            ),
            "image_url": (
                standings_url
            ),
        }

        # ====================================================
        # STAT LEADERS
        #
        # 2026 REGULAR SEASON
        # ESPN SEASON TYPE 2
        # ====================================================

        payload[
            "stat_leaders"
        ][
            "regular"
        ] = (
            upload_stat_leaders(
                tmpdir=tmpdir,
                season=(
                    STAT_LEADERS_REGULAR_SEASON
                ),
                seasontype=2,
                phase="regular",
            )
        )

        # ====================================================
        # STAT LEADERS
        #
        # KEEP 2025 POSTSEASON
        # ESPN SEASON TYPE 3
        # ====================================================

        payload[
            "stat_leaders"
        ][
            "postseason"
        ] = (
            upload_stat_leaders(
                tmpdir=tmpdir,
                season=(
                    STAT_LEADERS_POSTSEASON_SEASON
                ),
                seasontype=3,
                phase="postseason",
            )
        )

        # ====================================================
        # CURRENT METADATA
        # ====================================================

        local_json = os.path.join(
            tmpdir,
            "nightly_posters_current.json",
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
                "nightly_posters/current.json",
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
                        "nightly_posters/history/"
                        f"{STANDINGS_SEASON}/"
                        "metadata.json"
                    ),
                )
            )

        return payload


# ============================================================
# CURRENT PUBLIC PAYLOAD
# ============================================================

def get_current_posters_payload() -> dict:

    return {
        "metadata_url": (
            public_storage_url(
                "nightly_posters/current.json"
            )
        ),

        "standings_url": (
            public_storage_url(
                "standings/current.png"
            )
        ),
    }


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
        publish_posters(
            keep_versioned=(
                args.keep_versioned
            )
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
