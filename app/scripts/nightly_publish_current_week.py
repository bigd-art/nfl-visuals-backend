#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from app.generator.week_posters import generate_week
from app.services.storage_supabase import upload_file_return_url
from app.scripts.current_week_context import resolve_current_regular_context


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def publish_current_week_posters(keep_versioned: bool = False) -> dict:
    season, week, seasontype = resolve_current_regular_context()
    local_dir = generate_week(season, week, seasontype)

    pngs = sorted(
        [
            os.path.join(local_dir, f)
            for f in os.listdir(local_dir)
            if f.lower().endswith(".png")
        ]
    )

    if not pngs:
        raise RuntimeError("No week poster PNGs were generated.")

    urls = []
    for local_path in pngs:
        filename = os.path.basename(local_path)
        urls.append(
            upload_file_return_url(local_path, f"week_posters/current/{filename}")
        )

    payload = {
        "season": season,
        "week": week,
        "seasontype": seasontype,
        "count": len(urls),
        "poster_urls": urls,
    }

    local_json = f"/tmp/week_posters_{season}_week_{week}.json"
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    payload["metadata_url"] = upload_file_return_url(local_json, "week_posters/current.json")

    if keep_versioned:
        payload["versioned_metadata_url"] = upload_file_return_url(
            local_json,
            f"week_posters/history/{season}/week{week:02d}/metadata.json"
        )

    return payload


def get_current_week_posters_payload() -> dict:
    return {
        "metadata_url": public_storage_url("week_posters/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_current_week_posters(keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
