#!/usr/bin/env python3
import argparse
import mimetypes
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from app.scripts.nfl_team_needs_generate import generate_all_team_needs_posters

PNG_CONTENT_TYPE = "image/png"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def upload_to_supabase(
    local_path: str,
    bucket: str,
    object_path: str,
    supabase_url: str,
    service_role_key: str,
    upsert: bool = True,
) -> None:
    with open(local_path, "rb") as f:
        data = f.read()

    url = f"{supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{object_path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
        "Content-Type": mimetypes.guess_type(local_path)[0] or PNG_CONTENT_TYPE,
        "x-upsert": "true" if upsert else "false",
    }

    resp = requests.post(url, headers=headers, data=data, timeout=120)

    if resp.status_code in (200, 201):
        return

    # Supabase may return 400/409 on existing file when POSTing with upsert;
    # fallback to PUT for safety.
    put_resp = requests.put(url, headers=headers, data=data, timeout=120)
    if put_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Upload failed for {object_path}. "
            f"POST {resp.status_code}: {resp.text[:300]} | "
            f"PUT {put_resp.status_code}: {put_resp.text[:300]}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate and upload nightly NFL team needs posters.")
    parser.add_argument("--keep_versioned", action="store_true")
    parser.add_argument("--folder", default="team_needs", help="Bucket folder for latest posters")
    args = parser.parse_args()

    supabase_url = require_env("SUPABASE_URL")
    service_role_key = require_env("SUPABASE_SERVICE_ROLE_KEY")
    bucket = require_env("SUPABASE_BUCKET")

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs, failures = generate_all_team_needs_posters(tmpdir)

        if not outputs:
            raise RuntimeError("No team needs posters were generated.")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        for team, local_path in outputs.items():
            filename = Path(local_path).name

            latest_object_path = f"{args.folder}/{filename}"
            upload_to_supabase(
                local_path=local_path,
                bucket=bucket,
                object_path=latest_object_path,
                supabase_url=supabase_url,
                service_role_key=service_role_key,
                upsert=True,
            )
            print(f"Uploaded latest: {latest_object_path}")

            if args.keep_versioned:
                versioned_object_path = f"{args.folder}/versioned/{timestamp}/{filename}"
                upload_to_supabase(
                    local_path=local_path,
                    bucket=bucket,
                    object_path=versioned_object_path,
                    supabase_url=supabase_url,
                    service_role_key=service_role_key,
                    upsert=True,
                )
                print(f"Uploaded versioned: {versioned_object_path}")

        if failures:
            print("\nSome teams failed during generation:")
            for team, err in failures.items():
                print(f" - {team}: {err}")

        print(f"\nDone. Uploaded {len(outputs)} team-needs posters.")


if __name__ == "__main__":
    main()
