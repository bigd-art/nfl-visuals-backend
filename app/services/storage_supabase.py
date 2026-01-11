# app/services/storage_supabase.py

import os
from typing import List
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "nfl-posters")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_file_return_url(local_path: str, key: str) -> str:
    """
    Upload a file to Supabase storage and return its public URL
    """
    with open(local_path, "rb") as f:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=key,
            file=f,
            file_options={"content-type": "image/png"},
        )

    return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(key)


def cached_urls_for_prefix(prefix: str) -> List[str]:
    """
    STRICT prefix match.
    Only returns files that start with the prefix exactly.
    """
    results: List[str] = []
    offset = 0
    limit = 100

    while True:
        resp = supabase.storage.from_(SUPABASE_BUCKET).list(
            path=prefix,
            options={"limit": limit, "offset": offset},
        )

        if not resp:
            break

        for obj in resp:
            name = obj.get("name")
            if not name:
                continue

            full_key = f"{prefix}{name}"
            results.append(
                supabase.storage.from_(SUPABASE_BUCKET).get_public_url(full_key)
            )

        if len(resp) < limit:
            break

        offset += limit

    return sorted(results)

