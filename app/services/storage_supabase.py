# app/services/storage_supabase.py

import os
from typing import List, Optional
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "nfl-posters")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not set (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _clean_key(key: str) -> str:
    """
    Normalize object keys so we don't accidentally create double slashes, leading slashes, etc.
    """
    k = (key or "").strip()
    # remove leading slash so bucket paths are consistent
    k = k.lstrip("/")
    # collapse accidental double slashes
    while "//" in k:
        k = k.replace("//", "/")
    return k


def _clean_prefix(prefix: str) -> str:
    """
    Normalize a folder prefix used for list().
    Supabase list() expects a "folder" path.
    """
    p = (prefix or "").strip()
    p = p.lstrip("/")
    while "//" in p:
        p = p.replace("//", "/")
    # allow root listing with ""
    if p and not p.endswith("/"):
        p += "/"
    return p


def upload_file_return_url(local_path: str, key: str, upsert: bool = True) -> str:
    """
    Upload a local PNG file to Supabase storage and return its public URL.

    - Expo can only display public HTTPS URLs, not /tmp paths.
    - Set upsert=True to overwrite existing files (needed for nightly refresh).
    """
    storage = supabase.storage.from_(SUPABASE_BUCKET)
    object_key = _clean_key(key)

    with open(local_path, "rb") as f:
        storage.upload(
            path=object_key,
            file=f,
            file_options={
                "content-type": "image/png",
                # IMPORTANT: overwrite existing poster keys so the same URL always points to newest image
                "upsert": "true" if upsert else "false",
                # good practice for CDN
                "cacheControl": "0",
            },
        )

    return storage.get_public_url(object_key)


def _list_folder(prefix: str, limit: int = 1000, offset: int = 0):
    """
    Wrapper around Supabase list() for a single folder page.
    """
    storage = supabase.storage.from_(SUPABASE_BUCKET)
    return storage.list(path=prefix, options={"limit": limit, "offset": offset})


def cached_urls_for_prefix(prefix: str) -> List[str]:
    """
    Returns PUBLIC URLs for all files under the given folder prefix.

    IMPORTANT:
    - This is a *folder listing*, not a "starts_with" search across the entire bucket.
    - If you want "strict prefix", pass a folder-like prefix:
        "stat_leaders/2025/" or "team_stat_leaders/ari/"
    """
    storage = supabase.storage.from_(SUPABASE_BUCKET)

    folder = _clean_prefix(prefix)
    results: List[str] = []

    # We do a recursive walk because Supabase list() is folder-based
    def walk(current_folder: str):
        offset = 0
        page_limit = 1000

        while True:
            resp = _list_folder(current_folder, limit=page_limit, offset=offset)
            if not resp:
                break

            for obj in resp:
                name = obj.get("name")
                if not name:
                    continue

                # Supabase list() returns folders sometimes with "id" null and "metadata" null-ish
                # The easiest safe rule:
                # - if it has "metadata" and "metadata" contains size -> it's a file
                # - otherwise treat as folder and recurse
                metadata = obj.get("metadata") or {}
                is_file = isinstance(metadata, dict) and ("size" in metadata or "mimetype" in metadata)

                if is_file:
                    full_key = _clean_key(f"{current_folder}{name}")
                    results.append(storage.get_public_url(full_key))
                else:
                    # folder
                    sub_folder = _clean_prefix(f"{current_folder}{name}")
                    if sub_folder != current_folder:  # safety guard
                        walk(sub_folder)

            if len(resp) < page_limit:
                break
            offset += page_limit

    walk(folder)
    return sorted(results)
