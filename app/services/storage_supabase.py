# app/services/storage_supabase.py

import os
from typing import List, Optional

from supabase import create_client, Client


# ============================
# Config + Client
# ============================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Use service role key on the backend (Render), NOT anon key
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "nfl-posters")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "Missing Supabase env vars. Need SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ============================
# URL helpers
# ============================

def get_public_url(storage_key: str) -> str:
    storage_key = storage_key.lstrip("/")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_key}"


# ============================
# Upload
# ============================

def upload_file_return_url(local_path: str, storage_key: str) -> str:
    """
    Upload a local file to Supabase Storage and return the public URL.
    Uses upsert so re-uploads overwrite safely.
    """
    storage_key = storage_key.lstrip("/")

    with open(local_path, "rb") as f:
        data = f.read()

    # supabase-py storage upload (upsert=True)
    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_key,
            file=data,
            file_options={
                "content-type": "image/png",
                "upsert": "true",
            },
        )
    except Exception as e:
        # If upload fails, show the real error
        raise RuntimeError(f"Supabase upload failed for {storage_key}: {repr(e)}")

    return get_public_url(storage_key)


# ============================
# Listing / Cache helpers
# ============================

def _list_folder(folder: str, limit: int = 200):
    """
    Version-tolerant wrapper around Supabase Storage list().
    """
    # Normalize folder: must NOT start with "/", must NOT end with "//"
    folder = folder.strip("/")

    # Try common signatures across supabase-py versions
    try:
        return supabase.storage.from_(SUPABASE_BUCKET).list(folder, {"limit": limit})
    except TypeError:
        return supabase.storage.from_(SUPABASE_BUCKET).list(path=folder, options={"limit": limit})


def list_files_by_prefix(prefix: str, limit: int = 200) -> List[str]:
    """
    Return full storage keys (bucket paths) for objects directly under the prefix folder.
    NOTE: Supabase list() is NOT recursive.
    """
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    folder = prefix.strip("/")

    print(f"[list_files_by_prefix] folder={folder}")

    res = _list_folder(folder, limit=limit)
    print(f"[list_files_by_prefix] raw_res={res}")

    keys: List[str] = []
    for item in res or []:
        name = item.get("name")
        if not name:
            continue
        # full key = folder + name
        keys.append(f"{folder}/{name}")

    print(f"[list_files_by_prefix] keys_count={len(keys)}")
    return keys


def cached_urls_for_prefix(prefix: str) -> List[str]:
    """
    If PNGs exist under prefix (folder), return their public URLs.
    Otherwise return [].
    """
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    print(f"[cached_urls_for_prefix] LIST prefix={prefix}")

    try:
        keys = list_files_by_prefix(prefix)
    except Exception as e:
        # If listing fails, we want it visible in Render logs
        print(f"[cached_urls_for_prefix] ERROR {repr(e)}")
        return []

    png_keys = [k for k in keys if k.lower().endswith(".png")]

    if not png_keys:
        print(f"[cached_urls_for_prefix] EMPTY prefix={prefix}")
        return []

    urls = [get_public_url(k) for k in png_keys]
    print(f"[cached_urls_for_prefix] FOUND {len(urls)} files")
    return urls

