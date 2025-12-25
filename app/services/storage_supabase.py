import os
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "nfl-posters")

def upload_file_return_url(local_path: str, storage_key: str) -> str:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Missing Supabase environment variables")

    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{storage_key}"

    with open(local_path, "rb") as f:
        response = requests.post(
            upload_url,
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "x-upsert": "true",
                "Content-Type": "image/png",
            },
            data=f,
            timeout=60,
        )

    response.raise_for_status()

    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_key}"

# ============================
# CACHE / LISTING HELPERS
# ============================

from typing import List


def list_files_by_prefix(prefix: str, limit: int = 200) -> List[str]:
    folder = prefix.strip("/")

    try:
        # Most compatible form across supabase-py versions:
        res = supabase.storage.from_(SUPABASE_BUCKET).list(folder, {"limit": limit})
    except TypeError:
        # Fallback for clients that require keyword args:
        res = supabase.storage.from_(SUPABASE_BUCKET).list(path=folder, options={"limit": limit})
    except Exception as e:
        print(f"[list_files_by_prefix] ERROR folder={folder} err={e}")
        return []

    keys = []
    for item in res or []:
        name = item.get("name")
        if not name:
            continue
        keys.append(f"{folder}/{name}")

    print(f"[list_files_by_prefix] folder={folder} count={len(keys)}")
    return keys


def get_public_url(storage_key: str) -> str:
    """
    Return the public URL for a storage key.
    Assumes the bucket is PUBLIC.
    """
    return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_key)

def cached_urls_for_prefix(prefix: str) -> List[str]:
    """
    List public URLs for all PNG objects under a Supabase Storage prefix.
    Prefix MUST behave like a folder.
    """
    # normalize prefix to folder semantics
    prefix = prefix.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"

    print(f"[cached_urls_for_prefix] LIST prefix={prefix}")

    try:
        objects = supabase.storage.from_(SUPABASE_BUCKET).list(path=prefix)
    except Exception as e:
        print(f"[cached_urls_for_prefix] ERROR {e}")
        return []

    if not objects:
        print(f"[cached_urls_for_prefix] EMPTY prefix={prefix}")
        return []

    urls = []
    for obj in objects:
        name = obj.get("name")
        if not name or not name.lower().endswith(".png"):
            continue

        storage_key = f"{prefix}{name}"
        urls.append(get_public_url(storage_key))

    print(f"[cached_urls_for_prefix] FOUND {len(urls)} files")
    return urls

