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
    """
    List files under a storage prefix (folder).
    Example prefix: 'posters/2025/week13'
    """
    folder = prefix.strip("/")

    try:
        res = supabase.storage.from_(SUPABASE_BUCKET).list(
            path=folder,
            options={"limit": limit},
        )
    except Exception:
        return []

    keys = []
    for item in res or []:
        name = item.get("name")
        if not name:
            continue
        keys.append(f"{folder}/{name}")

    return keys


def get_public_url(storage_key: str) -> str:
    """
    Return the public URL for a storage key.
    Assumes the bucket is PUBLIC.
    """
    return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_key)
def cached_urls_for_prefix(prefix: str) -> List[str]:
    """
    If PNGs exist under prefix, return their public URLs.
    Otherwise return [].
    """
    keys = list_files_by_prefix(prefix)
    png_keys = [k for k in keys if k.lower().endswith(".png")]
    return [get_public_url(k) for k in png_keys]
# --- DEBUG PROOF (safe to remove later) ---
STORAGE_SUPABASE_VERSION = "v3_has_cached_urls_for_prefix"

from typing import List

def cached_urls_for_prefix(prefix: str) -> List[str]:
    # minimal safe implementation (won't crash import)
    try:
        keys = list_files_by_prefix(prefix)
        png_keys = [k for k in keys if k.lower().endswith(".png")]
        return [get_public_url(k) for k in png_keys]
    except Exception:
        return []

