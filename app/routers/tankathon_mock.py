from fastapi import APIRouter, HTTPException
import requests

from app.scripts.nightly_publish_tankathon_mock import get_current_tankathon_mock_payload

router = APIRouter(prefix="/tankathon-mock", tags=["tankathon-mock"])


def _looks_like_image_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    return (
        (v.startswith("http://") or v.startswith("https://"))
        and any(ext in v for ext in [".png", ".jpg", ".jpeg", ".webp"])
    )


def _collect_image_urls(obj, found=None):
    if found is None:
        found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            # Prefer common image keys
            if isinstance(v, str) and k.lower() in {
                "url",
                "image_url",
                "poster_url",
                "poster",
                "image",
            } and _looks_like_image_url(v):
                found.append(v)
            else:
                _collect_image_urls(v, found)

    elif isinstance(obj, list):
        for item in obj:
            _collect_image_urls(item, found)

    elif isinstance(obj, str):
        if _looks_like_image_url(obj):
            found.append(obj)

    return found


@router.get("/current")
def tankathon_mock_current():
    try:
        payload = get_current_tankathon_mock_payload()

        metadata_url = payload.get("metadata_url")
        if not metadata_url:
            return {
                "season": payload.get("season"),
                "urls": [],
                "detail": "No metadata_url found in payload.",
            }

        resp = requests.get(metadata_url, timeout=20)
        if not resp.ok:
            return {
                "season": payload.get("season"),
                "urls": [],
                "detail": f"Metadata fetch failed with status {resp.status_code}",
                "metadata_url": metadata_url,
            }

        meta = resp.json()

        # First try the whole metadata object
        urls = _collect_image_urls(meta)

        # Remove metadata_url itself if it got picked up for any reason
        urls = [u for u in urls if u != metadata_url]

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                deduped.append(u)

        return {
            "season": meta.get("season", payload.get("season")),
            "url": deduped[0] if len(deduped) == 1 else None,
            "urls": deduped,
            "metadata_url": metadata_url,
            "meta": meta,  # keep this temporarily for debugging
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
