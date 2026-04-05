from fastapi import APIRouter, HTTPException
import requests

from app.scripts.nightly_publish_tankathon_mock import get_current_tankathon_mock_payload

router = APIRouter(prefix="/tankathon-mock", tags=["tankathon-mock"])


def _extract_urls(value):
    urls = []

    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            urls.append(value)

    elif isinstance(value, list):
        for item in value:
            urls.extend(_extract_urls(item))

    elif isinstance(value, dict):
        for item in value.values():
            urls.extend(_extract_urls(item))

    return urls


@router.get("/current")
def tankathon_mock_current():
    try:
        payload = get_current_tankathon_mock_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)

        if not resp.ok:
            return payload

        meta = resp.json()
        posters = meta.get("posters", {}) or {}

        urls = _extract_urls(posters)

        return {
            "season": meta.get("season"),
            "posters": posters,
            "urls": urls,
            "metadata_url": payload.get("metadata_url"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
