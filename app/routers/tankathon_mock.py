from fastapi import APIRouter, HTTPException
import requests

from app.scripts.nightly_publish_tankathon_mock import get_current_tankathon_mock_payload

router = APIRouter(prefix="/tankathon-mock", tags=["tankathon-mock"])


@router.get("/current")
def tankathon_mock_current():
    try:
        payload = get_current_tankathon_mock_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)

        if not resp.ok:
            return payload

        meta = resp.json()

        posters = meta.get("posters", {}) or {}

        if isinstance(posters, dict):
            urls = [url for _, url in sorted(posters.items()) if url]
        elif isinstance(posters, list):
            urls = [url for url in posters if url]
        else:
            urls = []

        return {
            "season": meta.get("season"),
            "posters": posters,
            "urls": urls,
            "metadata_url": payload.get("metadata_url"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
