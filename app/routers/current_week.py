from fastapi import APIRouter, HTTPException
import requests

from app.scripts.nightly_publish_current_week import get_current_week_posters_payload

router = APIRouter(prefix="/week-posters", tags=["week-posters"])


@router.get("/current")
def current_week_posters():
    try:
        payload = get_current_week_posters_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if resp.ok:
            return resp.json()
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
