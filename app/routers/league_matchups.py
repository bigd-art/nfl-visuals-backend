from fastapi import APIRouter, HTTPException
import requests

from app.scripts.nightly_publish_league_matchups import get_current_league_matchups_payload

router = APIRouter(prefix="/league-matchups", tags=["league-matchups"])


@router.get("/current")
def league_matchups_current():
    try:
        payload = get_current_league_matchups_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if resp.ok:
            meta = resp.json()
            return meta
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
