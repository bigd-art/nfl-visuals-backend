from fastapi import APIRouter, HTTPException
import requests

from app.scripts.nightly_publish_tankathon_mock import get_current_tankathon_mock_payload

router = APIRouter(prefix="/tankathon-mock", tags=["tankathon-mock"])


@router.get("/current")
def tankathon_mock_current():
    try:
        payload = get_current_tankathon_mock_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if resp.ok:
            return resp.json()
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
