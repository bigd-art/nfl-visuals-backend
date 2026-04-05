from fastapi import APIRouter, HTTPException, Query
import requests

from app.scripts.nightly_publish_current_favorite_team import get_current_favorite_team_payload

router = APIRouter(prefix="/favorite-team", tags=["favorite-team"])


@router.get("/current")
def current_favorite_team(team: str = Query(default="")):
    try:
        payload = get_current_favorite_team_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if not resp.ok:
            return payload

        meta = resp.json()

        if team:
            team_abbr = team.strip().upper()
            return {
                "season": meta.get("season"),
                "week": meta.get("week"),
                "seasontype": meta.get("seasontype"),
                "team": team_abbr,
                "image_url": meta.get("posters", {}).get(team_abbr),
                "metadata_url": meta.get("metadata_url"),
            }

        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
