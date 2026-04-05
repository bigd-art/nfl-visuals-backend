from fastapi import APIRouter, HTTPException, Query
import requests

from app.scripts.nightly_publish_team_schedules import get_current_team_schedules_payload

router = APIRouter(prefix="/team-schedules", tags=["team-schedules"])


@router.get("/current")
def team_schedules_current(team: str = Query(default="")):
    try:
        payload = get_current_team_schedules_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if not resp.ok:
            return payload

        meta = resp.json()

        if team:
            team_abbr = team.strip().upper()
            return {
                "season": meta.get("season"),
                "team": team_abbr,
                "image_url": meta.get("posters", {}).get(team_abbr),
                "metadata_url": meta.get("metadata_url"),
            }

        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
