from fastapi import APIRouter, HTTPException, Query
import requests

from app.scripts.nightly_publish_team_rosters import get_current_team_rosters_payload

router = APIRouter(prefix="/team-rosters", tags=["team-rosters"])


@router.get("/current")
def team_rosters_current(
    team: str = Query(default=""),
    unit: str = Query(default="")
):
    try:
        payload = get_current_team_rosters_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if not resp.ok:
            return payload

        meta = resp.json()

        if team and unit:
            team_abbr = team.strip().upper()
            unit_key = unit.strip().lower()
            return {
                "team": team_abbr,
                "unit": unit_key,
                "image_url": meta.get("posters", {}).get(team_abbr, {}).get(unit_key),
                "metadata_url": meta.get("metadata_url"),
            }

        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
