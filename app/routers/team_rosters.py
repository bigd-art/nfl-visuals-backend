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
        posters = meta.get("posters", {}) or {}

        if team and unit:
            team_abbr = team.strip().upper()
            unit_key = unit.strip().lower()
            image_url = posters.get(team_abbr, {}).get(unit_key)

            return {
                "team": team_abbr,
                "unit": unit_key,
                "url": image_url,
                "metadata_url": payload.get("metadata_url"),
            }

        urls = []
        for team_map in posters.values():
            if isinstance(team_map, dict):
                urls.extend([url for url in team_map.values() if url])

        return {
            "posters": posters,
            "urls": urls,
            "metadata_url": payload.get("metadata_url"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
