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

        team_abbr = team.strip().upper() if team else ""
        unit_key = unit.strip().lower() if unit else ""

        # Case 1: team + unit => single URL
        if team_abbr and unit_key:
            image_url = posters.get(team_abbr, {}).get(unit_key)
            return {
                "team": team_abbr,
                "unit": unit_key,
                "url": image_url,
                "metadata_url": payload.get("metadata_url"),
            }

        # Case 2: team only => return only that team's units
        if team_abbr:
            team_posters = posters.get(team_abbr, {}) or {}
            urls = [url for url in team_posters.values() if url]

            return {
                "team": team_abbr,
                "posters": {team_abbr: team_posters},
                "urls": urls,
                "metadata_url": payload.get("metadata_url"),
            }

        # Case 3: no team => return everything
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
