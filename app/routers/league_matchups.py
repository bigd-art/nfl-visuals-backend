import os
import tempfile
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app.scripts.league_matchups_poster import generate_league_matchups_poster
from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)

router = APIRouter(prefix="/league-matchups", tags=["league-matchups"])


class LeagueMatchupsRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2


def _week_folder(week: int) -> str:
    return f"week{str(week).zfill(2)}"


def _kind_from_seasontype(seasontype: int) -> str:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix_matchups(year: int, kind: str, week: int) -> str:
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/league_matchups/"


@router.post("/generate")
def generate_league_matchups(req: LeagueMatchupsRequest) -> Dict[str, Any]:
    kind = _kind_from_seasontype(req.seasontype)
    prefix = _prefix_matchups(req.year, kind, req.week)

    try:
        cached = cached_urls_for_prefix(prefix)
        if cached:
            return {
                "ok": True,
                "cache_hit": True,
                "url": cached[0],
                "urls": cached,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "matchups.png")
            generate_league_matchups_poster(req.year, req.week, req.seasontype, local_path)

            key = f"{prefix}matchups.png"
            url = upload_file_return_url(local_path, key)

        return {
            "ok": True,
            "cache_hit": False,
            "url": url,
            "urls": [url],
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "url": None,
            "urls": [],
        }
