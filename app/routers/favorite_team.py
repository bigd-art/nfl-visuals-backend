import os
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app.generator.favorite_team_poster import generate_favorite_team_poster
from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)

router = APIRouter(prefix="/favorite-team", tags=["favorite-team"])


class FavoriteTeamRequest(BaseModel):
    team: str
    year: int
    week: int
    seasontype: int = 2


def _normalize_team(team: str) -> str:
    t = (team or "").strip().upper()
    return "WSH" if t == "WAS" else t


def _week_folder(week: int) -> str:
    return f"week{str(week).zfill(2)}"


def _kind_from_seasontype(seasontype: int) -> str:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix_favorite(year: int, kind: str, week: int, team: str) -> str:
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/favorite/{team}/"


@router.post("/generate")
def generate_favorite_team(req: FavoriteTeamRequest) -> Dict[str, Any]:
    team = _normalize_team(req.team)
    kind = _kind_from_seasontype(req.seasontype)
    prefix = _prefix_favorite(req.year, kind, req.week, team)

    try:
        cached = cached_urls_for_prefix(prefix)
        if cached:
            return {
                "ok": True,
                "cache_hit": True,
                "url": cached[0],
                "urls": cached,
            }

        png_path = generate_favorite_team_poster(
            year=req.year,
            week=req.week,
            seasontype=req.seasontype,
            team=team,
        )

        if not png_path or not os.path.exists(png_path):
            raise RuntimeError("Favorite team poster not generated.")

        key = f"{prefix}{os.path.basename(png_path)}"
        url = upload_file_return_url(png_path, key)

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
