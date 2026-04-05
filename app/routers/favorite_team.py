import os
from fastapi import APIRouter
from pydantic import BaseModel
import requests

from app.generator.favorite_team_poster import generate_favorite_team_poster
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/favorite-team", tags=["favorite-team"])


class FavoriteTeamRequest(BaseModel):
    team: str
    year: int
    week: int
    seasontype: int = 2


def _normalize_team(team: str) -> str:
    t = (team or "").strip().upper()
    return "WSH" if t == "WAS" else t


def _public_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def _url_exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


@router.post("/generate")
def generate_favorite_team(req: FavoriteTeamRequest):
    team = _normalize_team(req.team)
    kind = "regular" if req.seasontype == 2 else "playoffs"
    folder = f"week{str(req.week).zfill(2)}"

    storage_key = f"posters_v3/{req.year}/{kind}/{folder}/favorite/{team}/favorite_team_poster.png"
    public_url = _public_url(storage_key)

    try:
        if _url_exists(public_url):
            return {"ok": True, "url": public_url}

        local_path = generate_favorite_team_poster(
            year=req.year,
            week=req.week,
            seasontype=req.seasontype,
            team=team,
        )

        uploaded = upload_file_return_url(local_path, storage_key)
        return {"ok": True, "url": uploaded}

    except Exception as e:
        return {"ok": False, "error": str(e), "url": None}
