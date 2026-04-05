import os
import tempfile
from fastapi import APIRouter
from pydantic import BaseModel
import requests

from app.generator.league_matchups_poster import generate_league_matchups_poster
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/league-matchups", tags=["league-matchups"])


class LeagueMatchupsRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2


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
def generate_league_matchups(req: LeagueMatchupsRequest):
    kind = "regular" if req.seasontype == 2 else "playoffs"
    folder = f"week{str(req.week).zfill(2)}"
    storage_key = f"posters_v3/{req.year}/{kind}/{folder}/league_matchups/matchups.png"
    public_url = _public_url(storage_key)

    try:
        if _url_exists(public_url):
            return {"ok": True, "url": public_url}

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "matchups.png")
            generate_league_matchups_poster(req.year, req.week, req.seasontype, local_path)
            uploaded = upload_file_return_url(local_path, storage_key)

        return {"ok": True, "url": uploaded}

    except Exception as e:
        return {"ok": False, "error": str(e), "url": None}
