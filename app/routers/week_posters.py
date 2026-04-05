import os
from fastapi import APIRouter
from pydantic import BaseModel
import requests

from app.generator.week_posters import generate_week
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/week-posters", tags=["week-posters"])


class WeekPosterRequest(BaseModel):
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
def generate_week_posters(req: WeekPosterRequest):
    kind = "regular" if req.seasontype == 2 else "playoffs"
    folder = f"week{str(req.week).zfill(2)}"

    try:
        out_dir = generate_week(req.year, req.week, req.seasontype)

        urls = []
        for filename in sorted(os.listdir(out_dir)):
            if not filename.lower().endswith(".png"):
                continue

            local_path = os.path.join(out_dir, filename)
            storage_key = f"posters_v3/{req.year}/{kind}/{folder}/games/{filename}"
            public_url = _public_url(storage_key)

            if _url_exists(public_url):
                urls.append(public_url)
            else:
                urls.append(upload_file_return_url(local_path, storage_key))

        return {"ok": True, "count": len(urls), "urls": urls}

    except Exception as e:
        return {"ok": False, "error": str(e), "urls": []}
