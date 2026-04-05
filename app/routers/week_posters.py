import os
import glob
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)

router = APIRouter(prefix="/week-posters", tags=["week-posters"])


class WeekPosterRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2


def _week_folder(week: int) -> str:
    return f"week{str(week).zfill(2)}"


def _kind_from_seasontype(seasontype: int) -> str:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix_week(year: int, kind: str, week: int) -> str:
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/games/"


def _upload_all_pngs(out_dir: str, prefix: str) -> List[str]:
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        raise RuntimeError("No PNGs generated.")

    urls: List[str] = []
    for p in pngs:
        key = f"{prefix}{os.path.basename(p)}"
        urls.append(upload_file_return_url(p, key))
    return urls


@router.post("/generate")
def generate_week_posters(req: WeekPosterRequest) -> Dict[str, Any]:
    kind = _kind_from_seasontype(req.seasontype)
    prefix = _prefix_week(req.year, kind, req.week)

    try:
        cached = cached_urls_for_prefix(prefix)
        if cached:
            return {
                "ok": True,
                "cache_hit": True,
                "count": len(cached),
                "urls": cached,
            }

        out_dir = generate_week(req.year, req.week, req.seasontype)
        urls = _upload_all_pngs(out_dir, prefix)

        return {
            "ok": True,
            "cache_hit": False,
            "count": len(urls),
            "urls": urls,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "urls": [],
        }
