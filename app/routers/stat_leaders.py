# app/routers/stat_leaders.py

from typing import Dict, Any, List
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.storage_supabase import cached_urls_for_prefix
from app.scripts.stat_leaders_phone_v2 import generate_stat_leaders_and_upload

router = APIRouter(prefix="/stat-leaders", tags=["Stat Leaders"])


class StatLeadersRequest(BaseModel):
    season: int
    seasontype: int = 2  # 2=regular, 3=playoffs


def _kind(seasontype: int) -> str:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix(season: int, seasontype: int) -> str:
    # ONE TRUE SCHEMA:
    # posters_v3/{season}/{regular|playoffs}/leaders/
    return f"posters_v3/{season}/{_kind(seasontype)}/leaders/"


@router.post("/generate")
def generate(req: StatLeadersRequest) -> Dict[str, Any]:
    t0 = time.time()
    season = int(req.season)
    seasontype = int(req.seasontype)

    if seasontype not in (2, 3):
        raise HTTPException(status_code=400, detail="seasontype must be 2 (regular) or 3 (playoffs)")

    prefix = _prefix(season, seasontype)

    cached: List[str] = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "season": season,
            "seasontype": seasontype,
            "prefix": prefix,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # This function should:
    # - generate the 10 phone-friendly posters
    # - upload them to Supabase under the SAME prefix schema
    # - return the final public URLs
    urls = generate_stat_leaders_and_upload(season=season, seasontype=seasontype)

    return {
        "cache_hit": False,
        "season": season,
        "seasontype": seasontype,
        "prefix": prefix,
        "count": len(urls),
        "images": urls,
        "timing_ms": int((time.time() - t0) * 1000),
    }

