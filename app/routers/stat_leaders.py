# app/routers/stat_leaders.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any

from app.scripts.stat_leaders_phone_v2 import generate_stat_leaders_and_upload

router = APIRouter(prefix="/stat-leaders", tags=["Stat Leaders"])


class StatLeadersRequest(BaseModel):
    season: int
    seasontype: int  # 2=regular, 3=playoffs


@router.post("/generate")
def generate(req: StatLeadersRequest) -> Dict[str, Any]:
    urls = generate_stat_leaders_and_upload(
        season=req.season,
        seasontype=req.seasontype,
    )
    return {
        "season": req.season,
        "seasontype": req.seasontype,
        "count": len(urls),
        "images": urls,
    }

