# app/routers/stat_leaders.py

import os
import tempfile
from typing import Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.storage_supabase import upload_file_return_url
from app.scripts.nfl_stat_leaders_generate import (
    generate_all_stat_leader_posters,
    STAT_CONFIG,
)

router = APIRouter(prefix="/stat-leaders", tags=["Stat Leaders"])


class StatLeadersRequest(BaseModel):
    season: int
    seasontype: int  # 2=regular, 3=playoffs


@router.post("/generate")
def generate(req: StatLeadersRequest) -> Dict[str, Any]:
    phase = "regular" if req.seasontype == 2 else "postseason"

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = generate_all_stat_leader_posters(
            season=req.season,
            seasontype=req.seasontype,
            outdir=tmpdir,
        )

        urls = {}

        for slug, _full_title, _short_title in STAT_CONFIG:
            local_path = outputs[slug]
            storage_key = f"stat_leaders/current/{phase}/{slug}.png"

            urls[slug] = upload_file_return_url(
                local_path,
                storage_key,
            )

    return {
        "season": req.season,
        "seasontype": req.seasontype,
        "phase": phase,
        "count": len(urls),
        "images": urls,
    }
