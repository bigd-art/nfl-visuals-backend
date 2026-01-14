# app/routers/stat_leaders.py

import os
import subprocess
import tempfile
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)

router = APIRouter()


# =========================
# Request model
# =========================

class StatLeadersRequest(BaseModel):
    year: int
    seasontype: int  # 2 = regular, 3 = playoffs


# =========================
# Helpers
# =========================

def _kind(seasontype: int) -> str:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _leaders_prefix(year: int, kind: str) -> str:
    # ONE TRUE SCHEMA
    # posters_v3/{year}/{regular|playoffs}/leaders/
    return f"posters_v3/{year}/{kind}/leaders/"


# =========================
# Endpoint
# =========================

@router.post("/generate-stat-leaders")
def generate_stat_leaders(req: StatLeadersRequest) -> Dict[str, Any]:
    t0 = time.time()

    year = int(req.year)
    seasontype = int(req.seasontype)
    kind = _kind(seasontype)
    prefix = _leaders_prefix(year, kind)

    # 1) CACHE CHECK
    cached = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "year": year,
            "kind": kind,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # 2) GENERATE LOCALLY
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(
            os.getcwd(),
            "app",
            "scripts",
            "stat_leaders_single_posters.py",
        )

        if not os.path.exists(script_path):
            raise HTTPException(status_code=500, detail="Stat leaders script not found.")

        cmd = [
            "python3",
            script_path,
            "--season", str(year),
            "--seasontype", str(seasontype),
            "--outdir", tmpdir,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Stat leaders generation failed:\n{proc.stderr}",
            )

        # 3) UPLOAD ALL PNGS
        pngs = sorted(
            f for f in os.listdir(tmpdir) if f.lower().endswith(".png")
        )

        if not pngs:
            raise HTTPException(status_code=500, detail="No stat leader PNGs generated.")

        urls = []
        for fname in pngs:
            local_path = os.path.join(tmpdir, fname)
            key = f"{prefix}{fname}"
            url = upload_file_return_url(local_path, key)
            urls.append(url)

    return {
        "cache_hit": False,
        "year": year,
        "kind": kind,
        "count": len(urls),
        "images": urls,
        "timing_ms": int((time.time() - t0) * 1000),
    }

