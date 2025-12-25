# app/main.py

import os
import glob
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.generator.favorite_team_poster import generate_favorite_team_poster
from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)

app = FastAPI()


# ======================
# Request Models
# ======================

class WeekRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2  # 1=preseason, 2=regular, 3=postseason


class FavoriteTeamRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2
    team: str  # e.g. "SEA"


# ======================
# Health Check
# ======================

@app.get("/health")
def health():
    return {"ok": True}


# ======================
# Generate Week Posters
# ======================

@app.post("/generate-week")
def generate_week_endpoint(req: WeekRequest):
    t0 = time.time()
    week_str = str(req.week).zfill(2)

    # IMPORTANT: cache prefix MUST be a folder (trailing slash)
    cache_prefix = f"posters/{req.year}/week{week_str}/"
    print(f"[generate-week] cache check prefix={cache_prefix}")

    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[generate-week] CACHE HIT count={len(cached)} ms={dt_ms}")
        return {
            "cache_hit": True,
            "cache_prefix": cache_prefix,
            "cached_count": len(cached),
            "timing_ms": dt_ms,
            "year": req.year,
            "week": week_str,
            "seasontype": req.seasontype,
            "count": len(cached),
            "images": cached,
        }

    print(f"[generate-week] CACHE MISS prefix={cache_prefix}")

    # Generate posters
    gen0 = time.time()
    try:
        out_dir = generate_week(req.year, req.week, req.seasontype)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")
    gen_ms = int((time.time() - gen0) * 1000)

    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        raise HTTPException(status_code=500, detail="No PNGs generated")

    urls: List[str] = []
    up0 = time.time()
    for path in pngs:
        storage_key = f"{cache_prefix}{os.path.basename(path)}"
        url = upload_file_return_url(path, storage_key)
        urls.append(url)
    upload_ms = int((time.time() - up0) * 1000)

    total_ms = int((time.time() - t0) * 1000)

    print(
        f"[generate-week] DONE gen_ms={gen_ms} upload_ms={upload_ms} total_ms={total_ms}"
    )

    return {
        "cache_hit": False,
        "cache_prefix": cache_prefix,
        "generation_ms": gen_ms,
        "upload_ms": upload_ms,
        "timing_ms": total_ms,
        "year": req.year,
        "week": week_str,
        "seasontype": req.seasontype,
        "count": len(urls),
        "images": urls,
    }


# ======================
# Generate Favorite Team Poster
# ======================

@app.post("/generate-favorite-team")
def generate_favorite_team_endpoint(req: FavoriteTeamRequest):
    t0 = time.time()

    team = req.team.upper()
    week_str = str(req.week).zfill(2)

    # 🚨 THIS WAS THE BUG — MUST END WITH "/"
    cache_prefix = f"posters/{req.year}/week{week_str}/favorite/{team}/"
    print(f"[favorite-team] cache check prefix={cache_prefix}")

    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[favorite-team] CACHE HIT count={len(cached)} ms={dt_ms}")
        return {
            "cache_hit": True,
            "cache_prefix": cache_prefix,
            "cached_count": len(cached),
            "timing_ms": dt_ms,
            "year": req.year,
            "week": week_str,
            "seasontype": req.seasontype,
            "team": team,
            "count": len(cached),
            "images": cached,
        }

    print(f"[favorite-team] CACHE MISS prefix={cache_prefix}")

    # Generate poster
    gen0 = time.time()
    try:
        png_path = generate_favorite_team_poster(
            req.year, req.week, req.seasontype, team
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")
    gen_ms = int((time.time() - gen0) * 1000)

    if not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail="PNG not found")

    storage_key = f"{cache_prefix}{os.path.basename(png_path)}"

    up0 = time.time()
    url = upload_file_return_url(png_path, storage_key)
    upload_ms = int((time.time() - up0) * 1000)

    total_ms = int((time.time() - t0) * 1000)

    print(
        f"[favorite-team] DONE gen_ms={gen_ms} upload_ms={upload_ms} total_ms={total_ms}"
    )

    return {
        "cache_hit": False,
        "cache_prefix": cache_prefix,
        "generation_ms": gen_ms,
        "upload_ms": upload_ms,
        "timing_ms": total_ms,
        "year": req.year,
        "week": week_str,
        "seasontype": req.seasontype,
        "team": team,
        "count": 1,
        "images": [url],
    }

