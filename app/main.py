# app/main.py

import os
import glob
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.generator.favorite_team_poster import generate_favorite_team_poster
from app.services.storage_supabase import upload_file_return_url, cached_urls_for_prefix

app = FastAPI()


class WeekRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2  # 1=preseason, 2=regular, 3=postseason


class FavoriteTeamRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2
    team: str  # e.g. "SEA"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate-week")
def generate_week_endpoint(req: WeekRequest):
    t0 = time.time()

    # Normalize week formatting to avoid cache busting (week 1 == "01")
    week_str = str(req.week).zfill(2)

    # 0) CACHE CHECK (Supabase listing = storage-backed cache)
    cache_prefix = f"posters/{req.year}/week{week_str}"
    print(f"[generate-week] cache check prefix={cache_prefix}")

    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[generate-week] CACHE HIT prefix={cache_prefix} count={len(cached)} ms={dt_ms}")
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

    # 1) Generate posters locally
    gen0 = time.time()
    try:
        out_dir = generate_week(req.year, req.week, req.seasontype)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")
    gen_ms = int((time.time() - gen0) * 1000)

    # 2) Find all generated PNGs
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        raise HTTPException(status_code=500, detail=f"No PNGs found in {out_dir}")

    # 3) Upload each PNG to Supabase + collect URLs
    up0 = time.time()
    urls: List[str] = []
    for path in pngs:
        storage_key = f"posters/{req.year}/week{week_str}/{os.path.basename(path)}"
        try:
            url = upload_file_return_url(local_path=path, storage_key=storage_key)
            urls.append(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed for {path}: {e}")
    upload_ms = int((time.time() - up0) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    print(
        f"[generate-week] DONE prefix={cache_prefix} "
        f"generated={len(pngs)} gen_ms={gen_ms} upload_ms={upload_ms} total_ms={total_ms}"
    )

    return {
        "cache_hit": False,
        "cache_prefix": cache_prefix,
        "generated_count": len(pngs),
        "generation_ms": gen_ms,
        "upload_ms": upload_ms,
        "timing_ms": total_ms,
        "year": req.year,
        "week": week_str,
        "seasontype": req.seasontype,
        "count": len(urls),
        "images": urls,
    }


@app.post("/generate-favorite-team")
def generate_favorite_team_endpoint(req: FavoriteTeamRequest):
    t0 = time.time()

    team_upper = req.team.upper()
    week_str = str(req.week).zfill(2)

    # 0) CACHE CHECK (Supabase listing = storage-backed cache)
    cache_prefix = f"posters/{req.year}/week{week_str}/favorite/{team_upper}"
    print(f"[favorite-team] cache check prefix={cache_prefix}")

    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[favorite-team] CACHE HIT prefix={cache_prefix} count={len(cached)} ms={dt_ms}")
        return {
            "cache_hit": True,
            "cache_prefix": cache_prefix,
            "cached_count": len(cached),
            "timing_ms": dt_ms,
            "year": req.year,
            "week": week_str,
            "seasontype": req.seasontype,
            "team": team_upper,
            "count": len(cached),
            "images": cached,
        }

    print(f"[favorite-team] CACHE MISS prefix={cache_prefix}")

    # 1) Generate ONE poster locally
    gen0 = time.time()
    try:
        png_path = generate_favorite_team_poster(
            req.year,
            req.week,
            req.seasontype,
            team_upper,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")
    gen_ms = int((time.time() - gen0) * 1000)

    if not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail=f"PNG not found: {png_path}")

    # 2) Upload PNG to Supabase (match the same normalized week folder)
    storage_key = (
        f"posters/{req.year}/week{week_str}/favorite/{team_upper}/"
        f"{os.path.basename(png_path)}"
    )

    up0 = time.time()
    try:
        url = upload_file_return_url(local_path=png_path, storage_key=storage_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    upload_ms = int((time.time() - up0) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    print(
        f"[favorite-team] DONE prefix={cache_prefix} "
        f"gen_ms={gen_ms} upload_ms={upload_ms} total_ms={total_ms} key={storage_key}"
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
        "team": team_upper,
        "count": 1,
        "images": [url],
    }
