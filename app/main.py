# app/main.py

import os
import glob
import time
from typing import List, Dict, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.generator.favorite_team_poster import generate_favorite_team_poster

from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)
from app.routers.team_stat_leaders import router as team_stat_leaders_router
from app.routers.stat_leaders import router as stat_leaders_router

from app.routers.standings import router as standings_router
app.include_router(standings_router)


app = FastAPI()
app.include_router(stat_leaders_router)
app.include_router(team_stat_leaders_router)

# =========================
# Helpers
# =========================

def _week_folder(week: int) -> str:
    return f"week{str(week).zfill(2)}"


def _kind_from_seasontype(seasontype: int) -> Literal["regular", "playoffs"]:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix_week(year: int, kind: str, week: int) -> str:
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/"


def _prefix_favorite(year: int, kind: str, week: int, team: str) -> str:
    team = team.strip().upper()
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/favorite/{team}/"


def _upload_all_pngs(out_dir: str, prefix: str) -> List[str]:
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        raise HTTPException(status_code=500, detail="No PNGs generated.")

    urls: List[str] = []
    for p in pngs:
        key = f"{prefix}{os.path.basename(p)}"
        url = upload_file_return_url(p, key)
        urls.append(url)

    return urls


# =========================
# Request Models
# =========================

class WeekRequest(BaseModel):
    year: int
    week: int


class FavoriteTeamRequest(BaseModel):
    year: int
    week: int
    team: str


class OldWeekRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2


class OldFavoriteRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2
    team: str


# =========================
# Health
# =========================

@app.get("/health")
def health():
    return {"ok": True}


# =========================
# REGULAR SEASON
# =========================

@app.post("/generate-week-regular")
def generate_week_regular(req: WeekRequest) -> Dict[str, Any]:
    t0 = time.time()
    kind = "regular"
    prefix = _prefix_week(req.year, kind, req.week)

    cached = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    out_dir = generate_week(req.year, req.week, 2)
    urls = _upload_all_pngs(out_dir, prefix)

    return {
        "cache_hit": False,
        "images": urls,
        "timing_ms": int((time.time() - t0) * 1000),
    }


@app.post("/generate-favorite-regular")
def generate_favorite_regular(req: FavoriteTeamRequest) -> Dict[str, Any]:
    t0 = time.time()
    kind = "regular"
    prefix = _prefix_favorite(req.year, kind, req.week, req.team)

    cached = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    png_path = generate_favorite_team_poster(req.year, req.week, 2, req.team)
    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail="Favorite team poster not generated.")

    key = f"{prefix}{os.path.basename(png_path)}"
    url = upload_file_return_url(png_path, key)

    return {
        "cache_hit": False,
        "images": [url],
        "timing_ms": int((time.time() - t0) * 1000),
    }


# =========================
# PLAYOFFS
# =========================

@app.post("/generate-week-playoffs")
def generate_week_playoffs(req: WeekRequest) -> Dict[str, Any]:
    t0 = time.time()
    kind = "playoffs"
    prefix = _prefix_week(req.year, kind, req.week)

    cached = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    out_dir = generate_week(req.year, req.week, 3)
    urls = _upload_all_pngs(out_dir, prefix)

    return {
        "cache_hit": False,
        "images": urls,
        "timing_ms": int((time.time() - t0) * 1000),
    }


@app.post("/generate-favorite-playoffs")
def generate_favorite_playoffs(req: FavoriteTeamRequest) -> Dict[str, Any]:
    t0 = time.time()
    kind = "playoffs"
    prefix = _prefix_favorite(req.year, kind, req.week, req.team)

    cached = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    png_path = generate_favorite_team_poster(req.year, req.week, 3, req.team)
    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail="Favorite team poster not generated.")

    key = f"{prefix}{os.path.basename(png_path)}"
    url = upload_file_return_url(png_path, key)

    return {
        "cache_hit": False,
        "images": [url],
        "timing_ms": int((time.time() - t0) * 1000),
    }


# =========================
# BACKWARDS COMPATIBILITY
# =========================

@app.post("/generate-week")
def generate_week_old(req: OldWeekRequest):
    kind = _kind_from_seasontype(req.seasontype)
    if kind == "regular":
        return generate_week_regular(WeekRequest(year=req.year, week=req.week))
    return generate_week_playoffs(WeekRequest(year=req.year, week=req.week))


@app.post("/generate-favorite-team")
def generate_favorite_old(req: OldFavoriteRequest):
    kind = _kind_from_seasontype(req.seasontype)
    if kind == "regular":
        return generate_favorite_regular(
            FavoriteTeamRequest(year=req.year, week=req.week, team=req.team)
        )
    return generate_favorite_playoffs(
        FavoriteTeamRequest(year=req.year, week=req.week, team=req.team)
    )
