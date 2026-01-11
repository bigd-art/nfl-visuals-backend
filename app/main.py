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

app = FastAPI()


# =========================
# Helpers
# =========================

def _week_folder(week: int) -> str:
    return f"week{str(week).zfill(2)}"


def _kind_from_seasontype(seasontype: int) -> Literal["regular", "playoffs"]:
    # 2 = regular, 3 = playoffs
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix_week(year: int, kind: str, week: int) -> str:
    # ONE TRUE SCHEMA (DO NOT CHANGE)
    # posters_v3/{year}/{regular|playoffs}/weekXX/
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/"


def _prefix_favorite(year: int, kind: str, week: int, team: str) -> str:
    team = team.strip().upper()
    return f"posters_v3/{year}/{kind}/{_week_folder(week)}/favorite/{team}/"


def _upload_all_pngs(out_dir: str, prefix: str) -> List[str]:
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        raise HTTPException(status_code=500, detail=f"No PNGs found in output dir: {out_dir}")

    urls: List[str] = []
    for p in pngs:
        key = f"{prefix}{os.path.basename(p)}"
        url = upload_file_return_url(p, key)
        urls.append(url)

    urls.sort()
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
            "year": req.year,
            "week": str(req.week).zfill(2),
            "kind": kind,
            "prefix": prefix,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # IMPORTANT: seasontype=2 for regular
    out_dir = generate_week(req.year, req.week, 2)
    urls = _upload_all_pngs(out_dir, prefix)

    return {
        "cache_hit": False,
        "year": req.year,
        "week": str(req.week).zfill(2),
        "kind": kind,
        "prefix": prefix,
        "count": len(urls),
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
            "year": req.year,
            "week": str(req.week).zfill(2),
            "kind": kind,
            "team": req.team.strip().upper(),
            "prefix": prefix,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # IMPORTANT: seasontype=2 for regular
    png_path = generate_favorite_team_poster(req.year, req.week, 2, req.team.strip().upper())
    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail="Favorite team poster was not generated.")

    key = f"{prefix}{os.path.basename(png_path)}"
    url = upload_file_return_url(png_path, key)

    return {
        "cache_hit": False,
        "year": req.year,
        "week": str(req.week).zfill(2),
        "kind": kind,
        "team": req.team.strip().upper(),
        "prefix": prefix,
        "count": 1,
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
            "year": req.year,
            "week": str(req.week).zfill(2),
            "kind": kind,
            "prefix": prefix,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # IMPORTANT: seasontype=3 for playoffs
    out_dir = generate_week(req.year, req.week, 3)
    urls = _upload_all_pngs(out_dir, prefix)

    return {
        "cache_hit": False,
        "year": req.year,
        "week": str(req.week).zfill(2),
        "kind": kind,
        "prefix": prefix,
        "count": len(urls),
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
            "year": req.year,
            "week": str(req.week).zfill(2),
            "kind": kind,
            "team": req.team.strip().upper(),
            "prefix": prefix,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # IMPORTANT: seasontype=3 for playoffs
    png_path = generate_favorite_team_poster(req.year, req.week, 3, req.team.strip().upper())
    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail="Favorite team poster was not generated.")

    key = f"{prefix}{os.path.basename(png_path)}"
    url = upload_file_return_url(png_path, key)

    return {
        "cache_hit": False,
        "year": req.year,
        "week": str(req.week).zfill(2),
        "kind": kind,
        "team": req.team.strip().upper(),
        "prefix": prefix,
        "count": 1,
        "images": [url],
        "timing_ms": int((time.time() - t0) * 1000),
    }


# =========================
# Backwards-compatible endpoints
# =========================

class OldWeekRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2  # 2=regular, 3=playoffs


class OldFavoriteRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2
    team: str


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
        return generate_favorite_regular(FavoriteTeamRequest(year=req.year, week=req.week, team=req.team))
    return generate_favorite_playoffs(FavoriteTeamRequest(year=req.year, week=req.week, team=req.team))

