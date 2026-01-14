# app/main.py
import os
import glob
import time
import tempfile
from typing import List, Dict, Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.generator.favorite_team_poster import generate_favorite_team_poster

from app.services.storage_supabase import (
    upload_file_return_url,
    cached_urls_for_prefix,
)

# Stat leaders generator script (the one you added)
# app/scripts/nfl_stat_leaders_generate.py
from app.scripts import nfl_stat_leaders_generate as statgen

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


def _prefix_stat_leaders(year: int, kind: str) -> str:
    # posters_v3/{year}/{regular|playoffs}/stat_leaders/
    return f"posters_v3/{year}/{kind}/stat_leaders/"


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


def _upload_exact_pngs(paths: List[str], prefix: str) -> List[str]:
    urls: List[str] = []
    for p in paths:
        if not os.path.exists(p):
            raise HTTPException(status_code=500, detail=f"Missing generated file: {p}")
        key = f"{prefix}{os.path.basename(p)}"
        urls.append(upload_file_return_url(p, key))
    urls.sort()
    return urls


def _run_stat_leaders_generator(year: int, seasontype: int) -> List[str]:
    """
    Runs app/scripts/nfl_stat_leaders_generate.py in-process by faking argv,
    writes output to a temp folder, and returns the two generated PNG paths.
    """
    with tempfile.TemporaryDirectory() as td:
        outdir = td

        # Fake command-line args for the generator's argparse
        import sys
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "nfl_stat_leaders_generate.py",
                "--season", str(year),
                "--seasontype", str(seasontype),
                "--outdir", outdir,
            ]
            statgen.main()
        finally:
            sys.argv = old_argv

        tag = "reg" if int(seasontype) == 2 else "post"
        off_path = os.path.join(outdir, f"offense_stat_leaders_{year}_{tag}.png")
        def_path = os.path.join(outdir, f"defense_stat_leaders_{year}_{tag}.png")

        if not os.path.exists(off_path) or not os.path.exists(def_path):
            raise HTTPException(status_code=500, detail="Stat leaders PNGs were not generated.")

        # Copy them out of temp (because temp folder disappears)
        # We'll return bytes via upload immediately by re-reading paths here:
        # simplest: upload from temp before exiting context => do that in caller.
        # So instead of returning paths, we return content-ready temp paths by delaying exit.
        # BUT we are in a context manager; so we must upload before it exits.
        # Therefore: caller should upload inside this function. We'll do that instead elsewhere.
        return [off_path, def_path]


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


class StatLeadersRequest(BaseModel):
    year: int
    seasontype: int = 2  # 2=regular, 3=playoffs


# =========================
# Health
# =========================

@app.get("/health")
def health():
    return {"ok": True}


# =========================
# STAT LEADERS (NEW)
# =========================

@app.get("/stat-leaders")
def stat_leaders_cached(
    year: int = Query(...),
    seasontype: int = Query(2, description="2=regular, 3=playoffs"),
) -> Dict[str, Any]:
    """
    Fast endpoint for Expo: returns cached URLs only (no generation).
    Your nightly cron should keep this warm.
    """
    if int(seasontype) not in (2, 3):
        raise HTTPException(status_code=400, detail="seasontype must be 2 or 3")

    kind = _kind_from_seasontype(seasontype)
    prefix = _prefix_stat_leaders(year, kind)

    cached = cached_urls_for_prefix(prefix)
    return {
        "cache_hit": bool(cached),
        "year": year,
        "kind": kind,
        "prefix": prefix,
        "count": len(cached) if cached else 0,
        "images": cached or [],
    }


@app.post("/generate-stat-leaders")
def generate_stat_leaders(req: StatLeadersRequest) -> Dict[str, Any]:
    """
    Generates stat leaders posters (offense + defense) and uploads to Supabase.
    Uses the same cache-first behavior as the rest of your API.
    """
    t0 = time.time()

    if int(req.seasontype) not in (2, 3):
        raise HTTPException(status_code=400, detail="seasontype must be 2 or 3")

    kind = _kind_from_seasontype(req.seasontype)
    prefix = _prefix_stat_leaders(req.year, kind)

    cached = cached_urls_for_prefix(prefix)
    if cached:
        return {
            "cache_hit": True,
            "year": req.year,
            "kind": kind,
            "prefix": prefix,
            "count": len(cached),
            "images": cached,
            "timing_ms": int((time.time() - t0) * 1000),
        }

    # Generate into temp folder and upload immediately before it disappears
    with tempfile.TemporaryDirectory() as td:
        import sys
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "nfl_stat_leaders_generate.py",
                "--season", str(req.year),
                "--seasontype", str(req.seasontype),
                "--outdir", td,
            ]
            statgen.main()
        finally:
            sys.argv = old_argv

        # Upload BOTH PNGs from the temp dir (it contains only these 2)
        urls = _upload_all_pngs(td, prefix)

    return {
        "cache_hit": False,
        "year": req.year,
        "kind": kind,
        "prefix": prefix,
        "count": len(urls),
        "images": urls,
        "timing_ms": int((time.time() - t0) * 1000),
    }


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
