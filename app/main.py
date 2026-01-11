import glob
import os
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.generator.favorite_team_poster import generate_favorite_team_poster
from app.services.storage_supabase import upload_file_return_url, cached_urls_for_prefix

app = FastAPI()


# ---------------- Request Models ----------------

class WeekRequest(BaseModel):
    year: int
    week: int
    # ESPN: 1=preseason, 2=regular, 3=postseason
    seasontype: int = 2


class FavoriteTeamRequest(BaseModel):
    year: int
    week: int
    seasontype: int = 2
    team: str  # e.g. "SEA"


# ---------------- Health ----------------

@app.get("/health")
def health():
    return {"ok": True}


# ---------------- Helpers ----------------

def _validate(year: int, week: int, seasontype: int) -> None:
    if year < 2002 or year > 2035:
        raise HTTPException(status_code=400, detail="Year must be between 2002 and 2035.")
    if week < 1 or week > 23:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 23.")
    if seasontype not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="seasontype must be 1, 2, or 3.")


def _week_folder(week: int) -> str:
    return f"week{str(week).zfill(2)}"


def _list_pngs(out_dir: str) -> List[str]:
    if not out_dir or not os.path.isdir(out_dir):
        return []
    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def _norm_team(team: str) -> str:
    t = (team or "").strip().upper()
    if not t or not t.isalpha() or len(t) < 2 or len(t) > 4:
        raise HTTPException(status_code=400, detail="Team must be 2–4 letters (e.g., SEA, KC, LAR).")
    return t


# ---------------- Week Posters ----------------

@app.post("/generate-week")
def generate_week_endpoint(req: WeekRequest):
    _validate(req.year, req.week, req.seasontype)

    t0 = time.time()
    week_folder = _week_folder(req.week)

    # IMPORTANT: include seasontype in the path to prevent regular/playoff collisions
    cache_prefix = f"posters/{req.year}/seasontype{req.seasontype}/{week_folder}/"
    print(f"[generate-week] cache check prefix={cache_prefix}")

    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[generate-week] CACHE HIT count={len(cached)} ms={dt_ms}")
        return {
            "cache_hit": True,
            "cache_prefix": cache_prefix,
            "year": req.year,
            "week": str(req.week).zfill(2),
            "seasontype": req.seasontype,
            "count": len(cached),
            "images": cached,
            "timing_ms": dt_ms,
        }

    print(f"[generate-week] CACHE MISS prefix={cache_prefix}")

    gen0 = time.time()
    try:
        out_dir = generate_week(req.year, req.week, req.seasontype)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")
    gen_ms = int((time.time() - gen0) * 1000)

    pngs = _list_pngs(out_dir)
    if not pngs:
        raise HTTPException(status_code=500, detail=f"No posters generated in: {out_dir}")

    up0 = time.time()
    urls: List[str] = []
    for p in pngs:
        key = f"{cache_prefix}{os.path.basename(p)}"
        urls.append(upload_file_return_url(p, key))
    upload_ms = int((time.time() - up0) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    print(f"[generate-week] DONE gen_ms={gen_ms} upload_ms={upload_ms} total_ms={total_ms} count={len(urls)}")

    return {
        "cache_hit": False,
        "cache_prefix": cache_prefix,
        "year": req.year,
        "week": str(req.week).zfill(2),
        "seasontype": req.seasontype,
        "count": len(urls),
        "images": urls,
        "timing_ms": total_ms,
        "gen_ms": gen_ms,
        "upload_ms": upload_ms,
    }


# ---------------- Favorite Team ----------------

@app.post("/generate-favorite-team")
def generate_favorite_team_endpoint(req: FavoriteTeamRequest):
    team = _norm_team(req.team)
    _validate(req.year, req.week, req.seasontype)

    t0 = time.time()
    week_folder = _week_folder(req.week)

    cache_prefix = f"posters/{req.year}/seasontype{req.seasontype}/{week_folder}/favorite/{team}/"
    print(f"[favorite-team] cache check prefix={cache_prefix}")

    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[favorite-team] CACHE HIT count={len(cached)} ms={dt_ms}")
        return {
            "cache_hit": True,
            "cache_prefix": cache_prefix,
            "year": req.year,
            "week": str(req.week).zfill(2),
            "seasontype": req.seasontype,
            "team": team,
            "count": len(cached),
            "images": cached,
            "timing_ms": dt_ms,
        }

    print(f"[favorite-team] CACHE MISS prefix={cache_prefix}")

    gen0 = time.time()
    try:
        png_path = generate_favorite_team_poster(req.year, req.week, req.seasontype, team)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Favorite team generator failed: {e}")
    gen_ms = int((time.time() - gen0) * 1000)

    if not png_path or not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail="Favorite team poster was not generated.")

    up0 = time.time()
    key = f"{cache_prefix}{os.path.basename(png_path)}"
    url = upload_file_return_url(png_path, key)
    upload_ms = int((time.time() - up0) * 1000)

    total_ms = int((time.time() - t0) * 1000)
    print(f"[favorite-team] DONE gen_ms={gen_ms} upload_ms={upload_ms} total_ms={total_ms}")

    return {
        "cache_hit": False,
        "cache_prefix": cache_prefix,
        "year": req.year,
        "week": str(req.week).zfill(2),
        "seasontype": req.seasontype,
        "team": team,
        "count": 1,
        "images": [url],
        "timing_ms": total_ms,
        "gen_ms": gen_ms,
        "upload_ms": upload_ms,
    }
