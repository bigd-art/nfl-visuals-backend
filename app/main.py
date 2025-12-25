import os
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.generator.week_posters import generate_week
from app.generator.favorite_team_poster import generate_favorite_team_poster
from app.services.storage_supabase import upload_file_return_url
from app.services.storage_supabase import cached_urls_for_prefix

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
    # 0) CACHE CHECK (won't break scaling)
    cache_prefix = f"posters/{req.year}/week{req.week}"
    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        return {
            "year": req.year,
            "week": req.week,
            "seasontype": req.seasontype,
            "count": len(cached),
            "images": cached,
            "cached": True,
        }

    # 1) Generate posters locally
    try:
        out_dir = generate_week(req.year, req.week, req.seasontype)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")

    # 2) Find all generated PNGs
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png")))
    if not pngs:
        raise HTTPException(status_code=500, detail=f"No PNGs found in {out_dir}")

    # 3) Upload each PNG to Supabase + collect URLs
    urls = []
    for path in pngs:
        storage_key = f"posters/{req.year}/week{req.week}/{os.path.basename(path)}"
        try:
            url = upload_file_return_url(local_path=path, storage_key=storage_key)
            urls.append(url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed for {path}: {e}")

    return {
        "year": req.year,
        "week": req.week,
        "seasontype": req.seasontype,
        "count": len(urls),
        "images": urls,
    }
    
@app.post("/generate-favorite-team")
def generate_favorite_team_endpoint(req: FavoriteTeamRequest):
    team_upper = req.team.upper()

    # 0) CACHE CHECK (won't break scaling)
    cache_prefix = f"posters/{req.year}/week{req.week}/favorite/{team_upper}"
    cached = cached_urls_for_prefix(cache_prefix)
    if cached:
        return {
            "year": req.year,
            "week": req.week,
            "seasontype": req.seasontype,
            "team": team_upper,
            "count": len(cached),
            "images": cached,
            "cached": True,
        }

    # 1) Generate ONE poster locally
    try:
        png_path = generate_favorite_team_poster(
            req.year,
            req.week,
            req.seasontype,
            team_upper,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generator failed: {e}")

    if not os.path.exists(png_path):
        raise HTTPException(status_code=500, detail=f"PNG not found: {png_path}")

    # 2) Upload PNG to Supabase
    storage_key = (
        f"posters/{req.year}/week{req.week}/favorite/{team_upper}/"
        f"{os.path.basename(png_path)}"
    )

    try:
        url = upload_file_return_url(
            local_path=png_path,
            storage_key=storage_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {
        "year": req.year,
        "week": req.week,
        "seasontype": req.seasontype,
        "team": team_upper,
        "count": 1,
        "images": [url],
        "cached": False,
    }

