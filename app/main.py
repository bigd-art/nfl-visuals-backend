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

from app.routers.stat_leaders import router as stat_leaders_router

app = FastAPI()

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
    pngs = sorted(glob.glob(os.path.join(out_dir, "*.png
