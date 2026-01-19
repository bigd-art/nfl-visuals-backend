from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import os
import traceback
import re

# ✅ IMPORTANT:
# This should point to the NEW robust generator file:
# app/generator/team_stat_leaders.py  (from my previous message)
from app.generator.team_stat_leaders import generate_team_stat_leader_posters

from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/team-stat-leaders", tags=["Team Stat Leaders"])


class TeamStatLeadersRequest(BaseModel):
    team: str                    # e.g. "ARI"
    team_url: Optional[str] = None  # ESPN team stats URL (optional; generator can build from TEAM map)
    outdir: Optional[str] = None


_TEAM_RE = re.compile(r"^[A-Z]{2,4}$")


def _normalize_team(team: str) -> str:
    t = (team or "").strip().upper()
    if not t or not _TEAM_RE.match(t):
        raise ValueError("team must be 2–4 letters (e.g., ARI, KC, LAR).")
    # Normalize WAS/WSH
    if t == "WAS":
        t = "WSH"
    return t


def _validate_team_url(team_url: Optional[str]) -> Optional[str]:
    if not team_url:
        return None
    u = team_url.strip()
    if not u.startswith("https://www.espn.com/nfl/team/stats"):
        raise ValueError("team_url must be an ESPN NFL team stats URL.")
    return u


@router.post("/generate")
def generate_team_stat_leaders(req: TeamStatLeadersRequest) -> Dict[str, Any]:
    try:
        team = _normalize_team(req.team)
        team_url = _validate_team_url(req.team_url)

        outdir = (req.outdir or "/tmp").strip() or "/tmp"
        os.makedirs(outdir, exist_ok=True)

        # 1) Generate BOTH posters locally on server using robust ESPN detection
        # Returns file paths like /tmp/team_offense_leaders_SEA.png, /tmp/team_defense_leaders_SEA.png
        paths: List[str] = generate_team_stat_leader_posters(
            team=team,
            team_url=team_url,
            out_dir=outdir,
        )

        if not paths or len(paths) != 2:
            raise RuntimeError(f"Expected 2 poster paths, got {paths}")

        out_off, out_def = paths[0], paths[1]

        # 2) Upload to Supabase + return PUBLIC URLs
        # Cache-bust key so Expo doesn't show old/stale images
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        key_off = f"team_leaders/{team}/{team.lower()}_offense_{ts}.png"
        key_def = f"team_leaders/{team}/{team.lower()}_defense_{ts}.png"

        url_off = upload_file_return_url(out_off, key_off)
        url_def = upload_file_return_url(out_def, key_def)

        return {
            "ok": True,
            "team": team,
            "images": [url_off, url_def],
            "keys": [key_off, key_def],
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": e.__class__.__name__,
                "team": getattr(req, "team", None),
                "team_url": getattr(req, "team_url", None),
            },
        )
