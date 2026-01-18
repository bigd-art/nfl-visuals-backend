from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import os
import traceback

from app.scripts.nfl_team_stat_leaders_generate import (
    extract_team_leaders,
    draw_leaders_grid_poster,
)

router = APIRouter(prefix="/team-stat-leaders", tags=["Team Stat Leaders"])


class TeamStatLeadersRequest(BaseModel):
    team: str            # e.g. "ARI"
    team_url: str        # ESPN team stats URL
    outdir: str | None = None


@router.post("/generate")
def generate_team_stat_leaders(req: TeamStatLeadersRequest) -> Dict[str, Any]:
    try:
        team = req.team.upper()
        outdir = req.outdir or "/tmp"
        os.makedirs(outdir, exist_ok=True)

        # --- Core logic (unchanged) ---
        leaders = extract_team_leaders(req.team_url)

        updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        subtitle = f"{team} • Updated {updated}"

        offense_order = [
            "Passing Yards",
            "Passing TDs",
            "Interceptions Thrown",
            "Rushing Yards",
            "Rushing TDs",
            "Receiving Yards",
            "Receiving TDs",
        ]

        defense_order = ["Sacks", "Tackles", "Interceptions"]

        offense_sections = [
            (cat, leaders[cat][0], leaders[cat][1], leaders[cat][2])
            for cat in offense_order
        ]

        defense_sections = [
            (cat, leaders[cat][0], leaders[cat][1], leaders[cat][2])
            for cat in defense_order
        ]

        out_off = os.path.join(outdir, f"{team.lower()}_offense_leaders.png")
        out_def = os.path.join(outdir, f"{team.lower()}_defense_leaders.png")

        draw_leaders_grid_poster(
            out_off,
            "Offensive Statistical Leaders",
            subtitle,
            offense_sections,
            cols=2,
            rows=4,
        )

        draw_leaders_grid_poster(
            out_def,
            "Defensive Statistical Leaders",
            subtitle,
            defense_sections,
            cols=1,
            rows=3,
        )

        # ✅ ALWAYS return JSON
        return {
            "ok": True,
            "team": team,
            "images": [out_off, out_def],
        }

    except Exception as e:
        # 🔥 THIS IS THE CRITICAL FIX
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": e.__class__.__name__,
                "team": req.team,
                "team_url": req.team_url,
            },
        )
