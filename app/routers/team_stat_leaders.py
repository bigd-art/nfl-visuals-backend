from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
import os
import traceback

# KEEP OLD STYLE: use your scripts module
from app.scripts import nfl_team_stat_leaders_generate as team_gen
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/team-stat-leaders", tags=["Team Stat Leaders"])


class TeamStatLeadersRequest(BaseModel):
    team: str            # e.g. "ARI"
    team_url: str        # ESPN team stats URL
    outdir: str | None = None


@router.post("/generate")
def generate_team_stat_leaders(req: TeamStatLeadersRequest) -> Dict[str, Any]:
    try:
        team = req.team.upper().strip()
        outdir = req.outdir or "/tmp"
        os.makedirs(outdir, exist_ok=True)

        # 1) Scrape leaders (now fixed in scripts file)
        leaders = team_gen.extract_team_leaders(req.team_url)

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
            (cat, leaders[cat][0], leaders[cat][1], leaders[cat][3])
            for cat in offense_order
        ]

        defense_sections = [
            (cat, leaders[cat][0], leaders[cat][1], leaders[cat][3])
            for cat in defense_order
        ]


        # 2) Render using your OLD poster style function
        out_off = os.path.join(outdir, f"{team.lower()}_offense_leaders.png")
        out_def = os.path.join(outdir, f"{team.lower()}_defense_leaders.png")

        team_gen.draw_leaders_grid_poster(
            out_off,
            "Offensive Statistical Leaders",
            subtitle,
            offense_sections,
            cols=2,
            rows=4,
        )

        team_gen.draw_leaders_grid_poster(
            out_def,
            "Defensive Statistical Leaders",
            subtitle,
            defense_sections,
            cols=1,
            rows=3,
        )

        # 3) Upload to Supabase + return public URLs
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")  # cache-bust
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
                "team": req.team,
                "team_url": req.team_url,
            },
        )
