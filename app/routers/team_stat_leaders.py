from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
import os
import traceback

from app.scripts import nfl_team_stat_leaders_generate as team_gen
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/team-stat-leaders", tags=["Team Stat Leaders"])


class TeamStatLeadersRequest(BaseModel):
    team: str

    # NEW: allow selecting regular vs postseason by year + seasontype
    year: Optional[int] = None          # e.g. 2025
    seasontype: Optional[int] = None    # 2 regular, 3 postseason

    # Backward compatibility: older client can still send a full team_url
    team_url: Optional[str] = None

    outdir: Optional[str] = None


@router.post("/generate")
def generate_team_stat_leaders(req: TeamStatLeadersRequest) -> Dict[str, Any]:
    try:
        team = req.team.upper().strip()
        outdir = req.outdir or "/tmp"
        os.makedirs(outdir, exist_ok=True)

        # Prefer server-built URL (stable + consistent)
        year = None
        seasontype = None

        if req.year is not None and req.seasontype is not None:
            year = int(req.year)
            seasontype = int(req.seasontype)
            team_url = team_gen.build_team_stats_url(team, year, seasontype)
        elif req.team_url:
            team_url = req.team_url
        else:
            raise RuntimeError("Missing inputs: provide (year + seasontype) OR team_url.")

        # Scrape
        leaders = team_gen.extract_team_leaders(team_url)

        updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
        scope_label = "Regular Season" if seasontype == 2 else "Postseason" if seasontype == 3 else None
        if year and scope_label:
            subtitle = f"{team} • {year} {scope_label} • Updated {updated}"
        elif year:
            subtitle = f"{team} • {year} • Updated {updated}"
        else:
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

        required = offense_order + defense_order
        missing = [k for k in required if k not in leaders]
        if missing:
            raise RuntimeError(f"Leaders missing categories: {missing}. Keys present: {list(leaders.keys())}")

        offense_sections = [(cat, leaders[cat][0], leaders[cat][1], leaders[cat][3]) for cat in offense_order]
        defense_sections = [(cat, leaders[cat][0], leaders[cat][1], leaders[cat][3]) for cat in defense_order]

        out_off = os.path.join(outdir, f"{team.lower()}_offense_leaders.png")
        out_def = os.path.join(outdir, f"{team.lower()}_defense_leaders.png")

        # Keep poster design identical
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

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Separate keys so regular vs postseason never collide
        year_part = str(year) if year else "year?"
        st_part = f"st{seasontype}" if seasontype in (2, 3) else "st?"

        key_off = f"team_leaders/{team}/{year_part}/{st_part}/{team.lower()}_offense_{ts}.png"
        key_def = f"team_leaders/{team}/{year_part}/{st_part}/{team.lower()}_defense_{ts}.png"

        url_off = upload_file_return_url(out_off, key_off)
        url_def = upload_file_return_url(out_def, key_def)

        return {
            "ok": True,
            "team": team,
            "year": year,
            "seasontype": seasontype,
            "team_url": team_url,
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
                "year": getattr(req, "year", None),
                "seasontype": getattr(req, "seasontype", None),
            },
        )
