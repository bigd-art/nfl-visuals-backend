# app/routers/team_stat_leaders.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import traceback

from app.scripts import nfl_team_stat_leaders_generate as team_gen
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/team-stat-leaders", tags=["Team Stat Leaders"])


class TeamStatLeadersRequest(BaseModel):
    team: str                     # "SEA"
    season: int = 2025            # 2025
    mode: str = "regular"         # "regular" | "playoffs" | "both"
    outdir: Optional[str] = None  # "/tmp"


def _season_types(mode: str) -> List[int]:
    m = (mode or "").strip().lower()
    if m == "regular":
        return [2]
    if m in {"playoffs", "postseason"}:
        return [3]
    if m == "both":
        return [2, 3]
    raise ValueError("mode must be one of: regular, playoffs, both")


@router.post("/generate")
def generate_team_stat_leaders(req: TeamStatLeadersRequest) -> Dict[str, Any]:
    try:
        team = req.team.upper().strip()
        season = int(req.season)

        outdir = req.outdir or "/tmp"
        os.makedirs(outdir, exist_ok=True)

        season_types = _season_types(req.mode)

        images: List[str] = []
        keys: List[str] = []
        meta: List[Dict[str, Any]] = []

        for seasontype in season_types:
            leaders = team_gen.extract_team_leaders(team=team, season=season, seasontype=seasontype)

            # poster subtitle
            updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
            label = "Regular Season" if seasontype == 2 else "Postseason"
            subtitle = f"{team} • {label} • Updated {updated}"

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

            # local render paths
            out_off = os.path.join(outdir, f"{team.lower()}_{season}_{seasontype}_offense.png")
            out_def = os.path.join(outdir, f"{team.lower()}_{season}_{seasontype}_defense.png")

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

            # upload to supabase
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            scope = "regular" if seasontype == 2 else "postseason"

            key_off = f"team_leaders/{team}/{season}/{scope}/{team.lower()}_offense_{ts}.png"
            key_def = f"team_leaders/{team}/{season}/{scope}/{team.lower()}_defense_{ts}.png"

            url_off = upload_file_return_url(out_off, key_off)
            url_def = upload_file_return_url(out_def, key_def)

            images.extend([url_off, url_def])
            keys.extend([key_off, key_def])
            meta.append({"season": season, "seasontype": seasontype, "scope": scope})

        return {
            "ok": True,
            "team": team,
            "season": season,
            "mode": req.mode,
            "images": images,
            "keys": keys,
            "meta": meta,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": e.__class__.__name__,
                "team": getattr(req, "team", None),
                "season": getattr(req, "season", None),
                "mode": getattr(req, "mode", None),
            },
        )
