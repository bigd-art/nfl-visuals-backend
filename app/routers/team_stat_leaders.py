from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal, List
from datetime import datetime
import os
import traceback

from app.scripts import nfl_team_stat_leaders_generate as team_gen
from app.services.storage_supabase import upload_file_return_url

router = APIRouter(prefix="/team-stat-leaders", tags=["Team Stat Leaders"])

Scope = Literal["regular", "playoffs", "both"]

class TeamStatLeadersRequest(BaseModel):
    team: str
    team_url: str
    season: int = 2025
    seasontype: int = 2          # 2=regular, 3=postseason
    scope: Scope = "regular"     # regular | playoffs | both
    outdir: Optional[str] = None


@router.post("/generate")
def generate_team_stat_leaders(req: TeamStatLeadersRequest) -> Dict[str, Any]:
    try:
        team = req.team.upper().strip()
        outdir = req.outdir or "/tmp"
        os.makedirs(outdir, exist_ok=True)

        def make_one(season: int, seasontype: int, label: str) -> Dict[str, Any]:
            leaders = team_gen.extract_team_leaders(req.team_url, season=season, seasontype=seasontype)

            updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
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

            out_off = os.path.join(outdir, f"{team.lower()}_{label.lower().replace(' ','_')}_offense.png")
            out_def = os.path.join(outdir, f"{team.lower()}_{label.lower().replace(' ','_')}_defense.png")

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
            key_off = f"team_leaders/{team}/{team.lower()}_{label.lower().replace(' ','_')}_offense_{ts}.png"
            key_def = f"team_leaders/{team}/{team.lower()}_{label.lower().replace(' ','_')}_defense_{ts}.png"

            url_off = upload_file_return_url(out_off, key_off)
            url_def = upload_file_return_url(out_def, key_def)

            return {
                "label": label,
                "images": [url_off, url_def],
                "keys": [key_off, key_def],
            }

        outputs: List[Dict[str, Any]] = []

        if req.scope == "both":
            outputs.append(make_one(req.season, 2, "Regular Season"))
            outputs.append(make_one(req.season, 3, "Postseason"))
        elif req.scope == "playoffs":
            outputs.append(make_one(req.season, 3, "Postseason"))
        else:
            outputs.append(make_one(req.season, 2, "Regular Season"))

        # Flatten for Expo
        images = []
        keys = []
        labels = []
        for o in outputs:
            labels.append(o["label"])
            images.extend(o["images"])
            keys.extend(o["keys"])

        return {
            "ok": True,
            "team": team,
            "season": req.season,
            "scope": req.scope,
            "labels": labels,
            "images": images,
            "keys": keys,
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
                "season": req.season,
                "seasontype": req.seasontype,
                "scope": getattr(req, "scope", None),
            },
        )
