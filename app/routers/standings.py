from fastapi import APIRouter, HTTPException, Query
from app.scripts.nfl_standings_conference_generate import generate_and_upload_standings_conference

router = APIRouter(prefix="/standings", tags=["standings"])

@router.get("/conference")
def standings_conference(season: int = Query(..., ge=2002, le=2100)):
    try:
        public_url = generate_and_upload_standings_conference(season)
        return {"season": season, "image_url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
