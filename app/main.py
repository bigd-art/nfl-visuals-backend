# app/main.py

from fastapi import FastAPI

from app.routers.team_stat_leaders import router as team_stat_leaders_router
from app.routers.stat_leaders import router as stat_leaders_router
from app.routers.standings import router as standings_router
from app.routers.stat_of_the_day import router as stat_of_the_day_router
from app.routers.league_matchups import router as league_matchups_router
from app.routers.pff_big_board import router as pff_big_board_router
from app.routers.tankathon_mock import router as tankathon_mock_router
from app.routers.team_schedules import router as team_schedules_router
from app.routers.team_rosters import router as team_rosters_router

# NEW input-based routers
from app.routers.week_posters import router as week_posters_router
from app.routers.favorite_team import router as favorite_team_router

app = FastAPI()

# Existing feature routers
app.include_router(stat_leaders_router)
app.include_router(team_stat_leaders_router)
app.include_router(standings_router)
app.include_router(stat_of_the_day_router)
app.include_router(league_matchups_router)
app.include_router(pff_big_board_router)
app.include_router(tankathon_mock_router)
app.include_router(team_schedules_router)
app.include_router(team_rosters_router)

# New input-based week/favorite routers
app.include_router(week_posters_router)
app.include_router(favorite_team_router)


@app.get("/health")
def health():
    return {"ok": True}
