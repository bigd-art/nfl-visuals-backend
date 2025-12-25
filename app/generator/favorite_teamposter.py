import os
from typing import Dict, List

# Import ONLY from your existing week script (week_posters.py stays untouched)
from app.generator.week_posters import (
    fetch_url,
    fetch_summary,
    scoreboard_url,
    extract_game_ids_from_scoreboard_html,
    generate_poster_for_game,
)


def _normalize_team_abbr(team: str) -> str:
    t = (team or "").strip().upper()
    if not t:
        raise ValueError("team is required (e.g., SEA, DAL, KC).")
    if not (2 <= len(t) <= 4) or not t.isalpha():
        raise ValueError("team must be a 2–4 letter abbreviation (e.g., SEA, KC, LAR).")
    return t


def _extract_team_abbrs_from_summary(summary: Dict) -> List[str]:
    """
    Returns team abbreviations in this game, e.g. ["SEA","SF"].
    """
    try:
        comp = summary["header"]["competitions"][0]
        return [c["team"]["abbreviation"] for c in comp.get("competitors", [])]
    except Exception:
        return []


def find_game_for_team_in_week(year: int, week: int, seasontype: int, team_abbr: str) -> str:
    """
    Finds the first gameId in the given week where team_abbr participates.
    """
    team_abbr = _normalize_team_abbr(team_abbr)

    url = scoreboard_url(year, week, seasontype)
    html = fetch_url(url)
    game_ids = extract_game_ids_from_scoreboard_html(html)

    if not game_ids:
        raise RuntimeError("No gameIds found. ESPN scoreboard format may have changed.")

    # Iterate gameIds and check ESPN summary until the favorite team is found
    for gid in game_ids:
        try:
            s = fetch_summary(gid)
            abbrs = _extract_team_abbrs_from_summary(s)
            if team_abbr in abbrs:
                return gid
        except Exception:
            continue

    raise RuntimeError(
        f"No game found for team '{team_abbr}' in year={year}, week={week}, seasontype={seasontype}. "
        "Double-check the abbreviation and confirm the team played that week."
    )


def generate_favorite_team_poster(
    year: int,
    week: int,
    seasontype: int = 2,
    team: str = "",
) -> str:
    """
    Generates ONE poster for the user's favorite team in the given week.
    Returns the output PNG path.
    """
    team_abbr = _normalize_team_abbr(team)
    game_id = find_game_for_team_in_week(year, week, seasontype, team_abbr)

    out_dir = os.path.join("game_visuals", f"{year}_week{week}", team_abbr)
    os.makedirs(out_dir, exist_ok=True)

    success, msg = generate_poster_for_game(game_id, out_dir)
    if not success:
        raise RuntimeError(f"Poster generation failed: {msg}")

    # msg is the PNG path when success=True
    return msg
