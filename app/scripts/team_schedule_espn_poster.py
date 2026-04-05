#!/usr/bin/env python3
import argparse
import io
import sys
from datetime import datetime
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/"
    "{team_id}/schedule?season={year}&seasontype=2"
)

TEAM_COLORS = {
    "ARI": ("#97233F", "#000000"),
    "ATL": ("#A71930", "#000000"),
    "BAL": ("#241773", "#000000"),
    "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#101820"),
    "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"),
    "CLE": ("#311D00", "#FF3C00"),
    "DAL": ("#041E42", "#869397"),
    "DEN": ("#FB4F14", "#002244"),
    "DET": ("#0076B6", "#B0B7BC"),
    "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"),
    "IND": ("#002C5F", "#A2AAAD"),
    "JAX": ("#006778", "#101820"),
    "KC": ("#E31837", "#FFB81C"),
    "LV": ("#000000", "#A5ACAF"),
    "LAC": ("#0080C6", "#FFC20E"),
    "LAR": ("#003594", "#FFA300"),
    "MIA": ("#008E97", "#FC4C02"),
    "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"),
    "NO": ("#101820", "#D3BC8D"),
    "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#000000"),
    "PHI": ("#004C54", "#A5ACAF"),
    "PIT": ("#101820", "#FFB612"),
    "SF": ("#AA0000", "#B3995D"),
    "SEA": ("#002244", "#69BE28"),
    "TB": ("#D50A0A", "#34302B"),
    "TEN": ("#0C2340", "#4B92DB"),
    "WSH": ("#5A1414", "#FFB612"),
}


def get_json(url: str) -> dict:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_team_map() -> Dict[str, dict]:
    data = get_json(TEAMS_URL)

    sports = data.get("sports", [])
    if not sports:
        raise RuntimeError("No sports data returned from ESPN teams endpoint.")

    leagues = sports[0].get("leagues", [])
    if not leagues:
        raise RuntimeError("No leagues data returned from ESPN teams endpoint.")

    teams_block = leagues[0].get("teams", [])
    if not teams_block:
        raise RuntimeError("No teams found in ESPN teams endpoint.")

    out = {}
    for item in teams_block:
        team = item.get("team", {})
        abbr = str(team.get("abbreviation", "")).strip().upper()
        team_id = str(team.get("id", "")).strip()
        logos = team.get("logos", []) or []

        if not abbr or not team_id:
            continue

        logo_url = ""
        if logos and isinstance(logos[0], dict):
            logo_url = str(logos[0].get("href", "")).strip()

        out[abbr] = {
            "id": team_id,
            "display_name": str(team.get("displayName", abbr)),
            "logo": logo_url,
        }

    return out


def safe_get(d: dict, *keys, default=""):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def extract_team_entry(competition: dict, team_abbr: str) -> Optional[dict]:
    for comp in competition.get("competitors", []):
        abbr = str(safe_get(comp, "team", "abbreviation", default="")).upper()
        if abbr == team_abbr:
            return comp
    return None


def extract_opponent_entry(competition: dict, team_abbr: str) -> Optional[dict]:
    for comp in competition.get("competitors", []):
        abbr = str(safe_get(comp, "team", "abbreviation", default="")).upper()
        if abbr != team_abbr:
            return comp
    return None


def parse_games_only(team_abbr: str, data: dict, team_map: Dict[str, dict]) -> Dict[int, dict]:
    events = data.get("events", [])
    if not events:
        raise RuntimeError("No schedule events returned from ESPN schedule endpoint.")

    by_week: Dict[int, dict] = {}

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]

        week_num = (
            safe_get(event, "week", "number", default="")
            or safe_get(competition, "week", "number", default="")
        )

        if not str(week_num).isdigit():
            continue

        week_num = int(week_num)
        if not (1 <= week_num <= 18):
            continue

        team_entry = extract_team_entry(competition, team_abbr)
        opp_entry = extract_opponent_entry(competition, team_abbr)

        if not team_entry or not opp_entry:
            continue

        home_away = str(team_entry.get("homeAway", "")).lower()
        opp_abbr = str(safe_get(opp_entry, "team", "abbreviation", default="")).upper()
        opponent = f"@ {opp_abbr}" if home_away == "away" else f"vs {opp_abbr}"

        date_iso = event.get("date", "") or competition.get("date", "")

        opp_logo = str(safe_get(opp_entry, "team", "logos", default=""))
        logo_url = ""

        opp_team_logos = safe_get(opp_entry, "team", "logos", default=[])
        if isinstance(opp_team_logos, list) and opp_team_logos:
            first = opp_team_logos[0]
            if isinstance(first, dict):
                logo_url = str(first.get("href", "")).strip()

        if not logo_url and opp_abbr in team_map:
            logo_url = team_map[opp_abbr].get("logo", "")

        by_week[week_num] = {
            "week": week_num,
            "opponent": opponent,
            "date": date_iso,
            "logo_url": logo_url,
        }

    return by_week


def format_date_eastern(date_iso: str) -> str:
    if not date_iso or date_iso == "-":
        return "-"

    try:
        raw = date_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)

        if ZoneInfo is not None:
            eastern = dt.astimezone(ZoneInfo("America/New_York"))
            day_part = eastern.strftime("%a").upper()
            date_part = eastern.strftime("%m/%d/%Y")
            time_part = eastern.strftime("%I:%M %p").lstrip("0")
            return f"{day_part} {date_part} ({time_part} ET)"

        day_part = dt.strftime("%a").upper()
        date_part = dt.strftime("%m/%d/%Y")
        time_part = dt.strftime("%I:%M %p").lstrip("0")
        return f"{day_part} {date_part} ({time_part})"

    except Exception:
        return date_iso


def build_full_18_week_schedule(team_abbr: str, data: dict, team_map: Dict[str, dict]) -> List[dict]:
    games_by_week = parse_games_only(team_abbr, data, team_map)

    full_schedule = []
    for week in range(1, 19):
        if week in games_by_week:
            game = games_by_week[week]
            full_schedule.append(
                {
                    "week": week,
                    "opponent": game["opponent"],
                    "date": format_date_eastern(game["date"]),
                    "logo_url": game["logo_url"],
                }
            )
        else:
            full_schedule.append(
                {
                    "week": week,
                    "opponent": "BYE",
                    "date": "-",
                    "logo_url": "",
                }
            )

    return full_schedule


def get_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    return ImageFont.load_default()


def draw_centered(draw, text, font, y, width, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = (width - text_w) // 2
    draw.text((x, y), text, font=font, fill=fill)


def fit_text(draw, text, font, max_width):
    text = str(text)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text

    shortened = text
    while len(shortened) > 3:
        shortened = shortened[:-1]
        candidate = shortened + "..."
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            return candidate

    return "..."


def fetch_logo_image(url: str, size: int, cache: Dict[str, Image.Image]) -> Optional[Image.Image]:
    if not url:
        return None

    if url in cache:
        return cache[url].copy()

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)

        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y), img)

        cache[url] = canvas
        return canvas.copy()
    except Exception:
        return None


def make_poster(team_abbr: str, team_name: str, year: int, games: List[dict], output_path: str):
    width = 1400
    height = 2200

    primary, secondary = TEAM_COLORS.get(team_abbr, ("#111111", "#444444"))
    bg = Image.new("RGB", (width, height), primary)
    draw = ImageDraw.Draw(bg)

    title_font = get_font(80, bold=True)
    subtitle_font = get_font(42, bold=True)
    header_font = get_font(34, bold=True)
    row_font = get_font(31, bold=False)
    week_font = get_font(32, bold=True)

    draw.rectangle([0, 0, width, 32], fill=secondary)
    draw.rectangle([0, height - 32, width, height], fill=secondary)

    draw_centered(draw, team_abbr, title_font, 80, width, "white")
    draw_centered(draw, f"{team_name} {year} Schedule", subtitle_font, 178, width, "white")

    left = 90
    right = width - 90
    top = 285
    row_h = 82
    row_gap = 12

    draw.rounded_rectangle([left, top, right, top + 112], radius=24, fill=secondary)
    draw.text((left + 40, top + 34), "WEEK", font=header_font, fill="white")
    draw.text((left + 220, top + 34), "OPP", font=header_font, fill="white")
    draw.text((left + 355, top + 34), "MATCHUP", font=header_font, fill="white")
    draw.text((left + 700, top + 34), "DATE / TIME (ET)", font=header_font, fill="white")

    y = top + 132

    logo_x = left + 230
    logo_size = 56
    matchup_x = left + 355
    date_x = left + 700

    matchup_width = 300
    date_width = right - date_x - 35

    logo_cache: Dict[str, Image.Image] = {}

    for i, game in enumerate(games):
        row_fill = "#FFFFFF" if i % 2 == 0 else "#F2F2F2"
        text_fill = "#111111"

        draw.rounded_rectangle(
            [left, y, right, y + row_h],
            radius=18,
            fill=row_fill
        )

        matchup_text = fit_text(draw, game["opponent"], row_font, matchup_width)
        date_text = fit_text(draw, game["date"], row_font, date_width)

        draw.text((left + 45, y + 21), f"{game['week']}", font=week_font, fill=text_fill)

        if game["opponent"] != "BYE":
            logo_img = fetch_logo_image(game.get("logo_url", ""), logo_size, logo_cache)
            if logo_img is not None:
                bg.paste(logo_img, (logo_x, y + 13), logo_img)
            else:
                draw.text((logo_x + 10, y + 19), "-", font=row_font, fill=text_fill)
        else:
            draw.text((logo_x + 6, y + 19), "-", font=row_font, fill=text_fill)

        draw.text((matchup_x, y + 21), matchup_text, font=row_font, fill=text_fill)
        draw.text((date_x, y + 21), date_text, font=row_font, fill=text_fill)

        y += row_h + row_gap

    bg.save(output_path)
    print(f"Saved poster to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a schedule poster for one NFL team's regular season from ESPN API."
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--team", type=str, required=True, help="Team abbreviation, e.g. PHI, DAL, WSH")
    args = parser.parse_args()

    team_abbr = args.team.strip().upper()

    try:
        team_map = build_team_map()

        if team_abbr not in team_map:
            valid = ", ".join(sorted(team_map.keys()))
            raise RuntimeError(f"Invalid team '{team_abbr}'. Valid teams: {valid}")

        team_id = team_map[team_abbr]["id"]
        team_name = team_map[team_abbr]["display_name"]

        url = SCHEDULE_URL.format(team_id=team_id, year=args.year)
        data = get_json(url)

        games = build_full_18_week_schedule(team_abbr, data, team_map)

        output_path = f"{team_abbr.lower()}_{args.year}_schedule_poster.png"
        make_poster(team_abbr, team_name, args.year, games, output_path)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
