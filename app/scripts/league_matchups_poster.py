#!/usr/bin/env python3
import argparse
import io
import sys
from datetime import datetime
from typing import List, Dict, Optional

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

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?dates={year}&seasontype=2&week={week}"
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

TEAM_SLUGS = {
    "ARI": "ari",
    "ATL": "atl",
    "BAL": "bal",
    "BUF": "buf",
    "CAR": "car",
    "CHI": "chi",
    "CIN": "cin",
    "CLE": "cle",
    "DAL": "dal",
    "DEN": "den",
    "DET": "det",
    "GB": "gb",
    "HOU": "hou",
    "IND": "ind",
    "JAX": "jax",
    "KC": "kc",
    "LV": "lv",
    "LAC": "lac",
    "LAR": "lar",
    "MIA": "mia",
    "MIN": "min",
    "NE": "ne",
    "NO": "no",
    "NYG": "nyg",
    "NYJ": "nyj",
    "PHI": "phi",
    "PIT": "pit",
    "SF": "sf",
    "SEA": "sea",
    "TB": "tb",
    "TEN": "ten",
    "WSH": "wsh",
}

DEFAULT_PRIMARY = "#111111"
DEFAULT_SECONDARY = "#444444"


def get_json(url: str) -> dict:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def safe_get(d: dict, *keys, default=""):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def format_date_eastern(date_iso: str) -> str:
    if not date_iso or date_iso == "-":
        return "-"

    try:
        raw = date_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)

        if ZoneInfo is not None:
            eastern = dt.astimezone(ZoneInfo("America/New_York"))
        else:
            eastern = dt

        day_part = eastern.strftime("%a").upper()
        date_part = eastern.strftime("%m/%d/%y")
        time_part = eastern.strftime("%I:%M %p").lstrip("0")
        return f"{day_part} {date_part} {time_part} ET"

    except Exception:
        return date_iso


def parse_week_games(data: dict) -> List[Dict[str, str]]:
    events = data.get("events", [])
    if not events:
        raise RuntimeError("No games returned from ESPN scoreboard endpoint.")

    games = []

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        away_team = None
        home_team = None

        for comp in competitors:
            abbr = str(safe_get(comp, "team", "abbreviation", default="")).upper()
            home_away = str(comp.get("homeAway", "")).lower()

            if home_away == "away":
                away_team = abbr
            elif home_away == "home":
                home_team = abbr

        if not away_team or not home_team:
            continue

        games.append(
            {
                "away": away_team,
                "home": home_team,
                "date": format_date_eastern(event.get("date", "") or competition.get("date", "")),
            }
        )

    if not games:
        raise RuntimeError("No valid matchups found for that week.")

    return games


def get_font(size: int, bold: bool = False):
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def get_poster_colors(games: List[Dict[str, str]]):
    if not games:
        return DEFAULT_PRIMARY, DEFAULT_SECONDARY

    away = games[0]["away"]
    home = games[0]["home"]

    away_colors = TEAM_COLORS.get(away, (DEFAULT_PRIMARY, DEFAULT_SECONDARY))
    home_colors = TEAM_COLORS.get(home, (DEFAULT_PRIMARY, DEFAULT_SECONDARY))

    return away_colors[0], home_colors[1]


def fetch_logo(team_abbr: str, size: int = 64) -> Optional[Image.Image]:
    slug = TEAM_SLUGS.get(team_abbr)
    if not slug:
        return None

    urls = [
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500-dark/{slug}.png",
    ]

    for url in urls:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            img.thumbnail((size, size), Image.LANCZOS)
            return img
        except Exception:
            continue

    return None


def draw_matchup_row(
    bg: Image.Image,
    draw: ImageDraw.ImageDraw,
    matchup_left: int,
    matchup_width: int,
    matchup_top: int,
    away: str,
    home: str,
    font,
    text_fill: str,
    logo_cache: Dict[str, Optional[Image.Image]],
):
    away_logo = logo_cache.get(away)
    home_logo = logo_cache.get(home)

    logo_gap = 12
    side_gap = 22

    away_bbox = draw.textbbox((0, 0), away, font=font)
    at_bbox = draw.textbbox((0, 0), "@", font=font)
    home_bbox = draw.textbbox((0, 0), home, font=font)

    away_w = away_bbox[2] - away_bbox[0]
    at_w = at_bbox[2] - at_bbox[0]
    home_w = home_bbox[2] - home_bbox[0]

    away_logo_w = away_logo.width if away_logo else 0
    home_logo_w = home_logo.width if home_logo else 0

    total_w = (
        away_logo_w
        + (logo_gap if away_logo_w else 0)
        + away_w
        + side_gap
        + at_w
        + side_gap
        + home_w
        + (logo_gap if home_logo_w else 0)
        + home_logo_w
    )

    cur_x = matchup_left + max(0, (matchup_width - total_w) // 2)
    text_y = matchup_top + 8

    if away_logo:
        logo_y = matchup_top + max(0, (48 - away_logo.height) // 2)
        bg.paste(away_logo, (cur_x, logo_y), away_logo)
        cur_x += away_logo.width + logo_gap

    draw.text((cur_x, text_y), away, font=font, fill=text_fill)
    cur_x += away_w + side_gap

    draw.text((cur_x, text_y), "@", font=font, fill=text_fill)
    cur_x += at_w + side_gap

    draw.text((cur_x, text_y), home, font=font, fill=text_fill)
    cur_x += home_w

    if home_logo:
        cur_x += logo_gap
        logo_y = matchup_top + max(0, (48 - home_logo.height) // 2)
        bg.paste(home_logo, (cur_x, logo_y), home_logo)


def make_poster(year: int, week: int, games: List[dict], output_path: str):
    width = 1400
    height = 2200

    primary, secondary = get_poster_colors(games)

    bg = Image.new("RGB", (width, height), primary)
    draw = ImageDraw.Draw(bg)

    title_font = get_font(80, bold=True)
    subtitle_font = get_font(42, bold=True)
    header_font = get_font(34, bold=True)
    row_font = get_font(30, bold=False)
    row_font_bold = get_font(31, bold=True)
    date_font = get_font(25, bold=False)

    draw.rectangle([0, 0, width, 32], fill=secondary)
    draw.rectangle([0, height - 32, width, height], fill=secondary)

    draw_centered(draw, "NFL LEAGUE MATCHUPS", title_font, 65, width, "white")
    draw_centered(draw, f"Week {week} • {year}", subtitle_font, 160, width, "white")

    left = 55
    right = width - 55
    top = 245
    row_h = 96
    row_gap = 11

    header_h = 122
    draw.rounded_rectangle([left, top, right, top + header_h], radius=24, fill="#A9B0B4")

    game_col_x = left + 38
    matchup_col_x = left + 280
    matchup_col_w = 500
    date_col_x = left + 860

    draw.text((game_col_x, top + 40), "GAME", font=header_font, fill="white")
    draw.text((matchup_col_x + (matchup_col_w - 170) // 2, top + 40), "MATCHUP", font=header_font, fill="white")
    draw.text((date_col_x, top + 40), "DATE / TIME (ET)", font=header_font, fill="white")

    y = top + header_h + 14

    logo_cache: Dict[str, Optional[Image.Image]] = {}
    all_teams = set()
    for game in games:
        all_teams.add(game["away"])
        all_teams.add(game["home"])

    for team in all_teams:
        logo_cache[team] = fetch_logo(team, size=64)

    for i, game in enumerate(games, start=1):
        row_fill = "#F5F5F5" if i % 2 == 1 else "#E8E8E8"
        text_fill = "#111111"

        draw.rounded_rectangle(
            [left, y, right, y + row_h],
            radius=18,
            fill=row_fill
        )

        game_num = fit_text(draw, str(i), row_font_bold, 90)
        date_text = fit_text(draw, game["date"], date_font, right - date_col_x - 25)

        draw.text((game_col_x + 5, y + 29), game_num, font=row_font_bold, fill=text_fill)

        draw_matchup_row(
            bg=bg,
            draw=draw,
            matchup_left=matchup_col_x,
            matchup_width=matchup_col_w,
            matchup_top=y + 20,
            away=game["away"],
            home=game["home"],
            font=row_font,
            text_fill=text_fill,
            logo_cache=logo_cache,
        )

        draw.text((date_col_x, y + 31), date_text, font=date_font, fill=text_fill)

        y += row_h + row_gap

    bg.save(output_path)
    print(f"Saved poster to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create an NFL league matchups poster for one regular-season week using ESPN."
    )
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--week", type=int, required=True, help="Week number, e.g. 1")
    args = parser.parse_args()

    try:
        url = SCOREBOARD_URL.format(year=args.year, week=args.week)
        data = get_json(url)
        games = parse_week_games(data)

        output_path = f"league_matchups_{args.year}_week_{args.week}.png"
        make_poster(args.year, args.week, games, output_path)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
