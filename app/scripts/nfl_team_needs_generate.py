#!/usr/bin/env python3
import argparse
import io
import os
import re
from typing import Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

HEADERS = {"User-Agent": "Mozilla/5.0"}

TEAM_NEEDS = {
    "ARI": ["QB", "RB", "G", "T"],
    "ATL": ["WR", "TE", "DI", "CB"],
    "BAL": ["WR", "G", "DL"],
    "BUF": ["WR", "ED", "LB", "DB"],
    "CAR": ["WR", "DL", "LB"],
    "CHI": ["DL", "LB", "S"],
    "CIN": ["C", "G", "DI", "S"],
    "CLE": ["QB", "WR", "G", "T"],
    "DAL": ["RB", "ED", "LB", "S"],
    "DEN": ["RB", "TE", "C", "LB"],
    "DET": ["C", "T", "ED", "CB"],
    "GB": ["T", "DI", "CB"],
    "HOU": ["RB", "C", "G", "DI"],
    "IND": ["QB", "DI", "LB", "S"],
    "JAX": ["C", "G", "DI", "S"],
    "KC": ["RB", "WR", "G", "T", "DL"],
    "LV": ["QB", "WR", "G", "T", "DI", "LB", "CB"],
    "LAC": ["G", "C", "DL"],
    "LAR": ["QB", "T", "CB"],
    "MIA": ["QB", "WR", "G", "CB"],
    "MIN": ["RB", "C", "DB"],
    "NE": ["G", "T", "ED", "LB"],
    "NO": ["WR", "G", "DL"],
    "NYG": ["WR", "G", "T", "CB"],
    "NYJ": ["QB", "WR", "DI", "LB", "CB"],
    "PHI": ["TE", "G", "ED", "CB"],
    "PIT": ["QB", "WR", "T", "DB"],
    "SEA": ["RB", "C", "G", "LB", "CB"],
    "SF": ["WR", "G", "ED", "S"],
    "TB": ["TE", "G", "ED", "LB", "CB"],
    "TEN": ["RB", "WR", "C", "G", "ED", "CB"],
    "WSH": ["TE", "G", "ED", "LB", "DB"],
}

TEAM_META = {
    "ARI": ("ari", "arizona-cardinals", "Arizona Cardinals"),
    "ATL": ("atl", "atlanta-falcons", "Atlanta Falcons"),
    "BAL": ("bal", "baltimore-ravens", "Baltimore Ravens"),
    "BUF": ("buf", "buffalo-bills", "Buffalo Bills"),
    "CAR": ("car", "carolina-panthers", "Carolina Panthers"),
    "CHI": ("chi", "chicago-bears", "Chicago Bears"),
    "CIN": ("cin", "cincinnati-bengals", "Cincinnati Bengals"),
    "CLE": ("cle", "cleveland-browns", "Cleveland Browns"),
    "DAL": ("dal", "dallas-cowboys", "Dallas Cowboys"),
    "DEN": ("den", "denver-broncos", "Denver Broncos"),
    "DET": ("det", "detroit-lions", "Detroit Lions"),
    "GB": ("gb", "green-bay-packers", "Green Bay Packers"),
    "HOU": ("hou", "houston-texans", "Houston Texans"),
    "IND": ("ind", "indianapolis-colts", "Indianapolis Colts"),
    "JAX": ("jax", "jacksonville-jaguars", "Jacksonville Jaguars"),
    "KC": ("kc", "kansas-city-chiefs", "Kansas City Chiefs"),
    "LV": ("lv", "las-vegas-raiders", "Las Vegas Raiders"),
    "LAC": ("lac", "los-angeles-chargers", "Los Angeles Chargers"),
    "LAR": ("lar", "los-angeles-rams", "Los Angeles Rams"),
    "MIA": ("mia", "miami-dolphins", "Miami Dolphins"),
    "MIN": ("min", "minnesota-vikings", "Minnesota Vikings"),
    "NE": ("ne", "new-england-patriots", "New England Patriots"),
    "NO": ("no", "new-orleans-saints", "New Orleans Saints"),
    "NYG": ("nyg", "new-york-giants", "New York Giants"),
    "NYJ": ("nyj", "new-york-jets", "New York Jets"),
    "PHI": ("phi", "philadelphia-eagles", "Philadelphia Eagles"),
    "PIT": ("pit", "pittsburgh-steelers", "Pittsburgh Steelers"),
    "SEA": ("sea", "seattle-seahawks", "Seattle Seahawks"),
    "SF": ("sf", "san-francisco-49ers", "San Francisco 49ers"),
    "TB": ("tb", "tampa-bay-buccaneers", "Tampa Bay Buccaneers"),
    "TEN": ("ten", "tennessee-titans", "Tennessee Titans"),
    "WSH": ("wsh", "washington-commanders", "Washington Commanders"),
}

POSITION_MAP = {
    "QB": ["QB"],
    "RB": ["RB", "FB", "HB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "C": ["C"],
    "G": ["G", "OG", "LG", "RG"],
    "T": ["T", "OT", "LT", "RT"],
    "ED": ["DE", "OLB", "EDGE"],
    "DI": ["DT", "NT", "DL"],
    "DL": ["DE", "DT", "NT", "DL", "EDGE"],
    "LB": ["LB", "ILB", "OLB", "MLB"],
    "CB": ["CB"],
    "S": ["S", "FS", "SS"],
    "DB": ["CB", "S", "FS", "SS", "DB"],
}

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
    "SEA": ("#002244", "#69BE28"),
    "SF": ("#AA0000", "#B3995D"),
    "TB": ("#D50A0A", "#34302B"),
    "TEN": ("#0C2340", "#4B92DB"),
    "WSH": ("#5A1414", "#FFB612"),
}

ALIASES = {"WAS": "WSH"}


def load_font(size: int, bold: bool = False):
    font_paths = []
    if bold:
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def fetch_image(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return None


def get_logo(team: str) -> Optional[Image.Image]:
    slug = TEAM_META[team][0]
    logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
    return fetch_image(logo_url)


def clean_player_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\d+$", "", name).strip()
    return name


def get_players(team: str) -> Dict[str, List[str]]:
    slug, full_slug, _ = TEAM_META[team]
    url = f"https://www.espn.com/nfl/team/roster/_/name/{slug}/{full_slug}"

    html = requests.get(url, headers=HEADERS, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")

    roster: Dict[str, List[str]] = {}

    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        player = clean_player_name(cols[1].get_text(" ", strip=True))
        pos = cols[2].get_text(" ", strip=True).upper()

        if not player or not pos:
            continue

        roster.setdefault(pos, [])
        if player not in roster[pos]:
            roster[pos].append(player)

    return roster


def players_for_position(pos: str, roster: Dict[str, List[str]]) -> List[str]:
    result = []
    for mapped_pos in POSITION_MAP.get(pos, [pos]):
        for player in roster.get(mapped_pos, []):
            if player not in result:
                result.append(player)
    return result[:8]


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_players(draw: ImageDraw.ImageDraw, players: List[str], font, max_width: int) -> List[str]:
    if not players:
        return ["No players found"]

    lines = []
    current = players[0]

    for player in players[1:]:
        test = current + ", " + player
        w, _ = text_size(draw, test, font)
        if w <= max_width:
            current = test
        else:
            lines.append(current)
            current = player

    lines.append(current)
    return lines


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, font, fill, canvas_width: int) -> int:
    w, h = text_size(draw, text, font)
    x = (canvas_width - w) // 2
    draw.text((x, y), text, fill=fill, font=font)
    return y + h


def make_gradient(width: int, height: int, top_color: Tuple[int, int, int], bottom_color: Tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (width, height), top_color)
    px = img.load()

    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        for x in range(width):
            px[x, y] = (r, g, b)

    return img


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def poster(team: str, out_file: Optional[str] = None):
    team = ALIASES.get(team, team)

    if team not in TEAM_META:
        raise ValueError(f"Invalid team: {team}")

    roster = get_players(team)
    _, _, name = TEAM_META[team]
    primary_hex, accent_hex = TEAM_COLORS[team]

    W, H = 1600, 2000
    img = make_gradient(W, H, hex_to_rgb(primary_hex), (10, 10, 10)).convert("RGBA")
    draw = ImageDraw.Draw(img)

    title_font = load_font(84, True)
    team_font = load_font(58, True)
    pos_font = load_font(38, True)
    body_font = load_font(29, False)
    footer_font = load_font(22, False)

    white = "white"
    accent = accent_hex

    y = 70

    logo = get_logo(team)
    if logo:
        logo.thumbnail((220, 220), Image.LANCZOS)
        logo_x = (W - logo.width) // 2
        img.alpha_composite(logo, (logo_x, y))
        y += logo.height + 30

    y = draw_centered(draw, name.upper(), y, team_font, white, W) + 20
    y = draw_centered(draw, "TOP POSITIONS OF NEED", y, title_font, white, W) + 45

    panel_x1, panel_y1 = 110, y
    panel_x2, panel_y2 = W - 110, H - 120

    draw.rounded_rectangle(
        (panel_x1, panel_y1, panel_x2, panel_y2),
        radius=34,
        fill=(245, 245, 245, 235),
        outline=(255, 255, 255, 80),
        width=3,
    )

    y = panel_y1 + 40
    left = panel_x1 + 45
    right = panel_x2 - 45
    max_width = right - left

    for i, pos in enumerate(TEAM_NEEDS[team], 1):
        draw.text((left, y), f"{i}. {pos}", fill=accent, font=pos_font)
        y += 54

        players = players_for_position(pos, roster)
        player_lines = wrap_players(draw, players, body_font, max_width)

        for line in player_lines:
            draw.text((left + 20, y), line, fill="black", font=body_font)
            y += 40

        y += 18

        if i != len(TEAM_NEEDS[team]):
            draw.line((left, y, right, y), fill=(190, 190, 190), width=2)
            y += 28

    draw_centered(draw, f"{team} TEAM NEEDS", H - 60, footer_font, "#DADADA", W)

    out_file = out_file or f"{team.lower()}_team_needs.png"
    img.convert("RGB").save(out_file, quality=95)
    return out_file


def generate_all_team_needs_posters(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    outputs = {}
    failures = {}

    for team in TEAM_META:
        try:
            out_file = os.path.join(outdir, f"{team.lower()}_team_needs.png")
            poster(team, out_file=out_file)
            outputs[team] = out_file
        except Exception as e:
            failures[team] = str(e)

    return outputs, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("team", nargs="?", help="Example: BUF, DAL, WSH")
    parser.add_argument("--all", action="store_true", help="Generate all 32 teams")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    if args.all:
        outputs, failures = generate_all_team_needs_posters(args.outdir)
        print(f"Generated {len(outputs)} posters")
        if failures:
            print("Failures:")
            for team, err in failures.items():
                print(f"{team}: {err}")
        return

    if not args.team:
        raise SystemExit("Provide a team abbreviation like BUF, or use --all")

    team = ALIASES.get(args.team.upper(), args.team.upper())
    out_file = os.path.join(args.outdir, f"{team.lower()}_team_needs.png")
    saved = poster(team, out_file=out_file)
    print(f"Saved {saved}")


if __name__ == "__main__":
    main()
