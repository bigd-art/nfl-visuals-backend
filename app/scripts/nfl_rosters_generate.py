import os
import re
import sys
from io import BytesIO
from typing import Dict, List

import requests
from PIL import Image, ImageDraw, ImageFont

TEAM_INFO = {
    "ari": {"id": "22", "name": "Arizona Cardinals", "primary": "#97233F", "secondary": "#000000", "accent": "#FFB612"},
    "atl": {"id": "1", "name": "Atlanta Falcons", "primary": "#A71930", "secondary": "#000000", "accent": "#A5ACAF"},
    "bal": {"id": "33", "name": "Baltimore Ravens", "primary": "#241773", "secondary": "#000000", "accent": "#9E7C0C"},
    "buf": {"id": "2", "name": "Buffalo Bills", "primary": "#00338D", "secondary": "#C60C30", "accent": "#FFFFFF"},
    "car": {"id": "29", "name": "Carolina Panthers", "primary": "#0085CA", "secondary": "#101820", "accent": "#BFC0BF"},
    "chi": {"id": "3", "name": "Chicago Bears", "primary": "#0B162A", "secondary": "#C83803", "accent": "#FFFFFF"},
    "cin": {"id": "4", "name": "Cincinnati Bengals", "primary": "#FB4F14", "secondary": "#000000", "accent": "#FFFFFF"},
    "cle": {"id": "5", "name": "Cleveland Browns", "primary": "#311D00", "secondary": "#FF3C00", "accent": "#FFFFFF"},
    "dal": {"id": "6", "name": "Dallas Cowboys", "primary": "#041E42", "secondary": "#869397", "accent": "#FFFFFF"},
    "den": {"id": "7", "name": "Denver Broncos", "primary": "#FB4F14", "secondary": "#002244", "accent": "#FFFFFF"},
    "det": {"id": "8", "name": "Detroit Lions", "primary": "#0076B6", "secondary": "#B0B7BC", "accent": "#FFFFFF"},
    "gb": {"id": "9", "name": "Green Bay Packers", "primary": "#203731", "secondary": "#FFB612", "accent": "#FFFFFF"},
    "hou": {"id": "34", "name": "Houston Texans", "primary": "#03202F", "secondary": "#A71930", "accent": "#FFFFFF"},
    "ind": {"id": "11", "name": "Indianapolis Colts", "primary": "#002C5F", "secondary": "#A2AAAD", "accent": "#FFFFFF"},
    "jax": {"id": "30", "name": "Jacksonville Jaguars", "primary": "#006778", "secondary": "#101820", "accent": "#D7A22A"},
    "kc": {"id": "12", "name": "Kansas City Chiefs", "primary": "#E31837", "secondary": "#FFB81C", "accent": "#FFFFFF"},
    "lv": {"id": "13", "name": "Las Vegas Raiders", "primary": "#000000", "secondary": "#A5ACAF", "accent": "#FFFFFF"},
    "lac": {"id": "24", "name": "Los Angeles Chargers", "primary": "#0080C6", "secondary": "#FFC20E", "accent": "#FFFFFF"},
    "lar": {"id": "14", "name": "Los Angeles Rams", "primary": "#003594", "secondary": "#FFD100", "accent": "#FFFFFF"},
    "mia": {"id": "15", "name": "Miami Dolphins", "primary": "#008E97", "secondary": "#FC4C02", "accent": "#FFFFFF"},
    "min": {"id": "16", "name": "Minnesota Vikings", "primary": "#4F2683", "secondary": "#FFC62F", "accent": "#FFFFFF"},
    "ne": {"id": "17", "name": "New England Patriots", "primary": "#002244", "secondary": "#C60C30", "accent": "#FFFFFF"},
    "no": {"id": "18", "name": "New Orleans Saints", "primary": "#101820", "secondary": "#D3BC8D", "accent": "#FFFFFF"},
    "nyg": {"id": "19", "name": "New York Giants", "primary": "#0B2265", "secondary": "#A71930", "accent": "#FFFFFF"},
    "nyj": {"id": "20", "name": "New York Jets", "primary": "#125740", "secondary": "#000000", "accent": "#FFFFFF"},
    "phi": {"id": "21", "name": "Philadelphia Eagles", "primary": "#004C54", "secondary": "#A5ACAF", "accent": "#FFFFFF"},
    "pit": {"id": "23", "name": "Pittsburgh Steelers", "primary": "#101820", "secondary": "#FFB612", "accent": "#FFFFFF"},
    "sf": {"id": "25", "name": "San Francisco 49ers", "primary": "#AA0000", "secondary": "#B3995D", "accent": "#FFFFFF"},
    "sea": {"id": "26", "name": "Seattle Seahawks", "primary": "#002244", "secondary": "#69BE28", "accent": "#FFFFFF"},
    "tb": {"id": "27", "name": "Tampa Bay Buccaneers", "primary": "#D50A0A", "secondary": "#34302B", "accent": "#FF7900"},
    "ten": {"id": "10", "name": "Tennessee Titans", "primary": "#0C2340", "secondary": "#4B92DB", "accent": "#C8102E"},
    "wsh": {"id": "28", "name": "Washington Commanders", "primary": "#5A1414", "secondary": "#FFB612", "accent": "#FFFFFF"},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

OFFENSE_REQUIREMENTS = [
    ("QB", 3),
    ("RB", 3),
    ("WR", 4),
    ("TE", 3),
    ("G", 2),
    ("T", 2),
    ("C", 2),
]

DEFENSE_REQUIREMENTS = [
    ("DE", 4),
    ("DT", 3),
    ("LB", 5),
    ("CB", 4),
    ("S", 3),
]


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip())


def get_font(size, bold=False):
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass

    return ImageFont.load_default()


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def blend(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


def make_gradient_background(width, height, top_hex, bottom_hex):
    top = hex_to_rgb(top_hex)
    bottom = hex_to_rgb(bottom_hex)

    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        t = y / max(height - 1, 1)
        color = blend(top, bottom, t)
        for x in range(width):
            pixels[x, y] = color

    return image


def center_text(draw, text, font, fill, canvas_width, y):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = int((canvas_width - text_width) / 2)
    draw.text((x, y), text, font=font, fill=fill)


def fit_text(draw, text, font, max_width):
    text = str(text)

    if draw.textlength(text, font=font) <= max_width:
        return text

    while len(text) > 3 and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]

    return text.rstrip() + "…"


def wrap_text(draw, text, font, max_width):
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def fetch_team_logo(team_code):
    logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{team_code}.png"
    response = requests.get(logo_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGBA")


def paste_logo_centered(base_image, logo, center_x, top_y, max_width=135, max_height=135):
    logo = logo.copy()
    logo.thumbnail((max_width, max_height), Image.LANCZOS)
    x = int(center_x - logo.size[0] / 2)
    base_image.paste(logo, (x, top_y), logo)


def normalize_position(pos):
    p = clean_text(pos).upper()

    if p == "QB":
        return "QB"
    if p in {"RB", "HB", "FB"}:
        return "RB"
    if p == "WR":
        return "WR"
    if p == "TE":
        return "TE"

    if p in {"G", "OG", "LG", "RG"}:
        return "G"
    if p in {"T", "OT", "LT", "RT"}:
        return "T"
    if p == "C":
        return "C"

    if p in {"DE", "EDGE", "LDE", "RDE"}:
        return "DE"
    if p in {"DT", "NT"}:
        return "DT"
    if p in {"LB", "ILB", "OLB", "MLB"}:
        return "LB"
    if p == "CB":
        return "CB"
    if p in {"S", "FS", "SS"}:
        return "S"

    return p


def fetch_roster_json(team_code):
    team_id = TEAM_INFO[team_code]["id"]
    url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/teams/{team_id}/roster"

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_player(raw):
    name = clean_text(
        raw.get("displayName")
        or raw.get("fullName")
        or raw.get("shortName")
        or ""
    )

    pos_obj = raw.get("position") or {}
    pos = clean_text(
        pos_obj.get("abbreviation")
        or pos_obj.get("name")
        or raw.get("position")
        or ""
    )

    age = clean_text(raw.get("age") or "")
    height = clean_text(raw.get("displayHeight") or "")
    weight = clean_text(raw.get("displayWeight") or "")

    exp = ""
    experience = raw.get("experience")
    if isinstance(experience, dict):
        exp = clean_text(experience.get("years") or "")
    else:
        exp = clean_text(experience or "")

    college = ""
    college_obj = raw.get("college")
    if isinstance(college_obj, dict):
        college = clean_text(college_obj.get("name") or college_obj.get("shortName") or "")
    else:
        college = clean_text(college_obj or "")

    return {
        "name": name,
        "pos": pos,
        "display_pos": normalize_position(pos),
        "age": age,
        "height": height,
        "weight": weight.replace(" lbs", "").replace("lbs", "").strip(),
        "exp": exp,
        "college": college,
    }


def parse_team_roster(team_code):
    data = fetch_roster_json(team_code)

    sections = {
        "offense": [],
        "defense": [],
        "special_teams": [],
    }

    groups = data.get("positionGroups", [])

    for group in groups:
        group_type = clean_text(group.get("type", "")).lower()
        group_name = clean_text(group.get("displayName", "")).lower()
        items = group.get("athletes", [])

        if group_type == "offense" or "offense" in group_name:
            section_key = "offense"
        elif group_type == "defense" or "defense" in group_name:
            section_key = "defense"
        elif "special" in group_type or "special" in group_name:
            section_key = "special_teams"
        else:
            continue

        for raw in items:
            player = parse_player(raw)
            if player["name"] and player["pos"]:
                sections[section_key].append(player)

    print("Parsed rows:")
    print("Offense:", len(sections["offense"]))
    print("Defense:", len(sections["defense"]))
    print("Special Teams:", len(sections["special_teams"]))

    if not sections["offense"] and not sections["defense"] and not sections["special_teams"]:
        raise RuntimeError(f"No roster players parsed for {team_code}. JSON format changed.")

    return sections


def select_players_by_requirements(players, requirements):
    selected = []
    used_indices = set()

    for wanted_pos, wanted_count in requirements:
        count = 0

        for idx, player in enumerate(players):
            if idx in used_indices:
                continue

            if player.get("display_pos") == wanted_pos:
                selected.append(player)
                used_indices.add(idx)
                count += 1

                if count == wanted_count:
                    break

    return selected


def build_display_players(unit_key, players):
    if unit_key == "offense":
        selected = select_players_by_requirements(players, OFFENSE_REQUIREMENTS)
        return selected if selected else players[:18]

    if unit_key == "defense":
        selected = select_players_by_requirements(players, DEFENSE_REQUIREMENTS)
        return selected if selected else players[:18]

    return players


def draw_players_block(draw, players, start_x, start_y, content_width, row_height, fonts, colors):
    y = start_y

    for player in players:
        pos = player.get("display_pos", player["pos"])
        name = player["name"]

        draw.text((start_x, y), pos, font=fonts["pos"], fill=colors["accent"])

        name_x = start_x + 95
        display_name = fit_text(draw, name.upper(), fonts["name"], content_width - 95)
        draw.text((name_x, y), display_name, font=fonts["name"], fill=colors["name"])

        meta_parts = []

        if player["age"]:
            meta_parts.append(f"Age {player['age']}")
        if player["height"]:
            meta_parts.append(player["height"])
        if player["weight"]:
            meta_parts.append(f"{player['weight']} lbs")
        if player["exp"]:
            meta_parts.append(f"Exp {player['exp']}")
        if player["college"]:
            meta_parts.append(player["college"])

        meta = " • ".join(meta_parts)
        meta_lines = wrap_text(draw, meta, fonts["meta"], content_width - 95)

        meta_y = y + 32
        for line in meta_lines[:2]:
            draw.text((name_x, meta_y), line, font=fonts["meta"], fill="#333333")
            meta_y += 20

        divider_y = y + row_height - 8
        draw.line(
            (start_x, divider_y, start_x + content_width, divider_y),
            fill=colors["line"],
            width=1,
        )

        y += row_height

    return y


def create_single_poster(team_code, unit_key, players, output_dir):
    team = TEAM_INFO[team_code]

    if unit_key == "offense":
        unit_title = "OFFENSE"
        section_label = "FEATURED OFFENSE"
    elif unit_key == "defense":
        unit_title = "DEFENSE"
        section_label = "FEATURED DEFENSE"
    else:
        unit_title = "SPECIAL TEAMS"
        section_label = "FULL ROSTER"

    width = 1080
    height = 1920

    bg = make_gradient_background(width, height, team["primary"], team["secondary"])
    draw = ImageDraw.Draw(bg)

    big_font = get_font(62, True)
    team_font = get_font(28, True)
    section_font = get_font(26, True)
    pos_font = get_font(28, True)
    name_font = get_font(25, True)
    meta_font = get_font(18, False)
    footer_font = get_font(16, False)

    colors = {
        "accent": team["accent"] if team["accent"] != "#FFFFFF" else "#111111",
        "name": "#111111",
        "line": "#CFCFCF",
    }

    try:
        logo = fetch_team_logo(team_code)
        paste_logo_centered(bg, logo, width // 2, 50)
    except Exception as e:
        print(f"WARNING: logo failed for {team_code}: {e}")

    center_text(draw, team["name"].upper(), team_font, "white", width, 195)
    center_text(draw, unit_title + " ROSTER", big_font, "white", width, 235)

    card_margin = 80
    card_top = 330
    card_bottom = height - 100

    draw.rounded_rectangle(
        (card_margin, card_top, width - card_margin, card_bottom),
        radius=28,
        fill="#F4F4F4",
    )

    draw.text(
        (card_margin + 28, card_top + 24),
        section_label,
        font=section_font,
        fill="#111111",
    )

    start_x = card_margin + 28
    start_y = card_top + 72
    content_width = width - (card_margin + 28) * 2

    fonts = {
        "pos": pos_font,
        "name": name_font,
        "meta": meta_font,
    }

    display_players = build_display_players(unit_key, players)

    row_height = 72
    max_rows = int((card_bottom - start_y - 20) / row_height)
    visible_players = display_players[:max_rows]

    draw_players_block(
        draw,
        visible_players,
        start_x,
        start_y,
        content_width,
        row_height,
        fonts,
        colors,
    )

    footer = f"{team_code.upper()} {unit_title}"
    bbox = draw.textbbox((0, 0), footer, font=footer_font)
    footer_width = bbox[2] - bbox[0]

    draw.text(
        (int((width - footer_width) / 2), height - 42),
        footer,
        font=footer_font,
        fill="white",
    )

    os.makedirs(output_dir, exist_ok=True)

    file_name = f"{team_code}_{unit_key}_roster.png"
    output_path = os.path.join(output_dir, file_name)
    bg.save(output_path)

    return output_path


def get_team_code_from_args():
    if len(sys.argv) < 2:
        print("Usage: python3 nfl_rosters_generate.py <team_code>")
        print("Example: python3 nfl_rosters_generate.py phi")
        print("")
        print("Valid team codes:")
        print(", ".join(sorted(TEAM_INFO.keys())))
        sys.exit(1)

    team_code = sys.argv[1].strip().lower()

    if team_code not in TEAM_INFO:
        print(f"Invalid team code: {team_code}")
        print("Valid team codes:")
        print(", ".join(sorted(TEAM_INFO.keys())))
        sys.exit(1)

    return team_code


def main():
    team_code = get_team_code_from_args()

    print(f"Fetching roster API for {TEAM_INFO[team_code]['name']}...")
    sections = parse_team_roster(team_code)

    output_dir = "single_team_roster_posters"

    offense_path = create_single_poster(team_code, "offense", sections["offense"], output_dir)
    defense_path = create_single_poster(team_code, "defense", sections["defense"], output_dir)
    special_path = create_single_poster(team_code, "special_teams", sections["special_teams"], output_dir)

    print("")
    print("Done.")
    print("Saved:")
    print(offense_path)
    print(defense_path)
    print(special_path)


if __name__ == "__main__":
    main()
