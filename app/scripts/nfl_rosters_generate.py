import os
import re
from io import BytesIO, StringIO

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# CHANGE ONLY THIS TEAM CODE WHEN YOU WANT A DIFFERENT TEAM
# =========================================================
TEAM_CODE = "ari"

TEAM_INFO = {
    "ari": {
        "name": "Arizona Cardinals",
        "primary": "#97233F",
        "secondary": "#000000",
        "accent": "#FFB612",
    },
    "atl": {
        "name": "Atlanta Falcons",
        "primary": "#A71930",
        "secondary": "#000000",
        "accent": "#A5ACAF",
    },
    "bal": {
        "name": "Baltimore Ravens",
        "primary": "#241773",
        "secondary": "#000000",
        "accent": "#9E7C0C",
    },
    "buf": {
        "name": "Buffalo Bills",
        "primary": "#00338D",
        "secondary": "#C60C30",
        "accent": "#FFFFFF",
    },
    "car": {
        "name": "Carolina Panthers",
        "primary": "#0085CA",
        "secondary": "#101820",
        "accent": "#BFC0BF",
    },
    "chi": {
        "name": "Chicago Bears",
        "primary": "#0B162A",
        "secondary": "#C83803",
        "accent": "#FFFFFF",
    },
    "cin": {
        "name": "Cincinnati Bengals",
        "primary": "#FB4F14",
        "secondary": "#000000",
        "accent": "#FFFFFF",
    },
    "cle": {
        "name": "Cleveland Browns",
        "primary": "#311D00",
        "secondary": "#FF3C00",
        "accent": "#FFFFFF",
    },
    "dal": {
        "name": "Dallas Cowboys",
        "primary": "#041E42",
        "secondary": "#869397",
        "accent": "#FFFFFF",
    },
    "den": {
        "name": "Denver Broncos",
        "primary": "#FB4F14",
        "secondary": "#002244",
        "accent": "#FFFFFF",
    },
    "det": {
        "name": "Detroit Lions",
        "primary": "#0076B6",
        "secondary": "#B0B7BC",
        "accent": "#FFFFFF",
    },
    "gb": {
        "name": "Green Bay Packers",
        "primary": "#203731",
        "secondary": "#FFB612",
        "accent": "#FFFFFF",
    },
    "hou": {
        "name": "Houston Texans",
        "primary": "#03202F",
        "secondary": "#A71930",
        "accent": "#FFFFFF",
    },
    "ind": {
        "name": "Indianapolis Colts",
        "primary": "#002C5F",
        "secondary": "#A2AAAD",
        "accent": "#FFFFFF",
    },
    "jax": {
        "name": "Jacksonville Jaguars",
        "primary": "#006778",
        "secondary": "#101820",
        "accent": "#D7A22A",
    },
    "kc": {
        "name": "Kansas City Chiefs",
        "primary": "#E31837",
        "secondary": "#FFB81C",
        "accent": "#FFFFFF",
    },
    "lv": {
        "name": "Las Vegas Raiders",
        "primary": "#000000",
        "secondary": "#A5ACAF",
        "accent": "#FFFFFF",
    },
    "lac": {
        "name": "Los Angeles Chargers",
        "primary": "#0080C6",
        "secondary": "#FFC20E",
        "accent": "#FFFFFF",
    },
    "lar": {
        "name": "Los Angeles Rams",
        "primary": "#003594",
        "secondary": "#FFD100",
        "accent": "#FFFFFF",
    },
    "mia": {
        "name": "Miami Dolphins",
        "primary": "#008E97",
        "secondary": "#FC4C02",
        "accent": "#FFFFFF",
    },
    "min": {
        "name": "Minnesota Vikings",
        "primary": "#4F2683",
        "secondary": "#FFC62F",
        "accent": "#FFFFFF",
    },
    "ne": {
        "name": "New England Patriots",
        "primary": "#002244",
        "secondary": "#C60C30",
        "accent": "#FFFFFF",
    },
    "no": {
        "name": "New Orleans Saints",
        "primary": "#101820",
        "secondary": "#D3BC8D",
        "accent": "#FFFFFF",
    },
    "nyg": {
        "name": "New York Giants",
        "primary": "#0B2265",
        "secondary": "#A71930",
        "accent": "#FFFFFF",
    },
    "nyj": {
        "name": "New York Jets",
        "primary": "#125740",
        "secondary": "#000000",
        "accent": "#FFFFFF",
    },
    "phi": {
        "name": "Philadelphia Eagles",
        "primary": "#004C54",
        "secondary": "#A5ACAF",
        "accent": "#FFFFFF",
    },
    "pit": {
        "name": "Pittsburgh Steelers",
        "primary": "#101820",
        "secondary": "#FFB612",
        "accent": "#FFFFFF",
    },
    "sf": {
        "name": "San Francisco 49ers",
        "primary": "#AA0000",
        "secondary": "#B3995D",
        "accent": "#FFFFFF",
    },
    "sea": {
        "name": "Seattle Seahawks",
        "primary": "#002244",
        "secondary": "#69BE28",
        "accent": "#FFFFFF",
    },
    "tb": {
        "name": "Tampa Bay Buccaneers",
        "primary": "#D50A0A",
        "secondary": "#34302B",
        "accent": "#FF7900",
    },
    "ten": {
        "name": "Tennessee Titans",
        "primary": "#0C2340",
        "secondary": "#4B92DB",
        "accent": "#C8102E",
    },
    "wsh": {
        "name": "Washington Commanders",
        "primary": "#5A1414",
        "secondary": "#FFB612",
        "accent": "#FFFFFF",
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def clean_text(value):
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_get(row, colname):
    for col in row.index:
        if clean_text(col).lower() == colname.lower():
            return clean_text(row[col])
    return ""


def get_font(size, bold=False):
    font_paths = []

    if bold:
        font_paths.extend([
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ])
    else:
        font_paths.extend([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])

    for path in font_paths:
        if os.path.exists(path):
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
        t = float(y) / float(max(height - 1, 1))
        color = blend(top, bottom, t)
        for x in range(width):
            pixels[x, y] = color

    return image


def rounded_rectangle(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def center_text(draw, text, font, fill, canvas_width, y):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = int((canvas_width - text_width) / 2)
    draw.text((x, y), text, font=font, fill=fill)


def wrap_text(draw, text, font, max_width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def fetch_team_logo(team_code):
    logo_url = "https://a.espncdn.com/i/teamlogos/nfl/500/{0}.png".format(team_code)

    response = requests.get(logo_url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    logo = Image.open(BytesIO(response.content)).convert("RGBA")
    return logo


def paste_logo_centered(base_image, logo, center_x, top_y, max_width=120, max_height=120):
    logo = logo.copy()
    logo.thumbnail((max_width, max_height), Image.LANCZOS)

    x = int(center_x - logo.size[0] / 2)
    y = top_y

    base_image.paste(logo, (x, y), logo)


def fetch_roster_tables(team_code):
    url = "https://www.espn.com/nfl/team/roster/_/name/{0}".format(team_code)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def parse_team_roster(team_code):
    tables = fetch_roster_tables(team_code)

    sections = {
        "offense": [],
        "defense": [],
        "special_teams": [],
    }

    section_order = ["offense", "defense", "special_teams"]
    section_index = 0

    for table in tables:
        normalized_cols = [clean_text(c).lower() for c in table.columns]

        if "name" not in normalized_cols or "pos" not in normalized_cols:
            continue

        if section_index >= len(section_order):
            break

        section_name = section_order[section_index]
        section_index += 1

        for _, row in table.iterrows():
            name = safe_get(row, "Name")
            name = re.sub(r"\d+$", "", name).strip()
            pos = safe_get(row, "Pos")
            age = safe_get(row, "Age")
            ht = safe_get(row, "HT")
            wt = safe_get(row, "WT")
            exp = safe_get(row, "Exp")
            college = safe_get(row, "College")

            if not name or not pos:
                continue

            player = {
                "name": name,
                "pos": pos,
                "age": age,
                "height": ht,
                "weight": wt,
                "exp": exp,
                "college": college,
            }

            sections[section_name].append(player)

    return sections


def draw_players_block(draw, players, start_x, start_y, content_width, row_height, fonts, colors):
    pos_font = fonts["pos"]
    name_font = fonts["name"]
    meta_font = fonts["meta"]

    accent = colors["accent"]
    name_color = colors["name"]
    line_color = colors["line"]

    y = start_y

    for player in players:
        pos = player["pos"]
        name = player["name"]

        draw.text((start_x, y), pos, font=pos_font, fill=accent)

        name_x = start_x + 95
        draw.text((name_x, y), name.upper(), font=name_font, fill=name_color)

        meta_parts = []
        if player["age"]:
            meta_parts.append("Age {0}".format(player["age"]))
        if player["height"]:
            meta_parts.append(player["height"])
        if player["weight"]:
            meta_parts.append("{0} lbs".format(player["weight"]))
        if player["exp"]:
            meta_parts.append("Exp {0}".format(player["exp"]))
        if player["college"]:
            meta_parts.append(player["college"])

        meta = " • ".join(meta_parts)
        meta_lines = wrap_text(draw, meta, meta_font, content_width - 95)

        meta_y = y + 32
        for line in meta_lines:
            draw.text((name_x, meta_y), line, font=meta_font, fill="#333333")
            meta_y += 20

        divider_y = y + row_height - 8
        draw.line(
            (start_x, divider_y, start_x + content_width, divider_y),
            fill=line_color,
            width=1,
        )

        y += row_height

    return y


def create_single_poster(team_code, unit_key, players, output_dir):
    team = TEAM_INFO[team_code]
    team_name = team["name"]

    if unit_key == "offense":
        unit_title = "OFFENSE"
    elif unit_key == "defense":
        unit_title = "DEFENSE"
    else:
        unit_title = "SPECIAL TEAMS"

    width = 1080
    height = 1920

    bg = make_gradient_background(width, height, team["primary"], team["secondary"])
    draw = ImageDraw.Draw(bg)

    title_font = get_font(28, bold=True)
    big_font = get_font(62, bold=True)
    team_font = get_font(28, bold=True)
    section_font = get_font(26, bold=True)
    pos_font = get_font(28, bold=True)
    name_font = get_font(25, bold=True)
    meta_font = get_font(18, bold=False)
    footer_font = get_font(16, bold=False)

    colors = {
        "accent": team["accent"] if team["accent"] != "#FFFFFF" else "#111111",
        "name": "#111111",
        "line": "#CFCFCF",
    }

    # Logo
    try:
        logo = fetch_team_logo(team_code)
        paste_logo_centered(bg, logo, width // 2, 50, max_width=130, max_height=130)
    except Exception:
        # fallback: do nothing if logo fetch fails
        pass

    center_text(draw, team_name.upper(), team_font, "white", width, 195)
    center_text(draw, unit_title + " ROSTER", big_font, "white", width, 235)

    card_margin = 80
    card_top = 330
    card_bottom = height - 100

    rounded_rectangle(
        draw,
        (card_margin, card_top, width - card_margin, card_bottom),
        radius=28,
        fill="#F4F4F4",
    )

    draw.text(
        (card_margin + 28, card_top + 24),
        "FULL ROSTER",
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

    row_height = 72
    max_rows_that_fit = int((card_bottom - start_y - 20) / row_height)
    visible_players = players[:max_rows_that_fit]

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

    if len(players) > max_rows_that_fit:
        more_text = "+ {0} more players not shown".format(len(players) - max_rows_that_fit)
        bbox = draw.textbbox((0, 0), more_text, font=footer_font)
        more_width = bbox[2] - bbox[0]
        draw.text(
            (width - card_margin - more_width - 24, card_bottom - 36),
            more_text,
            font=footer_font,
            fill="#666666",
        )

    footer = "{0} {1}".format(team_code.upper(), unit_title)
    bbox = draw.textbbox((0, 0), footer, font=footer_font)
    footer_width = bbox[2] - bbox[0]
    draw.text(
        (int((width - footer_width) / 2), height - 42),
        footer,
        font=footer_font,
        fill="white",
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = "{0}_{1}_roster.png".format(team_code, unit_key)
    output_path = os.path.join(output_dir, file_name)
    bg.save(output_path)
    return output_path


def main():
    if TEAM_CODE not in TEAM_INFO:
        raise ValueError("Invalid TEAM_CODE: {0}".format(TEAM_CODE))

    print("Scraping roster for {0}...".format(TEAM_INFO[TEAM_CODE]["name"]))
    sections = parse_team_roster(TEAM_CODE)

    output_dir = "single_team_roster_posters"

    offense_path = create_single_poster(TEAM_CODE, "offense", sections["offense"], output_dir)
    defense_path = create_single_poster(TEAM_CODE, "defense", sections["defense"], output_dir)
    special_path = create_single_poster(TEAM_CODE, "special_teams", sections["special_teams"], output_dir)

    print("")
    print("Done.")
    print("Saved:")
    print(offense_path)
    print(defense_path)
    print(special_path)


if __name__ == "__main__":
    main()
