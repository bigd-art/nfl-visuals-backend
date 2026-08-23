#!/usr/bin/env python3

import os
import re
import sys
from typing import Any, Dict, List

import requests
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

DEFAULT_YEAR = int(
    os.getenv(
        "FOOTBALL_SEASON",
        "2025",
    )
)

# Internal ESPN API path.
CORE_API_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# TEAM INFORMATION
# ============================================================

TEAM_INFO = {
    "ari": {
        "id": "22",
        "name": "Arizona Cardinals",
        "primary": "#97233F",
        "secondary": "#000000",
        "accent": "#FFB612",
    },
    "atl": {
        "id": "1",
        "name": "Atlanta Falcons",
        "primary": "#A71930",
        "secondary": "#000000",
        "accent": "#A5ACAF",
    },
    "bal": {
        "id": "33",
        "name": "Baltimore Ravens",
        "primary": "#241773",
        "secondary": "#000000",
        "accent": "#9E7C0C",
    },
    "buf": {
        "id": "2",
        "name": "Buffalo Bills",
        "primary": "#00338D",
        "secondary": "#C60C30",
        "accent": "#FFFFFF",
    },
    "car": {
        "id": "29",
        "name": "Carolina Panthers",
        "primary": "#0085CA",
        "secondary": "#101820",
        "accent": "#BFC0BF",
    },
    "chi": {
        "id": "3",
        "name": "Chicago Bears",
        "primary": "#0B162A",
        "secondary": "#C83803",
        "accent": "#FFFFFF",
    },
    "cin": {
        "id": "4",
        "name": "Cincinnati Bengals",
        "primary": "#FB4F14",
        "secondary": "#000000",
        "accent": "#FFFFFF",
    },
    "cle": {
        "id": "5",
        "name": "Cleveland Browns",
        "primary": "#311D00",
        "secondary": "#FF3C00",
        "accent": "#FFFFFF",
    },
    "dal": {
        "id": "6",
        "name": "Dallas Cowboys",
        "primary": "#041E42",
        "secondary": "#869397",
        "accent": "#FFFFFF",
    },
    "den": {
        "id": "7",
        "name": "Denver Broncos",
        "primary": "#FB4F14",
        "secondary": "#002244",
        "accent": "#FFFFFF",
    },
    "det": {
        "id": "8",
        "name": "Detroit Lions",
        "primary": "#0076B6",
        "secondary": "#B0B7BC",
        "accent": "#FFFFFF",
    },
    "gb": {
        "id": "9",
        "name": "Green Bay Packers",
        "primary": "#203731",
        "secondary": "#FFB612",
        "accent": "#FFFFFF",
    },
    "hou": {
        "id": "34",
        "name": "Houston Texans",
        "primary": "#03202F",
        "secondary": "#A71930",
        "accent": "#FFFFFF",
    },
    "ind": {
        "id": "11",
        "name": "Indianapolis Colts",
        "primary": "#002C5F",
        "secondary": "#A2AAAD",
        "accent": "#FFFFFF",
    },
    "jax": {
        "id": "30",
        "name": "Jacksonville Jaguars",
        "primary": "#006778",
        "secondary": "#101820",
        "accent": "#D7A22A",
    },
    "kc": {
        "id": "12",
        "name": "Kansas City Chiefs",
        "primary": "#E31837",
        "secondary": "#FFB81C",
        "accent": "#FFFFFF",
    },
    "lv": {
        "id": "13",
        "name": "Las Vegas Raiders",
        "primary": "#000000",
        "secondary": "#A5ACAF",
        "accent": "#FFFFFF",
    },
    "lac": {
        "id": "24",
        "name": "Los Angeles Chargers",
        "primary": "#0080C6",
        "secondary": "#FFC20E",
        "accent": "#FFFFFF",
    },
    "lar": {
        "id": "14",
        "name": "Los Angeles Rams",
        "primary": "#003594",
        "secondary": "#FFD100",
        "accent": "#FFFFFF",
    },
    "mia": {
        "id": "15",
        "name": "Miami Dolphins",
        "primary": "#008E97",
        "secondary": "#FC4C02",
        "accent": "#FFFFFF",
    },
    "min": {
        "id": "16",
        "name": "Minnesota Vikings",
        "primary": "#4F2683",
        "secondary": "#FFC62F",
        "accent": "#FFFFFF",
    },
    "ne": {
        "id": "17",
        "name": "New England Patriots",
        "primary": "#002244",
        "secondary": "#C60C30",
        "accent": "#FFFFFF",
    },
    "no": {
        "id": "18",
        "name": "New Orleans Saints",
        "primary": "#101820",
        "secondary": "#D3BC8D",
        "accent": "#FFFFFF",
    },
    "nyg": {
        "id": "19",
        "name": "New York Giants",
        "primary": "#0B2265",
        "secondary": "#A71930",
        "accent": "#FFFFFF",
    },
    "nyj": {
        "id": "20",
        "name": "New York Jets",
        "primary": "#125740",
        "secondary": "#000000",
        "accent": "#FFFFFF",
    },
    "phi": {
        "id": "21",
        "name": "Philadelphia Eagles",
        "primary": "#004C54",
        "secondary": "#A5ACAF",
        "accent": "#FFFFFF",
    },
    "pit": {
        "id": "23",
        "name": "Pittsburgh Steelers",
        "primary": "#101820",
        "secondary": "#FFB612",
        "accent": "#FFFFFF",
    },
    "sf": {
        "id": "25",
        "name": "San Francisco 49ers",
        "primary": "#AA0000",
        "secondary": "#B3995D",
        "accent": "#FFFFFF",
    },
    "sea": {
        "id": "26",
        "name": "Seattle Seahawks",
        "primary": "#002244",
        "secondary": "#69BE28",
        "accent": "#FFFFFF",
    },
    "tb": {
        "id": "27",
        "name": "Tampa Bay Buccaneers",
        "primary": "#D50A0A",
        "secondary": "#34302B",
        "accent": "#FF7900",
    },
    "ten": {
        "id": "10",
        "name": "Tennessee Titans",
        "primary": "#0C2340",
        "secondary": "#4B92DB",
        "accent": "#C8102E",
    },
    "wsh": {
        "id": "28",
        "name": "Washington Commanders",
        "primary": "#5A1414",
        "secondary": "#FFB612",
        "accent": "#FFFFFF",
    },
}


# ============================================================
# DISPLAY REQUIREMENTS
# ============================================================

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


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .replace("\xa0", " ")
        .strip(),
    )


def normalize_name(name):
    name = clean_text(
        name
    ).lower()

    name = re.sub(
        r"[^a-z0-9 ]",
        "",
        name,
    )

    return re.sub(
        r"\s+",
        " ",
        name,
    ).strip()


def normalize_position(pos):
    p = clean_text(
        pos
    ).upper()

    if p == "QB":
        return "QB"

    if p in {
        "RB",
        "HB",
        "FB",
    }:
        return "RB"

    if p == "WR":
        return "WR"

    if p == "TE":
        return "TE"

    if p in {
        "G",
        "OG",
        "LG",
        "RG",
    }:
        return "G"

    if p in {
        "T",
        "OT",
        "LT",
        "RT",
    }:
        return "T"

    if p == "C":
        return "C"

    if p in {
        "DE",
        "EDGE",
        "LDE",
        "RDE",
    }:
        return "DE"

    if p in {
        "DT",
        "NT",
        "LDT",
        "RDT",
        "DL",
    }:
        return "DT"

    if p in {
        "LB",
        "ILB",
        "OLB",
        "MLB",
        "WLB",
        "SLB",
    }:
        return "LB"

    if p in {
        "CB",
        "LCB",
        "RCB",
        "NB",
        "DB",
    }:
        return "CB"

    if p in {
        "S",
        "FS",
        "SS",
    }:
        return "S"

    if p in {
        "PK",
        "K",
    }:
        return "K"

    if p == "P":
        return "P"

    if p == "LS":
        return "LS"

    if p in {
        "PR",
        "KR",
    }:
        return p

    return p


# ============================================================
# HTTP
# ============================================================

def normalize_ref(
    url: str,
) -> str:

    url = str(
        url or ""
    ).strip()

    if url.startswith(
        "http://"
    ):
        return (
            "https://"
            + url[len("http://"):]
        )

    return url


def fetch_json(
    url: str,
) -> dict:

    url = normalize_ref(
        url
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    print(
        f"HTTP {response.status_code}: "
        f"{response.url}"
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# FONT / IMAGE HELPERS
# ============================================================

def get_font(
    size,
    bold=False,
):

    paths = [
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else
            "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/Library/Fonts/Arial Bold.ttf"
            if bold
            else
            "/Library/Fonts/Arial.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]

    for path in paths:

        try:
            return ImageFont.truetype(
                path,
                size,
            )

        except Exception:
            pass

    return ImageFont.load_default()


def hex_to_rgb(
    hex_color,
):

    hex_color = (
        hex_color
        .lstrip("#")
    )

    return tuple(
        int(
            hex_color[
                i:i + 2
            ],
            16,
        )
        for i in (
            0,
            2,
            4,
        )
    )


def blend(
    c1,
    c2,
    t,
):

    return tuple(
        int(
            c1[i]
            * (1 - t)
            + c2[i]
            * t
        )
        for i in range(3)
    )


def make_gradient_background(
    width,
    height,
    top_hex,
    bottom_hex,
):

    top = hex_to_rgb(
        top_hex
    )

    bottom = hex_to_rgb(
        bottom_hex
    )

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
    )

    pixels = image.load()

    for y in range(
        height
    ):

        t = (
            y
            / max(
                height - 1,
                1,
            )
        )

        color = blend(
            top,
            bottom,
            t,
        )

        for x in range(
            width
        ):
            pixels[
                x,
                y,
            ] = color

    return image


def center_text(
    draw,
    text,
    font,
    fill,
    canvas_width,
    y,
):

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    x = int(
        (
            canvas_width
            - text_width
        )
        / 2
    )

    draw.text(
        (
            x,
            y,
        ),
        text,
        font=font,
        fill=fill,
    )


def fit_text(
    draw,
    text,
    font,
    max_width,
):

    text = str(
        text
    )

    if (
        draw.textlength(
            text,
            font=font,
        )
        <= max_width
    ):
        return text

    while (
        len(text) > 3
        and draw.textlength(
            text + "…",
            font=font,
        )
        > max_width
    ):
        text = text[:-1]

    return (
        text.rstrip()
        + "…"
    )


def wrap_text(
    draw,
    text,
    font,
    max_width,
):

    words = str(
        text
    ).split()

    if not words:
        return [""]

    lines = []

    current = words[0]

    for word in words[1:]:

        trial = (
            current
            + " "
            + word
        )

        if (
            draw.textlength(
                trial,
                font=font,
            )
            <= max_width
        ):
            current = trial

        else:
            lines.append(
                current
            )

            current = word

    lines.append(
        current
    )

    return lines


# ============================================================
# PIXEL ICON HELPERS
# ============================================================

def draw_star(
    draw,
    cx,
    cy,
    radius,
    fill,
):

    points = [
        (cx, cy - radius),
        (cx + radius // 4, cy - radius // 4),
        (cx + radius, cy - radius // 4),
        (cx + radius // 3, cy + radius // 5),
        (cx + radius // 2, cy + radius),
        (cx, cy + radius // 2),
        (cx - radius // 2, cy + radius),
        (cx - radius // 3, cy + radius // 5),
        (cx - radius, cy - radius // 4),
        (cx - radius // 4, cy - radius // 4),
    ]

    draw.polygon(
        points,
        fill=fill,
    )


def draw_paw(
    draw,
    cx,
    cy,
    fill,
):

    draw.ellipse(
        (
            cx - 14,
            cy - 2,
            cx + 14,
            cy + 22,
        ),
        fill=fill,
    )

    for dx, dy in [
        (-20, -18),
        (-7, -25),
        (7, -25),
        (20, -18),
    ]:
        draw.ellipse(
            (
                cx + dx - 5,
                cy + dy - 7,
                cx + dx + 5,
                cy + dy + 7,
            ),
            fill=fill,
        )


def draw_football(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.ellipse(
        (
            cx - 34,
            cy - 17,
            cx + 34,
            cy + 17,
        ),
        fill=fill,
    )

    draw.line(
        (
            cx - 13,
            cy,
            cx + 13,
            cy,
        ),
        fill=accent,
        width=3,
    )

    for offset in (
        -8,
        0,
        8,
    ):
        draw.line(
            (
                cx + offset,
                cy - 5,
                cx + offset,
                cy + 5,
            ),
            fill=accent,
            width=2,
        )


def draw_feather(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 27, cy + 26),
            (cx - 14, cy - 18),
            (cx + 27, cy - 33),
            (cx + 18, cy + 8),
            (cx - 9, cy + 27),
        ],
        fill=fill,
    )

    draw.line(
        (
            cx - 24,
            cy + 30,
            cx + 18,
            cy - 24,
        ),
        fill=accent,
        width=3,
    )


def draw_claws(
    draw,
    cx,
    cy,
    fill,
):

    for offset in (
        -19,
        0,
        19,
    ):
        draw.polygon(
            [
                (
                    cx + offset - 5,
                    cy + 27,
                ),
                (
                    cx + offset + 3,
                    cy - 30,
                ),
                (
                    cx + offset + 10,
                    cy - 34,
                ),
                (
                    cx + offset + 3,
                    cy + 27,
                ),
            ],
            fill=fill,
        )


def draw_stripes(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 31,
            cy - 30,
            cx + 31,
            cy + 30,
        ),
        fill=fill,
    )

    for offset in (
        -34,
        -9,
        16,
    ):
        draw.polygon(
            [
                (
                    cx + offset,
                    cy - 30,
                ),
                (
                    cx + offset + 10,
                    cy - 30,
                ),
                (
                    cx + offset + 35,
                    cy + 30,
                ),
                (
                    cx + offset + 25,
                    cy + 30,
                ),
            ],
            fill=accent,
        )


def draw_mountain(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 36, cy + 27),
            (cx, cy - 31),
            (cx + 36, cy + 27),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (cx - 9, cy - 17),
            (cx, cy - 31),
            (cx + 10, cy - 15),
            (cx + 2, cy - 20),
        ],
        fill=accent,
    )


def draw_texas(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 28, cy - 30),
            (cx + 7, cy - 30),
            (cx + 8, cy - 13),
            (cx + 30, cy - 12),
            (cx + 23, cy + 8),
            (cx + 7, cy + 15),
            (cx - 2, cy + 33),
            (cx - 16, cy + 16),
            (cx - 30, cy + 4),
        ],
        fill=fill,
    )

    draw_star(
        draw,
        cx,
        cy,
        8,
        accent,
    )


def draw_arc(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.arc(
        (
            cx - 31,
            cy - 31,
            cx + 31,
            cy + 31,
        ),
        205,
        335,
        fill=fill,
        width=9,
    )

    draw.arc(
        (
            cx - 20,
            cy - 20,
            cx + 20,
            cy + 20,
        ),
        205,
        335,
        fill=accent,
        width=4,
    )


def draw_pennant(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 30,
            cy - 32,
            cx - 26,
            cy + 30,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (cx - 26, cy - 27),
            (cx + 33, cy - 4),
            (cx - 26, cy + 17),
        ],
        fill=fill,
    )


def draw_lightning(
    draw,
    cx,
    cy,
    fill,
):

    draw.polygon(
        [
            (cx + 5, cy - 34),
            (cx - 22, cy + 1),
            (cx - 5, cy + 1),
            (cx - 14, cy + 34),
            (cx + 25, cy - 8),
            (cx + 8, cy - 8),
        ],
        fill=fill,
    )


def draw_wave(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 34, cy + 18),
            (cx - 24, cy - 5),
            (cx - 10, cy - 20),
            (cx + 4, cy - 23),
            (cx + 21, cy - 14),
            (cx + 34, cy + 4),
            (cx + 13, cy - 1),
            (cx, cy + 8),
            (cx - 7, cy + 20),
        ],
        fill=fill,
    )

    draw.line(
        (
            cx - 29,
            cy + 22,
            cx + 30,
            cy + 22,
        ),
        fill=accent,
        width=4,
    )


def draw_spiral(
    draw,
    cx,
    cy,
    fill,
):

    draw.arc(
        (
            cx - 28,
            cy - 28,
            cx + 28,
            cy + 28,
        ),
        20,
        340,
        fill=fill,
        width=8,
    )

    draw.arc(
        (
            cx - 15,
            cy - 15,
            cx + 15,
            cy + 15,
        ),
        20,
        300,
        fill=fill,
        width=6,
    )


def draw_ship(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 34, cy + 17),
            (cx + 34, cy + 17),
            (cx + 22, cy + 29),
            (cx - 22, cy + 29),
        ],
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 2,
            cy - 31,
            cx + 2,
            cy + 15,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (cx + 2, cy - 27),
            (cx + 23, cy - 5),
            (cx + 2, cy - 5),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (cx - 3, cy - 24),
            (cx - 22, cy - 4),
            (cx - 3, cy - 4),
        ],
        fill=fill,
    )


def draw_hat(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 33, cy + 16),
            (cx - 19, cy - 13),
            (cx, cy - 25),
            (cx + 19, cy - 13),
            (cx + 33, cy + 16),
            (cx + 10, cy + 9),
            (cx, cy + 21),
            (cx - 10, cy + 9),
        ],
        fill=fill,
    )

    draw_star(
        draw,
        cx,
        cy,
        8,
        accent,
    )


def draw_trumpet(
    draw,
    cx,
    cy,
    fill,
):

    draw.rectangle(
        (
            cx - 23,
            cy - 5,
            cx + 10,
            cy + 5,
        ),
        fill=fill,
    )

    draw.polygon(
        [
            (cx + 10, cy - 14),
            (cx + 32, cy - 22),
            (cx + 32, cy + 22),
            (cx + 10, cy + 14),
        ],
        fill=fill,
    )

    draw.arc(
        (
            cx - 29,
            cy - 1,
            cx - 11,
            cy + 24,
        ),
        10,
        190,
        fill=fill,
        width=4,
    )


def draw_skyline(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    buildings = [
        (-31, 11, -22, 29),
        (-20, -3, -8, 29),
        (-6, -29, 6, 29),
        (8, -10, 21, 29),
        (23, 3, 32, 29),
    ]

    for (
        x1,
        y1,
        x2,
        y2,
    ) in buildings:

        draw.rectangle(
            (
                cx + x1,
                cy + y1,
                cx + x2,
                cy + y2,
            ),
            fill=fill,
        )

    draw.line(
        (
            cx - 35,
            cy + 30,
            cx + 35,
            cy + 30,
        ),
        fill=accent,
        width=4,
    )


def draw_jet(
    draw,
    cx,
    cy,
    fill,
):

    draw.polygon(
        [
            (cx - 35, cy + 6),
            (cx - 7, cy - 4),
            (cx + 18, cy - 28),
            (cx + 25, cy - 23),
            (cx + 12, cy - 3),
            (cx + 34, cy + 7),
            (cx + 10, cy + 10),
            (cx + 3, cy + 27),
            (cx - 5, cy + 27),
            (cx - 6, cy + 11),
        ],
        fill=fill,
    )


def draw_diamonds(
    draw,
    cx,
    cy,
):

    data = [
        (
            cx,
            cy - 18,
            "#F2C230",
        ),
        (
            cx - 19,
            cy + 13,
            "#1E5FA8",
        ),
        (
            cx + 19,
            cy + 13,
            "#D32E3E",
        ),
    ]

    for (
        x,
        y,
        color,
    ) in data:

        draw.polygon(
            [
                (x, y - 9),
                (x + 9, y),
                (x, y + 9),
                (x - 9, y),
            ],
            fill=color,
        )


def draw_bridge(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 26,
            cy - 27,
            cx - 21,
            cy + 27,
        ),
        fill=fill,
    )

    draw.rectangle(
        (
            cx + 21,
            cy - 27,
            cx + 26,
            cy + 27,
        ),
        fill=fill,
    )

    draw.line(
        (
            cx - 34,
            cy + 18,
            cx + 34,
            cy + 18,
        ),
        fill=fill,
        width=5,
    )

    draw.arc(
        (
            cx - 24,
            cy - 22,
            cx + 24,
            cy + 31,
        ),
        180,
        360,
        fill=accent,
        width=3,
    )


def draw_flag(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 28,
            cy - 32,
            cx - 24,
            cy + 30,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (cx - 24, cy - 26),
            (cx + 29, cy - 21),
            (cx + 20, cy + 3),
            (cx - 24, cy + 8),
        ],
        fill=fill,
    )


def draw_sword(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 4, cy - 33),
            (cx + 5, cy - 33),
            (cx + 6, cy + 13),
            (cx - 6, cy + 13),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (cx - 4, cy - 33),
            (cx, cy - 42),
            (cx + 5, cy - 33),
        ],
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 18,
            cy + 11,
            cx + 18,
            cy + 16,
        ),
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 4,
            cy + 16,
            cx + 5,
            cy + 31,
        ),
        fill=accent,
    )


def draw_column(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 27,
            cy - 28,
            cx + 27,
            cy - 22,
        ),
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 31,
            cy + 23,
            cx + 31,
            cy + 29,
        ),
        fill=accent,
    )

    for x in (
        -19,
        -6,
        6,
        19,
    ):

        draw.rectangle(
            (
                cx + x - 3,
                cy - 19,
                cx + x + 3,
                cy + 21,
            ),
            fill=fill,
        )


# ============================================================
# TEAM-SPECIFIC PIXEL SYMBOLS
# ============================================================

def draw_team_symbol(
    draw,
    abbreviation,
    cx,
    cy,
    primary,
    secondary,
):

    if abbreviation == "ARI":

        # Desert cactus
        draw.rectangle(
            (
                cx - 5,
                cy - 29,
                cx + 5,
                cy + 27,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx - 21,
                cy - 6,
                cx - 4,
                cy + 3,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx - 21,
                cy - 18,
                cx - 13,
                cy + 3,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx + 4,
                cy - 10,
                cx + 20,
                cy - 2,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx + 13,
                cy - 21,
                cx + 20,
                cy - 2,
            ),
            fill=primary,
        )

    elif abbreviation in {
        "ATL",
        "BAL",
        "PHI",
    }:

        draw_feather(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "BUF":

        # Generic hoof marks
        draw.ellipse(
            (
                cx - 26,
                cy - 26,
                cx - 4,
                cy + 5,
            ),
            fill=primary,
        )

        draw.ellipse(
            (
                cx + 4,
                cy - 26,
                cx + 26,
                cy + 5,
            ),
            fill=primary,
        )

        draw.ellipse(
            (
                cx - 9,
                cy + 9,
                cx + 9,
                cy + 26,
            ),
            fill=secondary,
        )

    elif abbreviation == "CAR":

        draw_claws(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbreviation in {
        "CHI",
        "DET",
        "JAX",
    }:

        draw_paw(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbreviation == "CIN":

        draw_stripes(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation in {
        "CLE",
        "GB",
    }:

        # CLE is intentionally a generic football,
        # not a helmet.
        draw_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "DAL":

        draw_star(
            draw,
            cx,
            cy,
            30,
            primary,
        )

    elif abbreviation == "DEN":

        draw_mountain(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "HOU":

        draw_texas(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "IND":

        # Original speed/racing arc.
        draw_arc(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "KC":

        draw_pennant(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "LV":

        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "LAC":

        draw_lightning(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbreviation == "LAR":

        draw_spiral(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbreviation == "MIA":

        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "MIN":

        draw_ship(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "NE":

        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "NO":

        # Generic trumpet.
        draw_trumpet(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbreviation == "NYG":

        draw_skyline(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "NYJ":

        draw_jet(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbreviation == "PIT":

        draw_diamonds(
            draw,
            cx,
            cy,
        )

    elif abbreviation == "SF":

        draw_bridge(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "SEA":

        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "TB":

        draw_flag(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "TEN":

        draw_sword(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbreviation == "WSH":

        draw_column(
            draw,
            cx,
            cy,
            secondary,
            primary,
        )

    else:

        draw_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )


# ============================================================
# CREATE ORIGINAL PIXEL TEAM ICON
#
# Abbreviation is built into the icon.
# ============================================================

def create_team_icon(
    team_code,
    size=155,
):

    team_code = (
        str(
            team_code
            or ""
        )
        .strip()
        .lower()
    )

    abbreviation = (
        team_code.upper()
    )

    team = TEAM_INFO.get(
        team_code
    )

    if not team:

        return Image.new(
            "RGBA",
            (
                size,
                size,
            ),
            (
                0,
                0,
                0,
                0,
            ),
        )

    primary = hex_to_rgb(
        team[
            "primary"
        ]
    )

    secondary = hex_to_rgb(
        team[
            "secondary"
        ]
    )

    base_width = 112
    base_height = 130

    icon = Image.new(
        "RGBA",
        (
            base_width,
            base_height,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        icon
    )

    draw_team_symbol(
        draw,
        abbreviation,
        base_width // 2,
        50,
        primary,
        secondary,
    )

    # --------------------------------------------------------
    # ABBREVIATION UNDER SYMBOL
    # --------------------------------------------------------

    abbreviation_font = get_font(
        20,
        True,
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        abbreviation,
        font=abbreviation_font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    draw.rounded_rectangle(
        (
            17,
            101,
            base_width - 17,
            127,
        ),
        radius=4,
        fill=(
            8,
            12,
            20,
            225,
        ),
    )

    text_fill = primary

    if sum(
        primary
    ) < 120:

        text_fill = (
            225,
            225,
            225,
        )

    draw.text(
        (
            (
                base_width
                - text_width
            )
            // 2,
            103,
        ),
        abbreviation,
        font=abbreviation_font,
        fill=text_fill,
    )

    ratio = (
        size
        / base_height
    )

    output_width = max(
        1,
        int(
            base_width
            * ratio
        ),
    )

    output = icon.resize(
        (
            output_width,
            size,
        ),
        Image.Resampling.NEAREST,
    )

    return output


def paste_logo_centered(
    base_image,
    logo,
    center_x,
    top_y,
    max_width=155,
    max_height=155,
):

    logo = logo.copy()

    logo.thumbnail(
        (
            max_width,
            max_height,
        ),
        Image.Resampling.NEAREST,
    )

    x = int(
        center_x
        - logo.size[0]
        / 2
    )

    base_image.paste(
        logo,
        (
            x,
            top_y,
        ),
        logo,
    )


# ============================================================
# ROSTER API
# ============================================================

def fetch_roster_json(
    team_code,
):

    team_id = (
        TEAM_INFO[
            team_code
        ][
            "id"
        ]
    )

    # Internal ESPN API path.
    url = (
        "https://site.web.api.espn.com/"
        "apis/common/v3/"
        "sports/football/nfl/"
        f"teams/{team_id}/roster"
    )

    return fetch_json(
        url
    )


# ============================================================
# CORE DEPTH CHART
# ============================================================

def fetch_depthchart_json(
    team_code,
    year=DEFAULT_YEAR,
):

    team_id = (
        TEAM_INFO[
            team_code
        ][
            "id"
        ]
    )

    url = (
        f"{CORE_API_BASE}/"
        f"seasons/{year}/"
        f"teams/{team_id}/"
        "depthcharts"
    )

    print(
        f"Fetching Core depth chart: "
        f"{url}"
    )

    return fetch_json(
        url
    )


def resolve_ref_object(
    obj: Any,
) -> Any:

    if not isinstance(
        obj,
        dict,
    ):
        return obj

    ref = clean_text(
        obj.get(
            "$ref"
        )
    )

    if not ref:
        return obj

    useful_keys = {
        "name",
        "displayName",
        "positions",
        "athletes",
        "position",
    }

    if any(
        key in obj
        for key in useful_keys
    ):
        return obj

    try:
        return fetch_json(
            ref
        )

    except Exception as exc:
        print(
            "WARNING: Could not resolve "
            f"Core reference {ref}: {exc}"
        )

        return obj


def get_depthchart_groups(
    data: Any,
) -> List[dict]:

    groups: List[
        dict
    ] = []

    if not isinstance(
        data,
        dict,
    ):
        return groups

    old_depthchart = (
        data.get(
            "depthchart"
        )
        or []
    )

    if isinstance(
        old_depthchart,
        list,
    ):

        for item in old_depthchart:

            if isinstance(
                item,
                dict,
            ):

                groups.append(
                    resolve_ref_object(
                        item
                    )
                )

    items = (
        data.get(
            "items"
        )
        or []
    )

    if isinstance(
        items,
        list,
    ):

        for item in items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            resolved = (
                resolve_ref_object(
                    item
                )
            )

            if isinstance(
                resolved,
                dict,
            ):
                groups.append(
                    resolved
                )

    if (
        "positions"
        in data
        and isinstance(
            data.get(
                "positions"
            ),
            (
                dict,
                list,
            ),
        )
    ):

        groups.append(
            data
        )

    return groups


# ============================================================
# PLAYER PARSING
# ============================================================

def parse_player(
    raw,
):

    name = clean_text(
        raw.get(
            "displayName"
        )
        or raw.get(
            "fullName"
        )
        or raw.get(
            "shortName"
        )
        or raw.get(
            "name"
        )
        or ""
    )

    pos_obj = (
        raw.get(
            "position"
        )
        or {}
    )

    if isinstance(
        pos_obj,
        dict,
    ):

        pos = clean_text(
            pos_obj.get(
                "abbreviation"
            )
            or pos_obj.get(
                "name"
            )
            or ""
        )

    else:

        pos = clean_text(
            pos_obj
        )

    age = clean_text(
        raw.get(
            "age"
        )
        or ""
    )

    height = clean_text(
        raw.get(
            "displayHeight"
        )
        or raw.get(
            "height"
        )
        or ""
    )

    weight = clean_text(
        raw.get(
            "displayWeight"
        )
        or raw.get(
            "weight"
        )
        or ""
    )

    exp = ""

    experience = (
        raw.get(
            "experience"
        )
    )

    if isinstance(
        experience,
        dict,
    ):

        exp = clean_text(
            experience.get(
                "years"
            )
            or ""
        )

    else:

        exp = clean_text(
            experience
            or ""
        )

    college = ""

    college_obj = (
        raw.get(
            "college"
        )
    )

    if isinstance(
        college_obj,
        dict,
    ):

        college = clean_text(
            college_obj.get(
                "name"
            )
            or college_obj.get(
                "shortName"
            )
            or ""
        )

    else:

        college = clean_text(
            college_obj
            or ""
        )

    return {
        "name": name,
        "name_key": normalize_name(
            name
        ),
        "pos": pos,
        "display_pos": normalize_position(
            pos
        ),
        "age": age,
        "height": height,
        "weight": (
            weight
            .replace(
                " lbs",
                "",
            )
            .replace(
                "lbs",
                "",
            )
            .strip()
        ),
        "exp": exp,
        "college": college,
    }


def parse_team_roster(
    team_code,
):

    data = fetch_roster_json(
        team_code
    )

    sections = {
        "offense": [],
        "defense": [],
        "special_teams": [],
    }

    groups = (
        data.get(
            "positionGroups"
        )
        or []
    )

    for group in groups:

        group_type = clean_text(
            group.get(
                "type",
                "",
            )
        ).lower()

        group_name = clean_text(
            group.get(
                "displayName",
                "",
            )
        ).lower()

        items = (
            group.get(
                "athletes"
            )
            or []
        )

        if (
            group_type == "offense"
            or "offense"
            in group_name
        ):
            section_key = (
                "offense"
            )

        elif (
            group_type == "defense"
            or "defense"
            in group_name
        ):
            section_key = (
                "defense"
            )

        elif (
            "special"
            in group_type
            or "special"
            in group_name
        ):
            section_key = (
                "special_teams"
            )

        else:
            continue

        for raw in items:

            player = parse_player(
                raw
            )

            if (
                player[
                    "name"
                ]
                and player[
                    "pos"
                ]
            ):

                sections[
                    section_key
                ].append(
                    player
                )

    print(
        "Roster API rows:"
    )

    print(
        "Offense:",
        len(
            sections[
                "offense"
            ]
        ),
    )

    print(
        "Defense:",
        len(
            sections[
                "defense"
            ]
        ),
    )

    print(
        "Special Teams:",
        len(
            sections[
                "special_teams"
            ]
        ),
    )

    return sections


# ============================================================
# DEPTH CHART PARSING
# ============================================================

def depth_position_from_position_data(
    position_data,
):

    position_obj = (
        position_data.get(
            "position",
            {},
        )
        or {}
    )

    if isinstance(
        position_obj,
        dict,
    ):

        position_obj = (
            resolve_ref_object(
                position_obj
            )
        )

    parent_obj = {}

    if isinstance(
        position_obj,
        dict,
    ):

        parent_obj = (
            position_obj.get(
                "parent",
                {},
            )
            or {}
        )

        if isinstance(
            parent_obj,
            dict,
        ):

            parent_obj = (
                resolve_ref_object(
                    parent_obj
                )
            )

    own_abbr = ""
    parent_abbr = ""

    if isinstance(
        position_obj,
        dict,
    ):

        own_abbr = clean_text(
            position_obj.get(
                "abbreviation",
                "",
            )
            or position_obj.get(
                "name",
                "",
            )
        )

    if isinstance(
        parent_obj,
        dict,
    ):

        parent_abbr = clean_text(
            parent_obj.get(
                "abbreviation",
                "",
            )
            or parent_obj.get(
                "name",
                "",
            )
        )

    if own_abbr:

        normalized_own = (
            normalize_position(
                own_abbr
            )
        )

        if normalized_own not in {
            "OFF",
            "DEF",
            "ST",
        }:
            return normalized_own

    if parent_abbr:

        normalized_parent = (
            normalize_position(
                parent_abbr
            )
        )

        if normalized_parent not in {
            "OFF",
            "DEF",
            "ST",
        }:
            return normalized_parent

    return normalize_position(
        own_abbr
        or parent_abbr
    )


def resolve_depth_athlete(
    athlete: dict,
) -> dict:

    if not isinstance(
        athlete,
        dict,
    ):
        return {}

    if (
        athlete.get(
            "displayName"
        )
        or athlete.get(
            "fullName"
        )
        or athlete.get(
            "name"
        )
    ):
        return athlete

    nested = (
        athlete.get(
            "athlete"
        )
    )

    if isinstance(
        nested,
        dict,
    ):

        nested_resolved = (
            resolve_ref_object(
                nested
            )
        )

        if isinstance(
            nested_resolved,
            dict,
        ):

            merged = dict(
                athlete
            )

            merged.update(
                nested_resolved
            )

            return merged

    resolved = (
        resolve_ref_object(
            athlete
        )
    )

    if isinstance(
        resolved,
        dict,
    ):
        return resolved

    return athlete


def iter_positions(
    group: dict,
):

    positions = (
        group.get(
            "positions"
        )
        or {}
    )

    if isinstance(
        positions,
        dict,
    ):

        for (
            position_key,
            position_data,
        ) in positions.items():

            if isinstance(
                position_data,
                dict,
            ):

                yield (
                    position_key,
                    resolve_ref_object(
                        position_data
                    ),
                )

    elif isinstance(
        positions,
        list,
    ):

        for (
            index,
            position_data,
        ) in enumerate(
            positions
        ):

            if not isinstance(
                position_data,
                dict,
            ):
                continue

            resolved = (
                resolve_ref_object(
                    position_data
                )
            )

            if not isinstance(
                resolved,
                dict,
            ):
                continue

            position_key = clean_text(
                resolved.get(
                    "name"
                )
                or resolved.get(
                    "displayName"
                )
                or resolved.get(
                    "abbreviation"
                )
                or index
            )

            yield (
                position_key,
                resolved,
            )


def classify_depth_group(
    group: dict,
) -> str:

    group_name = clean_text(
        group.get(
            "name"
        )
        or group.get(
            "displayName"
        )
        or group.get(
            "type"
        )
        or ""
    ).lower()

    if "special" in group_name:
        return "special_teams"

    if (
        "defense"
        in group_name
        or "base"
        in group_name
        or "4-3"
        in group_name
        or "3-4"
        in group_name
    ):
        return "defense"

    if "offense" in group_name:
        return "offense"

    return "offense"


def parse_depthchart_order(
    team_code,
    year=DEFAULT_YEAR,
):

    data = fetch_depthchart_json(
        team_code,
        year=year,
    )

    groups = (
        get_depthchart_groups(
            data
        )
    )

    print(
        f"Core depth chart groups: "
        f"{len(groups)}"
    )

    depth_sections = {
        "offense": [],
        "defense": [],
        "special_teams": [],
    }

    seen_by_section = {
        "offense": set(),
        "defense": set(),
        "special_teams": set(),
    }

    for group in groups:

        if not isinstance(
            group,
            dict,
        ):
            continue

        section_key = (
            classify_depth_group(
                group
            )
        )

        for (
            position_key,
            position_data,
        ) in iter_positions(
            group
        ):

            if not isinstance(
                position_data,
                dict,
            ):
                continue

            display_pos = (
                depth_position_from_position_data(
                    position_data
                )
            )

            athletes = (
                position_data.get(
                    "athletes"
                )
                or position_data.get(
                    "entries"
                )
                or []
            )

            if not isinstance(
                athletes,
                list,
            ):
                continue

            for (
                depth_rank,
                athlete,
            ) in enumerate(
                athletes,
                start=1,
            ):

                if not isinstance(
                    athlete,
                    dict,
                ):
                    continue

                resolved_athlete = (
                    resolve_depth_athlete(
                        athlete
                    )
                )

                name = clean_text(
                    resolved_athlete.get(
                        "displayName"
                    )
                    or resolved_athlete.get(
                        "fullName"
                    )
                    or resolved_athlete.get(
                        "shortName"
                    )
                    or resolved_athlete.get(
                        "name"
                    )
                    or ""
                )

                if not name:
                    continue

                name_key = (
                    normalize_name(
                        name
                    )
                )

                if (
                    name_key
                    in seen_by_section[
                        section_key
                    ]
                ):
                    continue

                seen_by_section[
                    section_key
                ].add(
                    name_key
                )

                depth_sections[
                    section_key
                ].append(
                    {
                        "name": name,
                        "name_key": name_key,
                        "display_pos": display_pos,
                        "depth_position_key": clean_text(
                            position_key
                        ).upper(),
                        "depth_rank": depth_rank,
                        "group_name": clean_text(
                            group.get(
                                "name"
                            )
                            or group.get(
                                "displayName"
                            )
                            or ""
                        ),
                    }
                )

    print(
        "Core depth chart rows:"
    )

    print(
        "Offense:",
        len(
            depth_sections[
                "offense"
            ]
        ),
    )

    print(
        "Defense:",
        len(
            depth_sections[
                "defense"
            ]
        ),
    )

    print(
        "Special Teams:",
        len(
            depth_sections[
                "special_teams"
            ]
        ),
    )

    return depth_sections


# ============================================================
# ORDERING
# ============================================================

def order_section_by_depthchart(
    roster_players,
    depth_players,
):

    roster_by_name = {
        player[
            "name_key"
        ]: player
        for player
        in roster_players
    }

    ordered = []
    used = set()

    for depth_player in depth_players:

        key = (
            depth_player[
                "name_key"
            ]
        )

        if key in roster_by_name:

            player = (
                roster_by_name[
                    key
                ].copy()
            )

            player[
                "display_pos"
            ] = depth_player.get(
                "display_pos",
                player.get(
                    "display_pos",
                    player.get(
                        "pos",
                        "",
                    ),
                ),
            )

            player[
                "depth_position_key"
            ] = depth_player.get(
                "depth_position_key",
                "",
            )

            player[
                "depth_rank"
            ] = depth_player.get(
                "depth_rank",
                "",
            )

            ordered.append(
                player
            )

            used.add(
                key
            )

    for player in roster_players:

        if (
            player[
                "name_key"
            ]
            not in used
        ):
            ordered.append(
                player
            )

    return ordered


def select_players_by_requirements(
    players,
    requirements,
):

    selected = []
    used_indices = set()

    for (
        wanted_pos,
        wanted_count,
    ) in requirements:

        count = 0

        for (
            idx,
            player,
        ) in enumerate(
            players
        ):

            if idx in used_indices:
                continue

            if (
                player.get(
                    "display_pos"
                )
                == wanted_pos
            ):

                selected.append(
                    player
                )

                used_indices.add(
                    idx
                )

                count += 1

                if (
                    count
                    == wanted_count
                ):
                    break

    return selected


def apply_numbered_position_labels(
    players,
):

    counts = {}
    labeled = []

    for player in players:

        player = (
            player.copy()
        )

        base_pos = (
            player.get(
                "display_pos"
            )
            or player.get(
                "pos"
            )
            or ""
        )

        base_pos = (
            normalize_position(
                base_pos
            )
        )

        counts[
            base_pos
        ] = (
            counts.get(
                base_pos,
                0,
            )
            + 1
        )

        player[
            "poster_pos_label"
        ] = (
            f"{base_pos}"
            f"{counts[base_pos]}"
        )

        labeled.append(
            player
        )

    return labeled


def build_display_players(
    unit_key,
    players,
):

    if unit_key == "offense":

        selected = (
            select_players_by_requirements(
                players,
                OFFENSE_REQUIREMENTS,
            )
        )

        if len(
            selected
        ) >= 14:

            return (
                apply_numbered_position_labels(
                    selected
                )
            )

        return (
            apply_numbered_position_labels(
                players[:18]
            )
        )

    if unit_key == "defense":

        selected = (
            select_players_by_requirements(
                players,
                DEFENSE_REQUIREMENTS,
            )
        )

        if len(
            selected
        ) >= 14:

            return (
                apply_numbered_position_labels(
                    selected
                )
            )

        return (
            apply_numbered_position_labels(
                players[:18]
            )
        )

    return (
        apply_numbered_position_labels(
            players
        )
    )


# ============================================================
# DRAWING
# ============================================================

def draw_players_block(
    draw,
    players,
    start_x,
    start_y,
    content_width,
    row_height,
    fonts,
    colors,
):

    y = start_y

    for player in players:

        pos = (
            player.get(
                "poster_pos_label"
            )
            or player.get(
                "display_pos",
                player[
                    "pos"
                ],
            )
        )

        name = (
            player[
                "name"
            ]
        )

        draw.text(
            (
                start_x,
                y,
            ),
            pos,
            font=fonts[
                "pos"
            ],
            fill=colors[
                "accent"
            ],
        )

        name_x = (
            start_x + 105
        )

        display_name = (
            fit_text(
                draw,
                name.upper(),
                fonts[
                    "name"
                ],
                content_width
                - 105,
            )
        )

        draw.text(
            (
                name_x,
                y,
            ),
            display_name,
            font=fonts[
                "name"
            ],
            fill=colors[
                "name"
            ],
        )

        meta_parts = []

        if player[
            "age"
        ]:

            meta_parts.append(
                f"Age "
                f"{player['age']}"
            )

        if player[
            "height"
        ]:

            meta_parts.append(
                player[
                    "height"
                ]
            )

        if player[
            "weight"
        ]:

            meta_parts.append(
                f"{player['weight']} lbs"
            )

        if player[
            "exp"
        ]:

            meta_parts.append(
                f"Exp "
                f"{player['exp']}"
            )

        if player[
            "college"
        ]:

            meta_parts.append(
                player[
                    "college"
                ]
            )

        meta = (
            " • ".join(
                meta_parts
            )
        )

        meta_lines = (
            wrap_text(
                draw,
                meta,
                fonts[
                    "meta"
                ],
                content_width
                - 105,
            )
        )

        meta_y = (
            y + 32
        )

        for line in meta_lines[
            :2
        ]:

            draw.text(
                (
                    name_x,
                    meta_y,
                ),
                line,
                font=fonts[
                    "meta"
                ],
                fill="#333333",
            )

            meta_y += 20

        divider_y = (
            y
            + row_height
            - 8
        )

        draw.line(
            (
                start_x,
                divider_y,
                start_x
                + content_width,
                divider_y,
            ),
            fill=colors[
                "line"
            ],
            width=1,
        )

        y += row_height

    return y


# ============================================================
# POSTER
# ============================================================

def create_single_poster(
    team_code,
    unit_key,
    players,
    output_dir,
):

    team = (
        TEAM_INFO[
            team_code
        ]
    )

    if unit_key == "offense":

        unit_title = "OFFENSE"

        section_label = (
            "DEPTH CHART OFFENSE"
        )

    elif unit_key == "defense":

        unit_title = "DEFENSE"

        section_label = (
            "DEPTH CHART DEFENSE"
        )

    else:

        unit_title = (
            "SPECIAL TEAMS"
        )

        section_label = (
            "DEPTH CHART SPECIAL TEAMS"
        )

    width = 1080
    height = 1920

    bg = make_gradient_background(
        width,
        height,
        team[
            "primary"
        ],
        team[
            "secondary"
        ],
    )

    draw = ImageDraw.Draw(
        bg
    )

    big_font = get_font(
        62,
        True,
    )

    team_font = get_font(
        28,
        True,
    )

    section_font = get_font(
        26,
        True,
    )

    pos_font = get_font(
        27,
        True,
    )

    name_font = get_font(
        25,
        True,
    )

    meta_font = get_font(
        18,
        False,
    )

    footer_font = get_font(
        16,
        False,
    )

    colors = {
        "accent": (
            team[
                "accent"
            ]
            if (
                team[
                    "accent"
                ]
                != "#FFFFFF"
            )
            else "#111111"
        ),
        "name": "#111111",
        "line": "#CFCFCF",
    }

    # --------------------------------------------------------
    # ORIGINAL PIXEL TEAM ICON
    # --------------------------------------------------------

    logo = create_team_icon(
        team_code,
        size=145,
    )

    paste_logo_centered(
        bg,
        logo,
        width // 2,
        42,
        max_width=145,
        max_height=145,
    )

    # The abbreviation is already built into the icon.

    center_text(
        draw,
        team[
            "name"
        ].upper(),
        team_font,
        "white",
        width,
        195,
    )

    center_text(
        draw,
        unit_title
        + " ROSTER",
        big_font,
        "white",
        width,
        235,
    )

    card_margin = 80
    card_top = 330

    card_bottom = (
        height - 100
    )

    draw.rounded_rectangle(
        (
            card_margin,
            card_top,
            width
            - card_margin,
            card_bottom,
        ),
        radius=28,
        fill="#F4F4F4",
    )

    draw.text(
        (
            card_margin + 28,
            card_top + 24,
        ),
        section_label,
        font=section_font,
        fill="#111111",
    )

    start_x = (
        card_margin + 28
    )

    start_y = (
        card_top + 72
    )

    content_width = (
        width
        - (
            card_margin
            + 28
        )
        * 2
    )

    fonts = {
        "pos": pos_font,
        "name": name_font,
        "meta": meta_font,
    }

    display_players = (
        build_display_players(
            unit_key,
            players,
        )
    )

    row_height = 72

    max_rows = int(
        (
            card_bottom
            - start_y
            - 20
        )
        / row_height
    )

    visible_players = (
        display_players[
            :max_rows
        ]
    )

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

    footer = (
        f"{team_code.upper()} "
        f"{unit_title}"
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        footer,
        font=footer_font,
    )

    footer_width = (
        bbox[2]
        - bbox[0]
    )

    draw.text(
        (
            int(
                (
                    width
                    - footer_width
                )
                / 2
            ),
            height - 42,
        ),
        footer,
        font=footer_font,
        fill="white",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    file_name = (
        f"{team_code}_"
        f"{unit_key}_"
        "depthchart_"
        "labels_roster.png"
    )

    output_path = (
        os.path.join(
            output_dir,
            file_name,
        )
    )

    bg.save(
        output_path
    )

    return output_path


# ============================================================
# CLI
# ============================================================

def get_team_code_from_args():

    if len(
        sys.argv
    ) < 2:

        print(
            "Usage: python3 "
            "rosters_depthchart_labels.py "
            "<team_code> [year]"
        )

        print(
            "Example: python3 "
            "rosters_depthchart_labels.py "
            "ari 2025"
        )

        print()

        print(
            "Valid team codes:"
        )

        print(
            ", ".join(
                sorted(
                    TEAM_INFO.keys()
                )
            )
        )

        sys.exit(
            1
        )

    team_code = (
        sys.argv[1]
        .strip()
        .lower()
    )

    if (
        team_code
        not in TEAM_INFO
    ):

        print(
            f"Invalid team code: "
            f"{team_code}"
        )

        print(
            "Valid team codes:"
        )

        print(
            ", ".join(
                sorted(
                    TEAM_INFO.keys()
                )
            )
        )

        sys.exit(
            1
        )

    year = DEFAULT_YEAR

    if len(
        sys.argv
    ) >= 3:

        try:
            year = int(
                sys.argv[2]
            )

        except ValueError:
            print(
                "Year must be an integer."
            )

            sys.exit(
                1
            )

    return (
        team_code,
        year,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        team_code,
        year,
    ) = (
        get_team_code_from_args()
    )

    print(
        "=" * 80
    )

    print(
        f"TEAM ROSTER: "
        f"{TEAM_INFO[team_code]['name']}"
    )

    print(
        f"Season: {year}"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Fetching roster API..."
    )

    sections = (
        parse_team_roster(
            team_code
        )
    )

    print()

    print(
        "Fetching ESPN Core "
        "depth chart API..."
    )

    try:

        depth_sections = (
            parse_depthchart_order(
                team_code,
                year=year,
            )
        )

    except Exception as exc:

        print(
            "WARNING: Core depth "
            f"chart failed: {exc}"
        )

        depth_sections = {
            "offense": [],
            "defense": [],
            "special_teams": [],
        }

    if any(
        depth_sections.values()
    ):

        sections[
            "offense"
        ] = (
            order_section_by_depthchart(
                sections[
                    "offense"
                ],
                depth_sections[
                    "offense"
                ],
            )
        )

        sections[
            "defense"
        ] = (
            order_section_by_depthchart(
                sections[
                    "defense"
                ],
                depth_sections[
                    "defense"
                ],
            )
        )

        sections[
            "special_teams"
        ] = (
            order_section_by_depthchart(
                sections[
                    "special_teams"
                ],
                depth_sections[
                    "special_teams"
                ],
            )
        )

        print(
            "Using Core depth-chart "
            "ordering."
        )

    else:

        print(
            "WARNING: No Core "
            "depth-chart order found."
        )

        print(
            "Falling back to roster "
            "API order."
        )

    output_dir = (
        "single_team_roster_"
        "depthchart_labels_posters"
    )

    offense_path = (
        create_single_poster(
            team_code,
            "offense",
            sections[
                "offense"
            ],
            output_dir,
        )
    )

    defense_path = (
        create_single_poster(
            team_code,
            "defense",
            sections[
                "defense"
            ],
            output_dir,
        )
    )

    special_path = (
        create_single_poster(
            team_code,
            "special_teams",
            sections[
                "special_teams"
            ],
            output_dir,
        )
    )

    print()

    print(
        "Done."
    )

    print(
        "Saved:"
    )

    print(
        offense_path
    )

    print(
        defense_path
    )

    print(
        special_path
    )


if __name__ == "__main__":
    main()
