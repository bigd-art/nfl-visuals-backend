#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import time
from typing import Any, Dict, List, Tuple, Optional

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

HTTP_MAX_ATTEMPTS = 4
HTTP_TIMEOUT = 30

REF_CACHE: Dict[str, Dict] = {}


# ============================================================
# TEAM NEEDS
# ============================================================

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


# ============================================================
# TEAM META
# ============================================================

TEAM_META = {
    "ARI": ("22", "ari", "Arizona Cardinals"),
    "ATL": ("1", "atl", "Atlanta Falcons"),
    "BAL": ("33", "bal", "Baltimore Ravens"),
    "BUF": ("2", "buf", "Buffalo Bills"),
    "CAR": ("29", "car", "Carolina Panthers"),
    "CHI": ("3", "chi", "Chicago Bears"),
    "CIN": ("4", "cin", "Cincinnati Bengals"),
    "CLE": ("5", "cle", "Cleveland Browns"),
    "DAL": ("6", "dal", "Dallas Cowboys"),
    "DEN": ("7", "den", "Denver Broncos"),
    "DET": ("8", "det", "Detroit Lions"),
    "GB": ("9", "gb", "Green Bay Packers"),
    "HOU": ("34", "hou", "Houston Texans"),
    "IND": ("11", "ind", "Indianapolis Colts"),
    "JAX": ("30", "jax", "Jacksonville Jaguars"),
    "KC": ("12", "kc", "Kansas City Chiefs"),
    "LV": ("13", "lv", "Las Vegas Raiders"),
    "LAC": ("24", "lac", "Los Angeles Chargers"),
    "LAR": ("14", "lar", "Los Angeles Rams"),
    "MIA": ("15", "mia", "Miami Dolphins"),
    "MIN": ("16", "min", "Minnesota Vikings"),
    "NE": ("17", "ne", "New England Patriots"),
    "NO": ("18", "no", "New Orleans Saints"),
    "NYG": ("19", "nyg", "New York Giants"),
    "NYJ": ("20", "nyj", "New York Jets"),
    "PHI": ("21", "phi", "Philadelphia Eagles"),
    "PIT": ("23", "pit", "Pittsburgh Steelers"),
    "SEA": ("26", "sea", "Seattle Seahawks"),
    "SF": ("25", "sf", "San Francisco 49ers"),
    "TB": ("27", "tb", "Tampa Bay Buccaneers"),
    "TEN": ("10", "ten", "Tennessee Titans"),
    "WSH": ("28", "wsh", "Washington Commanders"),
}


# ============================================================
# POSITION MAP
# ============================================================

POSITION_MAP = {
    "QB": ["QB"],
    "RB": ["RB", "HB", "FB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "C": ["C"],
    "G": ["G", "OG", "LG", "RG"],
    "T": ["T", "OT", "LT", "RT"],
    "ED": ["DE", "EDGE", "OLB"],
    "DI": ["DT", "NT", "DL"],
    "DL": ["DE", "DT", "NT", "DL", "EDGE"],
    "LB": ["LB", "ILB", "OLB", "MLB"],
    "CB": ["CB"],
    "S": ["S", "FS", "SS"],
    "DB": ["CB", "S", "FS", "SS", "DB"],
}


# ============================================================
# TEAM COLORS
# ============================================================

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


ALIASES = {
    "WAS": "WSH",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def load_font(
    size: int,
    bold: bool = False,
):
    paths = [
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/Library/Fonts/Arial Bold.ttf"
            if bold
            else "/Library/Fonts/Arial.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]

    for path in paths:
        try:
            return ImageFont.truetype(
                path,
                size,
            )
        except Exception:
            continue

    return ImageFont.load_default()


def clean_text(
    value,
) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
        .replace(
            "\xa0",
            " ",
        )
        .strip(),
    )


def normalize_name(
    name: str,
) -> str:
    name = (
        clean_text(
            name
        ).lower()
    )

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


def normalize_position(
    pos: str,
) -> str:
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
        "DL",
        "LDT",
        "RDT",
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

    return p


# ============================================================
# HTTP
# ============================================================

def normalize_ref(
    url: str,
) -> str:
    url = clean_text(
        url
    )

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
    use_cache: bool = True,
) -> Dict:
    url = normalize_ref(
        url
    )

    if (
        use_cache
        and url in REF_CACHE
    ):
        return REF_CACHE[
            url
        ]

    last_error = None

    for attempt in range(
        1,
        HTTP_MAX_ATTEMPTS + 1,
    ):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=HTTP_TIMEOUT,
            )

            print(
                f"HTTP "
                f"{response.status_code}: "
                f"{response.url}"
            )

            if (
                response.status_code
                in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                and attempt
                < HTTP_MAX_ATTEMPTS
            ):
                wait_seconds = (
                    attempt * 2
                )

                print(
                    f"Temporary HTTP "
                    f"{response.status_code}. "
                    f"Retrying in "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

            data = (
                response.json()
            )

            if use_cache:
                REF_CACHE[
                    url
                ] = data

            return data

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            last_error = exc

            if (
                attempt
                >= HTTP_MAX_ATTEMPTS
            ):
                break

            wait_seconds = (
                attempt * 2
            )

            print(
                f"Request failed "
                f"(attempt "
                f"{attempt}/"
                f"{HTTP_MAX_ATTEMPTS}): "
                f"{exc}"
            )

            print(
                f"Retrying in "
                f"{wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Failed API request: "
        f"{url}\n"
        f"{last_error}"
    )


# ============================================================
# PIXEL ICON HELPERS
# ============================================================

def hex_to_rgb(
    hex_color: str,
) -> Tuple[
    int,
    int,
    int,
]:
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
# TEAM-SPECIFIC PIXEL SYMBOL
# ============================================================

def draw_team_symbol(
    draw,
    team: str,
    cx: int,
    cy: int,
    primary,
    secondary,
):
    if team == "ARI":
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

    elif team in {
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

    elif team == "BUF":
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

    elif team == "CAR":
        draw_claws(
            draw,
            cx,
            cy,
            primary,
        )

    elif team in {
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

    elif team == "CIN":
        draw_stripes(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team in {
        "CLE",
        "GB",
    }:
        draw_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "DAL":
        draw_star(
            draw,
            cx,
            cy,
            30,
            primary,
        )

    elif team == "DEN":
        draw_mountain(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "HOU":
        draw_texas(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "IND":
        draw_arc(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "KC":
        draw_pennant(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "LV":
        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "LAC":
        draw_lightning(
            draw,
            cx,
            cy,
            secondary,
        )

    elif team == "LAR":
        draw_spiral(
            draw,
            cx,
            cy,
            secondary,
        )

    elif team == "MIA":
        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "MIN":
        draw_ship(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "NE":
        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "NO":
        draw_trumpet(
            draw,
            cx,
            cy,
            secondary,
        )

    elif team == "NYG":
        draw_skyline(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "NYJ":
        draw_jet(
            draw,
            cx,
            cy,
            primary,
        )

    elif team == "PIT":
        draw_diamonds(
            draw,
            cx,
            cy,
        )

    elif team == "SF":
        draw_bridge(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "SEA":
        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "TB":
        draw_flag(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "TEN":
        draw_sword(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "WSH":
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
# ============================================================

def create_team_icon(
    team: str,
    size: int = 215,
) -> Optional[
    Image.Image
]:
    team = (
        str(
            team
            or ""
        )
        .strip()
        .upper()
    )

    team = ALIASES.get(
        team,
        team,
    )

    if (
        not team
        or team not in TEAM_COLORS
    ):
        return None

    (
        primary_hex,
        secondary_hex,
    ) = TEAM_COLORS[
        team
    ]

    primary = hex_to_rgb(
        primary_hex
    )

    secondary = hex_to_rgb(
        secondary_hex
    )

    base_width = 132
    base_height = 150

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

    draw.rounded_rectangle(
        (
            5,
            5,
            base_width - 5,
            base_height - 5,
        ),
        radius=18,
        fill=(
            255,
            255,
            255,
            250,
        ),
        outline=(
            215,
            218,
            224,
            255,
        ),
        width=2,
    )

    draw.rounded_rectangle(
        (
            10,
            10,
            base_width - 10,
            base_height - 10,
        ),
        radius=14,
        outline=(
            238,
            240,
            244,
            255,
        ),
        width=2,
    )

    draw_team_symbol(
        draw,
        team,
        base_width // 2,
        52,
        primary,
        secondary,
    )

    abbreviation_font = load_font(
        20,
        True,
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        team,
        font=abbreviation_font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    draw.rounded_rectangle(
        (
            27,
            108,
            base_width - 27,
            136,
        ),
        radius=6,
        fill=(
            15,
            18,
            25,
            255,
        ),
    )

    draw.text(
        (
            (
                base_width
                - text_width
            )
            // 2,
            110,
        ),
        team,
        font=abbreviation_font,
        fill=(
            255,
            255,
            255,
            255,
        ),
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

    return icon.resize(
        (
            output_width,
            size,
        ),
        Image.Resampling.NEAREST,
    )


# ============================================================
# CORE TEAM ATHLETE ROSTER API
# ============================================================

def fetch_roster_json(
    team: str,
    year: int = DEFAULT_YEAR,
) -> Dict:
    """
    Uses the working ESPN Core team-athletes endpoint.

    Example:
    /seasons/2026/teams/22/athletes?limit=200

    The first request returns athlete $refs.
    Each athlete $ref is then fetched individually.

    One failed athlete will be skipped instead of killing
    the entire team request.
    """

    team_id, _, _ = (
        TEAM_META[
            team
        ]
    )

    url = (
        f"{CORE_API_BASE}/"
        f"seasons/{year}/"
        f"teams/{team_id}/"
        "athletes"
        "?limit=200"
    )

    print(
        f"{team}: Fetching Core "
        f"team athletes: "
        f"{url}"
    )

    index_data = fetch_json(
        url
    )

    athlete_refs = (
        index_data.get(
            "items"
        )
        or []
    )

    print(
        f"{team}: Core athlete "
        f"references="
        f"{len(athlete_refs)}"
    )

    athletes: List[
        Dict
    ] = []

    for index, item in enumerate(
        athlete_refs,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        ref = clean_text(
            item.get(
                "$ref"
            )
        )

        if not ref:
            continue

        try:
            athlete = (
                fetch_json(
                    ref
                )
            )

            if not isinstance(
                athlete,
                dict,
            ):
                continue

            name = clean_text(
                athlete.get(
                    "displayName"
                )
                or athlete.get(
                    "fullName"
                )
                or ""
            )

            print(
                f"{team}: Athlete "
                f"{index}/"
                f"{len(athlete_refs)}: "
                f"{name or ref}"
            )

            athletes.append(
                athlete
            )

        except Exception as exc:
            print(
                f"WARNING {team}: "
                f"Could not fetch "
                f"athlete "
                f"{index}/"
                f"{len(athlete_refs)}: "
                f"{exc}"
            )

            continue

    if not athletes:
        raise RuntimeError(
            f"No Core athlete "
            f"records returned "
            f"for {team} "
            f"season {year}."
        )

    return {
        "count": len(
            athletes
        ),
        "items": athletes,
    }


# ============================================================
# CORE DEPTH CHART
# ============================================================

def fetch_depthchart_json(
    team: str,
    year: int,
) -> Dict:
    team_id, _, _ = (
        TEAM_META[
            team
        ]
    )

    url = (
        f"{CORE_API_BASE}/"
        f"seasons/{year}/"
        f"teams/{team_id}/"
        "depthcharts"
    )

    print(
        f"{team}: Fetching Core "
        f"depth chart: "
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
        "displayName",
        "name",
        "positions",
        "athletes",
        "position",
        "abbreviation",
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
            "WARNING: Failed to "
            f"resolve Core ref "
            f"{ref}: {exc}"
        )

        return obj


def get_depthchart_groups(
    data: Any,
) -> List[Dict]:
    groups: List[
        Dict
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
    ):
        groups.append(
            data
        )

    return groups


# ============================================================
# PLAYER PARSING
# ============================================================

def parse_player(
    raw: Dict,
) -> Optional[
    Dict
]:
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
        if (
            not pos_obj.get(
                "abbreviation"
            )
            and pos_obj.get(
                "$ref"
            )
        ):
            pos_obj = (
                resolve_ref_object(
                    pos_obj
                )
            )

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

    pos = normalize_position(
        pos
    )

    if (
        not name
        or not pos
    ):
        return None

    return {
        "name": name,
        "name_key": normalize_name(
            name
        ),
        "position": pos,
    }


def parse_roster_players(
    team: str,
    year: int,
) -> List[
    Dict
]:
    data = fetch_roster_json(
        team,
        year=year,
    )

    players: List[
        Dict
    ] = []

    seen = set()

    athletes = (
        data.get(
            "items"
        )
        or []
    )

    for raw in athletes:
        if not isinstance(
            raw,
            dict,
        ):
            continue

        parsed = (
            parse_player(
                raw
            )
        )

        if not parsed:
            continue

        key = (
            parsed[
                "name_key"
            ]
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        players.append(
            parsed
        )

    if not players:
        raise RuntimeError(
            "No roster players "
            f"parsed for {team} "
            f"season {year}."
        )

    print(
        f"{team}: parsed "
        f"{len(players)} "
        f"Core roster players."
    )

    return players


# ============================================================
# DEPTH CHART POSITION HELPERS
# ============================================================

def depth_position_from_position_data(
    position_data: Dict,
) -> str:
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
                "abbreviation"
            )
            or position_obj.get(
                "name"
            )
            or ""
        )

    if isinstance(
        parent_obj,
        dict,
    ):
        parent_abbr = clean_text(
            parent_obj.get(
                "abbreviation"
            )
            or parent_obj.get(
                "name"
            )
            or ""
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
    athlete: Dict,
) -> Dict:
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
        resolved_nested = (
            resolve_ref_object(
                nested
            )
        )

        if isinstance(
            resolved_nested,
            dict,
        ):
            merged = dict(
                athlete
            )

            merged.update(
                resolved_nested
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
    group: Dict,
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
            if not isinstance(
                position_data,
                dict,
            ):
                continue

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
        for index, position_data in enumerate(
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
                    "abbreviation"
                )
                or resolved.get(
                    "name"
                )
                or resolved.get(
                    "displayName"
                )
                or index
            )

            yield (
                position_key,
                resolved,
            )


# ============================================================
# DEPTH CHART PARSER
# ============================================================

def parse_depthchart_order(
    team: str,
    year: int,
) -> List[
    Dict
]:
    data = fetch_depthchart_json(
        team,
        year,
    )

    groups = (
        get_depthchart_groups(
            data
        )
    )

    print(
        f"{team}: Core depth "
        f"chart groups="
        f"{len(groups)}"
    )

    ordered_players: List[
        Dict
    ] = []

    seen = set()

    for group in groups:
        if not isinstance(
            group,
            dict,
        ):
            continue

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

            for depth_rank, raw_athlete in enumerate(
                athletes,
                start=1,
            ):
                if not isinstance(
                    raw_athlete,
                    dict,
                ):
                    continue

                athlete = (
                    resolve_depth_athlete(
                        raw_athlete
                    )
                )

                name = clean_text(
                    athlete.get(
                        "displayName"
                    )
                    or athlete.get(
                        "fullName"
                    )
                    or athlete.get(
                        "shortName"
                    )
                    or athlete.get(
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

                if name_key in seen:
                    continue

                seen.add(
                    name_key
                )

                ordered_players.append(
                    {
                        "name": name,
                        "name_key": name_key,
                        "position": display_pos,
                        "depth_position_key": clean_text(
                            position_key
                        ).upper(),
                        "depth_rank": depth_rank,
                    }
                )

    print(
        f"{team}: parsed "
        f"{len(ordered_players)} "
        f"Core depth-chart players."
    )

    return ordered_players


# ============================================================
# ORDER ROSTER
# ============================================================

def order_roster_by_depthchart(
    roster_players: List[Dict],
    depth_players: List[Dict],
) -> List[Dict]:
    roster_by_name = {
        player[
            "name_key"
        ]: player
        for player
        in roster_players
    }

    ordered: List[
        Dict
    ] = []

    used = set()

    for depth_player in depth_players:
        name_key = (
            depth_player[
                "name_key"
            ]
        )

        if (
            name_key
            not in roster_by_name
        ):
            continue

        player = (
            roster_by_name[
                name_key
            ].copy()
        )

        depth_position = (
            depth_player.get(
                "position"
            )
        )

        if depth_position:
            player[
                "position"
            ] = depth_position

        player[
            "depth_position_key"
        ] = (
            depth_player.get(
                "depth_position_key",
                "",
            )
        )

        player[
            "depth_rank"
        ] = (
            depth_player.get(
                "depth_rank",
                "",
            )
        )

        ordered.append(
            player
        )

        used.add(
            name_key
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


def get_players(
    team: str,
    year: int,
) -> List[
    Dict
]:
    roster_players = (
        parse_roster_players(
            team,
            year,
        )
    )

    try:
        depth_players = (
            parse_depthchart_order(
                team,
                year,
            )
        )

        if depth_players:
            ordered_players = (
                order_roster_by_depthchart(
                    roster_players,
                    depth_players,
                )
            )

            print(
                f"{team}: Using "
                f"Core depth chart "
                f"ordering for "
                f"{len(depth_players)} "
                f"players."
            )

            return ordered_players

        print(
            f"WARNING: No Core "
            f"depth-chart rows "
            f"found for {team}."
        )

        print(
            "Using Core team-"
            "athlete order."
        )

    except Exception as exc:
        print(
            f"WARNING: Core depth "
            f"chart request failed "
            f"for {team}: {exc}"
        )

        print(
            "Using Core team-"
            "athlete order."
        )

    return roster_players


# ============================================================
# POSITION MATCHING
# ============================================================

def players_for_position(
    pos: str,
    roster: List[Dict],
    max_players: int = 8,
) -> List[str]:
    allowed_positions = {
        normalize_position(
            mapped_pos
        )
        for mapped_pos
        in POSITION_MAP.get(
            pos,
            [pos],
        )
    }

    result: List[
        str
    ] = []

    for player in roster:
        player_position = (
            normalize_position(
                player.get(
                    "position",
                    "",
                )
            )
        )

        if (
            player_position
            not in allowed_positions
        ):
            continue

        name = player.get(
            "name",
            "",
        )

        if (
            name
            and name not in result
        ):
            result.append(
                name
            )

        if len(
            result
        ) >= max_players:
            break

    return result


# ============================================================
# DRAWING HELPERS
# ============================================================

def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
) -> Tuple[
    int,
    int,
]:
    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
    )

    return (
        bbox[2]
        - bbox[0],
        bbox[3]
        - bbox[1],
    )


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> str:
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


def draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font,
    fill,
    canvas_width: int,
) -> int:
    width, height = (
        text_size(
            draw,
            text,
            font,
        )
    )

    x = (
        canvas_width
        - width
    ) // 2

    draw.text(
        (
            x,
            y,
        ),
        text,
        fill=fill,
        font=font,
    )

    return (
        y + height
    )


def make_gradient(
    width: int,
    height: int,
    top_color: Tuple[
        int,
        int,
        int,
    ],
    bottom_color: Tuple[
        int,
        int,
        int,
    ],
) -> Image.Image:
    img = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        top_color,
    )

    pixels = img.load()

    for y in range(
        height
    ):
        ratio = (
            y
            / max(
                1,
                height - 1,
            )
        )

        red = int(
            top_color[0]
            * (1 - ratio)
            + bottom_color[0]
            * ratio
        )

        green = int(
            top_color[1]
            * (1 - ratio)
            + bottom_color[1]
            * ratio
        )

        blue = int(
            top_color[2]
            * (1 - ratio)
            + bottom_color[2]
            * ratio
        )

        for x in range(
            width
        ):
            pixels[
                x,
                y,
            ] = (
                red,
                green,
                blue,
            )

    return img


def draw_need_block(
    draw: ImageDraw.ImageDraw,
    pos: str,
    players: List[str],
    x: int,
    y: int,
    width: int,
    accent: str,
    pos_font,
    player_font,
    small_font,
) -> int:
    draw.text(
        (
            x,
            y,
        ),
        pos,
        fill=accent,
        font=pos_font,
    )

    label = (
        "POSITION GROUP NEED"
    )

    label_width, _ = (
        text_size(
            draw,
            label,
            small_font,
        )
    )

    draw.text(
        (
            x
            + width
            - label_width,
            y + 8,
        ),
        label,
        fill="#555555",
        font=small_font,
    )

    y += 52

    if not players:
        draw.text(
            (
                x + 18,
                y,
            ),
            "1 - No players found",
            fill="black",
            font=player_font,
        )

        return (
            y + 42
        )

    for index, player in enumerate(
        players,
        1,
    ):
        line = (
            f"{index} - "
            f"{player}"
        )

        line = fit_text(
            draw,
            line,
            player_font,
            width - 24,
        )

        draw.text(
            (
                x + 18,
                y,
            ),
            line,
            fill="black",
            font=player_font,
        )

        y += 38

    return y


# ============================================================
# POSTER
# ============================================================

def poster(
    team: str,
    year: int = DEFAULT_YEAR,
    out_file: Optional[
        str
    ] = None,
):
    team = ALIASES.get(
        team,
        team,
    )

    if (
        team
        not in TEAM_META
    ):
        raise ValueError(
            f"Invalid team: "
            f"{team}"
        )

    print()

    print(
        "=" * 80
    )

    print(
        f"GENERATING TEAM NEEDS: "
        f"{team} | "
        f"SEASON {year}"
    )

    print(
        "=" * 80
    )

    roster = get_players(
        team,
        year,
    )

    _, _, name = (
        TEAM_META[
            team
        ]
    )

    (
        primary_hex,
        accent_hex,
    ) = (
        TEAM_COLORS[
            team
        ]
    )

    width = 1600
    height = 2000

    img = (
        make_gradient(
            width,
            height,
            hex_to_rgb(
                primary_hex
            ),
            (
                10,
                10,
                10,
            ),
        )
        .convert(
            "RGBA"
        )
    )

    draw = ImageDraw.Draw(
        img
    )

    title_font = load_font(
        82,
        True,
    )

    team_font = load_font(
        56,
        True,
    )

    pos_font = load_font(
        38,
        True,
    )

    player_font = load_font(
        27,
        False,
    )

    small_font = load_font(
        19,
        True,
    )

    footer_font = load_font(
        22,
        False,
    )

    y = 50

    logo = create_team_icon(
        team,
        size=215,
    )

    if logo:
        logo_x = (
            width
            - logo.width
        ) // 2

        img.alpha_composite(
            logo,
            (
                logo_x,
                y,
            ),
        )

        y += (
            logo.height
            + 20
        )

    y = (
        draw_centered(
            draw,
            name.upper(),
            y,
            team_font,
            "white",
            width,
        )
        + 16
    )

    y = (
        draw_centered(
            draw,
            "TEAM NEEDS BOARD",
            y,
            title_font,
            "white",
            width,
        )
        + 34
    )

    panel_x1 = 105
    panel_y1 = y
    panel_x2 = (
        width - 105
    )
    panel_y2 = (
        height - 120
    )

    draw.rounded_rectangle(
        (
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y2,
        ),
        radius=34,
        fill=(
            245,
            245,
            245,
            238,
        ),
        outline=(
            255,
            255,
            255,
            85,
        ),
        width=3,
    )

    y = (
        panel_y1 + 38
    )

    left = (
        panel_x1 + 45
    )

    right = (
        panel_x2 - 45
    )

    block_width = (
        right - left
    )

    needs = (
        TEAM_NEEDS[
            team
        ]
    )

    for index, pos in enumerate(
        needs,
        1,
    ):
        players = (
            players_for_position(
                pos,
                roster,
                max_players=8,
            )
        )

        y = draw_need_block(
            draw=draw,
            pos=f"{index}. {pos}",
            players=players,
            x=left,
            y=y,
            width=block_width,
            accent=accent_hex,
            pos_font=pos_font,
            player_font=player_font,
            small_font=small_font,
        )

        y += 18

        if (
            index
            != len(needs)
        ):
            draw.line(
                (
                    left,
                    y,
                    right,
                    y,
                ),
                fill=(
                    190,
                    190,
                    190,
                ),
                width=2,
            )

            y += 26

        if (
            y
            > panel_y2
            - 120
        ):
            break

    draw_centered(
        draw,
        f"{team} TEAM NEEDS",
        height - 60,
        footer_font,
        "#DADADA",
        width,
    )

    out_file = (
        out_file
        or (
            f"{team.lower()}_"
            "team_needs.png"
        )
    )

    os.makedirs(
        os.path.dirname(
            out_file
        )
        or ".",
        exist_ok=True,
    )

    img.convert(
        "RGB"
    ).save(
        out_file,
        quality=95,
    )

    print(
        f"Saved: {out_file}"
    )

    return out_file


# ============================================================
# ALL TEAMS
#
# IMPORTANT:
# The nightly publisher imports this exact function name.
# ============================================================

def generate_all_team_needs_posters(
    outdir: str,
    year: int = DEFAULT_YEAR,
):
    if os.path.exists(
        outdir
    ):
        shutil.rmtree(
            outdir
        )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    outputs = {}
    failures = {}

    for team in TEAM_META:
        try:
            out_file = (
                os.path.join(
                    outdir,
                    f"{team.lower()}_"
                    "team_needs.png",
                )
            )

            poster(
                team,
                year=year,
                out_file=out_file,
            )

            outputs[
                team
            ] = out_file

        except Exception as exc:
            failures[
                team
            ] = str(
                exc
            )

            print(
                f"ERROR {team}: "
                f"{exc}"
            )

    return (
        outputs,
        failures,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "team",
        nargs="?",
        help=(
            "Example: "
            "BUF, DAL, WSH"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Generate all "
            "32 teams"
        ),
    )

    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=(
            "Season year. "
            "Defaults to "
            f"{DEFAULT_YEAR}."
        ),
    )

    parser.add_argument(
        "--outdir",
        default=(
            "team_needs_"
            "numbered_posters"
        ),
        help=(
            "Output directory"
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.all:
        (
            outputs,
            failures,
        ) = (
            generate_all_team_needs_posters(
                args.outdir,
                year=args.year,
            )
        )

        print()

        print(
            f"Generated "
            f"{len(outputs)} "
            f"posters in "
            f"{args.outdir}"
        )

        if failures:
            print(
                "Failures:"
            )

            for (
                team,
                error,
            ) in failures.items():
                print(
                    f"{team}: "
                    f"{error}"
                )

        return

    if not args.team:
        raise SystemExit(
            "Provide a team "
            "abbreviation like BUF, "
            "or use --all"
        )

    team = ALIASES.get(
        args.team.upper(),
        args.team.upper(),
    )

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    out_file = (
        os.path.join(
            args.outdir,
            f"{team.lower()}_"
            "team_needs.png",
        )
    )

    saved = poster(
        team,
        year=args.year,
        out_file=out_file,
    )

    print(
        f"Saved {saved}"
    )


if __name__ == "__main__":
    main()
