#!/usr/bin/env python3

import os
import shutil
import re
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# POSTER CONFIG
# ============================================================

W, H = 1080, 1920

BLUE = (128, 183, 255)
BG = (10, 14, 24)
CARD = (24, 29, 42)
BORDER = (64, 74, 98)
WHITE = (246, 248, 252)
MUTED = (188, 198, 217)
HEADER = (22, 38, 74)


# ============================================================
# ORIGINAL TEAM ICON COLORS
#
# These are used to create our own simplified pixel-style
# symbols directly in Python.
#
# No official team logo images are downloaded.
# ============================================================

TEAM_ICON_COLORS = {
    "ARI": ("#B91C2E", "#D98C3A"),
    "ATL": ("#C71F37", "#111111"),
    "BAL": ("#5A2D91", "#D5A824"),
    "BUF": ("#1769AA", "#C8102E"),
    "CAR": ("#0085CA", "#101820"),
    "CHI": ("#D66A13", "#5A3216"),
    "CIN": ("#FB4F14", "#111111"),
    "CLE": ("#E85D04", "#5B2C12"),
    "DAL": ("#17365D", "#B7C1CC"),
    "DEN": ("#F36C21", "#17365D"),
    "DET": ("#1673B1", "#A7B1BA"),
    "GB": ("#1F5132", "#E3B23C"),
    "HOU": ("#17365D", "#C51F35"),
    "IND": ("#194F90", "#FFFFFF"),
    "JAX": ("#D5A52A", "#008C95"),
    "KC": ("#C62828", "#EFB32B"),
    "LV": ("#252525", "#B8B8B8"),
    "LAC": ("#168DD0", "#F5C542"),
    "LAR": ("#E0A520", "#1C4C8C"),
    "MIA": ("#1499A5", "#ED6A29"),
    "MIN": ("#5B328A", "#E5B735"),
    "NE": ("#17365D", "#C6283E"),
    "NO": ("#C9A13B", "#171717"),
    "NYG": ("#1F5594", "#B92739"),
    "NYJ": ("#1F573D", "#FFFFFF"),
    "PHI": ("#126A70", "#D5D8DA"),
    "PIT": ("#E3B52B", "#CB273A"),
    "SF": ("#C5412D", "#D9A640"),
    "SEA": ("#17365D", "#58A64A"),
    "TB": ("#B82729", "#E36D2E"),
    "TEN": ("#4D85BD", "#17365D"),
    "WSH": ("#8F2433", "#E6B640"),
}


# ============================================================
# HTTP
# ============================================================

def fetch_url(
    url: str,
    timeout: int = 25,
) -> str:

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.text


def fetch_summary(
    event_id: str,
) -> Dict:

    # The lowercase league name in this URL is an internal
    # ESPN API path and is required for the request to work.
    api_url = (
        "https://site.web.api.espn.com/apis/site/v2/"
        "sports/football/nfl/summary"
        f"?event={event_id}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    response = requests.get(
        api_url,
        headers=headers,
        timeout=25,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# NUMERIC HELPERS
# ============================================================

def safe_int(
    value,
    default: Optional[int] = 0,
) -> Optional[int]:

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_int(
    value,
    default=0,
) -> int:

    try:
        return int(value)

    except Exception:

        try:
            return int(
                float(value)
            )

        except Exception:
            return default


# ============================================================
# FONT HELPERS
# ============================================================

def load_font(
    size: int,
    bold: bool = False,
):

    candidates = [
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

    for path in candidates:

        try:
            return ImageFont.truetype(
                path,
                size=size,
            )

        except Exception:
            pass

    return ImageFont.load_default()


# ============================================================
# COLOR HELPERS
# ============================================================

def hex_to_rgb(
    value: str,
) -> Tuple[int, int, int]:

    value = (
        value
        .replace("#", "")
        .strip()
    )

    return tuple(
        int(
            value[i:i + 2],
            16,
        )
        for i in (
            0,
            2,
            4,
        )
    )


# ============================================================
# ORIGINAL PIXEL ICON SYSTEM
#
# The icons below are generic symbolic representations,
# not reproductions of official marks.
#
# Every icon contains the abbreviation beneath the symbol.
# ============================================================

def draw_pixel_star(
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


def draw_pixel_paw(
    draw,
    cx,
    cy,
    fill,
):

    draw.ellipse(
        (
            cx - 17,
            cy - 2,
            cx + 17,
            cy + 28,
        ),
        fill=fill,
    )

    toe_positions = [
        (-23, -21),
        (-8, -30),
        (8, -30),
        (23, -21),
    ]

    for dx, dy in toe_positions:

        draw.ellipse(
            (
                cx + dx - 7,
                cy + dy - 9,
                cx + dx + 7,
                cy + dy + 9,
            ),
            fill=fill,
        )


def draw_pixel_football(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.ellipse(
        (
            cx - 42,
            cy - 22,
            cx + 42,
            cy + 22,
        ),
        fill=fill,
    )

    draw.line(
        (
            cx - 18,
            cy,
            cx + 18,
            cy,
        ),
        fill=accent,
        width=4,
    )

    for offset in (
        -12,
        -4,
        4,
        12,
    ):
        draw.line(
            (
                cx + offset,
                cy - 7,
                cx + offset,
                cy + 7,
            ),
            fill=accent,
            width=3,
        )


def draw_pixel_wave(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    points = [
        (cx - 44, cy + 20),
        (cx - 31, cy - 7),
        (cx - 14, cy - 25),
        (cx + 5, cy - 30),
        (cx + 27, cy - 18),
        (cx + 45, cy + 5),
        (cx + 18, cy - 2),
        (cx + 1, cy + 9),
        (cx - 8, cy + 25),
    ]

    draw.polygon(
        points,
        fill=fill,
    )

    draw.line(
        (
            cx - 38,
            cy + 26,
            cx + 40,
            cy + 26,
        ),
        fill=accent,
        width=6,
    )


def draw_pixel_lightning(
    draw,
    cx,
    cy,
    fill,
):

    points = [
        (cx + 7, cy - 47),
        (cx - 30, cy + 2),
        (cx - 6, cy + 2),
        (cx - 18, cy + 47),
        (cx + 34, cy - 10),
        (cx + 10, cy - 10),
    ]

    draw.polygon(
        points,
        fill=fill,
    )


def draw_pixel_feather(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    points = [
        (cx - 35, cy + 32),
        (cx - 20, cy - 24),
        (cx + 34, cy - 42),
        (cx + 22, cy + 10),
        (cx - 12, cy + 32),
    ]

    draw.polygon(
        points,
        fill=fill,
    )

    draw.line(
        (
            cx - 30,
            cy + 38,
            cx + 23,
            cy - 30,
        ),
        fill=accent,
        width=4,
    )


def draw_pixel_claws(
    draw,
    cx,
    cy,
    fill,
):

    for offset in (
        -24,
        0,
        24,
    ):

        draw.polygon(
            [
                (
                    cx + offset - 6,
                    cy + 35,
                ),
                (
                    cx + offset + 4,
                    cy - 38,
                ),
                (
                    cx + offset + 14,
                    cy - 44,
                ),
                (
                    cx + offset + 4,
                    cy + 35,
                ),
            ],
            fill=fill,
        )


def draw_pixel_stripes(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 42,
            cy - 42,
            cx + 42,
            cy + 42,
        ),
        fill=fill,
    )

    for offset in (
        -42,
        -10,
        22,
    ):

        draw.polygon(
            [
                (cx + offset, cy - 42),
                (cx + offset + 15, cy - 42),
                (cx + offset + 50, cy + 42),
                (cx + offset + 35, cy + 42),
            ],
            fill=accent,
        )


def draw_pixel_mountain(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 48, cy + 34),
            (cx, cy - 42),
            (cx + 48, cy + 34),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (cx - 11, cy - 24),
            (cx, cy - 42),
            (cx + 13, cy - 21),
            (cx + 3, cy - 27),
        ],
        fill=accent,
    )


def draw_pixel_texas(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    points = [
        (cx - 35, cy - 38),
        (cx + 10, cy - 38),
        (cx + 11, cy - 16),
        (cx + 38, cy - 15),
        (cx + 29, cy + 11),
        (cx + 9, cy + 19),
        (cx - 3, cy + 43),
        (cx - 20, cy + 21),
        (cx - 38, cy + 5),
    ]

    draw.polygon(
        points,
        fill=fill,
    )

    draw_pixel_star(
        draw,
        cx,
        cy - 1,
        12,
        accent,
    )


def draw_pixel_arc(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.arc(
        (
            cx - 42,
            cy - 42,
            cx + 42,
            cy + 42,
        ),
        200,
        340,
        fill=fill,
        width=12,
    )

    draw.arc(
        (
            cx - 28,
            cy - 28,
            cx + 28,
            cy + 28,
        ),
        200,
        340,
        fill=accent,
        width=6,
    )


def draw_pixel_pennant(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 40,
            cy - 43,
            cx - 34,
            cy + 40,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (cx - 34, cy - 36),
            (cx + 43, cy - 5),
            (cx - 34, cy + 22),
        ],
        fill=fill,
    )


def draw_pixel_pirate_hat(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 43, cy + 23),
            (cx - 28, cy - 13),
            (cx, cy - 30),
            (cx + 28, cy - 13),
            (cx + 43, cy + 23),
            (cx + 13, cy + 14),
            (cx, cy + 28),
            (cx - 13, cy + 14),
        ],
        fill=fill,
    )

    draw.ellipse(
        (
            cx - 9,
            cy - 7,
            cx + 9,
            cy + 11,
        ),
        fill=accent,
    )


def draw_pixel_spiral(
    draw,
    cx,
    cy,
    fill,
):

    draw.arc(
        (
            cx - 38,
            cy - 38,
            cx + 38,
            cy + 38,
        ),
        20,
        340,
        fill=fill,
        width=11,
    )

    draw.arc(
        (
            cx - 22,
            cy - 22,
            cx + 22,
            cy + 22,
        ),
        20,
        300,
        fill=fill,
        width=8,
    )


def draw_pixel_ship(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 45, cy + 22),
            (cx + 45, cy + 22),
            (cx + 29, cy + 39),
            (cx - 29, cy + 39),
        ],
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 3,
            cy - 42,
            cx + 3,
            cy + 20,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (cx + 2, cy - 35),
            (cx + 32, cy - 7),
            (cx + 2, cy - 7),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (cx - 4, cy - 31),
            (cx - 30, cy - 6),
            (cx - 4, cy - 6),
        ],
        fill=fill,
    )


def draw_pixel_hat(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 44, cy + 20),
            (cx - 24, cy - 18),
            (cx, cy - 33),
            (cx + 24, cy - 18),
            (cx + 44, cy + 20),
            (cx + 13, cy + 12),
            (cx, cy + 27),
            (cx - 13, cy + 12),
        ],
        fill=fill,
    )

    draw_pixel_star(
        draw,
        cx,
        cy,
        11,
        accent,
    )


def draw_pixel_trumpet(
    draw,
    cx,
    cy,
    fill,
):

    draw.rectangle(
        (
            cx - 30,
            cy - 7,
            cx + 14,
            cy + 7,
        ),
        fill=fill,
    )

    draw.polygon(
        [
            (cx + 14, cy - 19),
            (cx + 44, cy - 30),
            (cx + 44, cy + 30),
            (cx + 14, cy + 19),
        ],
        fill=fill,
    )

    draw.arc(
        (
            cx - 38,
            cy - 3,
            cx - 15,
            cy + 30,
        ),
        10,
        190,
        fill=fill,
        width=5,
    )


def draw_pixel_skyline(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    buildings = [
        (-42, 15, -29, 38),
        (-27, -4, -10, 38),
        (-8, -39, 8, 38),
        (10, -13, 28, 38),
        (30, 3, 43, 38),
    ]

    for x1, y1, x2, y2 in buildings:

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
            cx - 48,
            cy + 39,
            cx + 48,
            cy + 39,
        ),
        fill=accent,
        width=5,
    )


def draw_pixel_jet(
    draw,
    cx,
    cy,
    fill,
):

    draw.polygon(
        [
            (cx - 47, cy + 8),
            (cx - 10, cy - 6),
            (cx + 25, cy - 37),
            (cx + 34, cy - 31),
            (cx + 16, cy - 4),
            (cx + 46, cy + 9),
            (cx + 14, cy + 13),
            (cx + 4, cy + 36),
            (cx - 6, cy + 36),
            (cx - 8, cy + 14),
        ],
        fill=fill,
    )


def draw_pixel_diamonds(
    draw,
    cx,
    cy,
    colors,
):

    locations = [
        (
            cx,
            cy - 24,
            colors[0],
        ),
        (
            cx - 25,
            cy + 18,
            colors[1],
        ),
        (
            cx + 25,
            cy + 18,
            colors[2],
        ),
    ]

    for x, y, color in locations:

        draw.polygon(
            [
                (x, y - 13),
                (x + 13, y),
                (x, y + 13),
                (x - 13, y),
            ],
            fill=color,
        )


def draw_pixel_bridge(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 35,
            cy - 35,
            cx - 28,
            cy + 35,
        ),
        fill=fill,
    )

    draw.rectangle(
        (
            cx + 28,
            cy - 35,
            cx + 35,
            cy + 35,
        ),
        fill=fill,
    )

    draw.line(
        (
            cx - 45,
            cy + 23,
            cx + 45,
            cy + 23,
        ),
        fill=fill,
        width=6,
    )

    draw.arc(
        (
            cx - 32,
            cy - 30,
            cx + 32,
            cy + 42,
        ),
        180,
        360,
        fill=accent,
        width=4,
    )


def draw_pixel_flag(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 37,
            cy - 43,
            cx - 31,
            cy + 40,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (cx - 31, cy - 35),
            (cx + 37, cy - 28),
            (cx + 25, cy + 4),
            (cx - 31, cy + 11),
        ],
        fill=fill,
    )

    draw_pixel_star(
        draw,
        cx,
        cy - 12,
        9,
        accent,
    )


def draw_pixel_sword(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (cx - 5, cy - 45),
            (cx + 6, cy - 45),
            (cx + 9, cy + 17),
            (cx - 8, cy + 17),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (cx - 5, cy - 45),
            (cx + 1, cy - 57),
            (cx + 6, cy - 45),
        ],
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 24,
            cy + 14,
            cx + 24,
            cy + 21,
        ),
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 5,
            cy + 20,
            cx + 6,
            cy + 42,
        ),
        fill=accent,
    )


def draw_pixel_column(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 36,
            cy - 37,
            cx + 36,
            cy - 29,
        ),
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 42,
            cy + 31,
            cx + 42,
            cy + 39,
        ),
        fill=accent,
    )

    for x in (
        -27,
        -9,
        9,
        27,
    ):

        draw.rectangle(
            (
                cx + x - 4,
                cy - 26,
                cx + x + 4,
                cy + 29,
            ),
            fill=fill,
        )


def draw_generic_symbol(
    draw,
    abbr,
    cx,
    cy,
    primary,
    secondary,
):

    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

    if abbr == "ARI":
        # Desert cactus
        draw.rectangle(
            (
                cx - 7,
                cy - 38,
                cx + 7,
                cy + 35,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx - 29,
                cy - 9,
                cx - 5,
                cy + 4,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx - 29,
                cy - 26,
                cx - 17,
                cy + 4,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx + 5,
                cy - 15,
                cx + 28,
                cy - 3,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx + 17,
                cy - 30,
                cx + 28,
                cy - 3,
            ),
            fill=primary,
        )

    elif abbr == "ATL":
        draw_pixel_feather(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "BAL":
        draw_pixel_feather(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "BUF":
        # Abstract hoof marks
        draw.ellipse(
            (
                cx - 34,
                cy - 35,
                cx - 5,
                cy + 8,
            ),
            fill=primary,
        )

        draw.ellipse(
            (
                cx + 5,
                cy - 35,
                cx + 34,
                cy + 8,
            ),
            fill=primary,
        )

        draw.ellipse(
            (
                cx - 12,
                cy + 11,
                cx + 12,
                cy + 35,
            ),
            fill=secondary,
        )

    elif abbr == "CAR":
        draw_pixel_claws(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "CHI":
        draw_pixel_paw(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "CIN":
        draw_pixel_stripes(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "CLE":
        # Generic orange football
        draw_pixel_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    elif abbr == "DAL":
        draw_pixel_star(
            draw,
            cx,
            cy,
            42,
            primary,
        )

    elif abbr == "DEN":
        draw_pixel_mountain(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "DET":
        draw_pixel_paw(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "GB":
        draw_pixel_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "HOU":
        draw_pixel_texas(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "IND":
        # Original speed/racing arc instead of a horseshoe
        draw_pixel_arc(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "JAX":
        draw_pixel_paw(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "KC":
        draw_pixel_pennant(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    # --------------------------------------------------------
    # THIRD ROW
    # --------------------------------------------------------

    elif abbr == "LV":
        draw_pixel_pirate_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "LAC":
        draw_pixel_lightning(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbr == "LAR":
        draw_pixel_spiral(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "MIA":
        draw_pixel_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "MIN":
        draw_pixel_ship(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "NE":
        draw_pixel_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "NO":
        # Generic gold trumpet instead of a fleur-de-lis
        draw_pixel_trumpet(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "NYG":
        draw_pixel_skyline(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    # --------------------------------------------------------
    # FOURTH ROW
    # --------------------------------------------------------

    elif abbr == "NYJ":
        draw_pixel_jet(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "PHI":
        draw_pixel_feather(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "PIT":
        draw_pixel_diamonds(
            draw,
            cx,
            cy,
            (
                "#F2C230",
                "#1E5FA8",
                "#D32E3E",
            ),
        )

    elif abbr == "SF":
        draw_pixel_bridge(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "SEA":
        draw_pixel_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "TB":
        draw_pixel_flag(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "TEN":
        draw_pixel_sword(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "WSH":
        draw_pixel_column(
            draw,
            cx,
            cy,
            secondary,
            primary,
        )

    else:
        draw_pixel_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )


def create_team_icon(
    team_abbr: str,
    size: int = 155,
) -> Image.Image:
    """
    Create one original pixel-style icon.

    The abbreviation is part of the generated icon itself.
    """

    abbr = (
        str(
            team_abbr
            or ""
        )
        .strip()
        .upper()
    )

    primary_hex, secondary_hex = (
        TEAM_ICON_COLORS.get(
            abbr,
            (
                "#4A6A8A",
                "#C7D0DA",
            ),
        )
    )

    primary = hex_to_rgb(
        primary_hex
    )

    secondary = hex_to_rgb(
        secondary_hex
    )

    # Render at deliberately low resolution first.
    # Scaling with NEAREST gives the 8-bit/pixel effect.
    base_width = 112
    base_height = 130

    base = Image.new(
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
        base
    )

    symbol_cx = (
        base_width // 2
    )

    symbol_cy = 50

    draw_generic_symbol(
        draw,
        abbr,
        symbol_cx,
        symbol_cy,
        primary,
        secondary,
    )

    # --------------------------------------------------------
    # ABBREVIATION BUILT INTO ICON
    # --------------------------------------------------------

    abbreviation_font = load_font(
        20,
        True,
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        abbr,
        font=abbreviation_font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    text_x = (
        base_width
        - text_width
    ) // 2

    # Small dark backing makes abbreviation readable
    # on every poster background.
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

    draw.text(
        (
            text_x,
            103,
        ),
        abbr,
        font=abbreviation_font,
        fill=primary,
    )

    # --------------------------------------------------------
    # SCALE TO REQUESTED SIZE
    # --------------------------------------------------------

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

    output = base.resize(
        (
            output_width,
            size,
        ),
        Image.Resampling.NEAREST,
    )

    return output


# ============================================================
# SCOREBOARD
# ============================================================

def scoreboard_url(
    year: int,
    week: int,
    seasontype: int,
) -> str:

    # The lowercase league name is required inside ESPN's
    # URL and is not displayed anywhere in the poster.
    return (
        "https://www.espn.com/"
        f"nfl/scoreboard/_/week/{week}/"
        f"year/{year}/"
        f"seasontype/{seasontype}"
    )


def extract_game_ids_from_scoreboard_html(
    html: str,
) -> List[str]:

    ids = re.findall(
        r"gameId/(\d+)",
        html,
    )

    seen = set()

    output = []

    for game_id in ids:

        if game_id not in seen:

            seen.add(
                game_id
            )

            output.append(
                game_id
            )

    return output


# ============================================================
# HTML FALLBACK
# ============================================================

def _strip_html_to_text(
    html: str,
) -> str:

    html = re.sub(
        r"<script[\s\S]*?</script>",
        " ",
        html,
        flags=re.IGNORECASE,
    )

    html = re.sub(
        r"<style[\s\S]*?</style>",
        " ",
        html,
        flags=re.IGNORECASE,
    )

    html = re.sub(
        r"<[^>]+>",
        " ",
        html,
    )

    html = re.sub(
        r"&nbsp;|&#160;",
        " ",
        html,
    )

    html = re.sub(
        r"\s+",
        " ",
        html,
    ).strip()

    return html


# ============================================================
# SCORING BY PERIOD
# ============================================================

def get_scoring_periods_from_summary(
    event_id: str,
) -> Tuple[
    List[str],
    Dict[
        str,
        List[int],
    ],
]:

    url = (
        "https://site.web.api.espn.com/apis/site/v2/"
        "sports/football/nfl/summary"
        f"?event={event_id}"
    )

    response = requests.get(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0"
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    competition = (
        data["header"]
        ["competitions"][0]
    )

    temporary: Dict[
        str,
        List[int],
    ] = {}

    max_periods = 0

    for competitor in competition[
        "competitors"
    ]:

        abbreviation = (
            competitor[
                "team"
            ][
                "abbreviation"
            ]
        )

        linescores = (
            competitor.get(
                "linescores",
                [],
            )
            or []
        )

        values = [
            _safe_int(
                quarter.get(
                    "value",
                    quarter.get(
                        "displayValue",
                        0,
                    ),
                )
            )
            for quarter
            in linescores
        ]

        temporary[
            abbreviation
        ] = values

        max_periods = max(
            max_periods,
            len(values),
        )

    if (
        max_periods == 0
        or any(
            len(values) == 0
            for values
            in temporary.values()
        )
    ):

        labels, scraped = (
            fetch_scoring_periods_from_boxscore(
                event_id
            )
        )

        return (
            labels,
            scraped,
        )

    has_ot = (
        max_periods > 4
    )

    labels = [
        "1Q",
        "2Q",
        "3Q",
        "4Q",
    ]

    if has_ot:
        labels.append(
            "OT"
        )

    target_length = (
        5
        if has_ot
        else 4
    )

    output: Dict[
        str,
        List[int],
    ] = {}

    for (
        abbreviation,
        values,
    ) in temporary.items():

        output[
            abbreviation
        ] = (
            values
            + [0]
            * target_length
        )[
            :target_length
        ]

    return (
        labels,
        output,
    )


def fetch_scoring_periods_from_boxscore(
    event_id: str,
) -> Tuple[
    List[str],
    Dict[
        str,
        List[int],
    ],
]:

    url = (
        "https://www.espn.com/"
        f"nfl/boxscore/_/gameId/"
        f"{event_id}"
    )

    response = requests.get(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0"
        },
        timeout=15,
    )

    response.raise_for_status()

    text = _strip_html_to_text(
        response.text
    )

    match = re.search(
        r"\bFinal\b\s+"
        r"((?:(?:\d+|OT)\s+)+)"
        r"T\b",
        text,
    )

    if not match:

        return (
            [
                "1Q",
                "2Q",
                "3Q",
                "4Q",
            ],
            {},
        )

    raw_labels = (
        match.group(1)
        .strip()
        .split()
    )

    labels = [
        (
            f"{value}Q"
            if value.isdigit()
            else "OT"
        )
        for value
        in raw_labels
    ]

    count = len(
        labels
    )

    window = text[
        match.end():
        match.end()
        + 2200
    ]

    row_pattern = re.compile(
        rf"\b([A-Z]{{2,4}})\b\s+"
        rf"((?:\d+\s+){{{count}}}\d+)\b"
    )

    rows = (
        row_pattern.findall(
            window
        )[
            :2
        ]
    )

    output: Dict[
        str,
        List[int],
    ] = {}

    for (
        abbreviation,
        numbers_blob,
    ) in rows:

        numbers = [
            _safe_int(value)
            for value
            in numbers_blob
            .strip()
            .split()
        ]

        if (
            len(numbers)
            == count + 1
        ):

            output[
                abbreviation
            ] = numbers[
                :count
            ]

    return (
        labels,
        output,
    )


# ============================================================
# STAT PARSING
# ============================================================

def parse_stat_group(
    stat_group: Dict,
) -> List[Dict]:

    labels = (
        stat_group.get(
            "labels",
            [],
        )
    )

    athletes = (
        stat_group.get(
            "athletes",
            [],
        )
    )

    rows: List[
        Dict
    ] = []

    for athlete in athletes:

        athlete_info = (
            athlete.get(
                "athlete",
                {},
            )
            or {}
        )

        stats = (
            athlete.get(
                "stats",
                [],
            )
            or []
        )

        row = dict(
            zip(
                labels,
                stats,
            )
        )

        row[
            "player"
        ] = (
            athlete_info.get(
                "displayName"
            )
            or athlete_info.get(
                "shortName"
            )
        )

        rows.append(
            row
        )

    return rows


def extract_passing_leader(
    stat_group: Dict,
) -> Optional[Dict]:

    rows = parse_stat_group(
        stat_group
    )

    if not rows:
        return None

    row = rows[0]

    cmp_att = (
        row.get(
            "C/ATT"
        )
        or row.get(
            "CMP/ATT"
        )
        or ""
    )

    completions = None
    attempts = None

    if "/" in cmp_att:

        completion_string, attempt_string = (
            cmp_att.split(
                "/",
                1,
            )
        )

        completions = safe_int(
            completion_string,
            default=None,
        )

        attempts = safe_int(
            attempt_string,
            default=None,
        )

    return {
        "name": row.get(
            "player"
        ),
        "completions": completions,
        "attempts": attempts,
        "yards": safe_int(
            row.get(
                "YDS"
            ),
            0,
        ),
        "td": safe_int(
            row.get(
                "TD"
            ),
            0,
        ),
        "ints": safe_int(
            row.get(
                "INT"
            ),
            0,
        ),
    }


def extract_yardage_leader(
    stat_group: Dict,
    kind: str,
) -> Optional[Dict]:

    rows = parse_stat_group(
        stat_group
    )

    if not rows:
        return None

    leader = None

    max_yards = -1

    for row in rows:

        yards = safe_int(
            row.get(
                "YDS"
            ),
            0,
        )

        if (
            yards
            > max_yards
        ):

            max_yards = yards

            leader = row

    if not leader:
        return None

    if kind == "rushing":

        return {
            "name": leader.get(
                "player"
            ),
            "carries": safe_int(
                leader.get(
                    "CAR"
                ),
                0,
            ),
            "yards": safe_int(
                leader.get(
                    "YDS"
                ),
                0,
            ),
            "td": safe_int(
                leader.get(
                    "TD"
                ),
                0,
            ),
        }

    return {
        "name": leader.get(
            "player"
        ),
        "receptions": safe_int(
            leader.get(
                "REC"
            ),
            0,
        ),
        "yards": safe_int(
            leader.get(
                "YDS"
            ),
            0,
        ),
        "td": safe_int(
            leader.get(
                "TD"
            ),
            0,
        ),
    }


# ============================================================
# OFFENSIVE LEADERS
# ============================================================

def extract_team_leaders_from_players_block(
    team_block: Dict,
) -> Dict:

    team_info = (
        team_block.get(
            "team",
            {},
        )
        or {}
    )

    team_name = (
        team_info.get(
            "displayName"
        )
        or team_info.get(
            "name"
        )
    )

    leaders: Dict[
        str,
        Optional[Dict],
    ] = {
        "team": team_name,
        "passing_leader": None,
        "rushing_leader": None,
        "receiving_leader": None,
    }

    for stat_group in (
        team_block.get(
            "statistics",
            [],
        )
    ):

        group_name = (
            stat_group.get(
                "name"
            )
            or ""
        ).lower()

        if (
            group_name
            == "passing"
        ):

            leaders[
                "passing_leader"
            ] = (
                extract_passing_leader(
                    stat_group
                )
            )

        elif (
            group_name
            == "rushing"
        ):

            leaders[
                "rushing_leader"
            ] = (
                extract_yardage_leader(
                    stat_group,
                    "rushing",
                )
            )

        elif (
            group_name
            == "receiving"
        ):

            leaders[
                "receiving_leader"
            ] = (
                extract_yardage_leader(
                    stat_group,
                    "receiving",
                )
            )

    return leaders


def extract_all_team_leaders(
    summary: Dict,
) -> Dict[
    str,
    Dict,
]:

    boxscore = (
        summary.get(
            "boxscore",
            {},
        )
        or {}
    )

    player_blocks = (
        boxscore.get(
            "players",
            [],
        )
        or []
    )

    results: Dict[
        str,
        Dict,
    ] = {}

    for team_block in player_blocks:

        team_leaders = (
            extract_team_leaders_from_players_block(
                team_block
            )
        )

        team_name = (
            team_leaders[
                "team"
            ]
        )

        if team_name:

            results[
                team_name
            ] = team_leaders

    return results


# ============================================================
# DEFENSIVE LEADERS
# ============================================================

def extract_interception_leader_from_players_block(
    team_block: Dict,
) -> Optional[Dict]:

    stat_groups = (
        team_block.get(
            "statistics",
            [],
        )
        or []
    )

    best_player = None

    best_ints = 0

    for stat_group in stat_groups:

        name = (
            stat_group.get(
                "name"
            )
            or ""
        ).lower()

        display_name = (
            stat_group.get(
                "displayName"
            )
            or stat_group.get(
                "shortDisplayName"
            )
            or ""
        ).lower()

        if (
            "interception"
            not in name
            and "interception"
            not in display_name
        ):

            continue

        rows = parse_stat_group(
            stat_group
        )

        for row in rows:

            interceptions = safe_int(
                row.get(
                    "INT"
                )
                or row.get(
                    "INTS"
                )
                or row.get(
                    "NO."
                )
                or 0,
                0,
            )

            if (
                interceptions
                > best_ints
            ):

                best_ints = interceptions

                best_player = {
                    "name": (
                        row.get(
                            "player"
                        )
                        or "N/A"
                    ),
                    "ints": interceptions,
                }

    if (
        best_player
        and best_player[
            "ints"
        ] > 0
    ):

        return best_player

    return None


def extract_defensive_leaders_from_players_block(
    team_block: Dict,
) -> Dict:

    team_info = (
        team_block.get(
            "team",
            {},
        )
        or {}
    )

    team_name = (
        team_info.get(
            "displayName"
        )
        or team_info.get(
            "name"
        )
    )

    leaders = {
        "team": team_name,
        "tackles_leader": None,
        "sacks_leader": None,
        "ints_leader": None,
    }

    defensive_group = None

    for stat_group in (
        team_block.get(
            "statistics",
            [],
        )
    ):

        if (
            (
                stat_group.get(
                    "name"
                )
                or ""
            ).lower()
            == "defensive"
        ):

            defensive_group = (
                stat_group
            )

            break

    if defensive_group:

        rows = parse_stat_group(
            defensive_group
        )

        max_tackles = 0

        max_sacks = 0.0

        for row in rows:

            tackles = safe_int(
                row.get(
                    "TOT"
                )
                or row.get(
                    "Total"
                )
                or row.get(
                    "TKL"
                )
                or 0,
                0,
            )

            sacks = safe_float(
                row.get(
                    "SACKS"
                )
                or row.get(
                    "SACK"
                )
                or row.get(
                    "SK"
                )
                or 0,
                0.0,
            )

            name = (
                row.get(
                    "player"
                )
                or "N/A"
            )

            if (
                tackles
                > max_tackles
                and tackles > 0
            ):

                max_tackles = tackles

                leaders[
                    "tackles_leader"
                ] = {
                    "name": name,
                    "tackles": tackles,
                }

            if (
                sacks
                > max_sacks
                and sacks > 0
            ):

                max_sacks = sacks

                leaders[
                    "sacks_leader"
                ] = {
                    "name": name,
                    "sacks": sacks,
                }

    interception_leader = (
        extract_interception_leader_from_players_block(
            team_block
        )
    )

    if interception_leader:

        leaders[
            "ints_leader"
        ] = interception_leader

    return leaders


def extract_all_defensive_leaders(
    summary: Dict,
) -> Dict[
    str,
    Dict,
]:

    boxscore = (
        summary.get(
            "boxscore",
            {},
        )
        or {}
    )

    player_blocks = (
        boxscore.get(
            "players",
            [],
        )
        or []
    )

    results: Dict[
        str,
        Dict,
    ] = {}

    for team_block in player_blocks:

        team_leaders = (
            extract_defensive_leaders_from_players_block(
                team_block
            )
        )

        team_name = (
            team_leaders[
                "team"
            ]
        )

        if team_name:

            results[
                team_name
            ] = team_leaders

    return results


# ============================================================
# GAME META
# ============================================================

def extract_game_meta(
    summary: Dict,
    meta_event_id: str,
) -> Dict:

    header = (
        summary.get(
            "header",
            {},
        )
        or {}
    )

    competitions = (
        header.get(
            "competitions",
            [],
        )
        or []
    )

    competition = (
        competitions[0]
        if competitions
        else {}
    )

    competitors = (
        competition.get(
            "competitors",
            [],
        )
        or []
    )

    status_type = (
        competition.get(
            "status"
        )
        or {}
    ).get(
        "type"
    ) or {}

    completed = bool(
        status_type.get(
            "completed",
            False,
        )
    )

    teams = []

    for competitor in competitors:

        team = (
            competitor.get(
                "team",
                {},
            )
            or {}
        )

        name = team.get(
            "displayName"
        )

        abbreviation = (
            team.get(
                "abbreviation"
            )
        )

        records = (
            competitor.get(
                "records"
            )
            or []
        )

        record = ""

        if records:

            record = (
                records[0]
                .get(
                    "summary",
                    "",
                )
            )

        score = (
            competitor.get(
                "score"
            )
        )

        home_away = (
            competitor.get(
                "homeAway",
                "home",
            )
        )

        teams.append(
            {
                "name": name,
                "abbr": abbreviation,
                "record": record,
                "score": score,
                "home_away": home_away,
                "quarter_scores": [],
            }
        )

    teams_sorted = sorted(
        teams,
        key=lambda team: (
            team[
                "home_away"
            ]
            != "away"
        ),
    )

    (
        period_labels,
        periods_by_abbreviation,
    ) = (
        get_scoring_periods_from_summary(
            meta_event_id
        )
    )

    for team in teams_sorted:

        abbreviation = (
            team.get(
                "abbr"
            )
            or ""
        )

        team[
            "quarter_scores"
        ] = (
            periods_by_abbreviation.get(
                abbreviation,
                [],
            )
        )

    return {
        "teams": teams_sorted,
        "completed": completed,
        "period_labels": period_labels,
    }


# ============================================================
# TEAM YARDAGE
# ============================================================

def extract_team_yardage(
    summary: Dict,
) -> Dict[
    str,
    Dict,
]:

    results: Dict[
        str,
        Dict,
    ] = {}

    boxscore = (
        summary.get(
            "boxscore",
            {},
        )
        or {}
    )

    team_blocks = (
        boxscore.get(
            "teams",
            [],
        )
        or []
    )

    for team_block in team_blocks:

        team = (
            team_block.get(
                "team",
                {},
            )
            or {}
        )

        name = (
            team.get(
                "displayName"
            )
            or team.get(
                "name"
            )
        )

        stats = (
            team_block.get(
                "statistics",
                [],
            )
            or []
        )

        total = 0
        rush = 0
        passing = 0

        for stat in stats:

            label = (
                stat.get(
                    "label"
                )
                or ""
            ).lower()

            value_string = (
                stat.get(
                    "displayValue"
                )
                or "0"
            )

            value = safe_int(
                value_string.split(
                    " "
                )[0],
                0,
            )

            if (
                "total yards"
                in label
            ):

                total = value

            elif (
                "rushing yards"
                in label
            ):

                rush = value

            elif (
                "passing yards"
                in label
            ):

                passing = value

        if name:

            results[
                name
            ] = {
                "total_yards": total,
                "rush_yards": rush,
                "pass_yards": passing,
            }

    return results


# ============================================================
# TEXT HELPERS
# ============================================================

def fit_text(
    draw,
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


def wrap_line(
    draw,
    text: str,
    font,
    max_width: int,
    max_lines: int = 2,
) -> List[str]:

    words = (
        str(text)
        .split()
    )

    if not words:
        return [""]

    lines = []

    current = ""

    for word in words:

        test = (
            word
            if not current
            else current
            + " "
            + word
        )

        if (
            draw.textlength(
                test,
                font=font,
            )
            <= max_width
        ):

            current = test

        else:

            if current:

                lines.append(
                    current
                )

            current = word

    if current:

        lines.append(
            current
        )

    if (
        len(lines)
        > max_lines
    ):

        lines = lines[
            :max_lines
        ]

        lines[-1] = fit_text(
            draw,
            lines[-1],
            font,
            max_width,
        )

    return lines


def draw_center(
    draw,
    box,
    text: str,
    font,
    fill,
):

    x1, y1, x2, y2 = (
        box
    )

    text = str(
        text
    )

    text_width = (
        draw.textlength(
            text,
            font=font,
        )
    )

    bounding_box = (
        draw.textbbox(
            (
                0,
                0,
            ),
            text,
            font=font,
        )
    )

    text_height = (
        bounding_box[3]
        - bounding_box[1]
    )

    draw.text(
        (
            x1
            + (
                x2
                - x1
                - text_width
            )
            / 2,
            y1
            + (
                y2
                - y1
                - text_height
            )
            / 2
            - 2,
        ),
        text,
        font=font,
        fill=fill,
    )


# ============================================================
# BACKGROUND
# ============================================================

def make_background() -> Image.Image:

    image = Image.new(
        "RGB",
        (
            W,
            H,
        ),
        BG,
    )

    draw = ImageDraw.Draw(
        image
    )

    draw.rectangle(
        (
            0,
            0,
            W,
            190,
        ),
        fill=HEADER,
    )

    draw.rectangle(
        (
            0,
            190,
            W,
            199,
        ),
        fill=BLUE,
    )

    for y in range(
        210,
        H,
        30,
    ):

        color = (
            (
                14,
                18,
                28,
            )
            if (
                y // 30
            )
            % 2
            == 0
            else (
                12,
                16,
                26,
            )
        )

        draw.rectangle(
            (
                0,
                y,
                W,
                y + 15,
            ),
            fill=color,
        )

    return image


# ============================================================
# LEADER TEXT
# ============================================================

def leader_offense_lines(
    off_dict: Dict,
) -> List[str]:

    passing = (
        off_dict.get(
            "passing_leader"
        )
        or {}
    )

    rushing = (
        off_dict.get(
            "rushing_leader"
        )
        or {}
    )

    receiving = (
        off_dict.get(
            "receiving_leader"
        )
        or {}
    )

    lines = []

    if passing:

        lines.append(
            "PASS: "
            f"{passing.get('name', 'N/A')} • "
            f"{passing.get('yards', 0)} YDS, "
            f"{passing.get('td', 0)} TD, "
            f"{passing.get('ints', 0)} INT"
        )

    else:

        lines.append(
            "PASS: N/A"
        )

    if rushing:

        lines.append(
            "RUSH: "
            f"{rushing.get('name', 'N/A')} • "
            f"{rushing.get('yards', 0)} YDS, "
            f"{rushing.get('td', 0)} TD"
        )

    else:

        lines.append(
            "RUSH: N/A"
        )

    if receiving:

        lines.append(
            "REC: "
            f"{receiving.get('name', 'N/A')} • "
            f"{receiving.get('yards', 0)} YDS, "
            f"{receiving.get('td', 0)} TD"
        )

    else:

        lines.append(
            "REC: N/A"
        )

    return lines


def leader_defense_lines(
    def_dict: Dict,
) -> List[str]:

    lines = []

    tackles = (
        def_dict.get(
            "tackles_leader"
        )
    )

    sacks = (
        def_dict.get(
            "sacks_leader"
        )
    )

    interceptions = (
        def_dict.get(
            "ints_leader"
        )
    )

    if tackles:

        lines.append(
            f"TACKLES: "
            f"{tackles['name']} • "
            f"{tackles['tackles']}"
        )

    else:

        lines.append(
            "TACKLES: N/A"
        )

    if sacks:

        lines.append(
            f"SACKS: "
            f"{sacks['name']} • "
            f"{sacks['sacks']}"
        )

    else:

        lines.append(
            "SACKS: N/A"
        )

    if interceptions:

        lines.append(
            f"INT: "
            f"{interceptions['name']} • "
            f"{interceptions['ints']}"
        )

    else:

        lines.append(
            "INT: None"
        )

    return lines


# ============================================================
# LEADER CARDS
# ============================================================

def draw_leader_column(
    draw,
    x1,
    y1,
    x2,
    title,
    lines,
):

    section_font = load_font(
        23,
        True,
    )

    line_font = load_font(
        21,
        True,
    )

    draw.text(
        (
            x1,
            y1,
        ),
        title,
        font=section_font,
        fill=WHITE,
    )

    y = (
        y1 + 42
    )

    max_width = (
        x2 - x1
    )

    for line in lines:

        wrapped = wrap_line(
            draw,
            line,
            line_font,
            max_width,
            max_lines=2,
        )

        for wrapped_line in wrapped:

            draw.text(
                (
                    x1,
                    y,
                ),
                wrapped_line,
                font=line_font,
                fill=WHITE,
            )

            y += 26

        y += 8


def draw_team_leader_card(
    draw,
    x1,
    y1,
    x2,
    y2,
    team_name,
    off_dict,
    def_dict,
):

    team_font = load_font(
        33,
        True,
    )

    draw.rounded_rectangle(
        (
            x1,
            y1,
            x2,
            y2,
        ),
        radius=26,
        fill=CARD,
        outline=BORDER,
        width=3,
    )

    team_text = fit_text(
        draw,
        str(
            team_name
        ).upper(),
        team_font,
        x2 - x1 - 36,
    )

    draw.text(
        (
            x1 + 18,
            y1 + 18,
        ),
        team_text,
        font=team_font,
        fill=BLUE,
    )

    middle = (
        x1
        + (
            x2
            - x1
        )
        // 2
    )

    top = (
        y1 + 84
    )

    bottom = (
        y2 - 24
    )

    draw.line(
        (
            middle,
            top,
            middle,
            bottom,
        ),
        fill=BORDER,
        width=2,
    )

    draw_leader_column(
        draw,
        x1 + 24,
        top,
        middle - 24,
        "OFFENSE",
        leader_offense_lines(
            off_dict
        ),
    )

    draw_leader_column(
        draw,
        middle + 24,
        top,
        x2 - 24,
        "DEFENSE",
        leader_defense_lines(
            def_dict
        ),
    )


def stat_line(
    draw,
    x,
    y,
    label,
    value,
    label_font,
    value_font,
):

    draw.text(
        (
            x,
            y,
        ),
        label,
        font=label_font,
        fill=MUTED,
    )

    draw.text(
        (
            x,
            y + 30,
        ),
        value,
        font=value_font,
        fill=WHITE,
    )


# ============================================================
# POSTER
# ============================================================

def make_poster_style_image(
    meta: Dict,
    offensive_leaders: Dict[
        str,
        Dict,
    ],
    defensive_leaders: Dict[
        str,
        Dict,
    ],
    yardage: Dict[
        str,
        Dict,
    ],
    output_path: str,
    style: Dict = None,
) -> None:

    teams = meta[
        "teams"
    ]

    if len(
        teams
    ) < 2:

        print(
            "Not a standard "
            "two-team game, skipping."
        )

        return

    away, home = (
        teams[0],
        teams[1],
    )

    completed = meta.get(
        "completed",
        False,
    )

    away_name = (
        away[
            "name"
        ]
    )

    home_name = (
        home[
            "name"
        ]
    )

    away_off = (
        offensive_leaders.get(
            away_name,
            {},
        )
    )

    home_off = (
        offensive_leaders.get(
            home_name,
            {},
        )
    )

    away_def = (
        defensive_leaders.get(
            away_name,
            {},
        )
    )

    home_def = (
        defensive_leaders.get(
            home_name,
            {},
        )
    )

    away_yards = (
        yardage.get(
            away_name,
            {
                "total_yards": 0,
                "rush_yards": 0,
                "pass_yards": 0,
            },
        )
    )

    home_yards = (
        yardage.get(
            home_name,
            {
                "total_yards": 0,
                "rush_yards": 0,
                "pass_yards": 0,
            },
        )
    )

    image = (
        make_background()
    )

    draw = ImageDraw.Draw(
        image
    )

    title_font = load_font(
        52,
        True,
    )

    small_font = load_font(
        26,
        False,
    )

    record_font = load_font(
        28,
        True,
    )

    score_font = load_font(
        90,
        True,
    )

    at_font = load_font(
        24,
        True,
    )

    section_font = load_font(
        31,
        True,
    )

    quarter_font = load_font(
        31,
        True,
    )

    quarter_header_font = (
        load_font(
            25,
            True,
        )
    )

    stat_label_font = (
        load_font(
            22,
            True,
        )
    )

    stat_value_font = (
        load_font(
            38,
            True,
        )
    )

    # Visible league name removed.
    title = "GAME RECAP"

    subtitle = (
        "FINAL SCORE • GAME RECAP"
    )

    draw.text(
        (
            (
                W
                - draw.textlength(
                    title,
                    font=title_font,
                )
            )
            / 2,
            28,
        ),
        title,
        font=title_font,
        fill=WHITE,
    )

    draw.text(
        (
            (
                W
                - draw.textlength(
                    subtitle,
                    font=small_font,
                )
            )
            / 2,
            105,
        ),
        subtitle,
        font=small_font,
        fill=(
            208,
            218,
            238,
        ),
    )

    # --------------------------------------------------------
    # SCORE CARD
    # --------------------------------------------------------

    x0 = 42
    y0 = 230
    x1 = W - 42
    y1 = 555

    draw.rounded_rectangle(
        (
            x0,
            y0,
            x1,
            y1,
        ),
        radius=32,
        fill=CARD,
        outline=BORDER,
        width=3,
    )

    # --------------------------------------------------------
    # CUSTOM ORIGINAL PIXEL ICONS
    #
    # Each includes its abbreviation.
    # --------------------------------------------------------

    away_icon = create_team_icon(
        away.get(
            "abbr"
        )
        or "",
        170,
    )

    home_icon = create_team_icon(
        home.get(
            "abbr"
        )
        or "",
        170,
    )

    away_x = (
        x0 + 23
    )

    home_x = (
        x1
        - home_icon.width
        - 23
    )

    icon_y = (
        y0 + 23
    )

    image.paste(
        away_icon,
        (
            away_x,
            icon_y,
        ),
        away_icon,
    )

    image.paste(
        home_icon,
        (
            home_x,
            icon_y,
        ),
        home_icon,
    )

    # The abbreviation is already inside each icon,
    # so the old separate team abbreviation text is removed.
    # Records remain in their original location.

    draw_center(
        draw,
        (
            x0 + 28,
            y0 + 248,
            x0 + 185,
            y0 + 294,
        ),
        away.get(
            "record"
        )
        or "",
        record_font,
        MUTED,
    )

    draw_center(
        draw,
        (
            x1 - 185,
            y0 + 248,
            x1 - 28,
            y0 + 294,
        ),
        home.get(
            "record"
        )
        or "",
        record_font,
        MUTED,
    )

    left_score = (
        away.get(
            "score"
        )
        if (
            completed
            and away.get(
                "score"
            )
            is not None
        )
        else "–"
    )

    right_score = (
        home.get(
            "score"
        )
        if (
            completed
            and home.get(
                "score"
            )
            is not None
        )
        else "–"
    )

    draw.text(
        (
            368,
            y0 + 92,
        ),
        str(
            left_score
        ),
        font=score_font,
        fill=WHITE,
    )

    draw_center(
        draw,
        (
            498,
            y0 + 127,
            582,
            y0 + 175,
        ),
        "AT",
        at_font,
        MUTED,
    )

    draw.text(
        (
            600,
            y0 + 92,
        ),
        str(
            right_score
        ),
        font=score_font,
        fill=WHITE,
    )

    # --------------------------------------------------------
    # SCORING BY QUARTER
    # --------------------------------------------------------

    quarter_x0 = 42
    quarter_y0 = 590
    quarter_x1 = W - 42
    quarter_y1 = 850

    draw.rounded_rectangle(
        (
            quarter_x0,
            quarter_y0,
            quarter_x1,
            quarter_y1,
        ),
        radius=28,
        fill=CARD,
        outline=BORDER,
        width=3,
    )

    draw_center(
        draw,
        (
            quarter_x0,
            quarter_y0 + 18,
            quarter_x1,
            quarter_y0 + 70,
        ),
        "SCORING BY QUARTER",
        section_font,
        BLUE,
    )

    period_labels = (
        meta.get(
            "period_labels",
            [
                "1Q",
                "2Q",
                "3Q",
                "4Q",
            ],
        )
    )

    labels = (
        ["TEAM"]
        + period_labels
    )

    start_x = 145
    end_x = 890

    step = (
        (
            end_x
            - start_x
        )
        / max(
            1,
            len(labels)
            - 1,
        )
    )

    column_x = [
        int(
            start_x
            + index
            * step
        )
        for index
        in range(
            len(labels)
        )
    ]

    for (
        label,
        x,
    ) in zip(
        labels,
        column_x,
    ):

        draw_center(
            draw,
            (
                x - 55,
                quarter_y0 + 92,
                x + 55,
                quarter_y0 + 130,
            ),
            label,
            quarter_header_font,
            WHITE,
        )

    rows = [
        (
            away.get(
                "abbr"
            )
            or "",
            away.get(
                "quarter_scores"
            )
            or [],
        ),
        (
            home.get(
                "abbr"
            )
            or "",
            home.get(
                "quarter_scores"
            )
            or [],
        ),
    ]

    for (
        row_index,
        (
            abbreviation,
            scores,
        ),
    ) in enumerate(
        rows
    ):

        y = (
            quarter_y0
            + 148
            + row_index
            * 62
        )

        scores = (
            scores
            + [0]
            * len(
                period_labels
            )
        )[
            :len(
                period_labels
            )
        ]

        draw_center(
            draw,
            (
                column_x[0]
                - 55,
                y,
                column_x[0]
                + 55,
                y + 42,
            ),
            abbreviation,
            quarter_font,
            BLUE,
        )

        for (
            score,
            x,
        ) in zip(
            scores,
            column_x[1:],
        ):

            draw_center(
                draw,
                (
                    x - 55,
                    y,
                    x + 55,
                    y + 42,
                ),
                str(
                    score
                ),
                quarter_font,
                WHITE,
            )

    # --------------------------------------------------------
    # TEAM YARDAGE
    # --------------------------------------------------------

    stats_x0 = 42
    stats_y0 = 890
    stats_x1 = W - 42
    stats_y1 = 1090

    draw.rounded_rectangle(
        (
            stats_x0,
            stats_y0,
            stats_x1,
            stats_y1,
        ),
        radius=28,
        fill=CARD,
        outline=BORDER,
        width=3,
    )

    draw_center(
        draw,
        (
            stats_x0,
            stats_y0 + 20,
            stats_x1,
            stats_y0 + 70,
        ),
        "TEAM YARDAGE",
        section_font,
        BLUE,
    )

    away_total = safe_int(
        away_yards.get(
            "total_yards"
        ),
        0,
    )

    home_total = safe_int(
        home_yards.get(
            "total_yards"
        ),
        0,
    )

    difference = (
        home_total
        - away_total
    )

    difference_text = (
        str(
            abs(
                difference
            )
        )
    )

    stat_line(
        draw,
        stats_x0 + 80,
        stats_y0 + 95,
        f"{away.get('abbr')} TOTAL",
        f"{away_total}",
        stat_label_font,
        stat_value_font,
    )

    stat_line(
        draw,
        stats_x0 + 380,
        stats_y0 + 95,
        f"{home.get('abbr')} TOTAL",
        f"{home_total}",
        stat_label_font,
        stat_value_font,
    )

    stat_line(
        draw,
        stats_x0 + 670,
        stats_y0 + 95,
        "DIFFERENCE",
        difference_text,
        stat_label_font,
        stat_value_font,
    )

    # --------------------------------------------------------
    # TEAM LEADERS
    # --------------------------------------------------------

    draw_center(
        draw,
        (
            42,
            1120,
            W - 42,
            1170,
        ),
        "TEAM LEADERS",
        section_font,
        WHITE,
    )

    draw_team_leader_card(
        draw,
        42,
        1190,
        W - 42,
        1518,
        away_name,
        away_off,
        away_def,
    )

    draw_team_leader_card(
        draw,
        42,
        1545,
        W - 42,
        1878,
        home_name,
        home_off,
        home_def,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    image.save(
        output_path,
        "PNG",
    )


# ============================================================
# SINGLE GAME
# ============================================================

def generate_poster_for_game(
    game_id: str,
    out_dir: str,
) -> Tuple[
    bool,
    str,
]:

    try:

        summary = fetch_summary(
            game_id
        )

        meta = extract_game_meta(
            summary,
            game_id,
        )

        offensive_leaders = (
            extract_all_team_leaders(
                summary
            )
        )

        defensive_leaders = (
            extract_all_defensive_leaders(
                summary
            )
        )

        yardage = (
            extract_team_yardage(
                summary
            )
        )

        out_path = (
            os.path.join(
                out_dir,
                f"game_{game_id}_poster.png",
            )
        )

        make_poster_style_image(
            meta,
            offensive_leaders,
            defensive_leaders,
            yardage,
            out_path,
        )

        return (
            True,
            out_path,
        )

    except Exception as error:

        return (
            False,
            f"{game_id}: {error}",
        )


# ============================================================
# WEEK GENERATION
# ============================================================

def no_poster_message(
    year: int,
    week: int,
) -> str:

    return (
        f"No poster available yet "
        f"for {year} week {week}. "
        "Please try another week "
        "or season type"
    )


def generate_week(
    year: int,
    week: int,
    seasontype: int = 2,
    limit: int = 0,
) -> str:

    url = scoreboard_url(
        year,
        week,
        seasontype,
    )

    html = fetch_url(
        url
    )

    game_ids = (
        extract_game_ids_from_scoreboard_html(
            html
        )
    )

    if (
        limit
        and limit > 0
    ):

        game_ids = (
            game_ids[
                :limit
            ]
        )

    if not game_ids:

        raise RuntimeError(
            no_poster_message(
                year,
                week,
            )
        )

    summaries: Dict[
        str,
        Dict,
    ] = {}

    for game_id in game_ids:

        summary = (
            fetch_summary(
                game_id
            )
        )

        summaries[
            game_id
        ] = summary

        meta = (
            extract_game_meta(
                summary,
                game_id,
            )
        )

        if not meta.get(
            "completed",
            False,
        ):

            raise RuntimeError(
                no_poster_message(
                    year,
                    week,
                )
            )

    kind = (
        "regular"
        if seasontype == 2
        else "playoffs"
    )

    out_dir = (
        os.path.join(
            "game_visuals",
            str(
                year
            ),
            kind,
            (
                f"week"
                f"{str(week).zfill(2)}"
            ),
        )
    )

    shutil.rmtree(
        out_dir,
        ignore_errors=True,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    for game_id in game_ids:

        summary = (
            summaries[
                game_id
            ]
        )

        offensive_leaders = (
            extract_all_team_leaders(
                summary
            )
        )

        defensive_leaders = (
            extract_all_defensive_leaders(
                summary
            )
        )

        yardage = (
            extract_team_yardage(
                summary
            )
        )

        meta = (
            extract_game_meta(
                summary,
                game_id,
            )
        )

        out_path = (
            os.path.join(
                out_dir,
                f"game_{game_id}_poster.png",
            )
        )

        make_poster_style_image(
            meta,
            offensive_leaders,
            defensive_leaders,
            yardage,
            out_path,
        )

    return out_dir


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate game posters "
            "for any ESPN week."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Season year, "
            "e.g. 2025"
        ),
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help=(
            "Week number, "
            "e.g. 13"
        ),
    )

    parser.add_argument(
        "--seasontype",
        type=int,
        default=2,
        help=(
            "1=preseason, "
            "2=regular, "
            "3=postseason"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Limit number "
            "of games, 0=all"
        ),
    )

    args = (
        parser.parse_args()
    )

    try:

        out_dir = generate_week(
            year=args.year,
            week=args.week,
            seasontype=args.seasontype,
            limit=args.limit,
        )

        print(
            f"Posters generated in: "
            f"{out_dir}"
        )

    except Exception as error:

        print(
            str(
                error
            )
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
