#!/usr/bin/env python3

import argparse
import gzip
import io
import json
import os
import random
import textwrap
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from app.services.storage_supabase import upload_file_return_url


W = 1080
H = 1350
MARGIN = 54

BG = (8, 12, 20)
TEXT = (245, 247, 250)
MUTED = (170, 178, 190)
LINE = (38, 48, 66)
CARD = (14, 20, 32)
CARD2 = (18, 26, 42)
GREEN = (66, 200, 120)
YELLOW = (255, 196, 66)
RED = (255, 100, 100)
BLUE = (88, 158, 255)

CATEGORY_ORDER = [
    "success_rate_by_down_and_distance",
    "qb_masterclass",
    "play_that_won_the_game",
    "better_than_expected",
    "clutch_gene",
]

CATEGORY_LABELS = {
    "success_rate_by_down_and_distance": "Success Rate by Down and Distance",
    "qb_masterclass": "QB Masterclass",
    "play_that_won_the_game": "Play That Won the Game",
    "better_than_expected": "Better Than Expected",
    "clutch_gene": "Clutch Gene",
}

TEAM_COLORS = {
    "ARI": (151, 35, 63),
    "ATL": (167, 25, 48),
    "BAL": (79, 50, 138),
    "BUF": (0, 51, 141),
    "CAR": (0, 133, 202),
    "CHI": (200, 78, 0),
    "CIN": (251, 79, 20),
    "CLE": (255, 60, 0),
    "DAL": (4, 30, 66),
    "DEN": (251, 79, 20),
    "DET": (0, 118, 182),
    "GB": (32, 55, 49),
    "HOU": (3, 32, 47),
    "IND": (0, 44, 95),
    "JAX": (0, 103, 120),
    "KC": (227, 24, 55),
    "LV": (35, 35, 35),
    "LAC": (0, 128, 198),
    "LAR": (0, 53, 148),
    "MIA": (0, 142, 151),
    "MIN": (79, 38, 131),
    "NE": (0, 34, 68),
    "NO": (211, 188, 141),
    "NYG": (11, 34, 101),
    "NYJ": (18, 87, 64),
    "PHI": (0, 76, 84),
    "PIT": (255, 182, 18),
    "SEA": (0, 34, 68),
    "SF": (170, 0, 0),
    "TB": (213, 10, 10),
    "TEN": (12, 35, 64),
    "WAS": (90, 20, 20),
    "WSH": (90, 20, 20),
}

TEAM_SECONDARY_COLORS = {
    "ARI": (218, 143, 55),
    "ATL": (20, 20, 20),
    "BAL": (213, 168, 36),
    "BUF": (198, 12, 48),
    "CAR": (16, 24, 32),
    "CHI": (92, 50, 22),
    "CIN": (20, 20, 20),
    "CLE": (91, 44, 18),
    "DAL": (183, 193, 204),
    "DEN": (0, 34, 68),
    "DET": (176, 183, 188),
    "GB": (255, 182, 18),
    "HOU": (167, 25, 48),
    "IND": (225, 225, 225),
    "JAX": (215, 162, 42),
    "KC": (255, 184, 28),
    "LV": (180, 180, 180),
    "LAC": (255, 194, 14),
    "LAR": (255, 163, 0),
    "MIA": (252, 76, 2),
    "MIN": (255, 198, 47),
    "NE": (198, 12, 48),
    "NO": (25, 25, 25),
    "NYG": (167, 25, 48),
    "NYJ": (225, 225, 225),
    "PHI": (165, 172, 175),
    "PIT": (203, 39, 58),
    "SEA": (105, 190, 40),
    "SF": (185, 150, 90),
    "TB": (255, 121, 0),
    "TEN": (75, 146, 219),
    "WAS": (255, 182, 18),
    "WSH": (255, 182, 18),
}

VALID_PLAY_TYPES = {
    "PASS",
    "RUSH",
    "SACK",
    "FIELD_GOAL",
    "PENALTY",
}

ROTATION_ANCHOR_DATE = date(2026, 1, 1)
MIN_SEASON = 2018

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def find_font_bold() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return ""


def find_font_regular() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return ""


FONT_BOLD_PATH = find_font_bold()
FONT_REG_PATH = find_font_regular()


def font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_REG_PATH

    if path:
        return ImageFont.truetype(
            path,
            size=size,
        )

    return ImageFont.load_default()


def ensure_dir(path: str) -> None:
    os.makedirs(
        path,
        exist_ok=True,
    )


def rounded_rect(
    draw,
    box,
    radius,
    fill,
    outline=None,
    width=1,
):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def wrap_text(
    text: str,
    width: int,
) -> str:
    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
        )
    )


def safe_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {
            1: "st",
            2: "nd",
            3: "rd",
        }.get(
            n % 10,
            "th",
        )

    return f"{n}{suffix}"


def normalize_team(team: str) -> str:
    if not team:
        return ""

    team = str(team).upper()

    if team == "WAS":
        return "WSH"

    return team


def clean_desc(
    text: str,
    max_len: int = None,
) -> str:
    text = (
        safe_text(text)
        .replace("\n", " ")
        .strip()
    )

    while "  " in text:
        text = text.replace(
            "  ",
            " ",
        )

    if (
        max_len is not None
        and len(text) > max_len
    ):
        text = (
            text[:max_len - 1]
            .rstrip()
            + "…"
        )

    return text


def yardline_bin(
    series: pd.Series,
    step: int = 5,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (
        np.floor(values / step)
        * step
        + step / 2
    ).astype("float")


def distance_bucket(
    series: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    output = pd.Series(
        index=series.index,
        dtype="object",
    )

    output[
        (values >= 1)
        & (values <= 3)
    ] = "1-3 YDS"

    output[
        (values >= 4)
        & (values <= 6)
    ] = "4-6 YDS"

    output[
        (values >= 7)
        & (values <= 10)
    ] = "7-10 YDS"

    output[
        values >= 11
    ] = "11+ YDS"

    return output


def format_down_distance(
    down,
    ydstogo,
) -> str:
    try:
        d = int(
            float(down)
        )

        y = int(
            round(
                float(ydstogo)
            )
        )

        return (
            f"{ordinal(d)} "
            f"Down & {y}"
        )

    except Exception:
        return "High-leverage snap"


def weekly_context(
    season: int,
    week: int,
) -> str:
    return (
        f"{season} • "
        f"Week {week} • "
        "Regular Season"
    )


def week_limit_for_year(
    season: int,
) -> int:
    return (
        18
        if season >= 2021
        else 17
    )


def validate_week(
    season: int,
    week: int,
) -> None:
    max_week = week_limit_for_year(
        season
    )

    if (
        week < 1
        or week > max_week
    ):
        raise ValueError(
            f"Invalid week {week} "
            f"for season {season}. "
            f"Use 1-{max_week}."
        )


def fit_multiline_text(
    draw,
    text: str,
    max_width: int,
    max_height: int,
    start_size: int,
    min_size: int = 16,
    bold: bool = False,
    line_spacing: int = 6,
):
    text = safe_text(
        text
    ).strip()

    if not text:
        return (
            "",
            font(
                start_size,
                bold=bold,
            ),
            line_spacing,
        )

    for size in range(
        start_size,
        min_size - 1,
        -1,
    ):
        current_font = font(
            size,
            bold=bold,
        )

        approx_chars = max(
            18,
            int(
                max_width
                / max(
                    size * 0.55,
                    1,
                )
            ),
        )

        wrapped = wrap_text(
            text,
            approx_chars,
        )

        bbox = draw.multiline_textbbox(
            (0, 0),
            wrapped,
            font=current_font,
            spacing=line_spacing,
        )

        text_width = (
            bbox[2]
            - bbox[0]
        )

        text_height = (
            bbox[3]
            - bbox[1]
        )

        if (
            text_width <= max_width
            and text_height <= max_height
        ):
            return (
                wrapped,
                current_font,
                line_spacing,
            )

    current_font = font(
        min_size,
        bold=bold,
    )

    approx_chars = max(
        18,
        int(
            max_width
            / max(
                min_size * 0.55,
                1,
            )
        ),
    )

    wrapped = wrap_text(
        text,
        approx_chars,
    )

    return (
        wrapped,
        current_font,
        line_spacing,
    )


def now_eastern_date() -> date:
    return datetime.now(
        ZoneInfo(
            "America/New_York"
        )
    ).date()


def rotation_index_for_day(
    day: date,
) -> int:
    return (
        (
            day
            - ROTATION_ANCHOR_DATE
        ).days
        % len(CATEGORY_ORDER)
    )


def category_for_day(
    day: date,
) -> str:
    return CATEGORY_ORDER[
        rotation_index_for_day(
            day
        )
    ]


def public_storage_url(
    storage_key: str,
) -> str:
    base = (
        os.environ[
            "SUPABASE_URL"
        ].rstrip("/")
    )

    bucket = os.environ.get(
        "SUPABASE_BUCKET",
        "nfl-posters",
    )

    return (
        f"{base}/"
        "storage/v1/object/public/"
        f"{bucket}/"
        f"{storage_key}"
    )


# ============================================================
# CUSTOM TEAM ICONS
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

    for x, y, color in data:
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


def draw_team_symbol(
    draw,
    team,
    cx,
    cy,
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
            primary,
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


def create_team_icon(
    team: str,
    size: int = 155,
):
    if not team:
        return None

    team = normalize_team(
        team
    )

    if team not in TEAM_COLORS:
        return None

    primary = TEAM_COLORS[
        team
    ]

    secondary = TEAM_SECONDARY_COLORS.get(
        team,
        (
            220,
            220,
            220,
        ),
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
        team,
        base_width // 2,
        50,
        primary,
        secondary,
    )

    abbreviation_font = font(
        20,
        bold=True,
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
            230,
        ),
    )

    text_fill = primary

    if sum(primary) < 140:
        text_fill = (
            235,
            235,
            235,
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
        team,
        font=abbreviation_font,
        fill=text_fill,
    )

    scale = (
        size
        / base_height
    )

    output_width = max(
        1,
        int(
            base_width
            * scale
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
# DATA LOADING
# ============================================================

def load_csv_gz_url(
    url: str,
) -> pd.DataFrame:
    response = requests.get(
        url,
        timeout=180,
        headers={
            "User-Agent": USER_AGENT
        },
    )

    response.raise_for_status()

    bio = io.BytesIO(
        response.content
    )

    with gzip.GzipFile(
        fileobj=bio
    ) as gz:
        return pd.read_csv(
            gz,
            low_memory=False,
        )


def load_pbp_one_season(
    season: int,
) -> pd.DataFrame:
    urls = [
        (
            "https://github.com/"
            "nflverse/nflverse-data/"
            "releases/download/pbp/"
            f"play_by_play_{season}.csv.gz"
        ),
        (
            "https://raw.githubusercontent.com/"
            "guga31bb/nflfastR-data/"
            "master/data/"
            f"play_by_play_{season}.csv.gz"
        ),
    ]

    last_error = None

    for url in urls:
        try:
            print(
                f"trying {url}"
            )

            return load_csv_gz_url(
                url
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not load "
        f"play-by-play for "
        f"{season}. "
        f"Last error: "
        f"{last_error}"
    )


def prep_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "week",
        "qtr",
        "down",
        "ydstogo",
        "yardline_100",
        "ep",
        "wp",
        "wpa",
        "epa",
        "cpoe",
        "air_yards",
        "yards_gained",
        "score_differential",
        "complete_pass",
        "incomplete_pass",
        "interception",
        "pass_touchdown",
        "rushing_yards",
        "passing_yards",
        "receiving_yards",
        "rush_touchdown",
        "receiving_touchdown",
        "game_seconds_remaining",
        "pass",
        "rush",
    ]

    for column in numeric_cols:
        if column in df.columns:
            df[
                column
            ] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "play_type_nfl" in df.columns:
        df = df[
            df[
                "play_type_nfl"
            ]
            .astype(str)
            .isin(
                VALID_PLAY_TYPES
            )
        ].copy()

    df = df[
        df["ep"].notna()
    ].copy()

    df = df[
        df[
            "yardline_100"
        ].between(
            1,
            99,
            inclusive="both",
        )
    ].copy()

    if "success" not in df.columns:
        df[
            "success"
        ] = (
            pd.to_numeric(
                df["epa"],
                errors="coerce",
            )
            > 0
        ).astype(float)

    else:
        df[
            "success"
        ] = pd.to_numeric(
            df["success"],
            errors="coerce",
        )

    for column in [
        "posteam",
        "defteam",
        "home_team",
        "away_team",
    ]:
        if column in df.columns:
            df[
                column
            ] = df[
                column
            ].map(
                normalize_team
            )

    if (
        "desc"
        not in df.columns
        and "play_description"
        in df.columns
    ):
        df[
            "desc"
        ] = df[
            "play_description"
        ]

    if (
        "score_differential"
        in df.columns
    ):
        df[
            "score_margin"
        ] = pd.to_numeric(
            df[
                "score_differential"
            ],
            errors="coerce",
        )

    elif (
        "posteam_score"
        in df.columns
        and "defteam_score"
        in df.columns
    ):
        df[
            "score_margin"
        ] = (
            pd.to_numeric(
                df[
                    "posteam_score"
                ],
                errors="coerce",
            )
            - pd.to_numeric(
                df[
                    "defteam_score"
                ],
                errors="coerce",
            )
        )

    else:
        df[
            "score_margin"
        ] = np.nan

    if (
        "game_seconds_remaining"
        in df.columns
    ):
        df[
            "minutes_remaining"
        ] = (
            pd.to_numeric(
                df[
                    "game_seconds_remaining"
                ],
                errors="coerce",
            )
            / 60.0
        )

    else:
        df[
            "minutes_remaining"
        ] = np.nan

    return df


def filter_regular_week(
    df: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    output = df.copy()

    if "game_type" in output.columns:
        output = output[
            output[
                "game_type"
            ].astype(str)
            == "REG"
        ].copy()

    output = output[
        output[
            "week"
        ]
        == week
    ].copy()

    return output


# ============================================================
# STAT LINES
# ============================================================

def qb_week_statline(
    df_week: pd.DataFrame,
    player: str,
) -> str:
    data = df_week[
        df_week[
            "passer_player_name"
        ].astype(str)
        == str(player)
    ].copy()

    if data.empty:
        return "Stat line unavailable"

    completions = int(
        data.get(
            "complete_pass",
            pd.Series(
                dtype=float
            ),
        )
        .fillna(0)
        .sum()
    )

    incompletions = int(
        data.get(
            "incomplete_pass",
            pd.Series(
                dtype=float
            ),
        )
        .fillna(0)
        .sum()
    )

    interceptions = int(
        data.get(
            "interception",
            pd.Series(
                dtype=float
            ),
        )
        .fillna(0)
        .sum()
    )

    attempts = (
        completions
        + incompletions
        + interceptions
    )

    pass_yards = int(
        data.get(
            "passing_yards",
            pd.Series(
                dtype=float
            ),
        )
        .fillna(0)
        .sum()
    )

    pass_tds = int(
        data.get(
            "pass_touchdown",
            pd.Series(
                dtype=float
            ),
        )
        .fillna(0)
        .sum()
    )

    rushes = data[
        data.get(
            "rush",
            pd.Series(
                dtype=float
            ),
        ).fillna(0)
        == 1
    ]

    rush_yards = (
        int(
            rushes.get(
                "rushing_yards",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not rushes.empty
        else 0
    )

    rush_tds = (
        int(
            rushes.get(
                "rush_touchdown",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not rushes.empty
        else 0
    )

    return (
        f"{completions}/{attempts}, "
        f"{pass_yards} Pass Yds, "
        f"{pass_tds} Pass TD, "
        f"{interceptions} INT, "
        f"{rush_yards} Rush Yds, "
        f"{rush_tds} Rush TD"
    )


def skill_week_statline(
    df_week: pd.DataFrame,
    player: str,
) -> str:
    recv = df_week[
        df_week.get(
            "receiver_player_name",
            pd.Series(
                dtype=object
            ),
        ).astype(str)
        == str(player)
    ].copy()

    rush = df_week[
        df_week.get(
            "rusher_player_name",
            pd.Series(
                dtype=object
            ),
        ).astype(str)
        == str(player)
    ].copy()

    receptions = (
        int(
            recv.get(
                "complete_pass",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not recv.empty
        else 0
    )

    rec_yards = (
        int(
            recv.get(
                "receiving_yards",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not recv.empty
        else 0
    )

    rec_tds = (
        int(
            recv.get(
                "receiving_touchdown",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not recv.empty
        else 0
    )

    carries = (
        len(rush)
        if not rush.empty
        else 0
    )

    rush_yards = (
        int(
            rush.get(
                "rushing_yards",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not rush.empty
        else 0
    )

    rush_tds = (
        int(
            rush.get(
                "rush_touchdown",
                pd.Series(
                    dtype=float
                ),
            )
            .fillna(0)
            .sum()
        )
        if not rush.empty
        else 0
    )

    if (
        receptions > 0
        and carries > 0
    ):
        return (
            f"{receptions} Rec, "
            f"{rec_yards} Rec Yds, "
            f"{rec_tds} Rec TD • "
            f"{carries} Car, "
            f"{rush_yards} Rush Yds, "
            f"{rush_tds} Rush TD"
        )

    if receptions > 0:
        return (
            f"{receptions} Rec, "
            f"{rec_yards} Rec Yds, "
            f"{rec_tds} Rec TD"
        )

    return (
        f"{carries} Car, "
        f"{rush_yards} Rush Yds, "
        f"{rush_tds} Rush TD"
    )


def actor_statline(
    df_week: pd.DataFrame,
    player: str,
) -> str:
    if player in set(
        df_week.get(
            "passer_player_name",
            pd.Series(
                dtype=object
            ),
        )
        .dropna()
        .astype(str)
    ):
        return qb_week_statline(
            df_week,
            player,
        )

    return skill_week_statline(
        df_week,
        player,
    )


# ============================================================
# CHART
# ============================================================

def apply_chart_style(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "#f5f5f5"
    )

    ax.grid(
        True,
        alpha=0.28,
        linewidth=0.8,
    )

    for spine in ax.spines.values():
        spine.set_alpha(
            0.45
        )

    ax.set_title(
        title,
        fontsize=28,
        fontweight="bold",
        loc="left",
        pad=14,
    )

    fig.text(
        0.125,
        0.02,
        subtitle,
        fontsize=14,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=16,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=16,
        fontweight="bold",
        labelpad=10,
    )

    ax.tick_params(
        labelsize=12
    )


def save_chart(
    fig: plt.Figure,
    path: str,
) -> None:
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"saved: {path}"
    )


def plot_success_rate_by_down_and_distance(
    df_week: pd.DataFrame,
    season: int,
    week: int,
    out_path: str,
) -> None:
    data = df_week.copy()

    data[
        "dist_bucket"
    ] = distance_bucket(
        data[
            "ydstogo"
        ]
    )

    data = data[
        data[
            "dist_bucket"
        ].notna()
    ].copy()

    data = data[
        data[
            "success"
        ].notna()
    ].copy()

    data = data[
        data[
            "down"
        ].between(
            1,
            4,
            inclusive="both",
        )
    ].copy()

    grouped = (
        data.groupby(
            [
                "down",
                "dist_bucket",
            ],
            as_index=False,
        )
        .agg(
            success_rate=(
                "success",
                "mean",
            ),
            plays=(
                "success",
                "size",
            ),
        )
    )

    grouped = grouped[
        grouped[
            "plays"
        ]
        >= 6
    ].copy()

    order = [
        "1-3 YDS",
        "4-6 YDS",
        "7-10 YDS",
        "11+ YDS",
    ]

    pivot = (
        grouped.pivot(
            index="dist_bucket",
            columns="down",
            values="success_rate",
        )
        .reindex(order)
    )

    labels = {
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
    }

    fig, ax = plt.subplots(
        figsize=(
            10.8,
            7.9,
        )
    )

    apply_chart_style(
        fig,
        ax,
        "Success Rate by Down and Distance",
        weekly_context(
            season,
            week,
        ),
        "Yards to go",
        "Success rate",
    )

    x = np.arange(
        len(order)
    )

    bar_width = 0.18

    for index, down in enumerate(
        [
            1,
            2,
            3,
            4,
        ]
    ):
        values = (
            pivot[
                down
            ].values
            if down
            in pivot.columns
            else np.full(
                len(order),
                np.nan,
            )
        )

        ax.bar(
            x
            + (
                index - 1.5
            )
            * bar_width,
            values,
            width=bar_width,
            label=labels[
                down
            ],
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        order
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend(
        title="Down",
        fontsize=12,
        title_fontsize=16,
    )

    save_chart(
        fig,
        out_path,
    )


# ============================================================
# PLAYER POSTERS
# ============================================================

@dataclass
class PosterItem:
    title: str
    subtitle: str
    player: str
    team: str
    big_value: str
    big_label: str
    description: str
    statline: str
    chip1: str
    chip2: str
    chip3: str
    accent_rgb: Tuple[
        int,
        int,
        int,
    ]
    visual_kind: str
    visual_values: Dict[
        str,
        float,
    ]


def add_gradient_background(
    img: Image.Image,
    accent_rgb: Tuple[
        int,
        int,
        int,
    ],
) -> None:
    overlay = Image.new(
        "RGBA",
        img.size,
        (
            0,
            0,
            0,
            0,
        ),
    )

    pixels = overlay.load()

    for y in range(H):
        alpha = int(
            95
            * (
                1
                - y / H
            )
        )

        red = int(
            accent_rgb[0]
            * 0.42
        )

        green = int(
            accent_rgb[1]
            * 0.42
        )

        blue = int(
            accent_rgb[2]
            * 0.42
        )

        for x in range(W):
            pixels[
                x,
                y,
            ] = (
                red,
                green,
                blue,
                alpha,
            )

    img.alpha_composite(
        overlay
    )


def paste_logo(
    base: Image.Image,
    team: str,
) -> None:
    icon = create_team_icon(
        team,
        size=155,
    )

    if icon is None:
        return

    lx = (
        W
        - MARGIN
        - icon.width
    )

    ly = 28

    base.alpha_composite(
        icon,
        (
            lx,
            ly,
        ),
    )


def draw_header(
    draw,
    item: PosterItem,
):
    draw.text(
        (
            MARGIN,
            34,
        ),
        "STAT OF THE DAY",
        font=font(
            24,
            bold=True,
        ),
        fill=MUTED,
    )

    draw.text(
        (
            MARGIN,
            74,
        ),
        item.title,
        font=font(
            54,
            bold=True,
        ),
        fill=TEXT,
    )

    draw.text(
        (
            MARGIN,
            140,
        ),
        item.subtitle,
        font=font(24),
        fill=MUTED,
    )


def draw_player_row(
    draw,
    item: PosterItem,
):
    y = 184

    rounded_rect(
        draw,
        (
            MARGIN,
            y,
            W - MARGIN,
            y + 108,
        ),
        24,
        fill=CARD,
        outline=LINE,
        width=2,
    )

    max_width = (
        W
        - 2 * MARGIN
        - 44
    )

    max_height = (
        108 - 28
    )

    (
        wrapped,
        fitted_font,
        spacing,
    ) = fit_multiline_text(
        draw=draw,
        text=item.player,
        max_width=max_width,
        max_height=max_height,
        start_size=46,
        min_size=28,
        bold=True,
        line_spacing=4,
    )

    bbox = draw.multiline_textbbox(
        (
            0,
            0,
        ),
        wrapped,
        font=fitted_font,
        spacing=spacing,
    )

    text_h = (
        bbox[3]
        - bbox[1]
    )

    text_y = (
        y
        + (
            108
            - text_h
        )
        // 2
        - 2
    )

    draw.multiline_text(
        (
            MARGIN + 22,
            text_y,
        ),
        wrapped,
        font=fitted_font,
        fill=TEXT,
        spacing=spacing,
    )


def draw_big_stat(
    draw,
    item: PosterItem,
):
    y = 314

    rounded_rect(
        draw,
        (
            MARGIN,
            y,
            W - MARGIN,
            y + 172,
        ),
        28,
        fill=CARD2,
        outline=item.accent_rgb,
        width=3,
    )

    draw.text(
        (
            MARGIN + 24,
            y + 18,
        ),
        item.big_label,
        font=font(
            26,
            bold=True,
        ),
        fill=MUTED,
    )

    max_width = (
        W
        - 2 * MARGIN
        - 48
    )

    (
        wrapped,
        fitted_font,
        spacing,
    ) = fit_multiline_text(
        draw=draw,
        text=item.big_value,
        max_width=max_width,
        max_height=96,
        start_size=92,
        min_size=60,
        bold=True,
        line_spacing=4,
    )

    draw.multiline_text(
        (
            MARGIN + 24,
            y + 56,
        ),
        wrapped,
        font=fitted_font,
        fill=TEXT,
        spacing=spacing,
    )


def draw_description(
    draw,
    item: PosterItem,
):
    y = 510

    rounded_rect(
        draw,
        (
            MARGIN,
            y,
            W - MARGIN,
            y + 138,
        ),
        22,
        fill=CARD,
        outline=LINE,
        width=2,
    )

    max_width = (
        W
        - 2 * MARGIN
        - 36
    )

    max_height = (
        138 - 28
    )

    (
        wrapped,
        fitted_font,
        spacing,
    ) = fit_multiline_text(
        draw=draw,
        text=item.description,
        max_width=max_width,
        max_height=max_height,
        start_size=28,
        min_size=18,
        bold=False,
        line_spacing=7,
    )

    bbox = draw.multiline_textbbox(
        (
            0,
            0,
        ),
        wrapped,
        font=fitted_font,
        spacing=spacing,
    )

    text_h = (
        bbox[3]
        - bbox[1]
    )

    text_y = (
        y
        + (
            138
            - text_h
        )
        // 2
        - 2
    )

    draw.multiline_text(
        (
            MARGIN + 18,
            text_y,
        ),
        wrapped,
        font=fitted_font,
        fill=TEXT,
        spacing=spacing,
    )


def draw_statline(
    draw,
    item: PosterItem,
):
    y = 664

    rounded_rect(
        draw,
        (
            MARGIN,
            y,
            W - MARGIN,
            y + 126,
        ),
        22,
        fill=CARD,
        outline=LINE,
        width=2,
    )

    draw.text(
        (
            MARGIN + 18,
            y + 14,
        ),
        "STAT LINE",
        font=font(
            20,
            bold=True,
        ),
        fill=MUTED,
    )

    max_width = (
        W
        - 2 * MARGIN
        - 36
    )

    max_height = (
        126 - 44
    )

    (
        wrapped,
        fitted_font,
        spacing,
    ) = fit_multiline_text(
        draw=draw,
        text=item.statline,
        max_width=max_width,
        max_height=max_height,
        start_size=28,
        min_size=17,
        bold=False,
        line_spacing=6,
    )

    draw.multiline_text(
        (
            MARGIN + 18,
            y + 44,
        ),
        wrapped,
        font=fitted_font,
        fill=TEXT,
        spacing=spacing,
    )


def draw_visual(
    draw,
    item: PosterItem,
):
    y = 814

    rounded_rect(
        draw,
        (
            MARGIN,
            y,
            W - MARGIN,
            y + 252,
        ),
        28,
        fill=CARD,
        outline=LINE,
        width=2,
    )

    draw.text(
        (
            MARGIN + 20,
            y + 16,
        ),
        "VISUAL BREAKDOWN",
        font=font(
            22,
            bold=True,
        ),
        fill=MUTED,
    )

    if (
        item.visual_kind
        == "before_after"
    ):
        left = (
            MARGIN + 30
        )

        right = (
            W
            - MARGIN
            - 30
        )

        before = max(
            0.0,
            min(
                1.0,
                item.visual_values.get(
                    "before",
                    0.0,
                ),
            ),
        )

        after = max(
            0.0,
            min(
                1.0,
                item.visual_values.get(
                    "after",
                    0.0,
                ),
            ),
        )

        draw.text(
            (
                left,
                y + 56,
            ),
            (
                f"Before: "
                f"{before * 100:.0f}%"
            ),
            font=font(
                24,
                bold=True,
            ),
            fill=TEXT,
        )

        draw.text(
            (
                left + 470,
                y + 56,
            ),
            (
                f"After: "
                f"{after * 100:.0f}%"
            ),
            font=font(
                24,
                bold=True,
            ),
            fill=TEXT,
        )

        bar_y1 = (
            y + 104
        )

        bar_y2 = (
            y + 140
        )

        rounded_rect(
            draw,
            (
                left,
                bar_y1,
                right,
                bar_y2,
            ),
            18,
            fill=(
                28,
                34,
                48,
            ),
        )

        rounded_rect(
            draw,
            (
                left,
                bar_y1,
                left
                + int(
                    (
                        right
                        - left
                    )
                    * before
                ),
                bar_y2,
            ),
            18,
            fill=YELLOW,
        )

        bar2_y1 = (
            y + 176
        )

        bar2_y2 = (
            y + 212
        )

        rounded_rect(
            draw,
            (
                left,
                bar2_y1,
                right,
                bar2_y2,
            ),
            18,
            fill=(
                28,
                34,
                48,
            ),
        )

        rounded_rect(
            draw,
            (
                left,
                bar2_y1,
                left
                + int(
                    (
                        right
                        - left
                    )
                    * after
                ),
                bar2_y2,
            ),
            18,
            fill=GREEN,
        )

    elif (
        item.visual_kind
        == "percentile"
    ):
        pct = max(
            1,
            min(
                99,
                int(
                    item.visual_values.get(
                        "percentile",
                        50,
                    )
                ),
            ),
        )

        left = (
            MARGIN + 30
        )

        right = (
            W
            - MARGIN
            - 30
        )

        draw.text(
            (
                left,
                y + 54,
            ),
            f"{pct}th percentile",
            font=font(
                50,
                bold=True,
            ),
            fill=TEXT,
        )

        draw.text(
            (
                left,
                y + 112,
            ),
            (
                "Compared to other "
                "performances that week"
            ),
            font=font(20),
            fill=MUTED,
        )

        bar_y1 = (
            y + 172
        )

        bar_y2 = (
            y + 210
        )

        rounded_rect(
            draw,
            (
                left,
                bar_y1,
                right,
                bar_y2,
            ),
            18,
            fill=(
                28,
                34,
                48,
            ),
        )

        fill_x = (
            left
            + int(
                (
                    right
                    - left
                )
                * (
                    pct
                    / 100.0
                )
            )
        )

        fill_color = (
            GREEN
            if pct >= 75
            else YELLOW
            if pct >= 50
            else RED
        )

        rounded_rect(
            draw,
            (
                left,
                bar_y1,
                fill_x,
                bar_y2,
            ),
            18,
            fill=fill_color,
        )

    elif (
        item.visual_kind
        == "expected_vs_actual"
    ):
        expected = max(
            0.0,
            float(
                item.visual_values.get(
                    "expected",
                    0.0,
                )
            ),
        )

        actual = max(
            0.0,
            float(
                item.visual_values.get(
                    "actual",
                    0.0,
                )
            ),
        )

        top = max(
            expected,
            actual,
            1.0,
        )

        chart_left = (
            MARGIN + 150
        )

        chart_right = (
            W
            - MARGIN
            - 150
        )

        base_y = (
            y + 208
        )

        max_h = 120
        bar_w = 150

        expected_h = int(
            max_h
            * (
                expected
                / top
            )
        )

        actual_h = int(
            max_h
            * (
                actual
                / top
            )
        )

        draw.line(
            (
                chart_left - 60,
                base_y,
                chart_right + 60,
                base_y,
            ),
            fill=LINE,
            width=3,
        )

        lx1 = (
            chart_left
        )

        lx2 = (
            chart_left
            + bar_w
        )

        rx1 = (
            chart_right
            - bar_w
        )

        rx2 = (
            chart_right
        )

        rounded_rect(
            draw,
            (
                lx1,
                base_y
                - expected_h,
                lx2,
                base_y,
            ),
            18,
            fill=YELLOW,
        )

        rounded_rect(
            draw,
            (
                rx1,
                base_y
                - actual_h,
                rx2,
                base_y,
            ),
            18,
            fill=GREEN,
        )

        draw.text(
            (
                lx1 + 28,
                base_y
                - expected_h
                - 34,
            ),
            f"{expected:.0f}",
            font=font(
                28,
                bold=True,
            ),
            fill=TEXT,
        )

        draw.text(
            (
                rx1 + 28,
                base_y
                - actual_h
                - 34,
            ),
            f"{actual:.0f}",
            font=font(
                28,
                bold=True,
            ),
            fill=TEXT,
        )

        draw.text(
            (
                lx1 + 12,
                base_y + 12,
            ),
            "Expected",
            font=font(
                22,
                bold=True,
            ),
            fill=TEXT,
        )

        draw.text(
            (
                rx1 + 28,
                base_y + 12,
            ),
            "Actual",
            font=font(
                22,
                bold=True,
            ),
            fill=TEXT,
        )


def draw_chips(
    draw,
    item: PosterItem,
):
    y = 1092

    gap = 16

    chip_w = (
        (
            W
            - 2 * MARGIN
            - 2 * gap
        )
        // 3
    )

    chips = [
        item.chip1,
        item.chip2,
        item.chip3,
    ]

    for index, text in enumerate(
        chips
    ):
        x1 = (
            MARGIN
            + index
            * (
                chip_w
                + gap
            )
        )

        x2 = (
            x1
            + chip_w
        )

        rounded_rect(
            draw,
            (
                x1,
                y,
                x2,
                y + 88,
            ),
            22,
            fill=CARD2,
            outline=LINE,
            width=2,
        )

        draw.multiline_text(
            (
                x1 + 14,
                y + 14,
            ),
            wrap_text(
                text,
                18,
            ),
            font=font(
                20,
                bold=True,
            ),
            fill=TEXT,
            spacing=4,
        )


def draw_footer(
    draw,
    season: int,
    week: int,
):
    draw.text(
        (
            MARGIN,
            H - 36,
        ),
        (
            f"{season} "
            f"Week {week} • "
            "Regular Season"
        ),
        font=font(18),
        fill=MUTED,
    )


def render_player_poster(
    item: PosterItem,
    season: int,
    week: int,
    out_path: str,
) -> None:
    base = Image.new(
        "RGBA",
        (
            W,
            H,
        ),
        BG + (255,),
    )

    add_gradient_background(
        base,
        item.accent_rgb,
    )

    draw = ImageDraw.Draw(
        base
    )

    draw_header(
        draw,
        item,
    )

    draw_player_row(
        draw,
        item,
    )

    draw_big_stat(
        draw,
        item,
    )

    draw_description(
        draw,
        item,
    )

    draw_statline(
        draw,
        item,
    )

    draw_visual(
        draw,
        item,
    )

    draw_chips(
        draw,
        item,
    )

    draw_footer(
        draw,
        season,
        week,
    )

    paste_logo(
        base,
        item.team,
    )

    base.convert(
        "RGB"
    ).save(
        out_path,
        quality=95,
    )

    print(
        f"saved: {out_path}"
    )


# ============================================================
# CATEGORY BUILDERS
# ============================================================

def choose_actor(
    row: pd.Series,
) -> str:
    for key in [
        "receiver_player_name",
        "rusher_player_name",
        "passer_player_name",
    ]:
        value = safe_text(
            row.get(key)
        )

        if value:
            return value

    return "Impact Player"


def percentile_from_series(
    series: pd.Series,
    value: float,
) -> int:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 50

    pct = (
        values <= value
    ).mean()

    return max(
        1,
        min(
            99,
            int(
                round(
                    pct * 100
                )
            ),
        ),
    )


def build_qb_masterclass(
    df_week: pd.DataFrame,
    season: int,
    week: int,
) -> PosterItem:
    candidates = df_week.copy()

    candidates = candidates[
        candidates[
            "passer_player_name"
        ].notna()
    ].copy()

    candidates = candidates[
        candidates[
            "epa"
        ].notna()
    ].copy()

    if (
        "qb_dropback"
        in candidates.columns
    ):
        candidates = candidates[
            (
                candidates[
                    "pass"
                ]
                == 1
            )
            |
            (
                candidates[
                    "qb_dropback"
                ]
                == 1
            )
        ].copy()

    else:
        candidates = candidates[
            candidates[
                "pass"
            ]
            == 1
        ].copy()

    grouped = (
        candidates.groupby(
            [
                "posteam",
                "passer_player_name",
            ],
            dropna=False,
        )
        .agg(
            attempts=(
                "epa",
                "size",
            ),
            epa_per_play=(
                "epa",
                "mean",
            ),
            total_epa=(
                "epa",
                "sum",
            ),
            cpoe=(
                "cpoe",
                "mean",
            ),
            success_rate=(
                "success",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped[
            "attempts"
        ]
        >= 15
    ].copy()

    grouped = grouped.sort_values(
        [
            "epa_per_play",
            "total_epa",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if grouped.empty:
        raise RuntimeError(
            "No QB Masterclass "
            "candidate found for "
            "that week."
        )

    row = grouped.iloc[
        0
    ]

    player = safe_text(
        row[
            "passer_player_name"
        ]
    )

    team = normalize_team(
        safe_text(
            row[
                "posteam"
            ]
        )
    )

    statline = qb_week_statline(
        df_week,
        player,
    )

    pct = percentile_from_series(
        grouped[
            "epa_per_play"
        ],
        float(
            row[
                "epa_per_play"
            ]
        ),
    )

    return PosterItem(
        title="QB MASTERCLASS",
        subtitle=weekly_context(
            season,
            week,
        ),
        player=player,
        team=team,
        big_value=(
            f"{float(row['epa_per_play']):+.2f}"
        ),
        big_label="IMPACT PER PLAY",
        description=(
            f"{player} created "
            "efficient offense all "
            "game long, not just "
            "empty passing volume."
        ),
        statline=statline,
        chip1=(
            f"{int(row['attempts'])} "
            "dropbacks"
        ),
        chip2=(
            f"{float(row['cpoe']):+.1f}% "
            "accuracy vs expected"
            if pd.notna(
                row[
                    "cpoe"
                ]
            )
            else
            "Accuracy data unavailable"
        ),
        chip3=(
            f"{float(row['success_rate']):.0%} "
            "positive plays"
        ),
        accent_rgb=TEAM_COLORS.get(
            team,
            BLUE,
        ),
        visual_kind="percentile",
        visual_values={
            "percentile": pct
        },
    )


def build_play_that_won_game(
    df_week: pd.DataFrame,
    season: int,
    week: int,
) -> PosterItem:
    candidates = df_week.copy()

    candidates = candidates[
        candidates[
            "wpa"
        ].notna()
    ].copy()

    candidates[
        "abs_wpa"
    ] = candidates[
        "wpa"
    ].abs()

    candidates = candidates[
        candidates[
            "abs_wpa"
        ]
        > 0.05
    ].copy()

    candidates = candidates[
        candidates[
            "desc"
        ].notna()
    ].copy()

    candidates = candidates.sort_values(
        [
            "abs_wpa",
            "epa",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if candidates.empty:
        raise RuntimeError(
            "No Play That Won "
            "the Game candidate "
            "found for that week."
        )

    row = candidates.iloc[
        0
    ]

    player = choose_actor(
        row
    )

    team = normalize_team(
        safe_text(
            row.get(
                "posteam"
            )
        )
    )

    wp_before = (
        float(
            row.get(
                "wp"
            )
        )
        if pd.notna(
            row.get(
                "wp"
            )
        )
        else 0.0
    )

    wpa = (
        float(
            row.get(
                "wpa"
            )
        )
        if pd.notna(
            row.get(
                "wpa"
            )
        )
        else 0.0
    )

    wp_after = max(
        0.0,
        min(
            1.0,
            wp_before + wpa,
        ),
    )

    teams_line = ""

    away = safe_text(
        row.get(
            "away_team"
        )
    )

    home = safe_text(
        row.get(
            "home_team"
        )
    )

    if (
        away
        and home
    ):
        teams_line = (
            f"{away} at {home}"
        )

    statline = actor_statline(
        df_week,
        player,
    )

    return PosterItem(
        title=(
            "PLAY THAT WON THE GAME"
        ),
        subtitle=(
            f"{weekly_context(season, week)} "
            f"• {teams_line}"
            if teams_line
            else weekly_context(
                season,
                week,
            )
        ),
        player=player,
        team=team,
        big_value=(
            f"{wpa * 100:+.1f}%"
        ),
        big_label=(
            "WIN CHANCE SWING"
        ),
        description=clean_desc(
            row.get(
                "desc"
            ),
            max_len=None,
        ),
        statline=statline,
        chip1=(
            f"Impact "
            f"{float(row['epa']):+.2f}"
            if pd.notna(
                row.get(
                    "epa"
                )
            )
            else
            "Impact unavailable"
        ),
        chip2=(
            f"Q{int(row['qtr'])}"
            if pd.notna(
                row.get(
                    "qtr"
                )
            )
            else
            "Quarter unavailable"
        ),
        chip3=format_down_distance(
            row.get(
                "down"
            ),
            row.get(
                "ydstogo"
            ),
        ),
        accent_rgb=TEAM_COLORS.get(
            team,
            BLUE,
        ),
        visual_kind="before_after",
        visual_values={
            "before": wp_before,
            "after": wp_after,
        },
    )


def build_better_than_expected(
    df_week: pd.DataFrame,
    season: int,
    week: int,
) -> PosterItem:
    candidates = df_week.copy()

    candidates = candidates[
        candidates[
            "yards_gained"
        ].notna()
    ].copy()

    candidates = candidates[
        (
            candidates[
                "rush"
            ]
            == 1
        )
        |
        (
            candidates[
                "receiver_player_name"
            ].notna()
        )
    ].copy()

    candidates[
        "ydstogo_bucket"
    ] = distance_bucket(
        candidates[
            "ydstogo"
        ]
    )

    candidates[
        "yard_bin"
    ] = yardline_bin(
        candidates[
            "yardline_100"
        ],
        step=20,
    )

    candidates[
        "down_bucket"
    ] = (
        candidates[
            "down"
        ]
        .fillna(0)
        .astype(int)
    )

    baseline = (
        candidates.groupby(
            [
                "down_bucket",
                "ydstogo_bucket",
                "yard_bin",
            ],
            dropna=False,
        )
        .agg(
            expected_yards=(
                "yards_gained",
                "mean",
            ),
            sample=(
                "yards_gained",
                "size",
            ),
        )
        .reset_index()
    )

    baseline = baseline[
        baseline[
            "sample"
        ]
        >= 3
    ].copy()

    candidates = candidates.merge(
        baseline[
            [
                "down_bucket",
                "ydstogo_bucket",
                "yard_bin",
                "expected_yards",
            ]
        ],
        on=[
            "down_bucket",
            "ydstogo_bucket",
            "yard_bin",
        ],
        how="left",
    )

    candidates[
        "expected_yards"
    ] = candidates[
        "expected_yards"
    ].fillna(
        candidates[
            "yards_gained"
        ].median()
    )

    candidates[
        "yards_over_expected"
    ] = (
        candidates[
            "yards_gained"
        ]
        - candidates[
            "expected_yards"
        ]
    )

    candidates[
        "player_name"
    ] = candidates[
        "receiver_player_name"
    ].fillna(
        candidates[
            "rusher_player_name"
        ]
    )

    candidates = candidates[
        candidates[
            "player_name"
        ].notna()
    ].copy()

    grouped = (
        candidates.groupby(
            [
                "posteam",
                "player_name",
            ],
            dropna=False,
        )
        .agg(
            total_yoe=(
                "yards_over_expected",
                "sum",
            ),
            total_actual=(
                "yards_gained",
                "sum",
            ),
            total_expected=(
                "expected_yards",
                "sum",
            ),
            plays=(
                "yards_over_expected",
                "size",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped[
            "plays"
        ]
        >= 4
    ].copy()

    grouped = grouped.sort_values(
        [
            "total_yoe",
            "total_actual",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if grouped.empty:
        raise RuntimeError(
            "No Better Than Expected "
            "candidate found for "
            "that week."
        )

    row = grouped.iloc[
        0
    ]

    player = safe_text(
        row[
            "player_name"
        ]
    )

    team = normalize_team(
        safe_text(
            row[
                "posteam"
            ]
        )
    )

    statline = skill_week_statline(
        df_week,
        player,
    )

    return PosterItem(
        title=(
            "BETTER THAN EXPECTED"
        ),
        subtitle=weekly_context(
            season,
            week,
        ),
        player=player,
        team=team,
        big_value=(
            f"{float(row['total_yoe']):+.1f}"
        ),
        big_label=(
            "YARDS ABOVE EXPECTATION"
        ),
        description=(
            f"{player} got much "
            "more out of his touches "
            "than an average player "
            "would in the same "
            "situations."
        ),
        statline=statline,
        chip1=(
            f"Actual "
            f"{float(row['total_actual']):.0f} "
            "yds"
        ),
        chip2=(
            f"Expected "
            f"{float(row['total_expected']):.0f} "
            "yds"
        ),
        chip3=(
            f"{int(row['plays'])} plays"
        ),
        accent_rgb=TEAM_COLORS.get(
            team,
            BLUE,
        ),
        visual_kind=(
            "expected_vs_actual"
        ),
        visual_values={
            "expected": float(
                row[
                    "total_expected"
                ]
            ),
            "actual": float(
                row[
                    "total_actual"
                ]
            ),
        },
    )


def build_clutch_gene(
    df_week: pd.DataFrame,
    season: int,
    week: int,
) -> PosterItem:
    candidates = df_week.copy()

    candidates = candidates[
        candidates[
            "epa"
        ].notna()
    ].copy()

    candidates[
        "actor"
    ] = candidates.apply(
        choose_actor,
        axis=1,
    )

    candidates = candidates[
        candidates[
            "actor"
        ]
        != "Impact Player"
    ].copy()

    candidates = candidates[
        candidates[
            "qtr"
        ]
        .fillna(0)
        >= 4
    ].copy()

    candidates = candidates[
        candidates[
            "wp"
        ].between(
            0.20,
            0.80,
            inclusive="both",
        )
    ].copy()

    grouped = (
        candidates.groupby(
            [
                "posteam",
                "actor",
            ],
            dropna=False,
        )
        .agg(
            clutch_epa=(
                "epa",
                "sum",
            ),
            plays=(
                "epa",
                "size",
            ),
            avg_epa=(
                "epa",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped[
            "plays"
        ]
        >= 3
    ].copy()

    grouped = grouped.sort_values(
        [
            "clutch_epa",
            "avg_epa",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if grouped.empty:
        raise RuntimeError(
            "No Clutch Gene "
            "candidate found for "
            "that week."
        )

    row = grouped.iloc[
        0
    ]

    player = safe_text(
        row[
            "actor"
        ]
    )

    team = normalize_team(
        safe_text(
            row[
                "posteam"
            ]
        )
    )

    statline = actor_statline(
        df_week,
        player,
    )

    pct = percentile_from_series(
        grouped[
            "clutch_epa"
        ],
        float(
            row[
                "clutch_epa"
            ]
        ),
    )

    return PosterItem(
        title="CLUTCH GENE",
        subtitle=weekly_context(
            season,
            week,
        ),
        player=player,
        team=team,
        big_value=(
            f"{float(row['clutch_epa']):+.2f}"
        ),
        big_label=(
            "CLUTCH IMPACT"
        ),
        description=(
            f"{player} delivered "
            "late when the game "
            "still felt up for grabs."
        ),
        statline=statline,
        chip1=(
            f"{int(row['plays'])} "
            "clutch plays"
        ),
        chip2=(
            f"{float(row['avg_epa']):+.2f} "
            "per play"
        ),
        chip3=(
            "4th quarter, live game state"
        ),
        accent_rgb=TEAM_COLORS.get(
            team,
            BLUE,
        ),
        visual_kind="percentile",
        visual_values={
            "percentile": pct
        },
    )


# ============================================================
# SEASON / WEEK PICKING
# ============================================================

def load_prepped_regular_week(
    season: int,
    week: int,
) -> pd.DataFrame:
    validate_week(
        season,
        week,
    )

    df = load_pbp_one_season(
        season
    )

    df = prep_df(
        df
    )

    df_week = filter_regular_week(
        df,
        week,
    )

    if df_week.empty:
        raise RuntimeError(
            f"No regular-season plays "
            f"found for {season} "
            f"week {week}."
        )

    return df_week


def category_candidate_works(
    category_key: str,
    df_week: pd.DataFrame,
    season: int,
    week: int,
) -> bool:
    try:
        if (
            category_key
            == "success_rate_by_down_and_distance"
        ):
            data = df_week.copy()

            data[
                "dist_bucket"
            ] = distance_bucket(
                data[
                    "ydstogo"
                ]
            )

            data = data[
                data[
                    "dist_bucket"
                ].notna()
            ].copy()

            data = data[
                data[
                    "success"
                ].notna()
            ].copy()

            data = data[
                data[
                    "down"
                ].between(
                    1,
                    4,
                    inclusive="both",
                )
            ].copy()

            grouped = (
                data.groupby(
                    [
                        "down",
                        "dist_bucket",
                    ],
                    as_index=False,
                )
                .agg(
                    success_rate=(
                        "success",
                        "mean",
                    ),
                    plays=(
                        "success",
                        "size",
                    ),
                )
            )

            grouped = grouped[
                grouped[
                    "plays"
                ]
                >= 6
            ].copy()

            return (
                not grouped.empty
            )

        if (
            category_key
            == "qb_masterclass"
        ):
            build_qb_masterclass(
                df_week,
                season,
                week,
            )

            return True

        if (
            category_key
            == "play_that_won_the_game"
        ):
            build_play_that_won_game(
                df_week,
                season,
                week,
            )

            return True

        if (
            category_key
            == "better_than_expected"
        ):
            build_better_than_expected(
                df_week,
                season,
                week,
            )

            return True

        if (
            category_key
            == "clutch_gene"
        ):
            build_clutch_gene(
                df_week,
                season,
                week,
            )

            return True

        return False

    except Exception:
        return False


def choose_random_valid_season_week(
    category_key: str,
    start_season: int = MIN_SEASON,
    end_season: int = None,
    max_tries: int = 60,
) -> Tuple[
    int,
    int,
    pd.DataFrame,
]:
    if end_season is None:
        end_season = (
            now_eastern_date()
            .year
        )

    candidates = []

    for season in range(
        start_season,
        end_season + 1,
    ):
        max_week = (
            week_limit_for_year(
                season
            )
        )

        for week in range(
            1,
            max_week + 1,
        ):
            candidates.append(
                (
                    season,
                    week,
                )
            )

    random.shuffle(
        candidates
    )

    last_error = None

    for (
        season,
        week,
    ) in candidates[
        :max_tries
    ]:
        try:
            print(
                f"trying candidate "
                f"season={season} "
                f"week={week} "
                f"for category="
                f"{category_key}"
            )

            df_week = (
                load_prepped_regular_week(
                    season,
                    week,
                )
            )

            if category_candidate_works(
                category_key,
                df_week,
                season,
                week,
            ):
                return (
                    season,
                    week,
                    df_week,
                )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not find valid "
        f"season/week for "
        f"category={category_key}. "
        f"Last error: "
        f"{last_error}"
    )


# ============================================================
# GENERATION
# ============================================================

def generate_single_category(
    category_key: str,
    season: int,
    week: int,
    df_week: pd.DataFrame,
    out_path: str,
) -> Dict[
    str,
    str,
]:
    if (
        category_key
        == "success_rate_by_down_and_distance"
    ):
        plot_success_rate_by_down_and_distance(
            df_week,
            season,
            week,
            out_path,
        )

        return {
            "title": CATEGORY_LABELS[
                category_key
            ],
            "team": "",
            "player": "",
        }

    if (
        category_key
        == "qb_masterclass"
    ):
        item = build_qb_masterclass(
            df_week,
            season,
            week,
        )

        render_player_poster(
            item,
            season,
            week,
            out_path,
        )

        return {
            "title": item.title,
            "team": item.team,
            "player": item.player,
        }

    if (
        category_key
        == "play_that_won_the_game"
    ):
        item = (
            build_play_that_won_game(
                df_week,
                season,
                week,
            )
        )

        render_player_poster(
            item,
            season,
            week,
            out_path,
        )

        return {
            "title": item.title,
            "team": item.team,
            "player": item.player,
        }

    if (
        category_key
        == "better_than_expected"
    ):
        item = (
            build_better_than_expected(
                df_week,
                season,
                week,
            )
        )

        render_player_poster(
            item,
            season,
            week,
            out_path,
        )

        return {
            "title": item.title,
            "team": item.team,
            "player": item.player,
        }

    if (
        category_key
        == "clutch_gene"
    ):
        item = build_clutch_gene(
            df_week,
            season,
            week,
        )

        render_player_poster(
            item,
            season,
            week,
            out_path,
        )

        return {
            "title": item.title,
            "team": item.team,
            "player": item.player,
        }

    raise ValueError(
        f"Unknown category: "
        f"{category_key}"
    )


def publish_stat_of_the_day(
    run_day: date = None,
    keep_versioned: bool = False,
) -> Dict[
    str,
    str,
]:
    if run_day is None:
        run_day = (
            now_eastern_date()
        )

    category_key = (
        category_for_day(
            run_day
        )
    )

    random.seed(
        f"stat-of-day::"
        f"{run_day.isoformat()}::"
        f"{category_key}"
    )

    (
        season,
        week,
        df_week,
    ) = (
        choose_random_valid_season_week(
            category_key=category_key
        )
    )

    local_png = (
        f"/tmp/"
        f"stat_of_the_day_"
        f"{run_day.isoformat()}.png"
    )

    info = generate_single_category(
        category_key,
        season,
        week,
        df_week,
        local_png,
    )

    current_storage_key = (
        "stat_of_the_day/current.png"
    )

    current_url = (
        upload_file_return_url(
            local_png,
            current_storage_key,
        )
    )

    metadata = {
        "date": (
            run_day.isoformat()
        ),
        "category_key": (
            category_key
        ),
        "category_label": (
            CATEGORY_LABELS[
                category_key
            ]
        ),
        "season": season,
        "week": week,
        "image_url": current_url,
        "title": info.get(
            "title",
            "",
        ),
        "team": info.get(
            "team",
            "",
        ),
        "player": info.get(
            "player",
            "",
        ),
        "storage_key": (
            current_storage_key
        ),
    }

    local_json = (
        f"/tmp/"
        f"stat_of_the_day_"
        f"{run_day.isoformat()}.json"
    )

    with open(
        local_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    current_meta_key = (
        "stat_of_the_day/current.json"
    )

    metadata_url = (
        upload_file_return_url(
            local_json,
            current_meta_key,
        )
    )

    versioned_image_url = ""
    versioned_meta_url = ""

    if keep_versioned:
        versioned_prefix = (
            "stat_of_the_day/"
            "history/"
            f"{run_day.isoformat()}"
        )

        versioned_image_key = (
            f"{versioned_prefix}/"
            "poster.png"
        )

        versioned_meta_key = (
            f"{versioned_prefix}/"
            "metadata.json"
        )

        versioned_image_url = (
            upload_file_return_url(
                local_png,
                versioned_image_key,
            )
        )

        versioned_meta_url = (
            upload_file_return_url(
                local_json,
                versioned_meta_key,
            )
        )

    metadata[
        "metadata_url"
    ] = metadata_url

    if versioned_image_url:
        metadata[
            "versioned_image_url"
        ] = versioned_image_url

    if versioned_meta_url:
        metadata[
            "versioned_metadata_url"
        ] = versioned_meta_url

    return metadata


def get_current_stat_of_the_day_payload() -> Dict[
    str,
    str,
]:
    return {
        "image_url": (
            public_storage_url(
                "stat_of_the_day/"
                "current.png"
            )
        ),
        "metadata_url": (
            public_storage_url(
                "stat_of_the_day/"
                "current.json"
            )
        ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Publish the nightly "
            "stat of the day poster."
        )
    )

    parser.add_argument(
        "--date",
        type=str,
        default="",
        help=(
            "Optional YYYY-MM-DD "
            "in America/New_York "
            "rotation logic"
        ),
    )

    parser.add_argument(
        "--keep_versioned",
        action="store_true",
        help=(
            "Also upload "
            "date-stamped "
            "history copy"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_day = (
        date.fromisoformat(
            args.date
        )
        if args.date
        else None
    )

    result = (
        publish_stat_of_the_day(
            run_day=run_day,
            keep_versioned=(
                args.keep_versioned
            ),
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
