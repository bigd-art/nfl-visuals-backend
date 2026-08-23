#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ============================================================
# CONFIG
# ============================================================

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Lowercase league path is required internally by ESPN.
CORE_API_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)


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
    "SF": ("#AA0000", "#B3995D"),
    "SEA": ("#002244", "#69BE28"),
    "TB": ("#D50A0A", "#34302B"),
    "TEN": ("#0C2340", "#4B92DB"),
    "WSH": ("#5A1414", "#FFB612"),
}

DEFAULT_PRIMARY = "#111111"
DEFAULT_SECONDARY = "#444444"


# ============================================================
# USER-FACING ERROR
# ============================================================

def no_poster_message(
    year: int,
    week: int,
) -> str:

    return (
        f"No poster available yet for "
        f"{year} week {week}. "
        "Please try another week or season type"
    )


# ============================================================
# CORE API URL
# ============================================================

def scoreboard_url(
    year: int,
    week: int,
    seasontype: int,
) -> str:

    return (
        f"{CORE_API_BASE}/"
        f"seasons/{year}/"
        f"types/{seasontype}/"
        f"weeks/{week}/"
        "events?limit=100"
    )


# ============================================================
# HTTP
# ============================================================

def normalize_ref(
    url: str,
) -> str:

    url = str(
        url or ""
    ).strip()

    if url.startswith("http://"):
        return (
            "https://"
            + url[len("http://"):]
        )

    return url


def get_json(
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
# DATE FORMAT
# ============================================================

def format_date_eastern(
    date_iso: str,
) -> str:

    if (
        not date_iso
        or date_iso == "-"
    ):
        return "-"

    try:
        raw = date_iso.replace(
            "Z",
            "+00:00",
        )

        dt = datetime.fromisoformat(
            raw
        )

        if ZoneInfo is not None:
            eastern = dt.astimezone(
                ZoneInfo(
                    "America/New_York"
                )
            )
        else:
            eastern = dt

        day_part = (
            eastern.strftime(
                "%a"
            ).upper()
        )

        date_part = (
            eastern.strftime(
                "%m/%d/%y"
            )
        )

        time_part = (
            eastern.strftime(
                "%I:%M %p"
            )
            .lstrip("0")
        )

        return (
            f"{day_part} "
            f"{date_part} "
            f"{time_part} ET"
        )

    except Exception:
        return date_iso


# ============================================================
# EVENT COLLECTION
# ============================================================

def resolve_events(
    data: dict,
) -> List[dict]:

    items = (
        data.get("items")
        or []
    )

    events: List[
        dict
    ] = []

    print(
        f"Weekly events collection "
        f"contains {len(items)} items"
    )

    for index, item in enumerate(
        items,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            ref = str(
                item.get("$ref")
                or ""
            ).strip()

            if ref:
                event = get_json(
                    ref
                )
            else:
                event = item

            if isinstance(
                event,
                dict,
            ):
                events.append(
                    event
                )

        except Exception as exc:
            print(
                f"WARNING: Event "
                f"{index} failed: "
                f"{exc}"
            )

    print(
        f"Resolved "
        f"{len(events)} events"
    )

    return events


# ============================================================
# COMPETITION
# ============================================================

def resolve_competition(
    event: dict,
) -> Optional[dict]:

    competitions = (
        event.get("competitions")
        or []
    )

    if not isinstance(
        competitions,
        list,
    ):
        return None

    if not competitions:
        return None

    competition = (
        competitions[0]
    )

    if not isinstance(
        competition,
        dict,
    ):
        return None

    if isinstance(
        competition.get(
            "competitors"
        ),
        list,
    ):
        return competition

    ref = str(
        competition.get("$ref")
        or ""
    ).strip()

    if ref:
        try:
            resolved = get_json(
                ref
            )

            if isinstance(
                resolved,
                dict,
            ):
                return resolved

        except Exception as exc:
            print(
                f"WARNING: competition "
                f"failed: {exc}"
            )

    return competition


# ============================================================
# TEAM RESOLUTION
# ============================================================

def resolve_team(
    team_obj: Any,
) -> dict:

    if not isinstance(
        team_obj,
        dict,
    ):
        return {}

    if team_obj.get(
        "abbreviation"
    ):
        return team_obj

    ref = str(
        team_obj.get("$ref")
        or ""
    ).strip()

    if ref:
        try:
            resolved = get_json(
                ref
            )

            if isinstance(
                resolved,
                dict,
            ):
                return resolved

        except Exception as exc:
            print(
                f"WARNING: team "
                f"resolution failed: "
                f"{exc}"
            )

    return team_obj


def team_abbreviation(
    team_obj: dict,
) -> str:

    return str(
        team_obj.get(
            "abbreviation"
        )
        or ""
    ).strip().upper()


# ============================================================
# COMPETITOR PARSING
# ============================================================

def parse_competitors(
    competition: dict,
) -> tuple[
    Optional[str],
    Optional[str],
]:

    competitors = (
        competition.get(
            "competitors"
        )
        or []
    )

    away_team = None
    home_team = None

    for competitor in competitors:

        if not isinstance(
            competitor,
            dict,
        ):
            continue

        team = resolve_team(
            competitor.get("team")
            or {}
        )

        abbreviation = (
            team_abbreviation(
                team
            )
        )

        if not abbreviation:
            continue

        home_away = str(
            competitor.get(
                "homeAway"
            )
            or ""
        ).strip().lower()

        if home_away == "away":
            away_team = abbreviation

        elif home_away == "home":
            home_team = abbreviation

    return (
        away_team,
        home_team,
    )


# ============================================================
# PARSE WEEK GAMES
# ============================================================

def parse_week_games(
    data: dict,
) -> List[
    Dict[str, str]
]:

    events = resolve_events(
        data
    )

    if not events:
        raise RuntimeError(
            "NO_GAMES"
        )

    games: List[
        Dict[str, str]
    ] = []

    for index, event in enumerate(
        events,
        start=1,
    ):

        try:
            competition = (
                resolve_competition(
                    event
                )
            )

            if not competition:
                print(
                    f"Skipping event "
                    f"{index}: "
                    f"no competition"
                )
                continue

            (
                away_team,
                home_team,
            ) = parse_competitors(
                competition
            )

            if (
                not away_team
                or not home_team
            ):
                print(
                    f"Skipping event "
                    f"{index}: "
                    f"away/home teams "
                    f"could not be resolved"
                )
                continue

            date_iso = str(
                event.get("date")
                or competition.get(
                    "date"
                )
                or ""
            ).strip()

            game = {
                "away": away_team,
                "home": home_team,
                "date": format_date_eastern(
                    date_iso
                ),
            }

            games.append(
                game
            )

            print(
                f"Game {len(games)}: "
                f"{away_team} @ "
                f"{home_team} | "
                f"{game['date']}"
            )

        except Exception as exc:
            print(
                f"WARNING: Event "
                f"{index} parse "
                f"failed: {exc}"
            )

    if not games:
        raise RuntimeError(
            "NO_GAMES"
        )

    return games


# ============================================================
# FONT HELPERS
# ============================================================

def get_font(
    size: int,
    bold: bool = False,
):

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
            return ImageFont.truetype(
                path,
                size,
            )
        except Exception:
            continue

    return ImageFont.load_default()


def draw_centered(
    draw,
    text,
    font,
    y,
    width,
    fill,
):

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    x = (
        width
        - text_width
    ) // 2

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
        draw.textbbox(
            (0, 0),
            text,
            font=font,
        )[2]
        <= max_width
    ):
        return text

    shortened = text

    while (
        len(shortened)
        > 3
    ):

        shortened = (
            shortened[:-1]
        )

        candidate = (
            shortened
            + "..."
        )

        if (
            draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )[2]
            <= max_width
        ):
            return candidate

    return "..."


# ============================================================
# POSTER COLORS
# ============================================================

def get_poster_colors(
    games: List[
        Dict[str, str]
    ],
):

    if not games:
        return (
            DEFAULT_PRIMARY,
            DEFAULT_SECONDARY,
        )

    away = games[0]["away"]
    home = games[0]["home"]

    away_colors = TEAM_COLORS.get(
        away,
        (
            DEFAULT_PRIMARY,
            DEFAULT_SECONDARY,
        ),
    )

    home_colors = TEAM_COLORS.get(
        home,
        (
            DEFAULT_PRIMARY,
            DEFAULT_SECONDARY,
        ),
    )

    return (
        away_colors[0],
        home_colors[1],
    )


# ============================================================
# PIXEL ICON HELPERS
# ============================================================

def hex_to_rgb(
    value: str,
):

    value = value.replace(
        "#",
        "",
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
    abbr: str,
    cx: int,
    cy: int,
    primary,
    secondary,
):

    if abbr == "ARI":
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

    elif abbr in {
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

    elif abbr == "BUF":
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

    elif abbr == "CAR":
        draw_claws(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr in {
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

    elif abbr == "CIN":
        draw_stripes(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr in {
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

    elif abbr == "DAL":
        draw_star(
            draw,
            cx,
            cy,
            30,
            primary,
        )

    elif abbr == "DEN":
        draw_mountain(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "HOU":
        draw_texas(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "IND":
        # Blue racing/speed arc
        draw_arc(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "KC":
        draw_pennant(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "LV":
        # Generic pirate hat
        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "LAC":
        draw_lightning(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbr == "LAR":
        draw_spiral(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbr == "MIA":
        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "MIN":
        draw_ship(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "NE":
        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "NO":
        # Gold trumpet
        draw_trumpet(
            draw,
            cx,
            cy,
            secondary,
        )

    elif abbr == "NYG":
        draw_skyline(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "NYJ":
        draw_jet(
            draw,
            cx,
            cy,
            primary,
        )

    elif abbr == "PIT":
        draw_diamonds(
            draw,
            cx,
            cy,
        )

    elif abbr == "SF":
        draw_bridge(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "SEA":
        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "TB":
        draw_flag(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "TEN":
        draw_sword(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif abbr == "WSH":
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
# CREATE ORIGINAL TEAM ICON
#
# Every image contains the team abbreviation beneath it.
# ============================================================

def create_team_icon(
    team_abbr: str,
    size: int = 64,
) -> Optional[
    Image.Image
]:

    abbreviation = (
        str(
            team_abbr
            or ""
        )
        .strip()
        .upper()
    )

    if not abbreviation:
        return None

    (
        primary_hex,
        secondary_hex,
    ) = TEAM_COLORS.get(
        abbreviation,
        (
            "#4A6A8A",
            "#C7D0DA",
        ),
    )

    primary = hex_to_rgb(
        primary_hex
    )

    secondary = hex_to_rgb(
        secondary_hex
    )

    # Work at a larger low-resolution canvas first.
    base_width = 104
    base_height = 118

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
        45,
        primary,
        secondary,
    )

    # --------------------------------------------------------
    # ABBREVIATION INSIDE ICON
    # --------------------------------------------------------

    text_font = get_font(
        19,
        True,
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        abbreviation,
        font=text_font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    draw.rounded_rectangle(
        (
            13,
            88,
            base_width - 13,
            115,
        ),
        radius=4,
        fill=(
            8,
            12,
            18,
            235,
        ),
    )

    draw.text(
        (
            (
                base_width
                - text_width
            )
            // 2,
            90,
        ),
        abbreviation,
        font=text_font,
        fill=primary,
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
# MATCHUP ROW
# ============================================================

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
    logo_cache: Dict[
        str,
        Optional[
            Image.Image
        ],
    ],
):

    away_logo = (
        logo_cache.get(
            away
        )
    )

    home_logo = (
        logo_cache.get(
            home
        )
    )

    logo_gap = 10
    side_gap = 18

    away_bbox = draw.textbbox(
        (0, 0),
        away,
        font=font,
    )

    at_bbox = draw.textbbox(
        (0, 0),
        "@",
        font=font,
    )

    home_bbox = draw.textbbox(
        (0, 0),
        home,
        font=font,
    )

    away_width = (
        away_bbox[2]
        - away_bbox[0]
    )

    at_width = (
        at_bbox[2]
        - at_bbox[0]
    )

    home_width = (
        home_bbox[2]
        - home_bbox[0]
    )

    away_logo_width = (
        away_logo.width
        if away_logo
        else 0
    )

    home_logo_width = (
        home_logo.width
        if home_logo
        else 0
    )

    total_width = (
        away_logo_width
        + (
            logo_gap
            if away_logo_width
            else 0
        )
        + away_width
        + side_gap
        + at_width
        + side_gap
        + home_width
        + (
            logo_gap
            if home_logo_width
            else 0
        )
        + home_logo_width
    )

    current_x = (
        matchup_left
        + max(
            0,
            (
                matchup_width
                - total_width
            )
            // 2,
        )
    )

    text_y = (
        matchup_top
        + 8
    )

    if away_logo:
        logo_y = (
            matchup_top
            - 7
        )

        bg.paste(
            away_logo,
            (
                current_x,
                logo_y,
            ),
            away_logo,
        )

        current_x += (
            away_logo.width
            + logo_gap
        )

    draw.text(
        (
            current_x,
            text_y,
        ),
        away,
        font=font,
        fill=text_fill,
    )

    current_x += (
        away_width
        + side_gap
    )

    draw.text(
        (
            current_x,
            text_y,
        ),
        "@",
        font=font,
        fill=text_fill,
    )

    current_x += (
        at_width
        + side_gap
    )

    draw.text(
        (
            current_x,
            text_y,
        ),
        home,
        font=font,
        fill=text_fill,
    )

    current_x += (
        home_width
    )

    if home_logo:
        current_x += (
            logo_gap
        )

        logo_y = (
            matchup_top
            - 7
        )

        bg.paste(
            home_logo,
            (
                current_x,
                logo_y,
            ),
            home_logo,
        )


# ============================================================
# POSTER
# ============================================================

def make_poster(
    year: int,
    week: int,
    seasontype: int,
    games: List[dict],
    output_path: str,
):

    width = 1400
    height = 2200

    (
        primary,
        secondary,
    ) = get_poster_colors(
        games
    )

    bg = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        primary,
    )

    draw = ImageDraw.Draw(
        bg
    )

    title_font = get_font(
        80,
        bold=True,
    )

    subtitle_font = get_font(
        42,
        bold=True,
    )

    header_font = get_font(
        34,
        bold=True,
    )

    row_font = get_font(
        30,
        bold=False,
    )

    row_font_bold = get_font(
        31,
        bold=True,
    )

    date_font = get_font(
        25,
        bold=False,
    )

    # --------------------------------------------------------
    # BORDERS
    # --------------------------------------------------------

    draw.rectangle(
        [
            0,
            0,
            width,
            32,
        ],
        fill=secondary,
    )

    draw.rectangle(
        [
            0,
            height - 32,
            width,
            height,
        ],
        fill=secondary,
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    draw_centered(
        draw,
        "LEAGUE MATCHUPS",
        title_font,
        65,
        width,
        "white",
    )

    if seasontype == 1:
        season_label = (
            "Preseason"
        )

    elif seasontype == 2:
        season_label = (
            "Regular Season"
        )

    elif seasontype == 3:
        season_label = (
            "Playoffs"
        )

    else:
        season_label = (
            f"Season Type "
            f"{seasontype}"
        )

    draw_centered(
        draw,
        (
            f"Week {week} • "
            f"{year} • "
            f"{season_label}"
        ),
        subtitle_font,
        160,
        width,
        "white",
    )

    # --------------------------------------------------------
    # TABLE HEADER
    # --------------------------------------------------------

    left = 55
    right = width - 55
    top = 245

    row_height = 96
    row_gap = 11
    header_height = 122

    draw.rounded_rectangle(
        [
            left,
            top,
            right,
            top + header_height,
        ],
        radius=24,
        fill="#A9B0B4",
    )

    game_column_x = (
        left + 38
    )

    matchup_column_x = (
        left + 280
    )

    matchup_column_width = (
        500
    )

    date_column_x = (
        left + 860
    )

    draw.text(
        (
            game_column_x,
            top + 40,
        ),
        "GAME",
        font=header_font,
        fill="white",
    )

    draw.text(
        (
            matchup_column_x
            + (
                matchup_column_width
                - 170
            )
            // 2,
            top + 40,
        ),
        "MATCHUP",
        font=header_font,
        fill="white",
    )

    draw.text(
        (
            date_column_x,
            top + 40,
        ),
        "DATE / TIME (ET)",
        font=header_font,
        fill="white",
    )

    y = (
        top
        + header_height
        + 14
    )

    # --------------------------------------------------------
    # CUSTOM ICON CACHE
    # --------------------------------------------------------

    logo_cache: Dict[
        str,
        Optional[
            Image.Image
        ],
    ] = {}

    all_teams = set()

    for game in games:
        all_teams.add(
            game["away"]
        )

        all_teams.add(
            game["home"]
        )

    for team in all_teams:
        logo_cache[
            team
        ] = create_team_icon(
            team,
            size=62,
        )

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

    for index, game in enumerate(
        games,
        start=1,
    ):

        row_fill = (
            "#F5F5F5"
            if index % 2 == 1
            else "#E8E8E8"
        )

        text_fill = (
            "#111111"
        )

        draw.rounded_rectangle(
            [
                left,
                y,
                right,
                y + row_height,
            ],
            radius=18,
            fill=row_fill,
        )

        game_number = fit_text(
            draw,
            str(index),
            row_font_bold,
            90,
        )

        date_text = fit_text(
            draw,
            game["date"],
            date_font,
            right
            - date_column_x
            - 25,
        )

        draw.text(
            (
                game_column_x + 5,
                y + 29,
            ),
            game_number,
            font=row_font_bold,
            fill=text_fill,
        )

        draw_matchup_row(
            bg=bg,
            draw=draw,
            matchup_left=matchup_column_x,
            matchup_width=matchup_column_width,
            matchup_top=y + 20,
            away=game["away"],
            home=game["home"],
            font=row_font,
            text_fill=text_fill,
            logo_cache=logo_cache,
        )

        draw.text(
            (
                date_column_x,
                y + 31,
            ),
            date_text,
            font=date_font,
            fill=text_fill,
        )

        y += (
            row_height
            + row_gap
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    bg.save(
        output_path
    )


# ============================================================
# GENERATOR
# ============================================================

def generate_league_matchups_poster(
    year: int,
    week: int,
    seasontype: int,
    output_path: str,
) -> str:

    try:
        print()
        print("=" * 80)

        print(
            f"GENERATING LEAGUE "
            f"MATCHUPS "
            f"{year} WEEK {week} "
            f"TYPE {seasontype}"
        )

        print("=" * 80)

        url = scoreboard_url(
            year,
            week,
            seasontype,
        )

        print(
            f"Core events URL: "
            f"{url}"
        )

        data = get_json(
            url
        )

        games = parse_week_games(
            data
        )

        print()

        print(
            f"Parsed "
            f"{len(games)} "
            f"games."
        )

        if not games:
            raise RuntimeError(
                "NO_GAMES"
            )

        make_poster(
            year,
            week,
            seasontype,
            games,
            output_path,
        )

        print(
            f"Saved poster to "
            f"{output_path}"
        )

        return output_path

    except RuntimeError as exc:

        if str(exc) == "NO_GAMES":
            raise RuntimeError(
                no_poster_message(
                    year,
                    week,
                )
            )

        raise


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a league "
            "matchups poster for "
            "a specific week using "
            "ESPN Core API."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help=(
            "Season year, "
            "for example 2025"
        ),
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help=(
            "Week number, "
            "for example 1"
        ),
    )

    parser.add_argument(
        "--seasontype",
        type=int,
        default=2,
        help=(
            "1=preseason, "
            "2=regular season, "
            "3=playoffs"
        ),
    )

    args = parser.parse_args()

    try:
        output_path = (
            f"league_matchups_"
            f"{args.year}_"
            f"week_{args.week}_"
            f"type_{args.seasontype}.png"
        )

        generate_league_matchups_poster(
            args.year,
            args.week,
            args.seasontype,
            output_path,
        )

        print(
            f"Saved poster to "
            f"{output_path}"
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
