#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ============================================================
# CONFIG
# ============================================================

# Team schedules currently use the 2026 season.
DEFAULT_YEAR = 2026

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

# Lowercase league name is required internally by ESPN.
CORE_API_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)


# ============================================================
# COMPATIBILITY CONSTANTS
# ============================================================

TEAMS_URL = (
    CORE_API_BASE
    + "/seasons/{year}/teams?limit=50"
)

# Kept because nightly_publish_team_schedules.py
# imports SCHEDULE_URL.
SCHEDULE_URL = (
    CORE_API_BASE
    + "/seasons/{year}/teams/{team_id}/events?limit=100"
)

WEEK_EVENTS_URL = (
    CORE_API_BASE
    + "/seasons/{year}/types/2/weeks/{week}/events?limit=100"
)


# ============================================================
# TEAM COLORS
# ============================================================

TEAM_COLORS = {
    "ARI": ("#97233F", "#E6A13B"),
    "ATL": ("#A71930", "#111111"),
    "BAL": ("#4F2A86", "#D5A824"),
    "BUF": ("#1769AA", "#C8102E"),
    "CAR": ("#0085CA", "#101820"),
    "CHI": ("#C95C12", "#5A3216"),
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

def normalize_ref(url: str) -> str:
    url = str(url or "").strip()

    if url.startswith("http://"):
        return "https://" + url[len("http://"):]

    return url


def get_json(url: str) -> dict:
    url = normalize_ref(url)

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
# TEAM HELPERS
# ============================================================

def get_team_abbreviation(team: dict) -> str:
    return str(
        team.get("abbreviation")
        or ""
    ).strip().upper()


def get_team_name(
    team: dict,
    fallback: str = "",
) -> str:
    return str(
        team.get("displayName")
        or team.get("name")
        or team.get("shortDisplayName")
        or fallback
    ).strip()


# ============================================================
# TEAM MAP
# ============================================================

def build_team_map(
    year: int = DEFAULT_YEAR,
) -> Dict[str, dict]:

    url = TEAMS_URL.format(
        year=year
    )

    print()
    print("=" * 80)
    print(
        f"FETCHING TEAM MAP FOR "
        f"{year}"
    )
    print("=" * 80)

    payload = get_json(
        url
    )

    items = (
        payload.get("items")
        or []
    )

    team_map: Dict[
        str,
        dict,
    ] = {}

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
                team = get_json(
                    ref
                )
            else:
                team = item

            team_id = str(
                team.get("id")
                or ""
            ).strip()

            abbreviation = (
                get_team_abbreviation(
                    team
                )
            )

            if (
                not team_id
                or not abbreviation
            ):
                continue

            team_map[
                abbreviation
            ] = {
                "id": team_id,
                "display_name": get_team_name(
                    team,
                    abbreviation,
                ),
            }

        except Exception as exc:
            print(
                f"WARNING: team map item "
                f"{index} failed: {exc}"
            )

    print(
        f"Team map count: "
        f"{len(team_map)}"
    )

    if len(
        team_map
    ) < 32:

        raise RuntimeError(
            "Expected 32 teams, "
            f"parsed {len(team_map)}."
        )

    return team_map


# ============================================================
# WEEK EVENTS
# ============================================================

def get_week_events(
    year: int,
    week: int,
) -> List[dict]:

    url = WEEK_EVENTS_URL.format(
        year=year,
        week=week,
    )

    print()
    print(
        f"Fetching {year} Week {week}: "
        f"{url}"
    )

    payload = get_json(
        url
    )

    items = (
        payload.get("items")
        or []
    )

    events: List[
        dict
    ] = []

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
                f"WARNING: "
                f"{year} Week {week} "
                f"event {index} failed: "
                f"{exc}"
            )

    print(
        f"{year} Week {week}: "
        f"{len(events)} events"
    )

    return events


# ============================================================
# COMPETITION
# ============================================================

def extract_competition(
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
                "WARNING: competition "
                f"resolution failed: "
                f"{exc}"
            )

    return competition


# ============================================================
# COMPETITOR HELPERS
# ============================================================

def resolve_team_from_competitor(
    competitor: dict,
) -> dict:

    team = (
        competitor.get("team")
        or {}
    )

    if not isinstance(
        team,
        dict,
    ):
        return {}

    if team.get(
        "abbreviation"
    ):
        return team

    ref = str(
        team.get("$ref")
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
                f"WARNING: team ref "
                f"failed: {exc}"
            )

    return team


def find_team_competitor(
    competition: dict,
    team_abbr: str,
) -> Optional[dict]:

    competitors = (
        competition.get(
            "competitors"
        )
        or []
    )

    for competitor in competitors:

        if not isinstance(
            competitor,
            dict,
        ):
            continue

        team = (
            resolve_team_from_competitor(
                competitor
            )
        )

        abbreviation = (
            get_team_abbreviation(
                team
            )
        )

        if abbreviation == team_abbr:
            result = dict(
                competitor
            )

            result[
                "team"
            ] = team

            return result

    return None


def find_opponent_competitor(
    competition: dict,
    team_abbr: str,
) -> Optional[dict]:

    competitors = (
        competition.get(
            "competitors"
        )
        or []
    )

    for competitor in competitors:

        if not isinstance(
            competitor,
            dict,
        ):
            continue

        team = (
            resolve_team_from_competitor(
                competitor
            )
        )

        abbreviation = (
            get_team_abbreviation(
                team
            )
        )

        if (
            abbreviation
            and abbreviation
            != team_abbr
        ):
            result = dict(
                competitor
            )

            result[
                "team"
            ] = team

            return result

    return None


# ============================================================
# FIND TEAM GAME
# ============================================================

def find_team_game_for_week(
    team_abbr: str,
    year: int,
    week: int,
    team_map: Dict[str, dict],
) -> Optional[dict]:

    events = get_week_events(
        year,
        week,
    )

    for event in events:

        competition = (
            extract_competition(
                event
            )
        )

        if not competition:
            continue

        team_entry = (
            find_team_competitor(
                competition,
                team_abbr,
            )
        )

        if not team_entry:
            continue

        opponent_entry = (
            find_opponent_competitor(
                competition,
                team_abbr,
            )
        )

        if not opponent_entry:
            continue

        opponent_team = (
            opponent_entry.get(
                "team"
            )
            or {}
        )

        opponent_abbr = (
            get_team_abbreviation(
                opponent_team
            )
        )

        if not opponent_abbr:
            continue

        home_away = str(
            team_entry.get(
                "homeAway"
            )
            or ""
        ).strip().lower()

        if home_away == "away":

            matchup = (
                f"@ {opponent_abbr}"
            )

        else:

            matchup = (
                f"vs {opponent_abbr}"
            )

        date_iso = str(
            event.get("date")
            or competition.get(
                "date"
            )
            or ""
        ).strip()

        print(
            f"{year} Week {week}: "
            f"{team_abbr} "
            f"{matchup} "
            f"{date_iso}"
        )

        return {
            "week": week,
            "opponent": matchup,
            "opponent_abbr": (
                opponent_abbr
            ),
            "date": date_iso,
        }

    print(
        f"{year} Week {week}: "
        f"{team_abbr} BYE"
    )

    return None


# ============================================================
# DATE
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

        date_part = (
            eastern.strftime(
                "%m/%d/%Y"
            )
        )

        time_part = (
            eastern.strftime(
                "%I:%M"
            )
            .lstrip("0")
        )

        return (
            f"{date_part} "
            f"{time_part}"
        )

    except Exception:
        return date_iso


# ============================================================
# BUILD SCHEDULE
# ============================================================

def build_schedule_from_weeks(
    team_abbr: str,
    year: int,
    team_map: Dict[str, dict],
) -> List[dict]:

    print()
    print("=" * 80)

    print(
        f"BUILDING {team_abbr} "
        f"{year} SCHEDULE"
    )

    print("=" * 80)

    full_schedule: List[
        dict
    ] = []

    games_found = 0

    for week in range(
        1,
        19,
    ):

        game = (
            find_team_game_for_week(
                team_abbr,
                year,
                week,
                team_map,
            )
        )

        if game:

            games_found += 1

            full_schedule.append(
                {
                    "week": week,
                    "opponent": (
                        game[
                            "opponent"
                        ]
                    ),
                    "opponent_abbr": (
                        game[
                            "opponent_abbr"
                        ]
                    ),
                    "date": (
                        format_date_eastern(
                            game[
                                "date"
                            ]
                        )
                    ),
                }
            )

        else:

            full_schedule.append(
                {
                    "week": week,
                    "opponent": "BYE",
                    "opponent_abbr": "",
                    "date": "-",
                }
            )

    print()

    print(
        f"{team_abbr} {year}: "
        f"{games_found} games found"
    )

    # Prevent publishing obviously wrong schedules.
    if games_found < 16:

        raise RuntimeError(
            f"Only found {games_found} "
            f"games for "
            f"{team_abbr} {year}. "
            "Refusing to generate "
            "an incomplete poster."
        )

    return full_schedule


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def build_full_18_week_schedule(
    team_abbr: str,
    data,
    team_map: Dict[str, dict],
    year: int = DEFAULT_YEAR,
) -> List[dict]:
    """
    The nightly publisher currently calls this with:

        build_full_18_week_schedule(
            team_abbr,
            data,
            team_map,
        )

    The old data argument remains so existing
    imports and calls continue working.

    Team schedules explicitly default to 2026.
    """

    print()

    print(
        "build_full_18_week_schedule:"
    )

    print(
        f"Using schedule season "
        f"{year}"
    )

    return build_schedule_from_weeks(
        team_abbr,
        year,
        team_map,
    )


# ============================================================
# FONT HELPERS
# ============================================================

def get_font(
    size: int,
    bold: bool = False,
):

    if bold:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ]

    else:

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
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
            (
                0,
                0,
            ),
            text,
            font=font,
        )[2]
        <= max_width
    ):
        return text

    shortened = text

    while (
        len(
            shortened
        )
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
                (
                    0,
                    0,
                ),
                candidate,
                font=font,
            )[2]
            <= max_width
        ):
            return candidate

    return "..."


# ============================================================
# PIXEL ICON HELPERS
# ============================================================

def hex_to_rgb(
    value: str,
):

    value = (
        value
        .replace(
            "#",
            "",
        )
        .strip()
    )

    return tuple(
        int(
            value[
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
        (
            cx,
            cy - radius,
        ),
        (
            cx + radius // 4,
            cy - radius // 4,
        ),
        (
            cx + radius,
            cy - radius // 4,
        ),
        (
            cx + radius // 3,
            cy + radius // 5,
        ),
        (
            cx + radius // 2,
            cy + radius,
        ),
        (
            cx,
            cy + radius // 2,
        ),
        (
            cx - radius // 2,
            cy + radius,
        ),
        (
            cx - radius // 3,
            cy + radius // 5,
        ),
        (
            cx - radius,
            cy - radius // 4,
        ),
        (
            cx - radius // 4,
            cy - radius // 4,
        ),
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
        (
            -20,
            -18,
        ),
        (
            -7,
            -25,
        ),
        (
            7,
            -25,
        ),
        (
            20,
            -18,
        ),
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
            (
                cx - 27,
                cy + 26,
            ),
            (
                cx - 14,
                cy - 18,
            ),
            (
                cx + 27,
                cy - 33,
            ),
            (
                cx + 18,
                cy + 8,
            ),
            (
                cx - 9,
                cy + 27,
            ),
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
                    cx
                    + offset
                    - 5,
                    cy + 27,
                ),
                (
                    cx
                    + offset
                    + 3,
                    cy - 30,
                ),
                (
                    cx
                    + offset
                    + 10,
                    cy - 34,
                ),
                (
                    cx
                    + offset
                    + 3,
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
                    cx
                    + offset
                    + 10,
                    cy - 30,
                ),
                (
                    cx
                    + offset
                    + 35,
                    cy + 30,
                ),
                (
                    cx
                    + offset
                    + 25,
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
            (
                cx - 36,
                cy + 27,
            ),
            (
                cx,
                cy - 31,
            ),
            (
                cx + 36,
                cy + 27,
            ),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx - 9,
                cy - 17,
            ),
            (
                cx,
                cy - 31,
            ),
            (
                cx + 10,
                cy - 15,
            ),
            (
                cx + 2,
                cy - 20,
            ),
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
            (
                cx - 28,
                cy - 30,
            ),
            (
                cx + 7,
                cy - 30,
            ),
            (
                cx + 8,
                cy - 13,
            ),
            (
                cx + 30,
                cy - 12,
            ),
            (
                cx + 23,
                cy + 8,
            ),
            (
                cx + 7,
                cy + 15,
            ),
            (
                cx - 2,
                cy + 33,
            ),
            (
                cx - 16,
                cy + 16,
            ),
            (
                cx - 30,
                cy + 4,
            ),
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
            (
                cx - 26,
                cy - 27,
            ),
            (
                cx + 33,
                cy - 4,
            ),
            (
                cx - 26,
                cy + 17,
            ),
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
            (
                cx + 5,
                cy - 34,
            ),
            (
                cx - 22,
                cy + 1,
            ),
            (
                cx - 5,
                cy + 1,
            ),
            (
                cx - 14,
                cy + 34,
            ),
            (
                cx + 25,
                cy - 8,
            ),
            (
                cx + 8,
                cy - 8,
            ),
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
            (
                cx - 34,
                cy + 18,
            ),
            (
                cx - 24,
                cy - 5,
            ),
            (
                cx - 10,
                cy - 20,
            ),
            (
                cx + 4,
                cy - 23,
            ),
            (
                cx + 21,
                cy - 14,
            ),
            (
                cx + 34,
                cy + 4,
            ),
            (
                cx + 13,
                cy - 1,
            ),
            (
                cx,
                cy + 8,
            ),
            (
                cx - 7,
                cy + 20,
            ),
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
            (
                cx - 34,
                cy + 17,
            ),
            (
                cx + 34,
                cy + 17,
            ),
            (
                cx + 22,
                cy + 29,
            ),
            (
                cx - 22,
                cy + 29,
            ),
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
            (
                cx + 2,
                cy - 27,
            ),
            (
                cx + 23,
                cy - 5,
            ),
            (
                cx + 2,
                cy - 5,
            ),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx - 3,
                cy - 24,
            ),
            (
                cx - 22,
                cy - 4,
            ),
            (
                cx - 3,
                cy - 4,
            ),
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
            (
                cx - 33,
                cy + 16,
            ),
            (
                cx - 19,
                cy - 13,
            ),
            (
                cx,
                cy - 25,
            ),
            (
                cx + 19,
                cy - 13,
            ),
            (
                cx + 33,
                cy + 16,
            ),
            (
                cx + 10,
                cy + 9,
            ),
            (
                cx,
                cy + 21,
            ),
            (
                cx - 10,
                cy + 9,
            ),
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
            (
                cx + 10,
                cy - 14,
            ),
            (
                cx + 32,
                cy - 22,
            ),
            (
                cx + 32,
                cy + 22,
            ),
            (
                cx + 10,
                cy + 14,
            ),
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
        (
            -31,
            11,
            -22,
            29,
        ),
        (
            -20,
            -3,
            -8,
            29,
        ),
        (
            -6,
            -29,
            6,
            29,
        ),
        (
            8,
            -10,
            21,
            29,
        ),
        (
            23,
            3,
            32,
            29,
        ),
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
            (
                cx - 35,
                cy + 6,
            ),
            (
                cx - 7,
                cy - 4,
            ),
            (
                cx + 18,
                cy - 28,
            ),
            (
                cx + 25,
                cy - 23,
            ),
            (
                cx + 12,
                cy - 3,
            ),
            (
                cx + 34,
                cy + 7,
            ),
            (
                cx + 10,
                cy + 10,
            ),
            (
                cx + 3,
                cy + 27,
            ),
            (
                cx - 5,
                cy + 27,
            ),
            (
                cx - 6,
                cy + 11,
            ),
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
                (
                    x,
                    y - 9,
                ),
                (
                    x + 9,
                    y,
                ),
                (
                    x,
                    y + 9,
                ),
                (
                    x - 9,
                    y,
                ),
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
            (
                cx - 24,
                cy - 26,
            ),
            (
                cx + 29,
                cy - 21,
            ),
            (
                cx + 20,
                cy + 3,
            ),
            (
                cx - 24,
                cy + 8,
            ),
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
            (
                cx - 4,
                cy - 33,
            ),
            (
                cx + 5,
                cy - 33,
            ),
            (
                cx + 6,
                cy + 13,
            ),
            (
                cx - 6,
                cy + 13,
            ),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx - 4,
                cy - 33,
            ),
            (
                cx,
                cy - 42,
            ),
            (
                cx + 5,
                cy - 33,
            ),
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
            primary,
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


# ============================================================
# CREATE ORIGINAL TEAM ICON
#
# The abbreviation is part of the icon itself.
# ============================================================

def create_team_icon(
    team_abbr: str,
    size: int = 78,
) -> Optional[
    Image.Image
]:

    team_abbr = (
        str(
            team_abbr
            or ""
        )
        .strip()
        .upper()
    )

    if (
        not team_abbr
        or team_abbr
        not in TEAM_COLORS
    ):
        return None

    (
        primary_hex,
        secondary_hex,
    ) = TEAM_COLORS[
        team_abbr
    ]

    primary = hex_to_rgb(
        primary_hex
    )

    secondary = hex_to_rgb(
        secondary_hex
    )

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
        team_abbr,
        base_width // 2,
        45,
        primary,
        secondary,
    )

    abbreviation_font = (
        get_font(
            18,
            bold=True,
        )
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        team_abbr,
        font=abbreviation_font,
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

    text_fill = primary

    if sum(
        primary
    ) < 140:

        text_fill = (
            230,
            230,
            230,
        )

    draw.text(
        (
            (
                base_width
                - text_width
            )
            // 2,
            91,
        ),
        team_abbr,
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

    return icon.resize(
        (
            output_width,
            size,
        ),
        Image.Resampling.NEAREST,
    )


# ============================================================
# POSTER
# ============================================================

def make_poster(
    team_abbr: str,
    team_name: str,
    year: int,
    games: List[dict],
    output_path: str,
):

    width = 1800
    height = 3000

    (
        primary,
        secondary,
    ) = TEAM_COLORS.get(
        team_abbr,
        (
            "#111111",
            "#444444",
        ),
    )

    background = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        primary,
    )

    draw = ImageDraw.Draw(
        background
    )

    title_font = get_font(
        132,
        bold=True,
    )

    subtitle_font = get_font(
        68,
        bold=True,
    )

    header_font = get_font(
        45,
        bold=True,
    )

    row_font = get_font(
        40,
        bold=False,
    )

    week_font = get_font(
        45,
        bold=True,
    )

    draw.rectangle(
        [
            0,
            0,
            width,
            54,
        ],
        fill=secondary,
    )

    draw.rectangle(
        [
            0,
            height - 54,
            width,
            height,
        ],
        fill=secondary,
    )

    draw_centered(
        draw,
        team_abbr,
        title_font,
        86,
        width,
        "white",
    )

    draw_centered(
        draw,
        (
            f"{team_name} "
            f"{year} Schedule"
        ),
        subtitle_font,
        236,
        width,
        "white",
    )

    left = 62
    right = width - 62
    top = 334

    header_h = 126
    row_h = 120
    row_gap = 7

    draw.rounded_rectangle(
        [
            left,
            top,
            right,
            top + header_h,
        ],
        radius=28,
        fill=secondary,
    )

    week_x = (
        left + 30
    )

    opp_label_x = (
        left + 245
    )

    logo_x = (
        left + 250
    )

    matchup_x = (
        left + 405
    )

    date_x = (
        left + 1115
    )

    draw.text(
        (
            week_x,
            top + 38,
        ),
        "WEEK",
        font=header_font,
        fill="white",
    )

    draw.text(
        (
            opp_label_x,
            top + 38,
        ),
        "OPP",
        font=header_font,
        fill="white",
    )

    draw.text(
        (
            matchup_x,
            top + 38,
        ),
        "MATCHUP",
        font=header_font,
        fill="white",
    )

    draw.text(
        (
            date_x,
            top + 38,
        ),
        "DATE / TIME (ET)",
        font=header_font,
        fill="white",
    )

    y = (
        top
        + header_h
        + 12
    )

    # The icon height is deliberately smaller than the
    # 120px row so it has comfortable padding above/below.
    logo_size = 82

    matchup_width = 675

    date_width = (
        right
        - date_x
        - 22
    )

    logo_cache: Dict[
        str,
        Optional[
            Image.Image
        ],
    ] = {}

    for index, game in enumerate(
        games
    ):

        row_fill = (
            "#FFFFFF"
            if index % 2 == 0
            else "#F2F2F2"
        )

        text_fill = (
            "#111111"
        )

        draw.rounded_rectangle(
            [
                left,
                y,
                right,
                y + row_h,
            ],
            radius=18,
            fill=row_fill,
        )

        matchup_text = fit_text(
            draw,
            game[
                "opponent"
            ],
            row_font,
            matchup_width,
        )

        date_text = fit_text(
            draw,
            game[
                "date"
            ],
            row_font,
            date_width,
        )

        week_text = str(
            game[
                "week"
            ]
        )

        week_bbox = (
            draw.textbbox(
                (
                    0,
                    0,
                ),
                week_text,
                font=week_font,
            )
        )

        week_height = (
            week_bbox[3]
            - week_bbox[1]
        )

        draw.text(
            (
                week_x,
                y
                + (
                    row_h
                    - week_height
                )
                // 2
                - 4,
            ),
            week_text,
            font=week_font,
            fill=text_fill,
        )

        # ----------------------------------------------------
        # ORIGINAL PIXEL OPPONENT ICON
        # ----------------------------------------------------

        opponent_abbr = (
            game.get(
                "opponent_abbr",
                "",
            )
        )

        if (
            game[
                "opponent"
            ]
            != "BYE"
            and opponent_abbr
        ):

            if (
                opponent_abbr
                not in logo_cache
            ):

                logo_cache[
                    opponent_abbr
                ] = create_team_icon(
                    opponent_abbr,
                    size=logo_size,
                )

            logo_image = (
                logo_cache.get(
                    opponent_abbr
                )
            )

            if (
                logo_image
                is not None
            ):

                icon_x = (
                    logo_x
                    + (
                        90
                        - logo_image.width
                    )
                    // 2
                )

                icon_y = (
                    y
                    + (
                        row_h
                        - logo_image.height
                    )
                    // 2
                )

                background.paste(
                    logo_image,
                    (
                        icon_x,
                        icon_y,
                    ),
                    logo_image,
                )

            else:

                draw.text(
                    (
                        logo_x + 22,
                        y + 38,
                    ),
                    "-",
                    font=row_font,
                    fill=text_fill,
                )

        else:

            draw.text(
                (
                    logo_x + 22,
                    y + 38,
                ),
                "-",
                font=row_font,
                fill=text_fill,
            )

        matchup_bbox = (
            draw.textbbox(
                (
                    0,
                    0,
                ),
                matchup_text,
                font=row_font,
            )
        )

        matchup_height = (
            matchup_bbox[3]
            - matchup_bbox[1]
        )

        draw.text(
            (
                matchup_x,
                y
                + (
                    row_h
                    - matchup_height
                )
                // 2
                - 4,
            ),
            matchup_text,
            font=row_font,
            fill=text_fill,
        )

        date_bbox = (
            draw.textbbox(
                (
                    0,
                    0,
                ),
                date_text,
                font=row_font,
            )
        )

        date_height = (
            date_bbox[3]
            - date_bbox[1]
        )

        draw.text(
            (
                date_x,
                y
                + (
                    row_h
                    - date_height
                )
                // 2
                - 4,
            ),
            date_text,
            font=row_font,
            fill=text_fill,
        )

        y += (
            row_h
            + row_gap
        )

    background.save(
        output_path
    )

    print(
        f"Saved poster to "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a team regular-season "
            "schedule poster using "
            "ESPN Core weekly events."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=(
            "Schedule season. "
            "Defaults to 2026."
        ),
    )

    parser.add_argument(
        "--team",
        type=str,
        required=True,
    )

    args = (
        parser.parse_args()
    )

    team_abbr = (
        args.team
        .strip()
        .upper()
    )

    try:
        print()
        print("=" * 80)

        print(
            f"GENERATING "
            f"{team_abbr} "
            f"{args.year} "
            f"SCHEDULE"
        )

        print("=" * 80)

        team_map = build_team_map(
            args.year
        )

        if (
            team_abbr
            not in team_map
        ):

            raise RuntimeError(
                f"Invalid team: "
                f"{team_abbr}"
            )

        team_name = (
            team_map[
                team_abbr
            ][
                "display_name"
            ]
        )

        games = (
            build_schedule_from_weeks(
                team_abbr,
                args.year,
                team_map,
            )
        )

        output_path = (
            f"{team_abbr.lower()}_"
            f"{args.year}_"
            f"schedule_poster.png"
        )

        make_poster(
            team_abbr,
            team_name,
            args.year,
            games,
            output_path,
        )

        print()
        print(
            "✅ SCHEDULE COMPLETE"
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        raise


if __name__ == "__main__":
    main()
