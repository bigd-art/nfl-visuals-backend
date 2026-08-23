#!/usr/bin/env python3

import argparse
import io
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

DEFAULT_YEAR = 2025

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

CORE_API_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)


# ============================================================
# COMPATIBILITY CONSTANTS
# ============================================================

# Kept because nightly_publish_team_schedules.py imports them.

TEAMS_URL = (
    CORE_API_BASE
    + "/seasons/{year}/teams?limit=50"
)

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


def get_team_logo(team: dict) -> str:
    logos = (
        team.get("logos")
        or []
    )

    if isinstance(
        logos,
        list,
    ):
        for logo in logos:

            if not isinstance(
                logo,
                dict,
            ):
                continue

            href = str(
                logo.get("href")
                or ""
            ).strip()

            if href:
                return href

    logo = team.get("logo")

    if isinstance(
        logo,
        dict,
    ):
        return str(
            logo.get("href")
            or ""
        ).strip()

    return ""


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
        f"FETCHING TEAM MAP FOR {year}"
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
                "display_name": (
                    get_team_name(
                        team,
                        abbreviation,
                    )
                ),
                "logo": (
                    get_team_logo(
                        team
                    )
                ),
            }

        except Exception as exc:

            print(
                f"WARNING: "
                f"team map item "
                f"{index} failed: "
                f"{exc}"
            )

    print(
        f"Team map count: "
        f"{len(team_map)}"
    )

    if len(team_map) < 32:
        raise RuntimeError(
            "Expected 32 teams, "
            f"parsed {len(team_map)}."
        )

    return team_map


# ============================================================
# CORE REF HELPERS
# ============================================================

def resolve_ref_object(
    obj: Any,
) -> Any:

    if not isinstance(
        obj,
        dict,
    ):
        return obj

    ref = str(
        obj.get("$ref")
        or ""
    ).strip()

    if not ref:
        return obj

    try:

        return get_json(
            ref
        )

    except Exception as exc:

        print(
            f"WARNING: failed "
            f"reference {ref}: "
            f"{exc}"
        )

        return obj


# ============================================================
# WEEK EVENTS
# ============================================================

def get_week_events(
    year: int,
    week: int,
) -> List[dict]:
    """
    Fetch all regular-season events for one week.
    """

    url = WEEK_EVENTS_URL.format(
        year=year,
        week=week,
    )

    print()
    print(
        f"Fetching Week {week}: "
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
                f"WARNING: Week "
                f"{week} event "
                f"{index} failed: "
                f"{exc}"
            )

    print(
        f"Week {week}: "
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

        if (
            abbreviation
            == team_abbr
        ):

            result = dict(
                competitor
            )

            result["team"] = team

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

            result["team"] = team

            return result

    return None


# ============================================================
# FIND TEAM GAME FOR WEEK
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

        if (
            home_away
            == "away"
        ):
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

        logo_url = (
            get_team_logo(
                opponent_team
            )
        )

        if (
            not logo_url
            and opponent_abbr
            in team_map
        ):

            logo_url = str(
                team_map[
                    opponent_abbr
                ].get(
                    "logo",
                    "",
                )
            )

        print(
            f"Week {week}: "
            f"{matchup} "
            f"{date_iso}"
        )

        return {
            "week": week,
            "opponent": matchup,
            "date": date_iso,
            "logo_url": logo_url,
        }

    print(
        f"Week {week}: BYE"
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

        raw = (
            date_iso.replace(
                "Z",
                "+00:00",
            )
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

        date_part = (
            dt.strftime(
                "%m/%d/%Y"
            )
        )

        time_part = (
            dt.strftime(
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
# BUILD 18-WEEK SCHEDULE
# ============================================================

def build_schedule_from_weeks(
    team_abbr: str,
    year: int,
    team_map: Dict[str, dict],
) -> List[dict]:
    """
    Deterministically check Weeks 1-18.

    If the team does not appear in a week's events,
    that week is treated as the BYE.
    """

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
                    "date": (
                        format_date_eastern(
                            game[
                                "date"
                            ]
                        )
                    ),
                    "logo_url": (
                        game[
                            "logo_url"
                        ]
                    ),
                }
            )

        else:

            full_schedule.append(
                {
                    "week": week,
                    "opponent": "BYE",
                    "date": "-",
                    "logo_url": "",
                }
            )

    print()
    print(
        f"{team_abbr}: "
        f"{games_found} games "
        f"found across Weeks 1-18"
    )

    # NFL regular-season schedule should have 17 games.
    # Fail instead of silently producing another all-BYE poster.

    if games_found < 16:

        raise RuntimeError(
            f"Only found {games_found} "
            f"regular-season games "
            f"for {team_abbr} {year}. "
            "Refusing to generate "
            "an incomplete schedule poster."
        )

    return full_schedule


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION
# ============================================================

def build_full_18_week_schedule(
    team_abbr: str,
    data: Any,
    team_map: Dict[str, dict],
) -> List[dict]:
    """
    This function name/signature is preserved because your
    nightly publisher already imports it.

    The old raw team-events payload is deliberately ignored.
    We infer the year from that payload when possible,
    otherwise default to 2025.
    """

    year = DEFAULT_YEAR

    if isinstance(
        data,
        dict,
    ):

        season_obj = (
            data.get("season")
            or {}
        )

        if isinstance(
            season_obj,
            dict,
        ):

            candidate = (
                season_obj.get("year")
                or season_obj.get(
                    "value"
                )
            )

            try:

                if candidate:
                    year = int(
                        candidate
                    )

            except Exception:
                pass

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

    while len(
        shortened
    ) > 3:

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
# LOGO IMAGE
# ============================================================

def fetch_logo_image(
    url: str,
    size: int,
    cache: Dict[
        str,
        Image.Image,
    ],
) -> Optional[
    Image.Image
]:

    if not url:
        return None

    if url in cache:
        return (
            cache[
                url
            ].copy()
        )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )

        response.raise_for_status()

        image = (
            Image.open(
                io.BytesIO(
                    response.content
                )
            )
            .convert(
                "RGBA"
            )
        )

        image.thumbnail(
            (
                size,
                size,
            ),
            Image.LANCZOS,
        )

        canvas = Image.new(
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

        x = (
            size
            - image.width
        ) // 2

        y = (
            size
            - image.height
        ) // 2

        canvas.paste(
            image,
            (
                x,
                y,
            ),
            image,
        )

        cache[
            url
        ] = canvas

        return (
            canvas.copy()
        )

    except Exception as exc:

        print(
            f"WARNING: logo "
            f"fetch failed: "
            f"{exc}"
        )

        return None


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
    ) = (
        TEAM_COLORS.get(
            team_abbr,
            (
                "#111111",
                "#444444",
            ),
        )
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
            top
            + header_h,
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
        left + 255
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

    logo_size = 78
    matchup_width = 675

    date_width = (
        right
        - date_x
        - 22
    )

    logo_cache: Dict[
        str,
        Image.Image,
    ] = {}

    for index, game in enumerate(
        games
    ):

        row_fill = (
            "#FFFFFF"
            if (
                index % 2
                == 0
            )
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

        matchup_text = (
            fit_text(
                draw,
                game[
                    "opponent"
                ],
                row_font,
                matchup_width,
            )
        )

        date_text = (
            fit_text(
                draw,
                game[
                    "date"
                ],
                row_font,
                date_width,
            )
        )

        week_text = str(
            game[
                "week"
            ]
        )

        week_bbox = (
            draw.textbbox(
                (0, 0),
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

        if (
            game[
                "opponent"
            ]
            != "BYE"
        ):

            logo_image = (
                fetch_logo_image(
                    game.get(
                        "logo_url",
                        "",
                    ),
                    logo_size,
                    logo_cache,
                )
            )

            if logo_image is not None:

                logo_y = (
                    y
                    + (
                        row_h
                        - logo_size
                    )
                    // 2
                )

                background.paste(
                    logo_image,
                    (
                        logo_x,
                        logo_y,
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
                (0, 0),
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
                (0, 0),
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
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a regular-season "
            "schedule poster using "
            "ESPN Core weekly events."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
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

        team_map = (
            build_team_map(
                args.year
            )
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
