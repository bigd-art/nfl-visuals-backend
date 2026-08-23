#!/usr/bin/env python3

import argparse
import io
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

CORE_API_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)

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
# HTTP HELPERS
# ============================================================

def normalize_ref(url: str) -> str:
    """
    ESPN Core frequently returns $ref URLs as http://.
    Always convert them to https:// before requesting.
    """

    url = str(url or "").strip()

    if url.startswith("http://"):
        return "https://" + url[len("http://"):]

    return url


def get_json(url: str) -> dict:
    """
    Fetch JSON from ESPN Core API.
    """

    url = normalize_ref(url)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    print(
        f"HTTP {response.status_code}: "
        f"{response.url}"
    )

    response.raise_for_status()

    return response.json()


def resolve_ref(
    obj: Any,
) -> Any:
    """
    If an ESPN object is only a $ref, fetch the referenced object.

    If it is already expanded, return it unchanged.
    """

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

    # If this object already contains useful expanded data,
    # do not make another request.
    useful_keys = {
        "id",
        "abbreviation",
        "displayName",
        "name",
        "competitors",
        "date",
        "stats",
    }

    if any(
        key in obj
        for key in useful_keys
        if key != "$ref"
    ):
        return obj

    return get_json(
        ref
    )


# ============================================================
# SAFE ACCESS
# ============================================================

def safe_get(
    data: dict,
    *keys,
    default="",
):
    current = data

    for key in keys:

        if not isinstance(
            current,
            dict,
        ):
            return default

        current = current.get(
            key
        )

        if current is None:
            return default

    return current


# ============================================================
# TEAM HELPERS
# ============================================================

def get_team_logo(
    team: dict,
) -> str:
    """
    Extract a logo URL from a Core API team object.
    """

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


def get_team_abbreviation(
    team: dict,
) -> str:
    return str(
        team.get("abbreviation")
        or team.get("shortDisplayName")
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
    year: int,
) -> Dict[str, dict]:
    """
    Build the full 32-team map from ESPN Core API.

    Endpoint:
        /seasons/{year}/teams?limit=50
    """

    url = (
        f"{CORE_API_BASE}/"
        f"seasons/{year}/teams"
        f"?limit=50"
    )

    print()
    print("=" * 80)
    print(
        f"FETCHING TEAM LIST FOR {year}"
    )
    print("=" * 80)

    data = get_json(
        url
    )

    items = (
        data.get("items")
        or []
    )

    print(
        f"Team list contains "
        f"{len(items)} items"
    )

    output: Dict[
        str,
        dict,
    ] = {}

    for index, item in enumerate(
        items,
        start=1,
    ):

        try:

            if not isinstance(
                item,
                dict,
            ):
                continue

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
                not abbreviation
                or not team_id
            ):

                print(
                    f"Skipping team item "
                    f"{index}: missing "
                    f"abbreviation or id"
                )

                continue

            display_name = (
                get_team_name(
                    team,
                    abbreviation,
                )
            )

            logo_url = (
                get_team_logo(
                    team
                )
            )

            output[
                abbreviation
            ] = {
                "id": team_id,
                "display_name": display_name,
                "logo": logo_url,
            }

            print(
                f"Team: "
                f"{abbreviation} | "
                f"id={team_id} | "
                f"{display_name}"
            )

        except Exception as exc:

            print(
                f"WARNING: Could not "
                f"resolve team item "
                f"{index}: {exc}"
            )

    print()
    print(
        f"Built team map with "
        f"{len(output)} teams"
    )

    if len(output) < 32:

        raise RuntimeError(
            f"Expected at least 32 teams "
            f"from ESPN Core API, "
            f"received {len(output)}."
        )

    return output


# ============================================================
# TEAM EVENT LIST
# ============================================================

def get_team_events(
    year: int,
    team_id: str,
) -> List[dict]:
    """
    Fetch all events associated with a team for a season.

    Endpoint:
        /seasons/{year}/teams/{team_id}/events
    """

    url = (
        f"{CORE_API_BASE}/"
        f"seasons/{year}/"
        f"teams/{team_id}/events"
        f"?limit=100"
    )

    print()
    print("=" * 80)
    print(
        f"FETCHING TEAM EVENTS "
        f"FOR TEAM ID {team_id}"
    )
    print("=" * 80)

    payload = get_json(
        url
    )

    items = (
        payload.get("items")
        or []
    )

    print(
        f"Team events endpoint "
        f"returned {len(items)} items"
    )

    events: List[
        dict
    ] = []

    for index, item in enumerate(
        items,
        start=1,
    ):

        try:

            if not isinstance(
                item,
                dict,
            ):
                continue

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
                f"WARNING: Could not "
                f"resolve event "
                f"{index}: {exc}"
            )

    print(
        f"Resolved "
        f"{len(events)} events"
    )

    return events


# ============================================================
# EVENT / COMPETITION RESOLUTION
# ============================================================

def resolve_competition(
    competition_obj: Any,
) -> Optional[dict]:

    if not isinstance(
        competition_obj,
        dict,
    ):
        return None

    competitors = (
        competition_obj.get(
            "competitors"
        )
    )

    if isinstance(
        competitors,
        list,
    ):
        return competition_obj

    ref = str(
        competition_obj.get("$ref")
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
                f"reference failed: {exc}"
            )

    return competition_obj


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

    return resolve_competition(
        competitions[0]
    )


# ============================================================
# COMPETITOR / TEAM RESOLUTION
# ============================================================

def resolve_competitor(
    competitor: dict,
) -> dict:
    """
    Resolve the competitor's team object if it is represented by $ref.
    """

    if not isinstance(
        competitor,
        dict,
    ):
        return {}

    team_obj = (
        competitor.get("team")
        or {}
    )

    if not isinstance(
        team_obj,
        dict,
    ):
        return competitor

    abbreviation = (
        get_team_abbreviation(
            team_obj
        )
    )

    if abbreviation:
        return competitor

    ref = str(
        team_obj.get("$ref")
        or ""
    ).strip()

    if not ref:
        return competitor

    try:

        full_team = get_json(
            ref
        )

        updated = dict(
            competitor
        )

        updated[
            "team"
        ] = full_team

        return updated

    except Exception as exc:

        print(
            f"WARNING: team reference "
            f"failed: {exc}"
        )

        return competitor


def extract_team_entry(
    competition: dict,
    team_abbr: str,
) -> Optional[dict]:

    competitors = (
        competition.get(
            "competitors"
        )
        or []
    )

    for raw_competitor in competitors:

        competitor = (
            resolve_competitor(
                raw_competitor
            )
        )

        abbreviation = (
            get_team_abbreviation(
                competitor.get(
                    "team"
                )
                or {}
            )
        )

        if abbreviation == team_abbr:

            return competitor

    return None


def extract_opponent_entry(
    competition: dict,
    team_abbr: str,
) -> Optional[dict]:

    competitors = (
        competition.get(
            "competitors"
        )
        or []
    )

    for raw_competitor in competitors:

        competitor = (
            resolve_competitor(
                raw_competitor
            )
        )

        abbreviation = (
            get_team_abbreviation(
                competitor.get(
                    "team"
                )
                or {}
            )
        )

        if (
            abbreviation
            and abbreviation
            != team_abbr
        ):

            return competitor

    return None


# ============================================================
# SEASON TYPE
# ============================================================

def extract_season_type(
    event: dict,
    competition: dict,
) -> Optional[int]:
    """
    Try several possible Core API locations for season type.

    Regular season = 2.
    """

    candidates = [
        safe_get(
            event,
            "season",
            "type",
            default=None,
        ),
        safe_get(
            event,
            "seasonType",
            "id",
            default=None,
        ),
        safe_get(
            event,
            "seasonType",
            "type",
            default=None,
        ),
        safe_get(
            competition,
            "season",
            "type",
            default=None,
        ),
        safe_get(
            competition,
            "type",
            "id",
            default=None,
        ),
    ]

    for value in candidates:

        if value is None:
            continue

        try:
            return int(
                value
            )
        except Exception:
            continue

    return None


# ============================================================
# WEEK RESOLUTION
# ============================================================

def resolve_week_object(
    week_obj: Any,
) -> Any:

    if not isinstance(
        week_obj,
        dict,
    ):
        return week_obj

    if (
        "number"
        in week_obj
    ):
        return week_obj

    ref = str(
        week_obj.get("$ref")
        or ""
    ).strip()

    if ref:

        try:
            return get_json(
                ref
            )
        except Exception:
            pass

    return week_obj


def extract_week_number(
    event: dict,
    competition: dict,
) -> Optional[int]:

    possible_week_objects = [
        event.get("week"),
        competition.get("week"),
    ]

    for week_obj in possible_week_objects:

        week_obj = (
            resolve_week_object(
                week_obj
            )
        )

        if isinstance(
            week_obj,
            dict,
        ):

            value = (
                week_obj.get(
                    "number"
                )
                or week_obj.get(
                    "value"
                )
                or week_obj.get(
                    "id"
                )
            )

            try:

                number = int(
                    value
                )

                if (
                    1
                    <= number
                    <= 18
                ):
                    return number

            except Exception:
                pass

        else:

            try:

                number = int(
                    week_obj
                )

                if (
                    1
                    <= number
                    <= 18
                ):
                    return number

            except Exception:
                pass

    return None


# ============================================================
# LOGO
# ============================================================

def extract_logo_url(
    team_obj: dict,
    abbreviation: str,
    team_map: Dict[str, dict],
) -> str:

    logo_url = (
        get_team_logo(
            team_obj
        )
    )

    if logo_url:
        return logo_url

    return str(
        team_map.get(
            abbreviation,
            {},
        ).get(
            "logo",
            "",
        )
    ).strip()


# ============================================================
# SCHEDULE PARSING
# ============================================================

def parse_games_only(
    team_abbr: str,
    events: List[dict],
    team_map: Dict[str, dict],
) -> Dict[int, dict]:
    """
    Convert Core API events into Week 1-18 schedule entries.
    """

    by_week: Dict[
        int,
        dict,
    ] = {}

    for index, event in enumerate(
        events,
        start=1,
    ):

        try:

            competition = (
                extract_competition(
                    event
                )
            )

            if not competition:

                print(
                    f"Skipping event {index}: "
                    f"no competition"
                )

                continue

            # ----------------------------------------------
            # REGULAR SEASON ONLY
            # ----------------------------------------------

            season_type = (
                extract_season_type(
                    event,
                    competition,
                )
            )

            if (
                season_type is not None
                and season_type != 2
            ):

                print(
                    f"Skipping event {index}: "
                    f"season type "
                    f"{season_type}"
                )

                continue

            # ----------------------------------------------
            # WEEK
            # ----------------------------------------------

            week_num = (
                extract_week_number(
                    event,
                    competition,
                )
            )

            if week_num is None:

                print(
                    f"Skipping event {index}: "
                    f"could not determine "
                    f"regular-season week"
                )

                continue

            # ----------------------------------------------
            # COMPETITORS
            # ----------------------------------------------

            team_entry = (
                extract_team_entry(
                    competition,
                    team_abbr,
                )
            )

            opponent_entry = (
                extract_opponent_entry(
                    competition,
                    team_abbr,
                )
            )

            if (
                not team_entry
                or not opponent_entry
            ):

                print(
                    f"Skipping week "
                    f"{week_num}: "
                    f"could not determine "
                    f"both teams"
                )

                continue

            # ----------------------------------------------
            # HOME / AWAY
            # ----------------------------------------------

            home_away = str(
                team_entry.get(
                    "homeAway"
                )
                or ""
            ).strip().lower()

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

                print(
                    f"Skipping week "
                    f"{week_num}: "
                    f"opponent abbreviation "
                    f"missing"
                )

                continue

            if home_away == "away":

                opponent_display = (
                    f"@ {opponent_abbr}"
                )

            else:

                opponent_display = (
                    f"vs {opponent_abbr}"
                )

            # ----------------------------------------------
            # DATE
            # ----------------------------------------------

            date_iso = str(
                event.get("date")
                or competition.get(
                    "date"
                )
                or ""
            ).strip()

            # ----------------------------------------------
            # LOGO
            # ----------------------------------------------

            logo_url = (
                extract_logo_url(
                    opponent_team,
                    opponent_abbr,
                    team_map,
                )
            )

            by_week[
                week_num
            ] = {
                "week": week_num,
                "opponent": opponent_display,
                "date": date_iso,
                "logo_url": logo_url,
            }

            print(
                f"Week {week_num}: "
                f"{opponent_display} | "
                f"{date_iso}"
            )

        except Exception as exc:

            print(
                f"WARNING: Could not parse "
                f"event {index}: {exc}"
            )

    return by_week


# ============================================================
# DATE FORMATTING
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

def build_full_18_week_schedule(
    team_abbr: str,
    events: List[dict],
    team_map: Dict[str, dict],
) -> List[dict]:

    games_by_week = (
        parse_games_only(
            team_abbr,
            events,
            team_map,
        )
    )

    print()
    print(
        f"Found games for "
        f"{len(games_by_week)} "
        f"regular-season weeks"
    )

    full_schedule: List[
        dict
    ] = []

    for week in range(
        1,
        19,
    ):

        if week in games_by_week:

            game = (
                games_by_week[
                    week
                ]
            )

            full_schedule.append(
                {
                    "week": week,
                    "opponent": game[
                        "opponent"
                    ],
                    "date": (
                        format_date_eastern(
                            game[
                                "date"
                            ]
                        )
                    ),
                    "logo_url": game[
                        "logo_url"
                    ],
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

    return full_schedule


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
        (x, y),
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

    shortened = (
        text
    )

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

        return cache[
            url
        ].copy()

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent":
                    USER_AGENT
            },
            timeout=30,
        )

        response.raise_for_status()

        image = (
            Image.open(
                io.BytesIO(
                    response.content
                )
            )
            .convert("RGBA")
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
            f"WARNING: logo fetch "
            f"failed: {exc}"
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

    primary, secondary = (
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

    # --------------------------------------------------------
    # TOP / BOTTOM BARS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

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
        f"{team_name} "
        f"{year} Schedule",
        subtitle_font,
        236,
        width,
        "white",
    )

    # --------------------------------------------------------
    # TABLE HEADER
    # --------------------------------------------------------

    left = 62

    right = (
        width - 62
    )

    top = 334

    header_height = 126
    row_height = 120
    row_gap = 7

    draw.rounded_rectangle(
        [
            left,
            top,
            right,
            top
            + header_height,
        ],
        radius=28,
        fill=secondary,
    )

    week_x = (
        left + 30
    )

    opponent_label_x = (
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
            opponent_label_x,
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
        + header_height
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

    # --------------------------------------------------------
    # ROWS
    # --------------------------------------------------------

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
                y + row_height,
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

        # ----------------------------------------------------
        # WEEK
        # ----------------------------------------------------

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
                    row_height
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
        # OPPONENT LOGO
        # ----------------------------------------------------

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

            if (
                logo_image
                is not None
            ):

                logo_y = (
                    y
                    + (
                        row_height
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

                dash_bbox = (
                    draw.textbbox(
                        (
                            0,
                            0,
                        ),
                        "-",
                        font=row_font,
                    )
                )

                dash_height = (
                    dash_bbox[3]
                    - dash_bbox[1]
                )

                draw.text(
                    (
                        logo_x + 22,
                        y
                        + (
                            row_height
                            - dash_height
                        )
                        // 2
                        - 2,
                    ),
                    "-",
                    font=row_font,
                    fill=text_fill,
                )

        else:

            dash_bbox = (
                draw.textbbox(
                    (
                        0,
                        0,
                    ),
                    "-",
                    font=row_font,
                )
            )

            dash_height = (
                dash_bbox[3]
                - dash_bbox[1]
            )

            draw.text(
                (
                    logo_x + 22,
                    y
                    + (
                        row_height
                        - dash_height
                    )
                    // 2
                    - 2,
                ),
                "-",
                font=row_font,
                fill=text_fill,
            )

        # ----------------------------------------------------
        # MATCHUP
        # ----------------------------------------------------

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
                    row_height
                    - matchup_height
                )
                // 2
                - 4,
            ),
            matchup_text,
            font=row_font,
            fill=text_fill,
        )

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

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
                    row_height
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
            row_height
            + row_gap
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

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
            "Create a schedule poster "
            "for one team's regular "
            "season using ESPN Core API."
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
        "--team",
        type=str,
        required=True,
        help=(
            "Team abbreviation, "
            "e.g. PHI, DAL, WSH"
        ),
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

        # ----------------------------------------------------
        # TEAM MAP
        # ----------------------------------------------------

        team_map = (
            build_team_map(
                args.year
            )
        )

        if (
            team_abbr
            not in team_map
        ):

            valid = ", ".join(
                sorted(
                    team_map.keys()
                )
            )

            raise RuntimeError(
                f"Invalid team "
                f"'{team_abbr}'. "
                f"Valid teams: "
                f"{valid}"
            )

        team_id = (
            team_map[
                team_abbr
            ][
                "id"
            ]
        )

        team_name = (
            team_map[
                team_abbr
            ][
                "display_name"
            ]
        )

        print()
        print(
            f"Selected team: "
            f"{team_abbr}"
        )

        print(
            f"Team ID: "
            f"{team_id}"
        )

        print(
            f"Team name: "
            f"{team_name}"
        )

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        events = (
            get_team_events(
                args.year,
                team_id,
            )
        )

        if not events:

            raise RuntimeError(
                "ESPN Core API returned "
                "no team events."
            )

        # ----------------------------------------------------
        # SCHEDULE
        # ----------------------------------------------------

        games = (
            build_full_18_week_schedule(
                team_abbr,
                events,
                team_map,
            )
        )

        actual_games = [
            game
            for game in games
            if (
                game[
                    "opponent"
                ]
                != "BYE"
            )
        ]

        print()
        print(
            f"Final schedule contains "
            f"{len(actual_games)} games "
            f"and "
            f"{18 - len(actual_games)} "
            f"BYE/empty weeks."
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

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
        print("=" * 80)

        print(
            "✅ SCHEDULE COMPLETE"
        )

        print("=" * 80)

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
