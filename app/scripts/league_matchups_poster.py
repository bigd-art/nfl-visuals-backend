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


TEAM_SLUGS = {
    "ARI": "ari",
    "ATL": "atl",
    "BAL": "bal",
    "BUF": "buf",
    "CAR": "car",
    "CHI": "chi",
    "CIN": "cin",
    "CLE": "cle",
    "DAL": "dal",
    "DEN": "den",
    "DET": "det",
    "GB": "gb",
    "HOU": "hou",
    "IND": "ind",
    "JAX": "jax",
    "KC": "kc",
    "LV": "lv",
    "LAC": "lac",
    "LAR": "lar",
    "MIA": "mia",
    "MIN": "min",
    "NE": "ne",
    "NO": "no",
    "NYG": "nyg",
    "NYJ": "nyj",
    "PHI": "phi",
    "PIT": "pit",
    "SF": "sf",
    "SEA": "sea",
    "TB": "tb",
    "TEN": "ten",
    "WSH": "wsh",
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
    """
    Kept under the old function name for compatibility.

    This no longer points to ESPN site.api scoreboard.
    It now returns the ESPN Core weekly events endpoint.
    """

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

    if url.startswith(
        "http://"
    ):

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
# SAFE GET
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

        raw = (
            date_iso
            .replace(
                "Z",
                "+00:00",
            )
        )

        dt = (
            datetime.fromisoformat(
                raw
            )
        )

        if (
            ZoneInfo
            is not None
        ):

            eastern = (
                dt.astimezone(
                    ZoneInfo(
                        "America/New_York"
                    )
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
# CORE REF RESOLUTION
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
        obj.get(
            "$ref"
        )
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
            f"Core reference "
            f"{ref}: {exc}"
        )

        return obj


# ============================================================
# EVENT COLLECTION
# ============================================================

def resolve_events(
    data: dict,
) -> List[dict]:
    """
    Core weekly events endpoint returns:

        {
            "items": [
                {"$ref": "..."},
                ...
            ]
        }

    Resolve each event reference.
    """

    items = (
        data.get(
            "items"
        )
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
                item.get(
                    "$ref"
                )
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
        event.get(
            "competitions"
        )
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
        competition.get(
            "$ref"
        )
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
                f"WARNING: "
                f"competition failed: "
                f"{exc}"
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
        team_obj.get(
            "$ref"
        )
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
    """
    Return (away_abbreviation, home_abbreviation).
    """

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

        team = (
            resolve_team(
                competitor.get(
                    "team"
                )
                or {}
            )
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

            away_team = (
                abbreviation
            )

        elif home_away == "home":

            home_team = (
                abbreviation
            )

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
    """
    Parse ESPN Core weekly event collection into:

        {
            away,
            home,
            date
        }
    """

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
            ) = (
                parse_competitors(
                    competition
                )
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
                event.get(
                    "date"
                )
                or competition.get(
                    "date"
                )
                or ""
            ).strip()

            game = {
                "away": away_team,
                "home": home_team,
                "date": (
                    format_date_eastern(
                        date_iso
                    )
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

            return (
                ImageFont.truetype(
                    path,
                    size,
                )
            )

        except Exception:

            continue

    return (
        ImageFont.load_default()
    )


def draw_centered(
    draw,
    text,
    font,
    y,
    width,
    fill,
):

    bbox = (
        draw.textbbox(
            (
                0,
                0,
            ),
            text,
            font=font,
        )
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

    away = (
        games[0][
            "away"
        ]
    )

    home = (
        games[0][
            "home"
        ]
    )

    away_colors = (
        TEAM_COLORS.get(
            away,
            (
                DEFAULT_PRIMARY,
                DEFAULT_SECONDARY,
            ),
        )
    )

    home_colors = (
        TEAM_COLORS.get(
            home,
            (
                DEFAULT_PRIMARY,
                DEFAULT_SECONDARY,
            ),
        )
    )

    return (
        away_colors[0],
        home_colors[1],
    )


# ============================================================
# TEAM LOGOS
# ============================================================

def fetch_logo(
    team_abbr: str,
    size: int = 64,
) -> Optional[
    Image.Image
]:

    slug = (
        TEAM_SLUGS.get(
            team_abbr
        )
    )

    if not slug:

        return None

    urls = [
        (
            "https://a.espncdn.com/"
            f"i/teamlogos/nfl/500/"
            f"{slug}.png"
        ),
        (
            "https://a.espncdn.com/"
            f"i/teamlogos/nfl/500-dark/"
            f"{slug}.png"
        ),
    ]

    for url in urls:

        try:

            response = (
                requests.get(
                    url,
                    headers=HEADERS,
                    timeout=20,
                )
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

            return image

        except Exception:

            continue

    return None


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

    logo_gap = 12

    side_gap = 22

    away_bbox = (
        draw.textbbox(
            (
                0,
                0,
            ),
            away,
            font=font,
        )
    )

    at_bbox = (
        draw.textbbox(
            (
                0,
                0,
            ),
            "@",
            font=font,
        )
    )

    home_bbox = (
        draw.textbbox(
            (
                0,
                0,
            ),
            home,
            font=font,
        )
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
            + max(
                0,
                (
                    48
                    - away_logo.height
                )
                // 2,
            )
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
            + max(
                0,
                (
                    48
                    - home_logo.height
                )
                // 2,
            )
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
    ) = (
        get_poster_colors(
            games
        )
    )

    bg = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        primary,
    )

    draw = (
        ImageDraw.Draw(
            bg
        )
    )

    title_font = (
        get_font(
            80,
            bold=True,
        )
    )

    subtitle_font = (
        get_font(
            42,
            bold=True,
        )
    )

    header_font = (
        get_font(
            34,
            bold=True,
        )
    )

    row_font = (
        get_font(
            30,
            bold=False,
        )
    )

    row_font_bold = (
        get_font(
            31,
            bold=True,
        )
    )

    date_font = (
        get_font(
            25,
            bold=False,
        )
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
        "NFL LEAGUE MATCHUPS",
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

    right = (
        width - 55
    )

    top = 245

    row_height = 96

    row_gap = 11

    header_height = 122

    draw.rounded_rectangle(
        [
            left,
            top,
            right,
            top
            + header_height,
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
    # LOGO CACHE
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
            game[
                "away"
            ]
        )

        all_teams.add(
            game[
                "home"
            ]
        )

    for team in all_teams:

        logo_cache[
            team
        ] = fetch_logo(
            team,
            size=64,
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
            if (
                index % 2
                == 1
            )
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
                y
                + row_height,
            ],
            radius=18,
            fill=row_fill,
        )

        game_number = (
            fit_text(
                draw,
                str(
                    index
                ),
                row_font_bold,
                90,
            )
        )

        date_text = (
            fit_text(
                draw,
                game[
                    "date"
                ],
                date_font,
                right
                - date_column_x
                - 25,
            )
        )

        draw.text(
            (
                game_column_x
                + 5,
                y + 29,
            ),
            game_number,
            font=row_font_bold,
            fill=text_fill,
        )

        draw_matchup_row(
            bg=bg,
            draw=draw,
            matchup_left=(
                matchup_column_x
            ),
            matchup_width=(
                matchup_column_width
            ),
            matchup_top=(
                y + 20
            ),
            away=game[
                "away"
            ],
            home=game[
                "home"
            ],
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

        if (
            str(exc)
            == "NO_GAMES"
        ):

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

    parser = (
        argparse.ArgumentParser(
            description=(
                "Create a league "
                "matchups poster for "
                "a specific week using "
                "ESPN Core API."
            )
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

    args = (
        parser.parse_args()
    )

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

        sys.exit(
            1
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
