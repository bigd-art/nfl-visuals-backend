#!/usr/bin/env python3

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

CORE_API_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)

SUPABASE_PUBLIC_BASE = (
    "https://rojsabkwywygludonpdf.supabase.co/"
    "storage/v1/object/public/nfl-posters"
)


# ============================================================
# TEAM -> DIVISION
# ============================================================

TEAM_TO_DIV: Dict[str, str] = {
    # AFC East
    "Buffalo Bills": "East",
    "Miami Dolphins": "East",
    "New England Patriots": "East",
    "New York Jets": "East",

    # AFC North
    "Baltimore Ravens": "North",
    "Cincinnati Bengals": "North",
    "Cleveland Browns": "North",
    "Pittsburgh Steelers": "North",

    # AFC South
    "Houston Texans": "South",
    "Indianapolis Colts": "South",
    "Jacksonville Jaguars": "South",
    "Tennessee Titans": "South",

    # AFC West
    "Denver Broncos": "West",
    "Kansas City Chiefs": "West",
    "Las Vegas Raiders": "West",
    "Los Angeles Chargers": "West",

    # NFC East
    "Dallas Cowboys": "East",
    "New York Giants": "East",
    "Philadelphia Eagles": "East",
    "Washington Commanders": "East",

    # NFC North
    "Chicago Bears": "North",
    "Detroit Lions": "North",
    "Green Bay Packers": "North",
    "Minnesota Vikings": "North",

    # NFC South
    "Atlanta Falcons": "South",
    "Carolina Panthers": "South",
    "New Orleans Saints": "South",
    "Tampa Bay Buccaneers": "South",

    # NFC West
    "Arizona Cardinals": "West",
    "Los Angeles Rams": "West",
    "San Francisco 49ers": "West",
    "Seattle Seahawks": "West",
}


def hardcoded_div(team_name: str) -> str:
    return TEAM_TO_DIV.get(
        (team_name or "").strip(),
        "",
    )


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class TeamRow:
    team_id: str
    team_name: str
    division: str
    w: int
    l: int
    t: int
    espn_seed: Optional[int] = None


# ============================================================
# HTTP
# ============================================================

def core_get_json(url: str) -> dict:
    """
    Fetch JSON from ESPN Core API.
    """

    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=25,
    )

    print(
        f"HTTP {response.status_code}: "
        f"{response.url}"
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# ESPN STANDINGS REQUEST
# ============================================================

def get_json(season: int) -> Dict[str, dict]:
    """
    Fetch the actual "overall" standings table for AFC and NFC.

    ESPN conference group IDs:
        AFC = 8
        NFC = 7

    First request:
        /groups/{group}/standings

    That gives us the available standings tables.

    We then select:
        name == "overall"

    and follow its $ref.
    """

    conference_ids = {
        "AFC": 8,
        "NFC": 7,
    }

    results: Dict[str, dict] = {}

    for conference, group_id in conference_ids.items():

        index_url = (
            f"{CORE_API_BASE}/"
            f"seasons/{season}/"
            f"types/2/"
            f"groups/{group_id}/"
            f"standings"
        )

        print()
        print("=" * 80)
        print(
            f"FETCHING {conference} STANDINGS INDEX"
        )
        print("=" * 80)

        index_payload = core_get_json(
            index_url
        )

        items = (
            index_payload.get("items")
            or []
        )

        overall_ref = None

        for item in items:

            if not isinstance(
                item,
                dict,
            ):
                continue

            if (
                str(
                    item.get("name")
                    or ""
                ).strip().lower()
                == "overall"
            ):

                overall_ref = str(
                    item.get("$ref")
                    or ""
                ).strip()

                break

        if not overall_ref:

            raise RuntimeError(
                f"Could not find the overall "
                f"{conference} standings table."
            )

        print(
            f"{conference} overall standings ref:"
        )

        print(
            overall_ref
        )

        print()
        print(
            f"FETCHING ACTUAL "
            f"{conference} STANDINGS"
        )

        standings_payload = core_get_json(
            overall_ref
        )

        results[
            conference
        ] = standings_payload

    return results


# ============================================================
# VALUE HELPERS
# ============================================================

def to_int(value: Any) -> int:
    try:

        return int(
            float(
                str(value).strip()
            )
        )

    except Exception:

        return 0


def team_id_from_ref(
    ref: str,
) -> str:

    if not ref:
        return ""

    clean = (
        ref
        .split("?")[0]
        .rstrip("/")
    )

    final_piece = (
        clean
        .split("/")[-1]
    )

    if final_piece.isdigit():
        return final_piece

    return ""


# ============================================================
# TEAM RESOLUTION
# ============================================================

def resolve_team(
    team_obj: Any,
) -> Tuple[str, str]:

    if not isinstance(
        team_obj,
        dict,
    ):

        return "", ""

    team_id = str(
        team_obj.get("id")
        or ""
    ).strip()

    team_name = str(
        team_obj.get("displayName")
        or team_obj.get("shortDisplayName")
        or team_obj.get("name")
        or ""
    ).strip()

    ref = str(
        team_obj.get("$ref")
        or ""
    ).strip()

    if not team_id and ref:

        team_id = team_id_from_ref(
            ref
        )

    if team_name:

        return (
            team_id,
            team_name,
        )

    if not ref:

        return (
            team_id,
            "",
        )

    team_data = core_get_json(
        ref
    )

    team_id = str(
        team_data.get("id")
        or team_id
    ).strip()

    team_name = str(
        team_data.get("displayName")
        or team_data.get("shortDisplayName")
        or team_data.get("name")
        or ""
    ).strip()

    return (
        team_id,
        team_name,
    )


# ============================================================
# RECORD PARSING
# ============================================================

def get_overall_record(
    standing_entry: dict,
) -> Optional[dict]:
    """
    Each standings row contains:

        "records": [
            {
                "id": "0",
                "name": "overall",
                ...
                "stats": [...]
            },
            ...
        ]

    Return the overall record only.
    """

    records = (
        standing_entry.get("records")
        or []
    )

    for record in records:

        if not isinstance(
            record,
            dict,
        ):
            continue

        record_id = str(
            record.get("id")
            or ""
        ).strip()

        record_name = str(
            record.get("name")
            or ""
        ).strip().lower()

        record_type = str(
            record.get("type")
            or ""
        ).strip().lower()

        if (
            record_id == "0"
            or record_name == "overall"
            or record_type == "total"
        ):

            return record

    return None


def stat_value(
    stats: List[dict],
    names: Tuple[str, ...],
    abbreviations: Tuple[str, ...] = (),
) -> Optional[Any]:

    normalized_names = {
        name.lower()
        for name in names
    }

    normalized_abbreviations = {
        abbreviation.lower()
        for abbreviation in abbreviations
    }

    for stat in stats:

        if not isinstance(
            stat,
            dict,
        ):
            continue

        name = str(
            stat.get("name")
            or ""
        ).strip().lower()

        stat_type = str(
            stat.get("type")
            or ""
        ).strip().lower()

        abbreviation = str(
            stat.get("abbreviation")
            or ""
        ).strip().lower()

        if (
            name in normalized_names
            or stat_type in normalized_names
            or abbreviation in normalized_abbreviations
        ):

            return stat.get(
                "value",
                stat.get("displayValue"),
            )

    return None


def extract_record_stats(
    standing_entry: dict,
) -> Tuple[
    int,
    int,
    int,
    Optional[int],
]:
    """
    Extract W/L/T and playoff seed from the overall record.
    """

    overall_record = get_overall_record(
        standing_entry
    )

    if overall_record is None:

        return (
            0,
            0,
            0,
            None,
        )

    stats = (
        overall_record.get("stats")
        or []
    )

    wins = to_int(
        stat_value(
            stats,
            ("wins",),
            ("w",),
        )
    )

    losses = to_int(
        stat_value(
            stats,
            ("losses",),
            ("l",),
        )
    )

    ties = to_int(
        stat_value(
            stats,
            ("ties",),
            ("t",),
        )
    )

    seed_raw = stat_value(
        stats,
        (
            "playoffseed",
            "seed",
        ),
        (
            "seed",
        ),
    )

    seed = to_int(
        seed_raw
    )

    if seed <= 0:
        seed = None

    return (
        wins,
        losses,
        ties,
        seed,
    )


# ============================================================
# CONFERENCE PARSING
# ============================================================

def parse_conference(
    conference: str,
    payload: dict,
) -> List[TeamRow]:
    """
    IMPORTANT:

    ESPN /standings/0 uses:

        {
            "standings": [
                {
                    "team": {...},
                    "records": [...]
                }
            ]
        }

    It does NOT use an "entries" array.
    """

    standings = (
        payload.get("standings")
        or []
    )

    print()
    print(
        f"{conference}: "
        f"received {len(standings)} "
        f"standings rows"
    )

    rows: List[TeamRow] = []

    for index, entry in enumerate(
        standings
    ):

        if not isinstance(
            entry,
            dict,
        ):

            continue

        team_obj = (
            entry.get("team")
            or {}
        )

        team_id, team_name = (
            resolve_team(
                team_obj
            )
        )

        if not team_name:

            print(
                f"{conference}: "
                f"skipping row "
                f"{index + 1}; "
                f"team could not be resolved"
            )

            continue

        wins, losses, ties, seed = (
            extract_record_stats(
                entry
            )
        )

        division = hardcoded_div(
            team_name
        )

        row = TeamRow(
            team_id=team_id,
            team_name=team_name,
            division=division,
            w=wins,
            l=losses,
            t=ties,
            espn_seed=seed,
        )

        rows.append(
            row
        )

        print(
            f"{conference} "
            f"{team_name}: "
            f"{wins}-{losses}"
            + (
                f"-{ties}"
                if ties
                else ""
            )
            + (
                f" | seed={seed}"
                if seed
                else ""
            )
        )

    seen = set()

    unique_rows: List[
        TeamRow
    ] = []

    for row in rows:

        key = (
            row.team_id
            or row.team_name
        )

        if (
            key
            and key not in seen
        ):

            seen.add(
                key
            )

            unique_rows.append(
                row
            )

    seeded_rows = [
        row
        for row in unique_rows
        if (
            row.espn_seed
            is not None
            and row.espn_seed > 0
        )
    ]

    if unique_rows:

        required_seed_count = max(
            4,
            int(
                len(unique_rows)
                * 0.8
            ),
        )

        if (
            len(seeded_rows)
            >= required_seed_count
        ):

            unique_rows = sorted(
                unique_rows,
                key=lambda row: (
                    row.espn_seed
                    if row.espn_seed
                    is not None
                    else 999
                ),
            )

    return unique_rows


def extract_conferences(
    data: Dict[str, dict],
) -> Dict[str, List[TeamRow]]:

    conferences = {
        "AFC": parse_conference(
            "AFC",
            data.get("AFC") or {},
        ),
        "NFC": parse_conference(
            "NFC",
            data.get("NFC") or {},
        ),
    }

    return conferences


# ============================================================
# FONTS
# ============================================================

def get_font(
    size: int,
    bold: bool = False,
):

    candidates = []

    if bold:

        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )

    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    )

    for path in candidates:

        try:

            return ImageFont.truetype(
                path,
                size=size,
            )

        except Exception:

            pass

    return ImageFont.load_default()


def fit_font(
    draw: ImageDraw.ImageDraw,
    text_value: str,
    max_width: int,
    starting_size: int,
    minimum_size: int = 18,
    bold: bool = False,
):

    size = starting_size

    while size >= minimum_size:

        font = get_font(
            size,
            bold=bold,
        )

        if (
            draw.textlength(
                text_value,
                font=font,
            )
            <= max_width
        ):

            return font

        size -= 1

    return get_font(
        minimum_size,
        bold=bold,
    )


# ============================================================
# POSTER RENDERING
# ============================================================

def render_conference_poster(
    season: int,
    conferences: Dict[
        str,
        List[TeamRow],
    ],
    out_path: str,
):

    width = 1080
    height = 1920

    # --------------------------------------------------------
    # COLORS
    # --------------------------------------------------------

    bg_top = (
        6,
        30,
        88,
    )

    bg_bottom = (
        3,
        10,
        28,
    )

    outer_border = (
        120,
        185,
        255,
    )

    panel = (
        10,
        28,
        72,
    )

    panel_2 = (
        14,
        39,
        96,
    )

    title_bar = (
        23,
        62,
        150,
    )

    title_bar_hi = (
        45,
        100,
        220,
    )

    table_header = (
        15,
        43,
        104,
    )

    row_a = (
        10,
        31,
        78,
    )

    row_b = (
        16,
        40,
        95,
    )

    grid = (
        78,
        132,
        228,
    )

    text_color = (
        245,
        248,
        255,
    )

    muted = (
        192,
        208,
        242,
    )

    accent = (
        154,
        204,
        255,
    )

    gold = (
        255,
        214,
        90,
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        bg_bottom,
    )

    draw = ImageDraw.Draw(
        image
    )

    # --------------------------------------------------------
    # GRADIENT
    # --------------------------------------------------------

    for yy in range(
        height
    ):

        ratio = (
            yy
            / max(
                1,
                height - 1,
            )
        )

        red = int(
            bg_top[0]
            * (1 - ratio)
            + bg_bottom[0]
            * ratio
        )

        green = int(
            bg_top[1]
            * (1 - ratio)
            + bg_bottom[1]
            * ratio
        )

        blue = int(
            bg_top[2]
            * (1 - ratio)
            + bg_bottom[2]
            * ratio
        )

        draw.line(
            (
                0,
                yy,
                width,
                yy,
            ),
            fill=(
                red,
                green,
                blue,
            ),
        )

    # --------------------------------------------------------
    # OUTER BORDER
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            18,
            18,
            width - 18,
            height - 18,
        ),
        radius=34,
        outline=outer_border,
        width=3,
    )

    draw.rounded_rectangle(
        (
            28,
            28,
            width - 28,
            height - 28,
        ),
        radius=30,
        outline=(
            40,
            90,
            190,
        ),
        width=1,
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    left = 38
    right = width - 38
    y = 38

    top_height = 150

    draw.rounded_rectangle(
        (
            left,
            y,
            right,
            y + top_height,
        ),
        radius=28,
        fill=panel,
        outline=outer_border,
        width=2,
    )

    draw.rounded_rectangle(
        (
            left + 10,
            y + 10,
            right - 10,
            y + top_height - 10,
        ),
        radius=24,
        fill=panel_2,
    )

    title = (
        f"STANDINGS {season}"
    )

    title_font = fit_font(
        draw,
        title,
        right - left - 60,
        76,
        52,
        bold=True,
    )

    title_width = (
        draw.textlength(
            title,
            font=title_font,
        )
    )

    title_height = (
        title_font.size
    )

    draw.text(
        (
            (
                width
                - title_width
            )
            / 2,
            y
            + (
                top_height
                - title_height
            )
            / 2
            - 8,
        ),
        title,
        fill=text_color,
        font=title_font,
    )

    y += (
        top_height
        + 24
    )

    # --------------------------------------------------------
    # SECTION DIMENSIONS
    # --------------------------------------------------------

    section_gap = 24
    bottom_margin = 34

    available_height = (
        height
        - y
        - bottom_margin
    )

    section_height = (
        available_height
        - section_gap
    ) // 2

    headers = [
        "#",
        "TEAM",
        "DIV",
        "W",
        "L",
        "T",
    ]

    column_fractions = [
        0.10,
        0.52,
        0.14,
        0.08,
        0.08,
        0.08,
    ]

    section_font = get_font(
        36,
        bold=True,
    )

    header_font = get_font(
        22,
        bold=True,
    )

    seed_font = get_font(
        28,
        bold=True,
    )

    stat_font = get_font(
        28,
    )

    # --------------------------------------------------------
    # CONFERENCE SECTION
    # --------------------------------------------------------

    def draw_section(
        top_y: int,
        section_title: str,
        rows: List[TeamRow],
    ):

        section_bottom = (
            top_y
            + section_height
        )

        draw.rounded_rectangle(
            (
                left,
                top_y,
                right,
                section_bottom,
            ),
            radius=28,
            fill=panel,
            outline=grid,
            width=2,
        )

        bar_margin = 16
        bar_height = 58

        draw.rounded_rectangle(
            (
                left + bar_margin,
                top_y + bar_margin,
                right - bar_margin,
                top_y
                + bar_margin
                + bar_height,
            ),
            radius=18,
            fill=title_bar,
        )

        draw.rounded_rectangle(
            (
                left + bar_margin,
                top_y + bar_margin,
                right - bar_margin,
                top_y
                + bar_margin
                + bar_height // 2,
            ),
            radius=18,
            fill=title_bar_hi,
        )

        section_title_width = (
            draw.textlength(
                section_title,
                font=section_font,
            )
        )

        draw.text(
            (
                (
                    width
                    - section_title_width
                )
                / 2,
                top_y
                + bar_margin
                + 9,
            ),
            section_title,
            fill=text_color,
            font=section_font,
        )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        table_left = (
            left + 16
        )

        table_right = (
            right - 16
        )

        table_width = (
            table_right
            - table_left
        )

        column_widths = [
            int(
                table_width
                * fraction
            )
            for fraction
            in column_fractions
        ]

        column_widths[-1] += (
            table_width
            - sum(
                column_widths
            )
        )

        header_y = (
            top_y
            + bar_margin
            + bar_height
            + 16
        )

        header_height = 46

        draw.rounded_rectangle(
            (
                table_left,
                header_y,
                table_right,
                header_y
                + header_height,
            ),
            radius=14,
            fill=table_header,
        )

        x = table_left

        for index, header in enumerate(
            headers
        ):

            if index in (
                0,
                1,
            ):

                draw.text(
                    (
                        x + 12,
                        header_y + 10,
                    ),
                    header,
                    fill=muted,
                    font=header_font,
                )

            else:

                header_width = (
                    draw.textlength(
                        header,
                        font=header_font,
                    )
                )

                draw.text(
                    (
                        x
                        + column_widths[
                            index
                        ]
                        - 12
                        - header_width,
                        header_y + 10,
                    ),
                    header,
                    fill=muted,
                    font=header_font,
                )

            x += column_widths[
                index
            ]

            if (
                index
                != len(headers) - 1
            ):

                draw.line(
                    (
                        x,
                        header_y + 7,
                        x,
                        header_y
                        + header_height
                        - 7,
                    ),
                    fill=grid,
                    width=1,
                )

        # ----------------------------------------------------
        # ROWS
        # ----------------------------------------------------

        rows_top = (
            header_y
            + header_height
            + 10
        )

        rows_bottom = (
            section_bottom
            - 18
        )

        number_of_rows = max(
            1,
            len(rows),
        )

        row_gap = 6

        usable_height = (
            rows_bottom
            - rows_top
            - row_gap
            * (
                number_of_rows - 1
            )
        )

        row_height = max(
            34,
            usable_height
            // number_of_rows,
        )

        current_y = (
            rows_top
        )

        for index, row in enumerate(
            rows
        ):

            fill_color = (
                row_a
                if index % 2 == 0
                else row_b
            )

            draw.rounded_rectangle(
                (
                    table_left,
                    current_y,
                    table_right,
                    current_y
                    + row_height,
                ),
                radius=14,
                fill=fill_color,
            )

            values = [
                str(
                    index + 1
                ),
                row.team_name,
                row.division,
                str(
                    row.w
                ),
                str(
                    row.l
                ),
                str(
                    row.t
                ),
            ]

            x = (
                table_left
            )

            for column_index, value in enumerate(
                values
            ):

                if (
                    column_index == 0
                ):

                    rank_color = (
                        gold
                        if index < 7
                        else accent
                    )

                    text_y = (
                        current_y
                        + (
                            row_height - 28
                        )
                        / 2
                        - 2
                    )

                    draw.text(
                        (
                            x + 14,
                            text_y,
                        ),
                        value,
                        fill=rank_color,
                        font=seed_font,
                    )

                elif (
                    column_index == 1
                ):

                    team_font = fit_font(
                        draw,
                        value,
                        column_widths[
                            column_index
                        ]
                        - 24,
                        28,
                        18,
                    )

                    text_y = (
                        current_y
                        + (
                            row_height
                            - team_font.size
                        )
                        / 2
                        - 2
                    )

                    draw.text(
                        (
                            x + 12,
                            text_y,
                        ),
                        value,
                        fill=text_color,
                        font=team_font,
                    )

                else:

                    value_width = (
                        draw.textlength(
                            value,
                            font=stat_font,
                        )
                    )

                    text_y = (
                        current_y
                        + (
                            row_height - 28
                        )
                        / 2
                        - 2
                    )

                    draw.text(
                        (
                            x
                            + column_widths[
                                column_index
                            ]
                            - 12
                            - value_width,
                            text_y,
                        ),
                        value,
                        fill=text_color,
                        font=stat_font,
                    )

                x += column_widths[
                    column_index
                ]

                if (
                    column_index
                    != len(values) - 1
                ):

                    draw.line(
                        (
                            x,
                            current_y + 7,
                            x,
                            current_y
                            + row_height
                            - 7,
                        ),
                        fill=grid,
                        width=1,
                    )

            current_y += (
                row_height
                + row_gap
            )

    # --------------------------------------------------------
    # AFC
    # --------------------------------------------------------

    draw_section(
        y,
        "AFC",
        conferences.get(
            "AFC",
            [],
        ),
    )

    # --------------------------------------------------------
    # NFC
    # --------------------------------------------------------

    draw_section(
        y
        + section_height
        + section_gap,
        "NFC",
        conferences.get(
            "NFC",
            [],
        ),
    )

    image.save(
        out_path
    )


# ============================================================
# GENERATE STANDINGS
# ============================================================

def generate_standings_conference_png(
    season: int,
    out_path: str,
) -> str:

    print()
    print("=" * 80)

    print(
        f"GENERATING STANDINGS "
        f"FOR SEASON {season}"
    )

    print("=" * 80)

    data = get_json(
        season
    )

    conferences = (
        extract_conferences(
            data
        )
    )

    afc_count = len(
        conferences.get(
            "AFC",
            [],
        )
    )

    nfc_count = len(
        conferences.get(
            "NFC",
            [],
        )
    )

    print()
    print("=" * 80)

    print(
        "FINAL STANDINGS COUNTS"
    )

    print(
        f"AFC = {afc_count}"
    )

    print(
        f"NFC = {nfc_count}"
    )

    print("=" * 80)

    if (
        afc_count != 16
        or nfc_count != 16
    ):

        raise RuntimeError(
            "Expected 16 teams in each conference, "
            f"but received AFC={afc_count}, "
            f"NFC={nfc_count}."
        )

    render_conference_poster(
        season,
        conferences,
        out_path,
    )

    print(
        f"✅ Standings poster saved: "
        f"{out_path}"
    )

    return out_path


# ============================================================
# PUBLIC URL HELPER
# ============================================================

def generate_and_upload_standings_conference(
    season: int,
) -> str:

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    return (
        f"{SUPABASE_PUBLIC_BASE}/"
        f"standings/current.png"
        f"?v={timestamp}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=(
            "standings_conference.png"
        ),
    )

    args = (
        parser.parse_args()
    )

    generate_standings_conference_png(
        args.season,
        args.out,
    )

    print(
        f"✅ Saved: "
        f"{args.out}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStopped."
        )

        sys.exit(
            130
        )

    except Exception as exc:

        print()
        print(
            f"❌ Error: {exc}"
        )

        raise
