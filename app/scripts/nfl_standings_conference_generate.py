#!/usr/bin/env python3

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

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
# TEAM → DIVISION MAPPING
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
# ESPN CORE API
# ============================================================

def core_get_json(url: str) -> dict:
    """
    Fetch JSON from ESPN's Core API.
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

    response.raise_for_status()

    return response.json()


def get_json(season: int) -> Dict[str, dict]:
    """
    Fetch the actual overall AFC and NFC standings tables.

    ESPN group IDs:

        AFC = 8
        NFC = 7

    standings/0 is the overall standings table.

    Example:

        /seasons/2025/types/2/groups/8/standings/0
    """

    conference_ids = {
        "AFC": 8,
        "NFC": 7,
    }

    data: Dict[str, dict] = {}

    for conference, group_id in conference_ids.items():

        url = (
            f"{CORE_API_BASE}/"
            f"seasons/{season}/"
            f"types/2/"
            f"groups/{group_id}/"
            f"standings/0"
        )

        print(
            f"Fetching {conference}: {url}"
        )

        data[conference] = core_get_json(
            url
        )

    return data


# ============================================================
# BASIC HELPERS
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


def normalize_division_name(
    name: str,
) -> str:

    if not name:
        return ""

    normalized = (
        name
        .strip()
        .lower()
    )

    if "east" in normalized:
        return "East"

    if "north" in normalized:
        return "North"

    if "south" in normalized:
        return "South"

    if "west" in normalized:
        return "West"

    return ""


# ============================================================
# STATS PARSING
# ============================================================

def extract_stats(
    entry: dict,
) -> Tuple[int, int, int]:

    wins = 0
    losses = 0
    ties = 0

    stats = (
        entry.get("stats", [])
        or []
    )

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

        display_name = str(
            stat.get("displayName")
            or ""
        ).strip().lower()

        abbreviation = str(
            stat.get("abbreviation")
            or ""
        ).strip().lower()

        value = stat.get(
            "value",
            stat.get(
                "displayValue"
            ),
        )

        if (
            name == "wins"
            or display_name == "wins"
            or abbreviation == "w"
        ):
            wins = to_int(
                value
            )

        elif (
            name == "losses"
            or display_name == "losses"
            or abbreviation == "l"
        ):
            losses = to_int(
                value
            )

        elif (
            name == "ties"
            or display_name == "ties"
            or abbreviation == "t"
        ):
            ties = to_int(
                value
            )

    return (
        wins,
        losses,
        ties,
    )


def extract_espn_seed(
    entry: dict,
) -> Optional[int]:

    stats = (
        entry.get("stats", [])
        or []
    )

    for stat in stats:

        if not isinstance(
            stat,
            dict,
        ):
            continue

        name = (
            stat.get("name")
            or ""
        ).lower().replace(
            "_",
            "",
        )

        display_name = (
            stat.get("displayName")
            or ""
        ).lower().replace(
            " ",
            "",
        )

        abbreviation = (
            stat.get("abbreviation")
            or ""
        ).lower()

        if (
            name in (
                "seed",
                "playoffseed",
                "rank",
                "conferencerank",
            )
            or display_name in (
                "seed",
                "playoffseed",
                "rank",
                "conferencerank",
            )
            or abbreviation == "seed"
        ):

            value = stat.get(
                "value",
                stat.get(
                    "displayValue"
                ),
            )

            seed = to_int(
                value
            )

            if seed > 0:
                return seed

    return None


# ============================================================
# ENTRY DISCOVERY
# ============================================================

def _find_entries(
    obj: Any,
) -> List[dict]:
    """
    Recursively search ESPN's JSON for the standings entries list.
    """

    if isinstance(
        obj,
        dict,
    ):

        entries = obj.get(
            "entries"
        )

        if (
            isinstance(
                entries,
                list,
            )
            and entries
        ):

            has_standing_rows = any(
                isinstance(
                    item,
                    dict,
                )
                and (
                    "team" in item
                    or "stats" in item
                )
                for item in entries
            )

            if has_standing_rows:

                return [
                    item
                    for item in entries
                    if isinstance(
                        item,
                        dict,
                    )
                ]

        for value in obj.values():

            found = _find_entries(
                value
            )

            if found:
                return found

    elif isinstance(
        obj,
        list,
    ):

        for item in obj:

            found = _find_entries(
                item
            )

            if found:
                return found

    return []


def _overall_entries(
    payload: dict,
) -> List[dict]:
    """
    We directly request standings/0.

    Therefore this payload should already represent the
    overall standings table.

    Simply search it for the team entries.
    """

    return _find_entries(
        payload
    )


# ============================================================
# TEAM RESOLUTION
# ============================================================

def _team_id_from_ref(
    ref: str,
) -> str:

    if not ref:
        return ""

    clean_ref = ref.rstrip(
        "/"
    )

    last_part = (
        clean_ref
        .split("/")[-1]
    )

    if last_part.isdigit():
        return last_part

    return ""


def _resolve_team(
    team_obj: Any,
) -> Tuple[str, str]:
    """
    ESPN Core API may provide either:

    1. A full team object
    2. A $ref pointing to the team

    Resolve either form.
    """

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
        or team_obj.get(
            "shortDisplayName"
        )
        or team_obj.get("name")
        or ""
    ).strip()

    ref = str(
        team_obj.get("$ref")
        or ""
    ).strip()

    if (
        not team_id
        and ref
    ):
        team_id = _team_id_from_ref(
            ref
        )

    if team_name:
        return (
            team_id,
            team_name,
        )

    if ref:

        resolved = core_get_json(
            ref
        )

        team_id = str(
            resolved.get("id")
            or team_id
        ).strip()

        team_name = str(
            resolved.get(
                "displayName"
            )
            or resolved.get(
                "shortDisplayName"
            )
            or resolved.get(
                "name"
            )
            or ""
        ).strip()

    return (
        team_id,
        team_name,
    )


# ============================================================
# CONFERENCE EXTRACTION
# ============================================================

def extract_conferences(
    data: Dict[str, dict],
) -> Dict[str, List[TeamRow]]:

    conferences: Dict[
        str,
        List[TeamRow],
    ] = {
        "AFC": [],
        "NFC": [],
    }

    for conference in (
        "AFC",
        "NFC",
    ):

        payload = (
            data.get(
                conference
            )
            or {}
        )

        entries = _overall_entries(
            payload
        )

        print(
            f"{conference}: "
            f"found {len(entries)} "
            f"raw standings entries"
        )

        rows: List[
            TeamRow
        ] = []

        for entry in entries:

            team_obj = (
                entry.get("team")
                or {}
            )

            team_id, team_name = (
                _resolve_team(
                    team_obj
                )
            )

            if not team_name:
                continue

            wins, losses, ties = (
                extract_stats(
                    entry
                )
            )

            seed = extract_espn_seed(
                entry
            )

            division = hardcoded_div(
                team_name
            )

            rows.append(
                TeamRow(
                    team_id=team_id,
                    team_name=team_name,
                    division=division,
                    w=wins,
                    l=losses,
                    t=ties,
                    espn_seed=seed,
                )
            )

        # --------------------------------------------
        # Remove duplicate teams while preserving
        # original ESPN ordering.
        # --------------------------------------------

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

        # --------------------------------------------
        # If ESPN provides conference seed/rank for
        # most teams, use that ordering.
        # --------------------------------------------

        seeded_rows = [
            row
            for row in unique_rows
            if (
                isinstance(
                    row.espn_seed,
                    int,
                )
                and row.espn_seed > 0
            )
        ]

        if len(
            unique_rows
        ) > 0:

            threshold = max(
                4,
                int(
                    0.8
                    * len(
                        unique_rows
                    )
                ),
            )

            if (
                len(
                    seeded_rows
                )
                >= threshold
            ):

                unique_rows = sorted(
                    unique_rows,
                    key=lambda row: (
                        row.espn_seed
                        if (
                            row.espn_seed
                            is not None
                        )
                        else 999
                    ),
                )

        conferences[
            conference
        ] = unique_rows

        print(
            f"{conference}: "
            f"parsed "
            f"{len(unique_rows)} teams"
        )

    return conferences


# ============================================================
# FONT HELPERS
# ============================================================

def get_font(
    size: int,
) -> ImageFont.FreeTypeFont:

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]

    for path in candidates:

        try:

            return (
                ImageFont.truetype(
                    path,
                    size=size,
                )
            )

        except Exception:
            pass

    return (
        ImageFont.load_default()
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

    def get_font_local(
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

                return (
                    ImageFont.truetype(
                        path,
                        size=size,
                    )
                )

            except Exception:
                pass

        return (
            ImageFont.load_default()
        )

    def fit_font(
        draw: ImageDraw.ImageDraw,
        text_value: str,
        max_width: int,
        start_size: int,
        min_size: int = 18,
        bold: bool = False,
    ):

        current_size = (
            start_size
        )

        while (
            current_size
            >= min_size
        ):

            font = get_font_local(
                current_size,
                bold=bold,
            )

            text_width = (
                draw.textlength(
                    text_value,
                    font=font,
                )
            )

            if (
                text_width
                <= max_width
            ):
                return font

            current_size -= 1

        return get_font_local(
            min_size,
            bold=bold,
        )

    def draw_vertical_gradient(
        draw: ImageDraw.ImageDraw,
        canvas_width: int,
        canvas_height: int,
        top_color,
        bottom_color,
    ):

        for yy in range(
            canvas_height
        ):

            ratio = (
                yy
                / max(
                    1,
                    canvas_height - 1,
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

            draw.line(
                (
                    0,
                    yy,
                    canvas_width,
                    yy,
                ),
                fill=(
                    red,
                    green,
                    blue,
                ),
            )

    # ========================================================
    # COLORS
    # ========================================================

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

    # ========================================================
    # CANVAS
    # ========================================================

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

    draw_vertical_gradient(
        draw,
        width,
        height,
        bg_top,
        bg_bottom,
    )

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

    # ========================================================
    # FONTS
    # ========================================================

    section_font = (
        get_font_local(
            36,
            bold=True,
        )
    )

    header_font = (
        get_font_local(
            22,
            bold=True,
        )
    )

    seed_font = (
        get_font_local(
            28,
            bold=True,
        )
    )

    stat_font = (
        get_font_local(
            28,
            bold=False,
        )
    )

    # ========================================================
    # TOP HEADER
    # ========================================================

    left = 38

    right = (
        width - 38
    )

    y = 38

    top_h = 150

    draw.rounded_rectangle(
        (
            left,
            y,
            right,
            y + top_h,
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
            y + top_h - 10,
        ),
        radius=24,
        fill=panel_2,
    )

    title = (
        f"NFL STANDINGS {season}"
    )

    max_title_width = (
        right
        - left
        - 60
    )

    title_font = fit_font(
        draw,
        title,
        max_title_width,
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
                top_h
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
        top_h
        + 24
    )

    # ========================================================
    # CONFERENCE SECTIONS
    # ========================================================

    section_gap = 24

    bottom_margin = 34

    available_height = (
        height
        - y
        - bottom_margin
    )

    section_height = (
        (
            available_height
            - section_gap
        )
        // 2
    )

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
                + (
                    bar_height // 2
                ),
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

        # ====================================================
        # TABLE
        # ====================================================

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
                        header_y
                        + 10,
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
                != len(
                    headers
                )
                - 1
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

        # ====================================================
        # ROWS
        # ====================================================

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
                if (
                    index % 2 == 0
                )
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

            seed = str(
                index + 1
            )

            division = (
                row.division
                or hardcoded_div(
                    row.team_name
                )
            )

            values = [
                seed,
                row.team_name,
                division,
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

                # --------------------------------------------
                # SEED
                # --------------------------------------------

                if (
                    column_index == 0
                ):

                    text_y = (
                        current_y
                        + (
                            row_height
                            - 28
                        )
                        / 2
                        - 2
                    )

                    seed_color = (
                        gold
                        if index < 7
                        else accent
                    )

                    draw.text(
                        (
                            x + 14,
                            text_y,
                        ),
                        value,
                        fill=seed_color,
                        font=seed_font,
                    )

                # --------------------------------------------
                # TEAM
                # --------------------------------------------

                elif (
                    column_index == 1
                ):

                    max_team_width = (
                        column_widths[
                            column_index
                        ]
                        - 24
                    )

                    team_font = (
                        fit_font(
                            draw,
                            value,
                            max_team_width,
                            28,
                            18,
                            bold=False,
                        )
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

                # --------------------------------------------
                # DIVISION / RECORD
                # --------------------------------------------

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
                            row_height
                            - 28
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
                    != len(
                        values
                    )
                    - 1
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

    # ========================================================
    # DRAW AFC
    # ========================================================

    draw_section(
        y,
        "AFC",
        conferences.get(
            "AFC",
            [],
        ),
    )

    # ========================================================
    # DRAW NFC
    # ========================================================

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

    # ========================================================
    # SAVE
    # ========================================================

    image.save(
        out_path
    )


# ============================================================
# GENERATE POSTER
# ============================================================

def generate_standings_conference_png(
    season: int,
    out_path: str,
) -> str:

    print(
        f"Generating standings "
        f"for season {season}"
    )

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

    print(
        "Final standings counts: "
        f"AFC={afc_count}, "
        f"NFC={nfc_count}"
    )

    if (
        afc_count == 0
        or nfc_count == 0
    ):

        raise RuntimeError(
            "ESPN Core API returned standings, "
            "but the parser could not find "
            "both conferences "
            f"(AFC={afc_count}, "
            f"NFC={nfc_count})."
        )

    render_conference_poster(
        season,
        conferences,
        out_path,
    )

    print(
        f"Standings poster saved: "
        f"{out_path}"
    )

    return out_path


# ============================================================
# EXISTING PUBLIC URL HELPER
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

        print(
            f"❌ Error: {exc}"
        )

        raise
