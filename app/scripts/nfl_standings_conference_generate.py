#!/usr/bin/env python3

import argparse
import json
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
# ESPN CORE API REQUEST
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
        f"HTTP {response.status_code} "
        f"for {response.url}"
    )

    response.raise_for_status()

    return response.json()


def get_json(season: int) -> Dict[str, dict]:
    """
    Fetch the documented conference standings endpoints.

    AFC group = 8
    NFC group = 7

    IMPORTANT:
    We intentionally stop at /standings here.

    We are NOT using /standings/0.
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
            f"standings"
        )

        print()
        print("=" * 80)
        print(
            f"FETCHING {conference} STANDINGS"
        )
        print("=" * 80)

        print(url)

        payload = core_get_json(url)

        data[conference] = payload

        debug_payload_structure(
            conference,
            payload,
        )

    return data


# ============================================================
# DEBUGGING
# ============================================================

def debug_payload_structure(
    conference: str,
    payload: Any,
) -> None:
    """
    Print enough of the real ESPN response structure to GitHub Actions
    so we can build the parser against the actual JSON.

    This avoids dumping an enormous JSON response while still showing:
      - root type
      - root keys
      - list lengths
      - item keys
      - any $ref values
      - shallow nested structure
      - a truncated JSON preview
    """

    print()
    print("#" * 80)
    print(
        f"{conference} RESPONSE STRUCTURE"
    )
    print("#" * 80)

    print(
        f"Root Python type: "
        f"{type(payload).__name__}"
    )

    if isinstance(payload, dict):

        print(
            f"Root keys: "
            f"{list(payload.keys())}"
        )

        for key, value in payload.items():

            print()
            print(
                f"ROOT KEY: {key}"
            )

            print(
                f"  type: "
                f"{type(value).__name__}"
            )

            if isinstance(value, dict):

                print(
                    f"  keys: "
                    f"{list(value.keys())}"
                )

                if "$ref" in value:
                    print(
                        f"  $ref: "
                        f"{value.get('$ref')}"
                    )

            elif isinstance(value, list):

                print(
                    f"  list length: "
                    f"{len(value)}"
                )

                for index, item in enumerate(
                    value[:5]
                ):

                    print(
                        f"  item[{index}] type: "
                        f"{type(item).__name__}"
                    )

                    if isinstance(item, dict):

                        print(
                            f"  item[{index}] keys: "
                            f"{list(item.keys())}"
                        )

                        if "$ref" in item:

                            print(
                                f"  item[{index}] $ref: "
                                f"{item.get('$ref')}"
                            )

                        for item_key, item_value in item.items():

                            if isinstance(
                                item_value,
                                dict,
                            ):

                                print(
                                    f"    {item_key}: "
                                    f"dict keys="
                                    f"{list(item_value.keys())}"
                                )

                                if "$ref" in item_value:

                                    print(
                                        f"    {item_key}.$ref="
                                        f"{item_value.get('$ref')}"
                                    )

                            elif isinstance(
                                item_value,
                                list,
                            ):

                                print(
                                    f"    {item_key}: "
                                    f"list length="
                                    f"{len(item_value)}"
                                )

                            else:

                                value_text = str(
                                    item_value
                                )

                                if len(
                                    value_text
                                ) > 250:
                                    value_text = (
                                        value_text[:250]
                                        + "..."
                                    )

                                print(
                                    f"    {item_key}: "
                                    f"{value_text}"
                                )

            else:

                value_text = str(
                    value
                )

                if len(
                    value_text
                ) > 500:
                    value_text = (
                        value_text[:500]
                        + "..."
                    )

                print(
                    f"  value: "
                    f"{value_text}"
                )

    elif isinstance(
        payload,
        list,
    ):

        print(
            f"Root list length: "
            f"{len(payload)}"
        )

        for index, item in enumerate(
            payload[:5]
        ):

            print(
                f"item[{index}] type: "
                f"{type(item).__name__}"
            )

            if isinstance(
                item,
                dict,
            ):

                print(
                    f"item[{index}] keys: "
                    f"{list(item.keys())}"
                )

    print()
    print(
        f"{conference} TRUNCATED JSON PREVIEW:"
    )

    try:

        pretty = json.dumps(
            payload,
            indent=2,
        )

        # Keep GitHub logs readable.
        max_chars = 12000

        if len(pretty) > max_chars:

            pretty = (
                pretty[:max_chars]
                + "\n... [TRUNCATED] ..."
            )

        print(pretty)

    except Exception as exc:

        print(
            f"Could not serialize payload: "
            f"{exc}"
        )

    print()
    print(
        f"END {conference} RESPONSE STRUCTURE"
    )

    print("#" * 80)
    print()


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


# ============================================================
# STAT PARSING
# ============================================================

def extract_stats(
    entry: dict,
) -> Tuple[int, int, int]:

    wins = 0
    losses = 0
    ties = 0

    stats = (
        entry.get("stats")
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
        entry.get("stats")
        or []
    )

    for stat in stats:

        if not isinstance(
            stat,
            dict,
        ):
            continue

        name = (
            str(
                stat.get("name")
                or ""
            )
            .lower()
            .replace(
                "_",
                "",
            )
        )

        display_name = (
            str(
                stat.get("displayName")
                or ""
            )
            .lower()
            .replace(
                " ",
                "",
            )
        )

        abbreviation = (
            str(
                stat.get("abbreviation")
                or ""
            )
            .lower()
        )

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
# GENERIC ENTRY DISCOVERY
# ============================================================

def find_entries(
    obj: Any,
    depth: int = 0,
) -> List[dict]:
    """
    Generic recursive search.

    For this diagnostic version we keep this parser,
    but the debug output above is what we care about.

    Once we see ESPN's real JSON layout, we can replace
    this with an exact parser.
    """

    if depth > 20:
        return []

    if isinstance(
        obj,
        dict,
    ):

        entries = obj.get(
            "entries"
        )

        if isinstance(
            entries,
            list,
        ):

            if entries:

                print(
                    f"Parser found an 'entries' list "
                    f"with {len(entries)} items "
                    f"at recursion depth {depth}"
                )

                first = entries[0]

                if isinstance(
                    first,
                    dict,
                ):

                    print(
                        "First entry keys: "
                        f"{list(first.keys())}"
                    )

                return [
                    item
                    for item in entries
                    if isinstance(
                        item,
                        dict,
                    )
                ]

        for key, value in obj.items():

            found = find_entries(
                value,
                depth + 1,
            )

            if found:
                return found

    elif isinstance(
        obj,
        list,
    ):

        for item in obj:

            found = find_entries(
                item,
                depth + 1,
            )

            if found:
                return found

    return []


# ============================================================
# TEAM RESOLUTION
# ============================================================

def team_id_from_ref(
    ref: str,
) -> str:

    if not ref:
        return ""

    clean = ref.rstrip(
        "/"
    )

    final_piece = (
        clean
        .split("/")[-1]
    )

    if final_piece.isdigit():
        return final_piece

    return ""


def resolve_team(
    team_obj: Any,
) -> Tuple[str, str]:

    if not isinstance(
        team_obj,
        dict,
    ):

        return (
            "",
            "",
        )

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

        team_id = team_id_from_ref(
            ref
        )

    if team_name:

        return (
            team_id,
            team_name,
        )

    if ref:

        print(
            f"Resolving team reference: "
            f"{ref}"
        )

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
# CONFERENCE PARSER
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

        print()
        print(
            "=" * 80
        )

        print(
            f"ATTEMPTING TO PARSE "
            f"{conference}"
        )

        print(
            "=" * 80
        )

        payload = (
            data.get(
                conference
            )
            or {}
        )

        entries = find_entries(
            payload
        )

        print(
            f"{conference}: "
            f"generic parser found "
            f"{len(entries)} entries"
        )

        rows: List[
            TeamRow
        ] = []

        for index, entry in enumerate(
            entries
        ):

            print(
                f"{conference} entry "
                f"{index + 1} keys: "
                f"{list(entry.keys())}"
            )

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
                    f"Skipping entry "
                    f"{index + 1}: "
                    f"could not resolve team"
                )

                continue

            wins, losses, ties = (
                extract_stats(
                    entry
                )
            )

            seed = extract_espn_seed(
                entry
            )

            rows.append(
                TeamRow(
                    team_id=team_id,
                    team_name=team_name,
                    division=hardcoded_div(
                        team_name
                    ),
                    w=wins,
                    l=losses,
                    t=ties,
                    espn_seed=seed,
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

        seeded = [
            row
            for row in unique_rows
            if (
                row.espn_seed
                is not None
                and row.espn_seed > 0
            )
        ]

        if unique_rows:

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
                len(seeded)
                >= threshold
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


# ============================================================
# POSTER
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

    # Gradient
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
        f"NFL STANDINGS {season}"
    )

    title_font = get_font(
        68,
        bold=True,
    )

    while (
        draw.textlength(
            title,
            font=title_font,
        )
        > (
            right
            - left
            - 60
        )
        and title_font.size > 48
    ):

        title_font = get_font(
            title_font.size - 1,
            bold=True,
        )

    title_width = draw.textlength(
        title,
        font=title_font,
    )

    draw.text(
        (
            (
                width
                - title_width
            )
            / 2,
            y + 37,
        ),
        title,
        fill=text_color,
        font=title_font,
    )

    y += (
        top_height
        + 24
    )

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

        current_y = rows_top

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
                str(row.w),
                str(row.l),
                str(row.t),
            ]

            x = table_left

            for column_index, value in enumerate(
                values
            ):

                if column_index == 0:

                    color = (
                        gold
                        if index < 7
                        else accent
                    )

                    draw.text(
                        (
                            x + 14,
                            current_y
                            + (
                                row_height
                                - 28
                            )
                            / 2
                            - 2,
                        ),
                        value,
                        fill=color,
                        font=seed_font,
                    )

                elif column_index == 1:

                    team_font = get_font(
                        28
                    )

                    while (
                        draw.textlength(
                            value,
                            font=team_font,
                        )
                        > (
                            column_widths[
                                column_index
                            ]
                            - 24
                        )
                        and team_font.size > 18
                    ):

                        team_font = get_font(
                            team_font.size - 1
                        )

                    draw.text(
                        (
                            x + 12,
                            current_y
                            + (
                                row_height
                                - team_font.size
                            )
                            / 2
                            - 2,
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

                    draw.text(
                        (
                            x
                            + column_widths[
                                column_index
                            ]
                            - 12
                            - value_width,
                            current_y
                            + (
                                row_height
                                - 28
                            )
                            / 2
                            - 2,
                        ),
                        value,
                        fill=text_color,
                        font=stat_font,
                    )

                x += column_widths[
                    column_index
                ]

            current_y += (
                row_height
                + row_gap
            )

    draw_section(
        y,
        "AFC",
        conferences.get(
            "AFC",
            [],
        ),
    )

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
# MAIN GENERATION
# ============================================================

def generate_standings_conference_png(
    season: int,
    out_path: str,
) -> str:

    print()
    print(
        "=" * 80
    )

    print(
        f"GENERATING STANDINGS "
        f"FOR SEASON {season}"
    )

    print(
        "=" * 80
    )

    data = get_json(
        season
    )

    print()
    print(
        "=" * 80
    )

    print(
        "API RESPONSES RECEIVED. "
        "NOW ATTEMPTING GENERIC PARSER."
    )

    print(
        "=" * 80
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
    print(
        "=" * 80
    )

    print(
        "FINAL STANDINGS COUNTS"
    )

    print(
        f"AFC = {afc_count}"
    )

    print(
        f"NFC = {nfc_count}"
    )

    print(
        "=" * 80
    )

    if (
        afc_count == 0
        or nfc_count == 0
    ):

        raise RuntimeError(
            "Diagnostic standings run completed, "
            "but the current generic parser "
            "did not find both conferences. "
            "Use the JSON structure printed above "
            "to build the exact parser. "
            f"AFC={afc_count}, NFC={nfc_count}."
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
