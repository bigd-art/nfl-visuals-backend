#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import time
from collections import defaultdict
from html.parser import HTMLParser
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG
# ============================================================

SEASON_DEFAULT = 2027

TOP_N = 5

OUTPUT_DIR = "big_board_posters"

DRAFTTEK_PAGE_URL = (
    "https://www.drafttek.com/"
    "{season}-NFL-Draft-Big-Board/"
    "Top-NFL-Draft-Prospects-{season}-Page-{page}.asp"
)

# DraftTek currently publishes the 2027 board across:
# Page 1 = 1-150
# Page 2 = 151-300
# Page 3 = 301-450
DRAFTTEK_PAGES = (
    1,
    2,
    3,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


# ============================================================
# TARGET POSTER GROUPS
#
# Keep exactly the same position groups the app previously
# published from the PFF Big Board.
# ============================================================

TARGET_POSITIONS = (
    "CB",
    "DI",
    "ED",
    "IOL",
    "LB",
    "QB",
    "RB",
    "S",
    "T",
    "TE",
    "WR",
)


# ============================================================
# DRAFTTEK POSITION -> APP POSITION
#
# DraftTek uses more granular position labels.
# Collapse them into the exact same groups our existing
# Big Board posters used.
# ============================================================

DRAFTTEK_POSITION_MAP = {
    # Quarterback
    "QB": "QB",

    # Running back
    "RB": "RB",
    "HB": "RB",
    "TB": "RB",

    # Wide receiver
    "WR": "WR",
    "WRS": "WR",
    "SWR": "WR",

    # Tight end
    "TE": "TE",

    # Offensive tackle
    "OT": "T",
    "T": "T",
    "LT": "T",
    "RT": "T",

    # Interior offensive line
    "OG": "IOL",
    "G": "IOL",
    "LG": "IOL",
    "RG": "IOL",
    "OC": "IOL",
    "C": "IOL",
    "IOL": "IOL",
    "OL": "IOL",

    # Edge defender
    "EDGE": "ED",
    "DE": "ED",
    "ED": "ED",

    # Interior defensive line
    "DL1T": "DI",
    "DL3T": "DI",
    "DL5T": "DI",
    "DT": "DI",
    "NT": "DI",
    "DL": "DI",
    "DI": "DI",

    # Linebacker
    "LB": "LB",
    "ILB": "LB",
    "OLB": "LB",
    "MLB": "LB",

    # Cornerback
    "CB": "CB",
    "CBN": "CB",
    "NCB": "CB",

    # Safety
    "S": "S",
    "FS": "S",
    "SS": "S",
}


# ============================================================
# POSTER DIMENSIONS
# ============================================================

POSTER_WIDTH = 1450
HEADER_HEIGHT = 280
ROW_HEIGHT = 190
BOTTOM_PADDING = 60
MARGIN = 44


# ============================================================
# OUTPUT
# ============================================================

def ensure_output_dir() -> None:

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


def safe_filename(
    name: str,
) -> str:

    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    ).strip("_")


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(
    value,
) -> str:

    text = str(
        value or ""
    )

    text = (
        text
        .replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def canonical_header(
    value: str,
) -> str:

    value = clean_text(
        value
    ).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def parse_rank(
    value: str,
) -> Optional[int]:

    value = clean_text(
        value
    )

    match = re.search(
        r"\b(\d{1,4})\b",
        value,
    )

    if not match:
        return None

    try:
        return int(
            match.group(1)
        )

    except Exception:
        return None


# ============================================================
# HTML TABLE PARSER
#
# Standard-library parser only.
# No BeautifulSoup dependency required.
# ============================================================

class DraftTekTableParser(
    HTMLParser,
):

    def __init__(
        self,
    ):

        super().__init__(
            convert_charrefs=True
        )

        self.rows: List[
            List[str]
        ] = []

        self._in_row = False
        self._in_cell = False

        self._row: List[str] = []
        self._cell_parts: List[str] = []

    def handle_starttag(
        self,
        tag,
        attrs,
    ):

        tag = tag.lower()

        if tag == "tr":

            self._in_row = True
            self._row = []

        elif (
            self._in_row
            and tag in (
                "td",
                "th",
            )
        ):

            self._in_cell = True
            self._cell_parts = []

    def handle_data(
        self,
        data,
    ):

        if self._in_cell:

            text = clean_text(
                data
            )

            if text:
                self._cell_parts.append(
                    text
                )

    def handle_endtag(
        self,
        tag,
    ):

        tag = tag.lower()

        if (
            self._in_cell
            and tag in (
                "td",
                "th",
            )
        ):

            value = clean_text(
                " ".join(
                    self._cell_parts
                )
            )

            self._row.append(
                value
            )

            self._in_cell = False
            self._cell_parts = []

        elif (
            self._in_row
            and tag == "tr"
        ):

            if self._row:

                self.rows.append(
                    self._row
                )

            self._in_row = False
            self._row = []


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    headers: List[str],
    names,
) -> Optional[int]:

    normalized = [
        canonical_header(
            header
        )
        for header in headers
    ]

    for index, header in enumerate(
        normalized
    ):

        for name in names:

            target = canonical_header(
                name
            )

            if (
                header == target
                or header.startswith(
                    target + " "
                )
            ):

                return index

    return None


def find_header_row(
    rows: List[List[str]],
):

    for row_index, row in enumerate(
        rows
    ):

        normalized = [
            canonical_header(
                cell
            )
            for cell in row
        ]

        has_rank = any(
            cell == "rank"
            for cell in normalized
        )

        has_prospect = any(
            cell in (
                "prospect",
                "player",
            )
            for cell in normalized
        )

        has_college = any(
            cell.startswith(
                "college"
            )
            for cell in normalized
        )

        has_position = any(
            cell.startswith(
                "pos"
            )
            or cell == "p1"
            for cell in normalized
        )

        if (
            has_rank
            and has_prospect
            and has_college
            and has_position
        ):

            return (
                row_index,
                row,
            )

    return (
        None,
        None,
    )


# ============================================================
# DRAFTTEK POSITION NORMALIZATION
# ============================================================

def normalize_position(
    value,
) -> str:

    raw = clean_text(
        value
    ).upper()

    raw = (
        raw
        .replace("-", "")
        .replace(" ", "")
    )

    return (
        DRAFTTEK_POSITION_MAP.get(
            raw,
            ""
        )
    )


# ============================================================
# PARSE ONE DRAFTTEK PAGE
# ============================================================

def parse_drafttek_page(
    html: str,
    season: int,
    page: int,
) -> List[dict]:

    parser = (
        DraftTekTableParser()
    )

    parser.feed(
        html
    )

    rows = parser.rows

    header_index, header_row = (
        find_header_row(
            rows
        )
    )

    if (
        header_index is None
        or header_row is None
    ):

        raise RuntimeError(
            f"Could not find DraftTek "
            f"Big Board table header "
            f"for season={season}, "
            f"page={page}."
        )

    rank_col = find_column(
        header_row,
        (
            "Rank",
        ),
    )

    name_col = find_column(
        header_row,
        (
            "Prospect",
            "Player",
        ),
    )

    college_col = find_column(
        header_row,
        (
            "College",
        ),
    )

    position_col = find_column(
        header_row,
        (
            "Pos",
            "P1",
            "Position",
        ),
    )

    height_col = find_column(
        header_row,
        (
            "Ht",
            "Height",
        ),
    )

    weight_col = find_column(
        header_row,
        (
            "Wt",
            "Weight",
        ),
    )

    class_col = find_column(
        header_row,
        (
            "Cls",
            "YR",
            "Class",
            "Elig",
        ),
    )

    required_columns = {
        "rank": rank_col,
        "name": name_col,
        "college": college_col,
        "position": position_col,
    }

    missing = [
        name
        for name, index
        in required_columns.items()
        if index is None
    ]

    if missing:

        raise RuntimeError(
            "DraftTek table is missing "
            f"required columns: {missing}. "
            f"Headers={header_row}"
        )

    players: List[
        dict
    ] = []

    for row in rows[
        header_index + 1:
    ]:

        needed_indices = [
            index
            for index in (
                rank_col,
                name_col,
                college_col,
                position_col,
            )
            if index is not None
        ]

        if (
            not needed_indices
            or len(row)
            <= max(
                needed_indices
            )
        ):

            continue

        rank = parse_rank(
            row[
                rank_col
            ]
        )

        if rank is None:
            continue

        name = clean_text(
            row[
                name_col
            ]
        )

        college = clean_text(
            row[
                college_col
            ]
        )

        source_position = (
            clean_text(
                row[
                    position_col
                ]
            )
            .upper()
        )

        position = normalize_position(
            source_position
        )

        height = (
            clean_text(
                row[
                    height_col
                ]
            )
            if (
                height_col is not None
                and height_col < len(row)
            )
            else "N/A"
        )

        weight = (
            clean_text(
                row[
                    weight_col
                ]
            )
            if (
                weight_col is not None
                and weight_col < len(row)
            )
            else "N/A"
        )

        player_class = (
            clean_text(
                row[
                    class_col
                ]
            )
            if (
                class_col is not None
                and class_col < len(row)
            )
            else "N/A"
        )

        if not name:
            continue

        players.append(
            {
                "rank": rank,
                "name": name,
                "college": college or "N/A",
                "source_position": source_position,
                "position": position,
                "height": height or "N/A",
                "weight": weight or "N/A",
                "class": player_class or "N/A",
            }
        )

    print(
        f"DraftTek page {page}: "
        f"parsed {len(players)} ranked prospects"
    )

    return players


# ============================================================
# HTTP
# ============================================================

def fetch_html(
    url: str,
) -> str:

    last_error = None

    for attempt in range(
        1,
        5,
    ):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=30,
            )

            print(
                f"HTTP {response.status_code}: "
                f"{response.url}"
            )

            if response.status_code in (
                429,
                500,
                502,
                503,
                504,
            ):

                raise RuntimeError(
                    f"temporary HTTP "
                    f"{response.status_code}"
                )

            response.raise_for_status()

            return response.text

        except Exception as error:

            last_error = error

            print(
                f"WARNING: DraftTek request "
                f"attempt {attempt}/4 failed: "
                f"{error}"
            )

            if attempt < 4:

                time.sleep(
                    attempt * 2
                )

    raise RuntimeError(
        f"DraftTek request failed "
        f"after 4 attempts: "
        f"{last_error}"
    )


# ============================================================
# FETCH COMPLETE BIG BOARD
# ============================================================

def fetch_big_board(
    season,
):

    all_players: List[
        dict
    ] = []

    for page in DRAFTTEK_PAGES:

        url = (
            DRAFTTEK_PAGE_URL.format(
                season=season,
                page=page,
            )
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"FETCHING DRAFTTEK "
            f"{season} BIG BOARD "
            f"PAGE {page}"
        )

        print(
            "=" * 80
        )

        html = fetch_html(
            url
        )

        page_players = (
            parse_drafttek_page(
                html=html,
                season=season,
                page=page,
            )
        )

        all_players.extend(
            page_players
        )

    # --------------------------------------------------------
    # DEDUPE
    # --------------------------------------------------------

    unique = {}

    for player in all_players:

        key = (
            player.get("rank"),
            clean_text(
                player.get(
                    "name"
                )
            ).lower(),
        )

        unique[
            key
        ] = player

    players = list(
        unique.values()
    )

    players.sort(
        key=lambda player: (
            int(
                player.get(
                    "rank",
                    9999,
                )
            )
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"DRAFTTEK BIG BOARD "
        f"{season}: "
        f"{len(players)} "
        f"UNIQUE PROSPECTS"
    )

    print(
        "=" * 80
    )

    return {
        "source": "DraftTek",
        "season": season,
        "players": players,
    }


# ============================================================
# PLAYER LIST
# ============================================================

def get_player_list(
    data,
):

    if isinstance(
        data,
        list,
    ):

        return data

    if isinstance(
        data,
        dict,
    ):

        players = data.get(
            "players"
        )

        if isinstance(
            players,
            list,
        ):

            return players

    raise ValueError(
        "Could not find DraftTek "
        "player list."
    )


# ============================================================
# GROUP TOP 5 BY OUR EXISTING APP POSITION GROUPS
# ============================================================

def group_top_players(
    players,
):

    grouped = defaultdict(
        list
    )

    unknown_positions = (
        defaultdict(
            int
        )
    )

    for player in players:

        if not isinstance(
            player,
            dict,
        ):

            continue

        position = clean_text(
            player.get(
                "position"
            )
        )

        source_position = clean_text(
            player.get(
                "source_position"
            )
        )

        if (
            not position
            or position
            not in TARGET_POSITIONS
        ):

            if source_position:

                unknown_positions[
                    source_position
                ] += 1

            continue

        grouped[
            position
        ].append(
            player
        )

    output: Dict[
        str,
        List[dict],
    ] = {}

    for position in TARGET_POSITIONS:

        position_players = (
            grouped.get(
                position,
                []
            )
        )

        position_players.sort(
            key=lambda player: (
                int(
                    player.get(
                        "rank",
                        9999,
                    )
                )
            )
        )

        output[
            position
        ] = (
            position_players[
                :TOP_N
            ]
        )

    if unknown_positions:

        print()
        print(
            "UNMAPPED DRAFTTEK POSITIONS:"
        )

        for (
            source_position,
            count,
        ) in sorted(
            unknown_positions.items()
        ):

            print(
                f"  {source_position}: "
                f"{count}"
            )

    print()
    print(
        "=" * 80
    )

    print(
        "TOP-5 POSITION COUNTS"
    )

    print(
        "=" * 80
    )

    for position in TARGET_POSITIONS:

        print(
            f"{position}: "
            f"{len(output[position])}"
        )

    return output


# ============================================================
# FONTS
# ============================================================

def get_font(
    size=30,
    bold=False,
):

    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans-Bold.ttf"
            if bold
            else
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans.ttf"
        ),
        (
            "/System/Library/Fonts/Supplemental/"
            "Arial Bold.ttf"
            if bold
            else
            "/System/Library/Fonts/Supplemental/"
            "Arial.ttf"
        ),
        (
            "/Library/Fonts/Arial Bold.ttf"
            if bold
            else
            "/Library/Fonts/Arial.ttf"
        ),
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]

    for path in candidates:

        try:

            return ImageFont.truetype(
                path,
                size,
            )

        except Exception:

            pass

    return ImageFont.load_default()


SUBTITLE_FONT = get_font(
    34,
    bold=False,
)

HEADER_FONT = get_font(
    34,
    bold=True,
)


# ============================================================
# TEXT FITTING
# ============================================================

def fit_font(
    draw,
    text,
    max_width,
    start_size,
    min_size=20,
    bold=False,
):

    size = start_size

    while (
        size >= min_size
    ):

        font = get_font(
            size,
            bold=bold,
        )

        if (
            draw.textlength(
                str(
                    text
                ),
                font=font,
            )
            <= max_width
        ):

            return font

        size -= 1

    return get_font(
        min_size,
        bold=bold,
    )


# ============================================================
# BACKGROUND
# ============================================================

def draw_vertical_gradient(
    draw,
    width,
    height,
    top_color,
    bottom_color,
):

    for y in range(
        height
    ):

        ratio = (
            y
            / max(
                1,
                height - 1,
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
                y,
                width,
                y,
            ),
            fill=(
                red,
                green,
                blue,
            ),
        )


# ============================================================
# POSTER
#
# Same visual layout/colors as the existing Big Board.
#
# Only data-field change:
# AGE -> CLASS
# because DraftTek exposes class rather than age.
# ============================================================

def create_poster(
    position,
    players,
    season,
):

    height = (
        HEADER_HEIGHT
        + len(
            players
        )
        * ROW_HEIGHT
        + BOTTOM_PADDING
    )

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

    text = (
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
            POSTER_WIDTH,
            height,
        ),
        bg_bottom,
    )

    draw = ImageDraw.Draw(
        image
    )

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    draw_vertical_gradient(
        draw,
        POSTER_WIDTH,
        height,
        bg_top,
        bg_bottom,
    )

    # --------------------------------------------------------
    # OUTER BORDER
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            18,
            18,
            POSTER_WIDTH - 18,
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
            POSTER_WIDTH - 28,
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

    left = MARGIN

    right = (
        POSTER_WIDTH
        - MARGIN
    )

    top_height = 180
    top_y = 34

    # --------------------------------------------------------
    # TITLE PANEL
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            left,
            top_y,
            right,
            top_y
            + top_height,
        ),
        radius=28,
        fill=panel,
        outline=outer_border,
        width=2,
    )

    draw.rounded_rectangle(
        (
            left + 10,
            top_y + 10,
            right - 10,
            top_y
            + top_height
            - 10,
        ),
        radius=24,
        fill=panel_2,
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = (
        f"{position} - TOP "
        f"{len(players)} PROSPECTS"
    )

    title_font = fit_font(
        draw,
        title,
        right
        - left
        - 60,
        84,
        44,
        bold=True,
    )

    title_width = (
        draw.textlength(
            title,
            font=title_font,
        )
    )

    draw.text(
        (
            (
                POSTER_WIDTH
                - title_width
            )
            / 2,
            top_y + 28,
        ),
        title,
        fill=text,
        font=title_font,
    )

    subtitle = (
        f"SEASON {season}"
    )

    subtitle_width = (
        draw.textlength(
            subtitle,
            font=SUBTITLE_FONT,
        )
    )

    draw.text(
        (
            (
                POSTER_WIDTH
                - subtitle_width
            )
            / 2,
            top_y + 120,
        ),
        subtitle,
        fill=muted,
        font=SUBTITLE_FONT,
    )

    # --------------------------------------------------------
    # TABLE SETUP
    # --------------------------------------------------------

    table_left = (
        left + 12
    )

    table_right = (
        right - 12
    )

    table_width = (
        table_right
        - table_left
    )

    column_fractions = [
        0.09,
        0.37,
        0.22,
        0.12,
        0.12,
        0.08,
    ]

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

    headers = [
        "RANK",
        "NAME",
        "COLLEGE",
        "HEIGHT",
        "WEIGHT",
        "CLASS",
    ]

    header_y = (
        top_y
        + top_height
        + 18
    )

    header_height = 58

    # --------------------------------------------------------
    # TABLE HEADER
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            table_left,
            header_y,
            table_right,
            header_y
            + header_height,
        ),
        radius=16,
        fill=title_bar,
    )

    draw.rounded_rectangle(
        (
            table_left,
            header_y,
            table_right,
            header_y
            + header_height
            // 2,
        ),
        radius=16,
        fill=title_bar_hi,
    )

    x = table_left

    for index, header in enumerate(
        headers
    ):

        if index in (
            0,
            1,
            2,
        ):

            draw.text(
                (
                    x + 14,
                    header_y + 12,
                ),
                header,
                fill=muted,
                font=HEADER_FONT,
            )

        else:

            header_font = fit_font(
                draw,
                header,
                column_widths[
                    index
                ] - 22,
                34,
                22,
                bold=True,
            )

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
                    - 14
                    - header_width,
                    header_y + 12,
                ),
                header,
                fill=muted,
                font=header_font,
            )

        x += (
            column_widths[
                index
            ]
        )

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
                    header_y + 8,
                    x,
                    header_y
                    + header_height
                    - 8,
                ),
                fill=grid,
                width=1,
            )

    # --------------------------------------------------------
    # PLAYER ROWS
    # --------------------------------------------------------

    row_y = (
        header_y
        + header_height
        + 14
    )

    for (
        row_index,
        player,
    ) in enumerate(
        players
    ):

        fill = (
            row_a
            if (
                row_index
                % 2
                == 0
            )
            else row_b
        )

        draw.rounded_rectangle(
            (
                table_left,
                row_y,
                table_right,
                row_y
                + ROW_HEIGHT
                - 12,
            ),
            radius=18,
            fill=fill,
        )

        values = [
            str(
                player.get(
                    "rank",
                    "N/A",
                )
            ),
            str(
                player.get(
                    "name",
                    "Unknown",
                )
            ),
            str(
                player.get(
                    "college",
                    "N/A",
                )
            ),
            str(
                player.get(
                    "height",
                    "N/A",
                )
            ),
            str(
                player.get(
                    "weight",
                    "N/A",
                )
            ),
            str(
                player.get(
                    "class",
                    "N/A",
                )
            ),
        ]

        x = table_left

        for (
            column_index,
            value,
        ) in enumerate(
            values
        ):

            column_width = (
                column_widths[
                    column_index
                ]
            )

            # ------------------------------------------------
            # RANK
            # ------------------------------------------------

            if (
                column_index
                == 0
            ):

                font = fit_font(
                    draw,
                    value,
                    column_width
                    - 28,
                    42,
                    24,
                    bold=True,
                )

                text_y = (
                    row_y + 24
                )

                draw.text(
                    (
                        x + 14,
                        text_y,
                    ),
                    value,
                    fill=gold,
                    font=font,
                )

            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            elif (
                column_index
                == 1
            ):

                font = fit_font(
                    draw,
                    value,
                    column_width
                    - 24,
                    44,
                    24,
                    bold=True,
                )

                text_y = (
                    row_y + 18
                )

                draw.text(
                    (
                        x + 14,
                        text_y,
                    ),
                    value,
                    fill=text,
                    font=font,
                )

            # ------------------------------------------------
            # COLLEGE
            # ------------------------------------------------

            elif (
                column_index
                == 2
            ):

                font = fit_font(
                    draw,
                    value,
                    column_width
                    - 24,
                    34,
                    20,
                    bold=False,
                )

                text_y = (
                    row_y + 64
                )

                draw.text(
                    (
                        x + 14,
                        text_y,
                    ),
                    value,
                    fill=accent,
                    font=font,
                )

            # ------------------------------------------------
            # HEIGHT / WEIGHT
            # ------------------------------------------------

            elif column_index in (
                3,
                4,
            ):

                font = fit_font(
                    draw,
                    value,
                    column_width
                    - 24,
                    38,
                    24,
                    bold=False,
                )

                text_width = (
                    draw.textlength(
                        value,
                        font=font,
                    )
                )

                text_y = (
                    row_y + 42
                )

                draw.text(
                    (
                        x
                        + column_width
                        - 14
                        - text_width,
                        text_y,
                    ),
                    value,
                    fill=text,
                    font=font,
                )

            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            else:

                font = fit_font(
                    draw,
                    value,
                    column_width
                    - 20,
                    36,
                    22,
                    bold=False,
                )

                text_width = (
                    draw.textlength(
                        value,
                        font=font,
                    )
                )

                text_y = (
                    row_y + 42
                )

                draw.text(
                    (
                        x
                        + column_width
                        - 14
                        - text_width,
                        text_y,
                    ),
                    value,
                    fill=text,
                    font=font,
                )

            x += (
                column_width
            )

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
                        row_y + 10,
                        x,
                        row_y
                        + ROW_HEIGHT
                        - 22,
                    ),
                    fill=grid,
                    width=1,
                )

        row_y += (
            ROW_HEIGHT
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    path = os.path.join(
        OUTPUT_DIR,
        (
            f"{safe_filename(position)}"
            "_top_5.png"
        ),
    )

    image.save(
        path
    )

    print(
        "Saved",
        path,
    )

    return path


# ============================================================
# VALIDATE POSITION COVERAGE
# ============================================================

def validate_position_groups(
    grouped,
) -> None:

    problems = []

    for position in TARGET_POSITIONS:

        count = len(
            grouped.get(
                position,
                []
            )
        )

        if count < TOP_N:

            problems.append(
                (
                    position,
                    count,
                )
            )

    if problems:

        message = ", ".join(
            f"{position}={count}"
            for (
                position,
                count,
            )
            in problems
        )

        raise RuntimeError(
            "DraftTek did not provide "
            "five usable prospects for "
            f"every required group: {message}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--season",
        type=int,
        default=SEASON_DEFAULT,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=OUTPUT_DIR,
    )

    args = (
        parser.parse_args()
    )

    global OUTPUT_DIR

    OUTPUT_DIR = (
        args.outdir
    )

    ensure_output_dir()

    print(
        f"Fetching DraftTek "
        f"Big Board for "
        f"{args.season}..."
    )

    data = fetch_big_board(
        args.season
    )

    players = get_player_list(
        data
    )

    print(
        f"Found "
        f"{len(players)} "
        f"DraftTek prospects."
    )

    grouped = group_top_players(
        players
    )

    validate_position_groups(
        grouped
    )

    for position in TARGET_POSITIONS:

        create_poster(
            position,
            grouped[
                position
            ],
            args.season,
        )

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()
