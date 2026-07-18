#!/usr/bin/env python3

import argparse
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


SEASON_DEFAULT = 2027
VERSION = 4
TOP_N = 5
OUTPUT_DIR = "pff_posters"

BOARD_URL = "https://www.pff.com/api/college/big_board"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

SKIP_POSITIONS = {
    "FB",
    "K",
    "P",
}

POSTER_WIDTH = 1450
HEADER_HEIGHT = 280
ROW_HEIGHT = 190
BOTTOM_PADDING = 60
MARGIN = 44


def ensure_output_dir() -> None:
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


def safe_filename(name: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(name),
    ).strip("_")


def get_font(
    size: int = 30,
    bold: bool = False,
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
            continue

    return ImageFont.load_default()


SUBTITLE_FONT = get_font(
    34,
    bold=False,
)

HEADER_FONT = get_font(
    34,
    bold=True,
)


def get_headers(
    season: int,
) -> Dict[str, str]:
    return {
        "accept": "application/json, text/plain, */*",
        "referer": (
            "https://www.pff.com/draft/"
            f"big-board?season={season}"
        ),
        "user-agent": USER_AGENT,
    }


def clean_text(
    value: Any,
    default: str = "N/A",
) -> str:
    if value in (
        None,
        "",
        [],
        {},
    ):
        return default

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def pick(
    data: Dict[str, Any],
    keys: List[str],
    default: Any = "N/A",
) -> Any:
    if not isinstance(
        data,
        dict,
    ):
        return default

    for key in keys:
        value = data.get(key)

        if value not in (
            None,
            "",
            [],
            {},
        ):
            return value

    return default


def normalize_position(
    position: Any,
) -> str:
    if isinstance(
        position,
        dict,
    ):
        position = pick(
            position,
            [
                "abbreviation",
                "shortName",
                "shortDisplayName",
                "displayName",
                "name",
                "code",
            ],
            "UNK",
        )

    position = clean_text(
        position,
        "UNK",
    ).upper()

    mapping = {
        "HALFBACK": "RB",
        "HB": "RB",
        "TB": "RB",
        "RUNNING BACK": "RB",
        "RUNNINGBACK": "RB",
        "WIDE RECEIVER": "WR",
        "TIGHT END": "TE",
        "OFFENSIVE TACKLE": "OT",
        "TACKLE": "OT",
        "OFFENSIVE GUARD": "IOL",
        "GUARD": "IOL",
        "CENTER": "IOL",
        "INTERIOR OFFENSIVE LINE": "IOL",
        "INTERIOR OFFENSIVE LINEMAN": "IOL",
        "G": "IOL",
        "C": "IOL",
        "OL": "IOL",
        "OG": "IOL",
        "OFFENSIVE LINE": "IOL",
        "DEFENSIVE END": "EDGE",
        "EDGE DEFENDER": "EDGE",
        "EDGE RUSHER": "EDGE",
        "DE": "EDGE",
        "DEFENSIVE TACKLE": "DL",
        "DEFENSIVE LINE": "DL",
        "DEFENSIVE LINEMAN": "DL",
        "NOSE TACKLE": "DL",
        "DT": "DL",
        "NT": "DL",
        "OUTSIDE LINEBACKER": "LB",
        "INSIDE LINEBACKER": "LB",
        "MIDDLE LINEBACKER": "LB",
        "LINEBACKER": "LB",
        "OLB": "LB",
        "ILB": "LB",
        "MLB": "LB",
        "CORNERBACK": "CB",
        "CORNER": "CB",
        "FREE SAFETY": "S",
        "STRONG SAFETY": "S",
        "SAFETY": "S",
        "FS": "S",
        "SS": "S",
    }

    return mapping.get(
        position,
        position,
    )


def get_nested_value(
    data: Any,
    keys: List[str],
    default: Any = None,
) -> Any:
    if isinstance(
        data,
        dict,
    ):
        for key in keys:
            value = data.get(key)

            if value not in (
                None,
                "",
                [],
                {},
            ):
                return value

        preferred_nested_keys = [
            "player",
            "athlete",
            "prospect",
            "profile",
            "bio",
            "measurements",
            "combine",
            "school",
            "team",
            "position",
            "data",
            "attributes",
        ]

        for nested_key in preferred_nested_keys:
            nested = data.get(nested_key)

            if isinstance(
                nested,
                dict,
            ):
                found = get_nested_value(
                    nested,
                    keys,
                    default=None,
                )

                if found not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    return found

    return default


def extract_player_name(
    raw: Dict[str, Any],
) -> Optional[str]:
    value = get_nested_value(
        raw,
        [
            "player_name",
            "playerName",
            "displayName",
            "display_name",
            "fullName",
            "full_name",
            "name",
        ],
        default=None,
    )

    if isinstance(
        value,
        dict,
    ):
        value = pick(
            value,
            [
                "displayName",
                "fullName",
                "name",
            ],
            None,
        )

    if value in (
        None,
        "",
    ):
        first_name = get_nested_value(
            raw,
            [
                "firstName",
                "first_name",
            ],
            default="",
        )

        last_name = get_nested_value(
            raw,
            [
                "lastName",
                "last_name",
            ],
            default="",
        )

        combined = clean_text(
            f"{first_name} {last_name}",
            "",
        )

        return combined or None

    cleaned = clean_text(
        value,
        "",
    )

    return cleaned or None


def extract_position(
    raw: Dict[str, Any],
) -> str:
    value = get_nested_value(
        raw,
        [
            "position",
            "position_name",
            "positionName",
            "position_abbreviation",
            "positionAbbreviation",
            "position_code",
            "positionCode",
            "pos",
        ],
        default=None,
    )

    return normalize_position(
        value,
    )


def looks_like_player(
    raw: Any,
) -> bool:
    if not isinstance(
        raw,
        dict,
    ):
        return False

    name = extract_player_name(
        raw,
    )

    position = extract_position(
        raw,
    )

    return bool(
        name
        and position
        and position != "UNK"
    )


def collect_candidate_lists(
    data: Any,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    candidates = []

    def walk(
        value: Any,
        path: str = "root",
    ) -> None:
        if isinstance(
            value,
            list,
        ):
            dictionary_items = [
                item
                for item in value
                if isinstance(
                    item,
                    dict,
                )
            ]

            if dictionary_items:
                candidates.append(
                    (
                        path,
                        dictionary_items,
                    )
                )

            for index, item in enumerate(
                value,
            ):
                if isinstance(
                    item,
                    (
                        dict,
                        list,
                    ),
                ):
                    walk(
                        item,
                        f"{path}[{index}]",
                    )

        elif isinstance(
            value,
            dict,
        ):
            for key, nested in value.items():
                if isinstance(
                    nested,
                    (
                        dict,
                        list,
                    ),
                ):
                    walk(
                        nested,
                        f"{path}.{key}",
                    )

    walk(
        data,
    )

    return candidates


def score_player_list(
    path: str,
    items: List[Dict[str, Any]],
) -> int:
    if not items:
        return -1

    sample = items[
        :min(
            len(items),
            50,
        )
    ]

    player_matches = sum(
        1
        for item in sample
        if looks_like_player(item)
    )

    name_matches = sum(
        1
        for item in sample
        if extract_player_name(item)
    )

    position_matches = sum(
        1
        for item in sample
        if extract_position(item) != "UNK"
    )

    path_key = path.lower()

    preferred_tokens = [
        "players",
        "prospects",
        "big_board",
        "bigboard",
        "rankings",
        "draft",
        "results",
    ]

    path_bonus = sum(
        25
        for token in preferred_tokens
        if token in path_key
    )

    invalid_penalty = sum(
        1
        for item in sample
        if not isinstance(
            item,
            dict,
        )
    )

    return (
        player_matches * 1000
        + name_matches * 100
        + position_matches * 100
        + min(
            len(items),
            200,
        )
        + path_bonus
        - invalid_penalty * 100
    )


def get_player_list(
    data: Any,
) -> List[Dict[str, Any]]:
    candidates = collect_candidate_lists(
        data,
    )

    if not candidates:
        raise RuntimeError(
            "Could not find any non-empty object lists "
            "inside the PFF response."
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda item: score_player_list(
            item[0],
            item[1],
        ),
        reverse=True,
    )

    for path, items in ranked_candidates:
        valid_players = [
            item
            for item in items
            if looks_like_player(item)
        ]

        if len(valid_players) >= 5:
            print(
                "Using PFF prospect list at "
                f"{path}: {len(valid_players)} valid players."
            )

            return valid_players

    diagnostic = [
        {
            "path": path,
            "items": len(items),
            "score": score_player_list(
                path,
                items,
            ),
            "valid_players": sum(
                1
                for item in items
                if looks_like_player(item)
            ),
        }
        for path, items in ranked_candidates[:10]
    ]

    raise RuntimeError(
        "PFF returned JSON, but no valid prospect list "
        f"could be identified. Candidates: {diagnostic}"
    )


def parse_rank(
    raw: Dict[str, Any],
    fallback_rank: int,
) -> int:
    value = get_nested_value(
        raw,
        [
            "rank",
            "overall_rank",
            "overallRank",
            "big_board_rank",
            "bigBoardRank",
            "boardRank",
            "ranking",
        ],
        default=fallback_rank,
    )

    try:
        rank = int(
            float(
                str(value).strip()
            )
        )

        if rank > 0:
            return rank

    except Exception:
        pass

    return fallback_rank


def parse_college(
    raw: Dict[str, Any],
) -> str:
    value = get_nested_value(
        raw,
        [
            "college",
            "college_name",
            "collegeName",
            "school",
            "school_name",
            "schoolName",
            "team_name",
            "teamName",
            "team",
        ],
        default="N/A",
    )

    if isinstance(
        value,
        dict,
    ):
        value = pick(
            value,
            [
                "displayName",
                "shortDisplayName",
                "fullName",
                "name",
                "abbreviation",
            ],
            "N/A",
        )

    return clean_text(
        value,
        "N/A",
    )


def parse_measurement(
    raw: Dict[str, Any],
    keys: List[str],
) -> str:
    value = get_nested_value(
        raw,
        keys,
        default="N/A",
    )

    if isinstance(
        value,
        dict,
    ):
        value = pick(
            value,
            [
                "displayValue",
                "display",
                "formatted",
                "value",
            ],
            "N/A",
        )

    return clean_text(
        value,
        "N/A",
    )


def parse_player(
    raw: Dict[str, Any],
    fallback_rank: int,
) -> Dict[str, Any]:
    return {
        "rank": parse_rank(
            raw,
            fallback_rank,
        ),
        "name": extract_player_name(
            raw,
        ) or "Unknown",
        "position": extract_position(
            raw,
        ),
        "college": parse_college(
            raw,
        ),
        "height": parse_measurement(
            raw,
            [
                "height",
                "height_display",
                "heightDisplay",
                "displayHeight",
                "formattedHeight",
            ],
        ),
        "weight": parse_measurement(
            raw,
            [
                "weight",
                "weight_display",
                "weightDisplay",
                "displayWeight",
                "formattedWeight",
            ],
        ),
        "age": parse_measurement(
            raw,
            [
                "age",
                "draftAge",
                "draft_age",
            ],
        ),
    }


def fetch_big_board(
    season: int,
) -> Any:
    response = requests.get(
        BOARD_URL,
        params={
            "season": season,
            "version": VERSION,
        },
        headers=get_headers(
            season,
        ),
        timeout=30,
    )

    response.raise_for_status()

    try:
        return response.json()

    except ValueError as error:
        preview = response.text[:500]

        raise RuntimeError(
            "PFF did not return valid JSON. "
            f"Response preview: {preview}"
        ) from error


def group_top_players(
    raw_players: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    parsed_players = []

    for index, raw in enumerate(
        raw_players,
        start=1,
    ):
        player = parse_player(
            raw,
            index,
        )

        if player["name"] == "Unknown":
            print(
                "WARNING: Skipping prospect with no name."
            )
            continue

        if player["position"] == "UNK":
            print(
                "WARNING: Skipping player with unknown "
                f"position: {player['name']}"
            )
            continue

        if player["position"] in SKIP_POSITIONS:
            continue

        parsed_players.append(
            player,
        )

    parsed_players.sort(
        key=lambda player: player["rank"],
    )

    grouped = defaultdict(
        list,
    )

    for player in parsed_players:
        position = player["position"]

        if len(
            grouped[position]
        ) < TOP_N:
            grouped[position].append(
                player,
            )

    output = dict(
        sorted(
            grouped.items(),
        )
    )

    if "UNK" in output:
        del output["UNK"]

    if not output:
        raise RuntimeError(
            "PFF data was fetched, but no valid "
            "position groups were parsed."
        )

    return output


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: Any,
    max_width: int,
    start_size: int,
    min_size: int = 20,
    bold: bool = False,
):
    size = start_size
    text = str(
        text,
    )

    while size >= min_size:
        font = get_font(
            size,
            bold=bold,
        )

        if draw.textlength(
            text,
            font=font,
        ) <= max_width:
            return font

        size -= 1

    return get_font(
        min_size,
        bold=bold,
    )


def draw_vertical_gradient(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int],
) -> None:
    for y in range(
        height,
    ):
        ratio = y / max(
            1,
            height - 1,
        )

        red = int(
            top_color[0] * (1 - ratio)
            + bottom_color[0] * ratio
        )

        green = int(
            top_color[1] * (1 - ratio)
            + bottom_color[1] * ratio
        )

        blue = int(
            top_color[2] * (1 - ratio)
            + bottom_color[2] * ratio
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


def create_poster(
    position: str,
    players: List[Dict[str, Any]],
    season: int,
) -> str:
    if position == "UNK":
        raise RuntimeError(
            "Refusing to create an UNK position poster."
        )

    height = (
        HEADER_HEIGHT
        + len(players) * ROW_HEIGHT
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

    title_bar_highlight = (
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
            POSTER_WIDTH,
            height,
        ),
        bg_bottom,
    )

    draw = ImageDraw.Draw(
        image,
    )

    draw_vertical_gradient(
        draw,
        POSTER_WIDTH,
        height,
        bg_top,
        bg_bottom,
    )

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
    right = POSTER_WIDTH - MARGIN

    top_height = 180
    top_y = 34

    draw.rounded_rectangle(
        (
            left,
            top_y,
            right,
            top_y + top_height,
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
            top_y + top_height - 10,
        ),
        radius=24,
        fill=panel_2,
    )

    title = (
        f"{position} - TOP {len(players)} "
        "PFF PROSPECTS"
    )

    title_font = fit_font(
        draw,
        title,
        right - left - 60,
        84,
        44,
        bold=True,
    )

    title_width = draw.textlength(
        title,
        font=title_font,
    )

    draw.text(
        (
            (
                POSTER_WIDTH
                - title_width
            ) / 2,
            top_y + 28,
        ),
        title,
        fill=text_color,
        font=title_font,
    )

    subtitle = f"SEASON {season}"

    subtitle_width = draw.textlength(
        subtitle,
        font=SUBTITLE_FONT,
    )

    draw.text(
        (
            (
                POSTER_WIDTH
                - subtitle_width
            ) / 2,
            top_y + 120,
        ),
        subtitle,
        fill=muted,
        font=SUBTITLE_FONT,
    )

    table_left = left + 12
    table_right = right - 12
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
            table_width * fraction
        )
        for fraction in column_fractions
    ]

    column_widths[-1] += (
        table_width
        - sum(
            column_widths,
        )
    )

    headers = [
        "RANK",
        "NAME",
        "COLLEGE",
        "HEIGHT",
        "WEIGHT",
        "AGE",
    ]

    header_y = (
        top_y
        + top_height
        + 18
    )

    header_height = 58

    draw.rounded_rectangle(
        (
            table_left,
            header_y,
            table_right,
            header_y + header_height,
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
            + header_height // 2,
        ),
        radius=16,
        fill=title_bar_highlight,
    )

    x = table_left

    for index, header in enumerate(
        headers,
    ):
        column_width = column_widths[
            index
        ]

        if index in (
            0,
            1,
            2,
        ):
            text_x = x + 14

        else:
            header_width = draw.textlength(
                header,
                font=HEADER_FONT,
            )

            text_x = (
                x
                + column_width
                - 14
                - header_width
            )

        draw.text(
            (
                text_x,
                header_y + 12,
            ),
            header,
            fill=muted,
            font=HEADER_FONT,
        )

        x += column_width

        if index != len(
            headers,
        ) - 1:
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

    row_y = (
        header_y
        + header_height
        + 14
    )

    for row_index, player in enumerate(
        players,
    ):
        row_fill = (
            row_a
            if row_index % 2 == 0
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
            fill=row_fill,
        )

        values = [
            str(
                player["rank"],
            ),
            str(
                player["name"],
            ),
            str(
                player["college"],
            ),
            str(
                player["height"],
            ),
            str(
                player["weight"],
            ),
            str(
                player["age"],
            ),
        ]

        x = table_left

        for column_index, value in enumerate(
            values,
        ):
            column_width = column_widths[
                column_index
            ]

            if column_index == 0:
                font = fit_font(
                    draw,
                    value,
                    column_width - 28,
                    42,
                    24,
                    bold=True,
                )

                text_x = x + 14
                text_y = row_y + 24
                fill = gold

            elif column_index == 1:
                font = fit_font(
                    draw,
                    value,
                    column_width - 24,
                    44,
                    24,
                    bold=True,
                )

                text_x = x + 14
                text_y = row_y + 18
                fill = text_color

            elif column_index == 2:
                font = fit_font(
                    draw,
                    value,
                    column_width - 24,
                    34,
                    20,
                    bold=False,
                )

                text_x = x + 14
                text_y = row_y + 64
                fill = accent

            elif column_index in (
                3,
                4,
            ):
                font = fit_font(
                    draw,
                    value,
                    column_width - 24,
                    38,
                    24,
                    bold=False,
                )

                value_width = draw.textlength(
                    value,
                    font=font,
                )

                text_x = (
                    x
                    + column_width
                    - 14
                    - value_width
                )

                text_y = row_y + 42
                fill = text_color

            else:
                font = fit_font(
                    draw,
                    value,
                    column_width - 20,
                    40,
                    26,
                    bold=False,
                )

                value_width = draw.textlength(
                    value,
                    font=font,
                )

                text_x = (
                    x
                    + column_width
                    - 14
                    - value_width
                )

                text_y = row_y + 42
                fill = text_color

            draw.text(
                (
                    text_x,
                    text_y,
                ),
                value,
                fill=fill,
                font=font,
            )

            x += column_width

            if column_index != len(
                values,
            ) - 1:
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

        row_y += ROW_HEIGHT

    path = os.path.join(
        OUTPUT_DIR,
        (
            f"{safe_filename(position)}"
            "_top_5.png"
        ),
    )

    image.save(
        path,
        "PNG",
    )

    print(
        f"Saved: {path}"
    )

    return path


def remove_old_unk_poster() -> None:
    unk_path = os.path.join(
        OUTPUT_DIR,
        "UNK_top_5.png",
    )

    if os.path.exists(
        unk_path,
    ):
        os.remove(
            unk_path,
        )

        print(
            f"Removed old invalid poster: {unk_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=SEASON_DEFAULT,
    )

    args = parser.parse_args()

    ensure_output_dir()
    remove_old_unk_poster()

    print(
        "Fetching PFF big board for "
        f"{args.season}..."
    )

    data = fetch_big_board(
        args.season,
    )

    players = get_player_list(
        data,
    )

    print(
        f"Parsed {len(players)} valid prospects."
    )

    grouped = group_top_players(
        players,
    )

    print(
        "Position groups found: "
        + ", ".join(
            grouped.keys(),
        )
    )

    outputs = {}

    for position, player_list in grouped.items():
        outputs[position] = create_poster(
            position,
            player_list,
            args.season,
        )

    if not outputs:
        raise RuntimeError(
            "No PFF Big Board posters were generated."
        )

    print(
        f"Done. Generated {len(outputs)} posters."
    )


if __name__ == "__main__":
    main()
