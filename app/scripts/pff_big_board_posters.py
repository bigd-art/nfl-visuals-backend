#!/usr/bin/env python3
import argparse
import os
import re
import sys
from collections import defaultdict

import requests
from PIL import Image, ImageDraw, ImageFont

SEASON_DEFAULT = 2026
VERSION = 4
TOP_N = 5
OUTPUT_DIR = "pff_posters"

BOARD_URL = "https://www.pff.com/api/college/big_board"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

SKIP_POSITIONS = {"FB", "K", "P"}

POSTER_WIDTH = 1450
HEADER_HEIGHT = 280
ROW_HEIGHT = 190
BOTTOM_PADDING = 60
MARGIN = 44


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def get_font(size=30, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf"
        if bold
        else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass

    return ImageFont.load_default()


TITLE_FONT = get_font(84, bold=True)
SUBTITLE_FONT = get_font(34, bold=False)
HEADER_FONT = get_font(34, bold=True)
TEXT_FONT = get_font(36, bold=False)
RANK_FONT = get_font(42, bold=True)
NAME_FONT = get_font(44, bold=True)
SMALL_TEXT_FONT = get_font(32, bold=False)


def get_headers(season):
    return {
        "accept": "application/json, text/plain, */*",
        "referer": f"https://www.pff.com/draft/big-board?season={season}",
        "user-agent": USER_AGENT,
    }


def pick(d, keys, default="N/A"):
    for key in keys:
        if key in d and d[key] not in (None, "", []):
            return d[key]

    return default


def normalize_position(pos):
    if not pos:
        return "UNK"

    pos = str(pos).strip().upper()

    mapping = {
        "HB": "RB",
        "TB": "RB",
        "OLB": "LB",
        "ILB": "LB",
        "MLB": "LB",
        "SS": "S",
        "FS": "S",
        "NT": "DL",
        "DT": "DL",
        "DE": "EDGE",
        "G": "IOL",
        "C": "IOL",
        "OL": "IOL",
    }

    return mapping.get(pos, pos)


def get_player_list(data):
    if isinstance(data, list):
        if data:
            return data

        return []

    if isinstance(data, dict):
        preferred_keys = [
            "players",
            "big_board",
            "bigBoard",
            "results",
            "prospects",
            "data",
        ]

        for key in preferred_keys:
            value = data.get(key)

            if isinstance(value, list) and value:
                return value

            if isinstance(value, dict):
                try:
                    nested_players = get_player_list(value)

                    if nested_players:
                        return nested_players

                except ValueError:
                    pass

        for value in data.values():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    return value

            if isinstance(value, dict):
                try:
                    nested_players = get_player_list(value)

                    if nested_players:
                        return nested_players

                except ValueError:
                    pass

    raise ValueError("Could not find a non-empty player list.")


def parse_player(raw, rank):
    return {
        "rank": rank,
        "name": str(
            pick(
                raw,
                ["player_name", "playerName", "name"],
                "Unknown",
            )
        ),
        "position": normalize_position(
            pick(
                raw,
                ["position", "pos"],
                "UNK",
            )
        ),
        "college": str(
            pick(
                raw,
                ["college", "school", "team_name", "teamName"],
                "N/A",
            )
        ),
        "height": str(
            pick(
                raw,
                ["height", "height_display", "heightDisplay"],
                "N/A",
            )
        ),
        "weight": str(
            pick(
                raw,
                ["weight", "weight_display", "weightDisplay"],
                "N/A",
            )
        ),
        "age": str(
            pick(
                raw,
                ["age"],
                "N/A",
            )
        ),
    }


def fetch_big_board(season):
    response = requests.get(
        BOARD_URL,
        params={
            "season": season,
            "version": VERSION,
        },
        headers=get_headers(season),
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def group_top_players(players):
    grouped = defaultdict(list)

    for i, raw in enumerate(players):
        player = parse_player(raw, i + 1)

        if player["position"] in SKIP_POSITIONS:
            continue

        grouped[player["position"]].append(player)

    for pos in grouped:
        grouped[pos] = grouped[pos][:TOP_N]

    return dict(sorted(grouped.items()))


def draw_text(draw, x, y, text, font):
    draw.text(
        (x, y),
        str(text),
        fill="black",
        font=font,
    )


def draw_header(draw, y):
    cols = [
        ("Rank", 60),
        ("Name", 170),
        ("College", 520),
        ("Height", 900),
        ("Weight", 1070),
        ("Age", 1240),
    ]

    for text, x in cols:
        draw_text(
            draw,
            x,
            y,
            text,
            HEADER_FONT,
        )


def draw_row(draw, y, player):
    cols = [
        (player["rank"], 60),
        (player["name"], 170),
        (player["college"], 520),
        (player["height"], 900),
        (player["weight"], 1070),
        (player["age"], 1240),
    ]

    for text, x in cols:
        draw_text(
            draw,
            x,
            y,
            text,
            TEXT_FONT,
        )


def fit_font(
    draw,
    text,
    max_width,
    start_size,
    min_size=20,
    bold=False,
):
    size = start_size

    while size >= min_size:
        font = get_font(
            size,
            bold=bold,
        )

        if draw.textlength(
            str(text),
            font=font,
        ) <= max_width:
            return font

        size -= 1

    return get_font(
        min_size,
        bold=bold,
    )


def draw_vertical_gradient(
    draw,
    width,
    height,
    top_color,
    bottom_color,
):
    for y in range(height):
        ratio = y / max(1, height - 1)

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
            (0, y, width, y),
            fill=(
                red,
                green,
                blue,
            ),
        )


def create_poster(position, players, season):
    height = (
        HEADER_HEIGHT
        + len(players) * ROW_HEIGHT
        + BOTTOM_PADDING
    )

    bg_top = (6, 30, 88)
    bg_bottom = (3, 10, 28)
    outer_border = (120, 185, 255)
    panel = (10, 28, 72)
    panel_2 = (14, 39, 96)
    title_bar = (23, 62, 150)
    title_bar_hi = (45, 100, 220)
    row_a = (10, 31, 78)
    row_b = (16, 40, 95)
    grid = (78, 132, 228)
    text = (245, 248, 255)
    muted = (192, 208, 242)
    accent = (154, 204, 255)
    gold = (255, 214, 90)

    img = Image.new(
        "RGB",
        (POSTER_WIDTH, height),
        bg_bottom,
    )

    draw = ImageDraw.Draw(img)

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
        outline=(40, 90, 190),
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

    title = f"{position} - TOP {len(players)} PFF PROSPECTS"

    title_font = fit_font(
        draw,
        title,
        (right - left) - 60,
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
            (POSTER_WIDTH - title_width) / 2,
            top_y + 28,
        ),
        title,
        fill=text,
        font=title_font,
    )

    subtitle = f"SEASON {season}"

    subtitle_width = draw.textlength(
        subtitle,
        font=SUBTITLE_FONT,
    )

    draw.text(
        (
            (POSTER_WIDTH - subtitle_width) / 2,
            top_y + 120,
        ),
        subtitle,
        fill=muted,
        font=SUBTITLE_FONT,
    )

    table_left = left + 12
    table_right = right - 12
    table_width = table_right - table_left

    col_fracs = [
        0.09,
        0.37,
        0.22,
        0.12,
        0.12,
        0.08,
    ]

    col_px = [
        int(table_width * fraction)
        for fraction in col_fracs
    ]

    col_px[-1] += (
        table_width
        - sum(col_px)
    )

    headers = [
        "RANK",
        "NAME",
        "COLLEGE",
        "HEIGHT",
        "WEIGHT",
        "AGE",
    ]

    header_y = top_y + top_height + 18
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
            header_y + (header_height // 2),
        ),
        radius=16,
        fill=title_bar_hi,
    )

    x = table_left

    for i, header in enumerate(headers):
        if i in (0, 1, 2):
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
            header_width = draw.textlength(
                header,
                font=HEADER_FONT,
            )

            draw.text(
                (
                    x
                    + col_px[i]
                    - 14
                    - header_width,
                    header_y + 12,
                ),
                header,
                fill=muted,
                font=HEADER_FONT,
            )

        x += col_px[i]

        if i != len(headers) - 1:
            draw.line(
                (
                    x,
                    header_y + 8,
                    x,
                    header_y + header_height - 8,
                ),
                fill=grid,
                width=1,
            )

    row_y = (
        header_y
        + header_height
        + 14
    )

    for idx, player in enumerate(players):
        fill = row_a if idx % 2 == 0 else row_b

        draw.rounded_rectangle(
            (
                table_left,
                row_y,
                table_right,
                row_y + ROW_HEIGHT - 12,
            ),
            radius=18,
            fill=fill,
        )

        rank = str(player["rank"])
        name = str(player["name"])
        college = str(player["college"])
        height_text = str(player["height"])
        weight_text = str(player["weight"])
        age_text = str(player["age"])

        values = [
            rank,
            name,
            college,
            height_text,
            weight_text,
            age_text,
        ]

        x = table_left

        for column_index, value in enumerate(values):
            column_width = col_px[column_index]

            if column_index == 0:
                font = fit_font(
                    draw,
                    value,
                    column_width - 28,
                    42,
                    24,
                    bold=True,
                )

                y_text = row_y + 24

                draw.text(
                    (
                        x + 14,
                        y_text,
                    ),
                    value,
                    fill=gold,
                    font=font,
                )

            elif column_index == 1:
                font = fit_font(
                    draw,
                    value,
                    column_width - 24,
                    44,
                    24,
                    bold=True,
                )

                y_text = row_y + 18

                draw.text(
                    (
                        x + 14,
                        y_text,
                    ),
                    value,
                    fill=text,
                    font=font,
                )

            elif column_index == 2:
                font = fit_font(
                    draw,
                    value,
                    column_width - 24,
                    34,
                    20,
                    bold=False,
                )

                y_text = row_y + 64

                draw.text(
                    (
                        x + 14,
                        y_text,
                    ),
                    value,
                    fill=accent,
                    font=font,
                )

            elif column_index in (3, 4):
                font = fit_font(
                    draw,
                    value,
                    column_width - 24,
                    38,
                    24,
                    bold=False,
                )

                text_width = draw.textlength(
                    value,
                    font=font,
                )

                y_text = row_y + 42

                draw.text(
                    (
                        x
                        + column_width
                        - 14
                        - text_width,
                        y_text,
                    ),
                    value,
                    fill=text,
                    font=font,
                )

            else:
                font = fit_font(
                    draw,
                    value,
                    column_width - 20,
                    40,
                    26,
                    bold=False,
                )

                text_width = draw.textlength(
                    value,
                    font=font,
                )

                y_text = row_y + 42

                draw.text(
                    (
                        x
                        + column_width
                        - 14
                        - text_width,
                        y_text,
                    ),
                    value,
                    fill=text,
                    font=font,
                )

            x += column_width

            if column_index != len(values) - 1:
                draw.line(
                    (
                        x,
                        row_y + 10,
                        x,
                        row_y + ROW_HEIGHT - 22,
                    ),
                    fill=grid,
                    width=1,
                )

        row_y += ROW_HEIGHT

    path = os.path.join(
        OUTPUT_DIR,
        f"{safe_filename(position)}_top_5.png",
    )

    img.save(path)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=SEASON_DEFAULT,
    )

    args = parser.parse_args()

    ensure_output_dir()

    print(
        f"Fetching PFF big board for "
        f"{args.season}..."
    )

    data = fetch_big_board(
        args.season
    )

    players = get_player_list(data)

    grouped = group_top_players(
        players
    )

    for position, player_list in grouped.items():
        create_poster(
            position,
            player_list,
            args.season,
        )

    print("Done.")


if __name__ == "__main__":
    main()
