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
HEADER_HEIGHT = 190
ROW_HEIGHT = 110
BOTTOM_PADDING = 50
MARGIN = 40


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def get_font(size=30, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


TITLE_FONT = get_font(52, bold=True)
SUBTITLE_FONT = get_font(22, bold=False)
HEADER_FONT = get_font(25, bold=True)
TEXT_FONT = get_font(24, bold=False)


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
        return data

    if isinstance(data, dict):
        for key in ["players", "big_board", "bigBoard", "results", "prospects", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        for v in data.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v

    raise ValueError("Could not find player list.")


def parse_player(raw, rank):
    return {
        "rank": rank,
        "name": str(pick(raw, ["player_name", "playerName", "name"], "Unknown")),
        "position": normalize_position(pick(raw, ["position", "pos"], "UNK")),
        "college": str(pick(raw, ["college", "school", "team_name", "teamName"], "N/A")),
        "height": str(pick(raw, ["height", "height_display", "heightDisplay"], "N/A")),
        "weight": str(pick(raw, ["weight", "weight_display", "weightDisplay"], "N/A")),
        "age": str(pick(raw, ["age"], "N/A")),
    }


def fetch_big_board(season):
    r = requests.get(
        BOARD_URL,
        params={"season": season, "version": VERSION},
        headers=get_headers(season),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


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
    draw.text((x, y), str(text), fill="black", font=font)


def draw_header(draw, y):
    cols = [
        ("Rank", 60),
        ("Name", 170),
        ("College", 520),
        ("Height", 900),
        ("Weight", 1070),
        ("Age", 1240),
    ]
    for t, x in cols:
        draw_text(draw, x, y, t, HEADER_FONT)


def draw_row(draw, y, p):
    cols = [
        (p["rank"], 60),
        (p["name"], 170),
        (p["college"], 520),
        (p["height"], 900),
        (p["weight"], 1070),
        (p["age"], 1240),
    ]
    for t, x in cols:
        draw_text(draw, x, y, t, TEXT_FONT)


def create_poster(position, players, season):
    h = HEADER_HEIGHT + len(players) * ROW_HEIGHT + BOTTOM_PADDING
    img = Image.new("RGB", (POSTER_WIDTH, h), "white")
    draw = ImageDraw.Draw(img)

    draw_text(draw, MARGIN, 30, f"{position} - Top {len(players)} PFF Prospects", TITLE_FONT)
    draw_text(draw, MARGIN, 95, f"Season {season}", SUBTITLE_FONT)

    y = 150
    draw.line((MARGIN, y - 10, POSTER_WIDTH - MARGIN, y - 10), fill="black", width=3)
    draw_header(draw, y)

    y += 50
    for p in players:
        draw.line((MARGIN, y - 10, POSTER_WIDTH - MARGIN, y - 10), fill="black", width=1)
        draw_row(draw, y + 5, p)
        y += ROW_HEIGHT

    path = os.path.join(OUTPUT_DIR, f"{safe_filename(position)}_top_5.png")
    img.save(path)
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=SEASON_DEFAULT)
    args = parser.parse_args()

    ensure_output_dir()

    print(f"Fetching PFF big board for {args.season}...")
    data = fetch_big_board(args.season)
    players = get_player_list(data)
    grouped = group_top_players(players)

    for pos, plist in grouped.items():
        create_poster(pos, plist, args.season)

    print("Done.")


if __name__ == "__main__":
    main()
