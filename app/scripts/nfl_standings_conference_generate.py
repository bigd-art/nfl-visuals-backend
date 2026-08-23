#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

CORE_API_BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
SUPABASE_PUBLIC_BASE = "https://rojsabkwywygludonpdf.supabase.co/storage/v1/object/public/nfl-posters"

# ✅ HARDCODED DIVISIONS (East/North/South/West) by ESPN displayName
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
    return TEAM_TO_DIV.get((team_name or "").strip(), "")


@dataclass
class TeamRow:
    team_id: str
    team_name: str
    division: str  # East/North/South/West (or "")
    w: int
    l: int
    t: int
    espn_seed: Optional[int] = None  # if ESPN provides seed/rank, use it


def core_get_json(url: str) -> dict:
    """
    Fetch JSON from ESPN's Core API.

    ESPN sometimes returns $ref URLs using http://. Convert those to https://
    before requesting them.
    """
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


def get_json(season: int) -> dict:
    """
    Fetch regular-season conference standings from ESPN Core API.

    Conference IDs:
      AFC = 8
      NFC = 7
    """
    conference_ids = {
        "AFC": 8,
        "NFC": 7,
    }

    data: Dict[str, dict] = {}

    for conference, group_id in conference_ids.items():
        url = (
            f"{CORE_API_BASE}/seasons/{season}/types/2/"
            f"groups/{group_id}/standings"
        )
        data[conference] = core_get_json(url)

    return data


def to_int(v: Any) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


def normalize_division_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    if "east" in n:
        return "East"
    if "north" in n:
        return "North"
    if "south" in n:
        return "South"
    if "west" in n:
        return "West"
    if n in ("east", "north", "south", "west"):
        return n.title()
    return ""


def extract_stats(entry: dict) -> Tuple[int, int, int]:
    w = l = t = 0

    for s in entry.get("stats", []) or []:
        if not isinstance(s, dict):
            continue

        name = str(s.get("name") or "").strip().lower()
        display_name = str(s.get("displayName") or "").strip().lower()
        abbreviation = str(s.get("abbreviation") or "").strip().lower()

        value = s.get("value", s.get("displayValue"))

        if name == "wins" or display_name == "wins" or abbreviation == "w":
            w = to_int(value)
        elif name == "losses" or display_name == "losses" or abbreviation == "l":
            l = to_int(value)
        elif name == "ties" or display_name == "ties" or abbreviation == "t":
            t = to_int(value)

    return w, l, t


def extract_espn_seed(entry: dict) -> Optional[int]:
    for s in entry.get("stats", []) or []:
        if not isinstance(s, dict):
            continue

        name = (s.get("name") or "").lower().replace("_", "")
        display_name = (s.get("displayName") or "").lower().replace(" ", "")
        abbreviation = (s.get("abbreviation") or "").lower()

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
            or abbreviation in ("seed",)
        ):
            val = s.get("value", s.get("displayValue"))
            seed = to_int(val)
            if seed > 0:
                return seed

    return None


def _find_entries(obj: Any) -> List[dict]:
    """
    Recursively find the first standings-style entries list.

    ESPN's Core API has changed nesting slightly over time, so this keeps the
    parser tolerant instead of depending on one exact JSON layout.
    """
    if isinstance(obj, dict):
        entries = obj.get("entries")
        if isinstance(entries, list) and entries:
            if any(
                isinstance(item, dict) and ("team" in item or "stats" in item)
                for item in entries
            ):
                return [item for item in entries if isinstance(item, dict)]

        for value in obj.values():
            found = _find_entries(value)
            if found:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = _find_entries(item)
            if found:
                return found

    return []


def _overall_entries(payload: dict) -> List[dict]:
    """
    Prefer the 'overall' standings table if ESPN returns multiple tables.
    """
    items = payload.get("items") or []

    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or "").strip().lower()
            display_name = str(item.get("displayName") or "").strip().lower()

            if name == "overall" or "overall standings" in display_name:
                entries = _find_entries(item)
                if entries:
                    return entries

    return _find_entries(payload)


def _team_id_from_ref(ref: str) -> str:
    if not ref:
        return ""

    clean = ref.rstrip("/")
    last = clean.split("/")[-1]
    return last if last.isdigit() else ""


def _resolve_team(team_obj: Any) -> Tuple[str, str]:
    """
    Return (team_id, display_name).

    Core standings entries commonly store the team as a $ref. If the team is
    already expanded, no extra request is made.
    """
    if not isinstance(team_obj, dict):
        return "", ""

    team_id = str(team_obj.get("id") or "").strip()
    team_name = str(
        team_obj.get("displayName")
        or team_obj.get("shortDisplayName")
        or team_obj.get("name")
        or ""
    ).strip()

    ref = str(team_obj.get("$ref") or "").strip()

    if not team_id and ref:
        team_id = _team_id_from_ref(ref)

    if team_name:
        return team_id, team_name

    if ref:
        resolved = core_get_json(ref)
        team_id = str(resolved.get("id") or team_id).strip()
        team_name = str(
            resolved.get("displayName")
            or resolved.get("shortDisplayName")
            or resolved.get("name")
            or ""
        ).strip()

    return team_id, team_name


def extract_conferences(data: dict) -> Dict[str, List[TeamRow]]:
    conferences: Dict[str, List[TeamRow]] = {"AFC": [], "NFC": []}

    for conference in ("AFC", "NFC"):
        payload = data.get(conference) or {}
        entries = _overall_entries(payload)

        rows: List[TeamRow] = []

        for entry in entries:
            team_obj = entry.get("team") or {}
            team_id, team_name = _resolve_team(team_obj)

            if not team_name:
                continue

            w, l, t = extract_stats(entry)
            seed = extract_espn_seed(entry)

            rows.append(
                TeamRow(
                    team_id=team_id,
                    team_name=team_name,
                    division=hardcoded_div(team_name),
                    w=w,
                    l=l,
                    t=t,
                    espn_seed=seed,
                )
            )

        # Remove duplicates while preserving ESPN's returned order.
        seen = set()
        unique_rows: List[TeamRow] = []
        for row in rows:
            key = row.team_id or row.team_name
            if key and key not in seen:
                seen.add(key)
                unique_rows.append(row)

        # If ESPN supplies usable seeds for most teams, use them.
        seeded = [
            row for row in unique_rows
            if isinstance(row.espn_seed, int) and row.espn_seed > 0
        ]
        if len(seeded) >= max(4, int(0.8 * len(unique_rows))):
            unique_rows = sorted(
                unique_rows,
                key=lambda row: (
                    row.espn_seed if row.espn_seed is not None else 999
                ),
            )

        conferences[conference] = unique_rows

    return conferences


def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_conference_poster(season: int, conferences: Dict[str, List[TeamRow]], out_path: str):
    width, height = 1080, 1920

    def get_font_local(size: int, bold: bool = False):
        candidates = []
        if bold:
            candidates.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ])
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ])
        for p in candidates:
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int = 18, bold: bool = False):
        size = start_size
        while size >= min_size:
            font = get_font_local(size, bold=bold)
            if draw.textlength(text, font=font) <= max_width:
                return font
            size -= 1
        return get_font_local(min_size, bold=bold)

    def draw_vertical_gradient(draw: ImageDraw.ImageDraw, width: int, height: int, top_color, bottom_color):
        for yy in range(height):
            t = yy / max(1, height - 1)
            r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
            g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
            b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
            draw.line((0, yy, width, yy), fill=(r, g, b))

    bg_top = (6, 30, 88)
    bg_bottom = (3, 10, 28)
    outer_border = (120, 185, 255)
    panel = (10, 28, 72)
    panel_2 = (14, 39, 96)
    title_bar = (23, 62, 150)
    title_bar_hi = (45, 100, 220)
    table_header = (15, 43, 104)
    row_a = (10, 31, 78)
    row_b = (16, 40, 95)
    grid = (78, 132, 228)
    text = (245, 248, 255)
    muted = (192, 208, 242)
    accent = (154, 204, 255)
    gold = (255, 214, 90)

    img = Image.new("RGB", (width, height), bg_bottom)
    draw = ImageDraw.Draw(img)
    draw_vertical_gradient(draw, width, height, bg_top, bg_bottom)

    draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=34, outline=outer_border, width=3)
    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=30, outline=(40, 90, 190), width=1)

    section_font = get_font_local(36, bold=True)
    header_font = get_font_local(22, bold=True)
    seed_font = get_font_local(28, bold=True)
    stat_font = get_font_local(28, bold=False)

    left = 38
    right = width - 38
    y = 38

    top_h = 150
    draw.rounded_rectangle((left, y, right, y + top_h), radius=28, fill=panel, outline=outer_border, width=2)
    draw.rounded_rectangle((left + 10, y + 10, right - 10, y + top_h - 10), radius=24, fill=panel_2)

    title = f"NFL STANDINGS {season}"
    max_title_width = (right - left) - 60
    fitted_title_font = fit_font(draw, title, max_title_width, 76, 52, bold=True)
    tw = draw.textlength(title, font=fitted_title_font)
    th = fitted_title_font.size
    draw.text(((width - tw) / 2, y + (top_h - th) / 2 - 8), title, fill=text, font=fitted_title_font)

    y += top_h + 24

    section_gap = 24
    bottom_margin = 34
    available_h = height - y - bottom_margin
    section_h = (available_h - section_gap) // 2

    headers = ["#", "TEAM", "DIV", "W", "L", "T"]
    col_fracs = [0.10, 0.52, 0.14, 0.08, 0.08, 0.08]

    def draw_section(top_y: int, title: str, rows: List[TeamRow]):
        sec_bottom = top_y + section_h
        draw.rounded_rectangle((left, top_y, right, sec_bottom), radius=28, fill=panel, outline=grid, width=2)

        bar_margin = 16
        bar_h = 58
        draw.rounded_rectangle(
            (left + bar_margin, top_y + bar_margin, right - bar_margin, top_y + bar_margin + bar_h),
            radius=18,
            fill=title_bar,
        )
        draw.rounded_rectangle(
            (left + bar_margin, top_y + bar_margin, right - bar_margin, top_y + bar_margin + (bar_h // 2)),
            radius=18,
            fill=title_bar_hi,
        )

        tw = draw.textlength(title, font=section_font)
        draw.text(((width - tw) / 2, top_y + bar_margin + 9), title, fill=text, font=section_font)

        table_left = left + 16
        table_right = right - 16
        table_w = table_right - table_left

        px = [int(table_w * f) for f in col_fracs]
        px[-1] += (table_w - sum(px))

        header_y = top_y + bar_margin + bar_h + 16
        header_h = 46
        draw.rounded_rectangle((table_left, header_y, table_right, header_y + header_h), radius=14, fill=table_header)

        x = table_left
        for i, h in enumerate(headers):
            if i in (0, 1):
                draw.text((x + 12, header_y + 10), h, fill=muted, font=header_font)
            else:
                tw = draw.textlength(h, font=header_font)
                draw.text((x + px[i] - 12 - tw, header_y + 10), h, fill=muted, font=header_font)

            x += px[i]
            if i != len(headers) - 1:
                draw.line((x, header_y + 7, x, header_y + header_h - 7), fill=grid, width=1)

        rows_top = header_y + header_h + 10
        rows_bottom = sec_bottom - 18
        n_rows = max(1, len(rows))
        gap = 6
        usable_h = rows_bottom - rows_top - gap * (n_rows - 1)
        row_h = max(34, usable_h // n_rows)

        current_y = rows_top
        for idx, r in enumerate(rows):
            fill = row_a if idx % 2 == 0 else row_b
            draw.rounded_rectangle((table_left, current_y, table_right, current_y + row_h), radius=14, fill=fill)

            seed = str(idx + 1)
            div = r.division or hardcoded_div(r.team_name)
            values = [seed, r.team_name, div, str(r.w), str(r.l), str(r.t)]

            x = table_left
            for c_i, val in enumerate(values):
                if c_i == 0:
                    text_y = current_y + (row_h - 28) / 2 - 2
                    color = gold if idx < 7 else accent
                    draw.text((x + 14, text_y), val, fill=color, font=seed_font)
                elif c_i == 1:
                    max_team_w = px[c_i] - 24
                    team_font = fit_font(draw, val, max_team_w, 28, 18, bold=False)
                    text_y = current_y + (row_h - team_font.size) / 2 - 2
                    draw.text((x + 12, text_y), val, fill=text, font=team_font)
                else:
                    tw = draw.textlength(val, font=stat_font)
                    text_y = current_y + (row_h - 28) / 2 - 2
                    draw.text((x + px[c_i] - 12 - tw, text_y), val, fill=text, font=stat_font)

                x += px[c_i]
                if c_i != len(values) - 1:
                    draw.line((x, current_y + 7, x, current_y + row_h - 7), fill=grid, width=1)

            current_y += row_h + gap

    draw_section(y, "AFC", conferences.get("AFC", []))
    draw_section(y + section_h + section_gap, "NFC", conferences.get("NFC", []))

    img.save(out_path)


def generate_standings_conference_png(season: int, out_path: str) -> str:
    data = get_json(season)
    conferences = extract_conferences(data)

    afc_count = len(conferences.get("AFC", []))
    nfc_count = len(conferences.get("NFC", []))

    if afc_count == 0 or nfc_count == 0:
        raise RuntimeError(
            "ESPN Core API returned standings, but the parser could not find "
            f"both conferences (AFC={afc_count}, NFC={nfc_count})."
        )

    render_conference_poster(season, conferences, out_path)
    return out_path


def generate_and_upload_standings_conference(season: int) -> str:
    """
    Return the existing current standings image from Supabase instead of
    generating and uploading a new one.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{SUPABASE_PUBLIC_BASE}/standings/current.png?v={timestamp}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=str, default="standings_conference.png")
    args = ap.parse_args()

    generate_standings_conference_png(args.season, args.out)
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
