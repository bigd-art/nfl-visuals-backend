#!/usr/bin/env python3
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

from app.services.storage_supabase import upload_file_return_url

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SITE_API = "https://site.api.espn.com/apis/v2/sports/football/nfl/standings"

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


def get_json(season: int) -> dict:
    r = requests.get(
        SITE_API,
        params={"season": season, "type": 2},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


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
        k = s.get("name")
        if k == "wins":
            w = to_int(s.get("value", s.get("displayValue")))
        elif k == "losses":
            l = to_int(s.get("value", s.get("displayValue")))
        elif k == "ties":
            t = to_int(s.get("value", s.get("displayValue")))
    return w, l, t


def extract_espn_seed(entry: dict) -> Optional[int]:
    # ESPN sometimes includes seed/rank in stats under one of these names
    for s in entry.get("stats", []) or []:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").lower()
        if name in (
            "seed",
            "playoffseed",
            "playoff_seed",
            "rank",
            "conferencerank",
            "conference_rank",
        ):
            val = s.get("value", s.get("displayValue"))
            seed = to_int(val)
            if seed > 0:
                return seed
    return None


def build_division_map(conf_obj: dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for div in conf_obj.get("children", []) or []:
        div_name = (div.get("shortName") or div.get("name") or "").strip()
        div_short = normalize_division_name(div_name)
        standings = (div.get("standings") or {}).get("entries") or []
        for e in standings:
            team = e.get("team") or {}
            tid = str(team.get("id") or "").strip()
            if tid:
                mapping[tid] = div_short
    return mapping


def extract_conferences(data: dict) -> Dict[str, List[TeamRow]]:
    conferences: Dict[str, List[TeamRow]] = {"AFC": [], "NFC": []}

    for conf in data.get("children", []) or []:
        conf_abbr = (
            conf.get("abbreviation")
            or conf.get("shortName")
            or conf.get("name")
            or ""
        ).strip().upper()
        if conf_abbr not in ("AFC", "NFC"):
            continue

        div_map = build_division_map(conf)
        conf_entries = (conf.get("standings") or {}).get("entries") or []

        # ✅ If ESPN provides conference standings entries, KEEP ESPN order
        if conf_entries:
            rows: List[TeamRow] = []
            seeds: List[Optional[int]] = []

            for e in conf_entries:
                team = e.get("team") or {}
                tid = str(team.get("id") or "").strip()
                name = str(team.get("displayName") or team.get("name") or "").strip()
                w, l, t = extract_stats(e)
                seed = extract_espn_seed(e)
                seeds.append(seed)

                rows.append(
                    TeamRow(
                        team_id=tid,
                        team_name=name,
                        division=div_map.get(tid, ""),
                        w=w,
                        l=l,
                        t=t,
                        espn_seed=seed,
                    )
                )

            # If most teams have a seed, sort by that seed; otherwise preserve API order
            non_null = sum(1 for s in seeds if isinstance(s, int) and s > 0)
            if non_null >= max(4, int(0.8 * len(rows))):
                rows = sorted(
                    rows,
                    key=lambda r: (r.espn_seed if r.espn_seed is not None else 999),
                )

            conferences[conf_abbr] = rows
            continue

        # Fallback: build from divisions (rare), keep old logic but stable
        fallback_rows: List[TeamRow] = []
        for div in conf.get("children", []) or []:
            div_name = (div.get("shortName") or div.get("name") or "").strip()
            div_short = normalize_division_name(div_name)
            div_entries = (div.get("standings") or {}).get("entries") or []
            for e in div_entries:
                team = e.get("team") or {}
                tid = str(team.get("id") or "").strip()
                name = str(team.get("displayName") or team.get("name") or "").strip()
                w, l, t = extract_stats(e)
                fallback_rows.append(
                    TeamRow(
                        team_id=tid,
                        team_name=name,
                        division=div_short,
                        w=w,
                        l=l,
                        t=t,
                    )
                )

        seen = set()
        uniq: List[TeamRow] = []
        for r in fallback_rows:
            if r.team_id and r.team_id not in seen:
                seen.add(r.team_id)
                uniq.append(r)

        conferences[conf_abbr] = uniq

    return conferences


def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # ✅ Ubuntu/GHA
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

    bg = (10, 12, 16)
    header_bar = (30, 36, 50)
    card = (18, 22, 30)
    grid = (40, 48, 66)
    text = (235, 238, 245)
    muted = (170, 176, 190)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    title_font = get_font(46)
    section_font = get_font(28)
    header_font = get_font(20)
    row_font = get_font(22)

    pad = 34
    y = pad

    draw.text((pad, y), f"NFL Standings {season} — By Conference", fill=text, font=title_font)
    y += 70

    headers = ["#", "TEAM", "DIV", "W", "L", "T"]
    col_fracs = [0.06, 0.68, 0.10, 0.06, 0.05, 0.05]

    def draw_section(title: str, rows: List[TeamRow]):
        nonlocal y

        bar_h = 36
        draw.rounded_rectangle((pad, y, width - pad, y + bar_h), radius=12, fill=header_bar)
        draw.text((pad + 12, y + 6), title, fill=text, font=section_font)
        y += bar_h + 8

        card_w = width - 2 * pad
        px = [int(card_w * f) for f in col_fracs]
        px[-1] += (card_w - sum(px))

        header_h = 30
        draw.rounded_rectangle((pad, y, width - pad, y + header_h), radius=10, fill=card)

        x = pad
        for i, h in enumerate(headers):
            if i in (0, 1):
                draw.text((x + 8, y + 6), h, fill=muted, font=header_font)
            else:
                tw = draw.textlength(h, font=header_font)
                draw.text((x + px[i] - 8 - tw, y + 6), h, fill=muted, font=header_font)

            x += px[i]
            if i != len(headers) - 1:
                draw.line((x, y + 5, x, y + header_h - 5), fill=grid, width=1)

        y += header_h + 4

        row_h = 28
        for idx, r in enumerate(rows):
            if y > height - 24:
                break

            fill = (14, 18, 26) if (idx % 2 == 0) else (12, 16, 22)
            draw.rounded_rectangle((pad, y, width - pad, y + row_h), radius=8, fill=fill)

            seed = str(idx + 1)
            div = r.division or hardcoded_div(r.team_name)
            values = [seed, r.team_name, div, str(r.w), str(r.l), str(r.t)]

            x = pad
            for c_i, val in enumerate(values):
                if c_i in (0, 1):
                    draw.text((x + 8, y + 4), val, fill=text, font=row_font)
                else:
                    tw = draw.textlength(val, font=row_font)
                    draw.text((x + px[c_i] - 8 - tw, y + 4), val, fill=text, font=row_font)

                x += px[c_i]
                if c_i != len(values) - 1:
                    draw.line((x, y + 4, x, y + row_h - 4), fill=grid, width=1)

            y += row_h + 3

        y += 10

    draw_section("AFC", conferences.get("AFC", []))
    draw_section("NFC", conferences.get("NFC", []))

    img.save(out_path)


def generate_standings_conference_png(season: int, out_path: str) -> str:
    data = get_json(season)
    conferences = extract_conferences(data)
    render_conference_poster(season, conferences, out_path)
    return out_path


def generate_and_upload_standings_conference(season: int) -> str:
    """
    Generates the PNG to /tmp and uploads to Supabase via upload_file_return_url().
    Returns public URL.
    """
    local_path = f"/tmp/standings_conference_{season}.png"
    generate_standings_conference_png(season, local_path)

    storage_key = f"standings/{season}/standings_conference.png"
    return upload_file_return_url(local_path, storage_key)


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
