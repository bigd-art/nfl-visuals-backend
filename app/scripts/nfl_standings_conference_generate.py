#!/usr/bin/env python3
import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SITE_API = "https://site.api.espn.com/apis/v2/sports/football/nfl/standings"

TEAM_TO_DIV: Dict[str, str] = {
    "Buffalo Bills": "East",
    "Miami Dolphins": "East",
    "New England Patriots": "East",
    "New York Jets": "East",
    "Baltimore Ravens": "North",
    "Cincinnati Bengals": "North",
    "Cleveland Browns": "North",
    "Pittsburgh Steelers": "North",
    "Houston Texans": "South",
    "Indianapolis Colts": "South",
    "Jacksonville Jaguars": "South",
    "Tennessee Titans": "South",
    "Denver Broncos": "West",
    "Kansas City Chiefs": "West",
    "Las Vegas Raiders": "West",
    "Los Angeles Chargers": "West",
    "Dallas Cowboys": "East",
    "New York Giants": "East",
    "Philadelphia Eagles": "East",
    "Washington Commanders": "East",
    "Chicago Bears": "North",
    "Detroit Lions": "North",
    "Green Bay Packers": "North",
    "Minnesota Vikings": "North",
    "Atlanta Falcons": "South",
    "Carolina Panthers": "South",
    "New Orleans Saints": "South",
    "Tampa Bay Buccaneers": "South",
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
    division: str
    w: int
    l: int
    t: int
    espn_seed: Optional[int] = None


def get_json(season: int) -> dict:
    r = requests.get(
        SITE_API,
        params={"season": season, "type": 2},
        headers={"User-Agent": USER_AGENT},
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


def to_int(v: Any) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


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


def extract_seed(entry: dict) -> Optional[int]:
    for s in entry.get("stats", []) or []:
        name = (s.get("name") or "").lower()
        if name in ("seed", "playoffseed", "conferencerank"):
            val = s.get("value", s.get("displayValue"))
            seed = to_int(val)
            if seed > 0:
                return seed
    return None


def extract_conferences(data: dict) -> Dict[str, List[TeamRow]]:
    conferences: Dict[str, List[TeamRow]] = {"AFC": [], "NFC": []}

    for conf in data.get("children", []) or []:
        conf_abbr = (conf.get("abbreviation") or "").upper()
        if conf_abbr not in ("AFC", "NFC"):
            continue

        entries = (conf.get("standings") or {}).get("entries") or []
        rows: List[TeamRow] = []

        for e in entries:
            team = e.get("team") or {}
            name = str(team.get("displayName") or "").strip()
            tid = str(team.get("id") or "").strip()
            w, l, t = extract_stats(e)
            seed = extract_seed(e)

            rows.append(
                TeamRow(
                    team_id=tid,
                    team_name=name,
                    division=hardcoded_div(name),
                    w=w,
                    l=l,
                    t=t,
                    espn_seed=seed,
                )
            )

        if any(r.espn_seed is not None for r in rows):
            rows = sorted(rows, key=lambda r: r.espn_seed if r.espn_seed else 999)

        conferences[conf_abbr] = rows

    return conferences


def get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
            draw.text((x + 8, y + 6), h, fill=text, font=header_font)
            x += px[i]

        y += header_h + 4

        row_h = 28
        for idx, r in enumerate(rows):
            if y > height - 24:
                break

            draw.rounded_rectangle((pad, y, width - pad, y + row_h), radius=8, fill=(14, 18, 26))

            seed = str(idx + 1)
            div = r.division

            values = [seed, r.team_name, div, str(r.w), str(r.l), str(r.t)]

            x = pad
            for c_i, val in enumerate(values):
                draw.text((x + 8, y + 4), val, fill=text, font=row_font)
                x += px[c_i]

            y += row_h + 3

        y += 10

    draw_section("AFC", conferences.get("AFC", []))
    draw_section("NFC", conferences.get("NFC", []))

    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=str, default="standings_conference.png")
    args = ap.parse_args()

    data = get_json(args.season)
    conferences = extract_conferences(data)
    render_conference_poster(args.season, conferences, args.out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
