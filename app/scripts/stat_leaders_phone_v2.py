# app/scripts/stat_leaders_phone_v2.py

import os
import re
import json
import time
import tempfile
from typing import Dict, Any, List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from app.services.storage_supabase import upload_file_return_url, cached_urls_for_prefix


# =========================
# Config
# =========================

W, H = 1440, 2560  # <-- phone-crisp base resolution
MARGIN_X = 90
TOP_Y = 140

ROW_H = 175
HEADER_H = 240

# ESPN leaders endpoint (JSON)
ESPN_LEADERS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/types/{seasontype}/leaders"
)

# Your 10 poster categories (keys match ESPN "name" fields)
CATEGORY_KEYS_ORDER = [
    "passingYards",
    "passingTouchdowns",
    "rushingYards",
    "rushingTouchdowns",
    "receivingYards",
    "receivingTouchdowns",
    "sacks",
    "totalTackles",
    "interceptions",
    "passesDefended",  # if this one is missing, we fall back cleanly
]

# If a category doesn't exist in the response, we try these fallbacks
CATEGORY_FALLBACKS = {
    "passingTouchdowns": ["passingTDs", "passingTouchdowns"],
    "rushingTouchdowns": ["rushingTDs", "rushingTouchdowns"],
    "receivingTouchdowns": ["receivingTDs", "receivingTouchdowns"],
    "totalTackles": ["totalTackles", "tackles"],
    "passesDefended": ["passesDefended", "passesDefensed", "passDeflections"],
}


# =========================
# Fonts
# =========================

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """
    Tries a few common font paths that exist on Linux (Render/GitHub Actions).
    Falls back to PIL default if needed.
    """
    candidates = []
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)

    return ImageFont.load_default()


FONT_TITLE = _load_font(74, bold=True)
FONT_SUB = _load_font(42, bold=False)
FONT_ROW_NAME = _load_font(52, bold=True)
FONT_ROW_TEAM = _load_font(42, bold=False)
FONT_ROW_VAL = _load_font(64, bold=True)
FONT_RANK = _load_font(46, bold=True)


# =========================
# Helpers
# =========================

def _kind(seasontype: int) -> str:
    return "regular" if int(seasontype) == 2 else "playoffs"


def _prefix(season: int, seasontype: int) -> str:
    return f"stat_leaders_v2/{season}/{_kind(seasontype)}/"


def _safe_filename(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "category"


def _http_get_json(url: str) -> Dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; P5TechStatLeaders/1.0)",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


def _fetch_leaders(season: int, seasontype: int) -> Dict[str, Any]:
    url = ESPN_LEADERS_URL.format(season=season, seasontype=seasontype)
    # Add a generous limit (ESPN may ignore, but harmless)
    if "?" not in url:
        url = url + "?limit=1000"
    return _http_get_json(url)


def _index_categories(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    ESPN returns:
      { categories: [ { name, displayName, leaders: [ ... ] }, ... ] }
    """
    cats = payload.get("categories", []) or []
    out: Dict[str, Dict[str, Any]] = {}
    for c in cats:
        name = c.get("name")
        if name:
            out[name] = c
    return out


def _resolve_category(cat_map: Dict[str, Dict[str, Any]], key: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (resolved_key, category_obj) or raises KeyError
    """
    if key in cat_map:
        return key, cat_map[key]

    for alt in CATEGORY_FALLBACKS.get(key, []):
        if alt in cat_map:
            return alt, cat_map[alt]

    raise KeyError(key)


def _leaders_top10(category_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    leaders = category_obj.get("leaders", []) or []
    # "leaders" is already a list of leader entries
    return leaders[:10]


def _leader_name_team_value(leader: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    leader example usually has:
      displayValue
      athlete -> { displayName }
      team -> { abbreviation }
    """
    display_value = str(leader.get("displayValue", "")).strip()

    athlete = leader.get("athlete") or {}
    name = str(athlete.get("displayName", "")).strip()

    team = leader.get("team") or {}
    abbr = str(team.get("abbreviation", "")).strip()

    if not name:
        name = "Unknown"
    if not abbr:
        abbr = "--"
    if not display_value:
        # sometimes only "value" exists
        v = leader.get("value", "")
        display_value = str(v) if v != "" else "--"

    return name, abbr, display_value


# =========================
# Drawing
# =========================

def _draw_bg(im: Image.Image) -> None:
    """
    Simple dark gradient background (looks good + compresses well as PNG)
    """
    px = im.load()
    for y in range(H):
        # dark -> slightly lighter
        t = y / (H - 1)
        base = int(12 + 28 * t)  # 12..40
        for x in range(W):
            px[x, y] = (base, base, base)


def _draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    # Title
    draw.text((MARGIN_X, TOP_Y), title, font=FONT_TITLE, fill=(255, 255, 255))
    # Subtitle
    draw.text((MARGIN_X, TOP_Y + 92), subtitle, font=FONT_SUB, fill=(190, 190, 190))

    # divider
    y = TOP_Y + HEADER_H - 35
    draw.line((MARGIN_X, y, W - MARGIN_X, y), fill=(85, 85, 85), width=3)


def _draw_rows(draw: ImageDraw.ImageDraw, rows: List[Tuple[str, str, str]]) -> None:
    start_y = TOP_Y + HEADER_H

    for i, (name, team, val) in enumerate(rows, start=1):
        y = start_y + (i - 1) * ROW_H

        # rank
        draw.text((MARGIN_X, y + 52), f"{i}", font=FONT_RANK, fill=(170, 170, 170))

        # name (big)
        name_x = MARGIN_X + 90
        draw.text((name_x, y + 32), name, font=FONT_ROW_NAME, fill=(255, 255, 255))

        # team (under name)
        draw.text((name_x, y + 98), team, font=FONT_ROW_TEAM, fill=(170, 170, 170))

        # value (right-aligned)
        val_w = draw.textlength(val, font=FONT_ROW_VAL)
        draw.text((W - MARGIN_X - val_w, y + 45), val, font=FONT_ROW_VAL, fill=(255, 255, 255))

        # row divider
        draw.line((MARGIN_X, y + ROW_H - 18, W - MARGIN_X, y + ROW_H - 18), fill=(55, 55, 55), width=2)


def _make_one_poster(
    season: int,
    seasontype: int,
    category_display: str,
    filename_slug: str,
    rows: List[Tuple[str, str, str]],
) -> str:
    im = Image.new("RGB", (W, H), (10, 10, 10))
    _draw_bg(im)
    draw = ImageDraw.Draw(im)

    st = "Regular Season" if int(seasontype) == 2 else "Postseason"
    title = category_display
    subtitle = f"{season} • {st} • Top 10"

    _draw_header(draw, title, subtitle)
    _draw_rows(draw, rows)

    out_dir = tempfile.mkdtemp(prefix=f"stat_leaders_{season}_{seasontype}_")
    out_path = os.path.join(out_dir, f"{filename_slug}.png")
    im.save(out_path, format="PNG", optimize=True)
    return out_path


# =========================
# Public API
# =========================

def generate_stat_leaders_and_upload(season: int, seasontype: int) -> List[str]:
    """
    Returns list of public URLs for the 10 posters.
    Uses Supabase cache prefix stat_leaders_v2/{season}/{regular|playoffs}/
    """
    t0 = time.time()
    prefix = _prefix(season, seasontype)

    cached = cached_urls_for_prefix(prefix)
    if cached:
        return cached

    payload = _fetch_leaders(season, seasontype)
    cat_map = _index_categories(payload)

    uploaded_urls: List[str] = []

    for requested_key in CATEGORY_KEYS_ORDER:
        try:
            resolved_key, cat = _resolve_category(cat_map, requested_key)
        except KeyError:
            # If ESPN doesn't provide this category in this season/type, skip it cleanly
            continue

        display_name = cat.get("displayName") or cat.get("name") or resolved_key
        leaders = _leaders_top10(cat)
        rows: List[Tuple[str, str, str]] = []
        for l in leaders:
            name, team, val = _leader_name_team_value(l)
            rows.append((name, team, val))

        slug = _safe_filename(display_name)

        png_path = _make_one_poster(
            season=season,
            seasontype=seasontype,
            category_display=display_name,
            filename_slug=slug,
            rows=rows,
        )

        key = f"{prefix}{os.path.basename(png_path)}"
        url = upload_file_return_url(png_path, key)
        uploaded_urls.append(url)

    # If nothing uploaded, surface a real error
    if not uploaded_urls:
        raise RuntimeError("No stat leader posters were generated/uploaded. ESPN response may have changed.")

    # Return sorted for stable ordering
    uploaded_urls.sort()
    return uploaded_urls


# Allow local/GHA runs:
if __name__ == "__main__":
    season = int(os.getenv("SEASON", "2025"))
    seasontype = int(os.getenv("SEASONTYPE", "2"))
    urls = generate_stat_leaders_and_upload(season, seasontype)
    print(json.dumps({"season": season, "seasontype": seasontype, "count": len(urls), "images": urls}, indent=2))
