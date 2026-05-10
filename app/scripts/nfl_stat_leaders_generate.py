import os
import re
import argparse
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict, Any

import requests
from PIL import Image, ImageDraw, ImageFont

TOP_N = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

Number = Union[int, float]

TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS", "WSH",
}

STAT_CONFIG = [
    ("passing_yards", "Passing Yards", "Passing Yards", ["passingYards", "passing yards"]),
    ("passing_tds", "Passing TDs", "Passing TDs", ["passingTouchdowns", "passing touchdowns", "passing tds"]),
    ("interceptions_thrown", "Interceptions Thrown", "Interceptions Thrown", ["interceptions", "interceptions thrown"]),
    ("rushing_yards", "Rushing Yards", "Rushing Yards", ["rushingYards", "rushing yards"]),
    ("rushing_tds", "Rushing TDs", "Rushing TDs", ["rushingTouchdowns", "rushing touchdowns", "rushing tds"]),
    ("receiving_yards", "Receiving Yards", "Receiving Yards", ["receivingYards", "receiving yards"]),
    ("receiving_tds", "Receiving TDs", "Receiving TDs", ["receivingTouchdowns", "receiving touchdowns", "receiving tds"]),
    ("sacks", "Sacks", "Sacks", ["sacks"]),
    ("tackles", "Tackles", "Tackles", ["totalTackles", "total tackles", "tackles"]),
    ("interceptions_defense", "Interceptions (Defense)", "Interceptions", ["defensiveInterceptions", "interceptions"]),
]

CORE_LEADERS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    "seasons/{season}/types/{seasontype}/leaders?lang=en&region=us"
)


def normalize_spaces(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u00ad", "")
    return re.sub(r"\s+", " ", s).strip()


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).replace(",", "").strip()
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in {"", "-", "."}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def safe_int(x) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    return int(round(f))


def fetch_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def resolve_ref(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict) and "$ref" in obj:
        try:
            return fetch_json(obj["$ref"])
        except Exception:
            return obj
    if isinstance(obj, dict):
        return obj
    return {}


def text_key(value: str) -> str:
    return normalize_spaces(value).lower().replace("_", " ").replace("-", " ")


def category_matches(category: Dict[str, Any], aliases: List[str], slug: str) -> bool:
    raw_fields = [
        category.get("name"),
        category.get("displayName"),
        category.get("shortDisplayName"),
        category.get("description"),
        category.get("abbreviation"),
    ]

    combined = " ".join(text_key(x) for x in raw_fields if x)

    for alias in aliases:
        a = text_key(alias)
        if a and a in combined:
            return True

    if slug == "interceptions_thrown":
        return "interception" in combined and "defensive" not in combined

    if slug == "interceptions_defense":
        return "interception" in combined and (
            "defensive" in combined or "defense" in combined or "def" in combined
        )

    return False


def find_leader_categories(data: Any) -> List[Dict[str, Any]]:
    found = []

    if isinstance(data, dict):
        if "leaders" in data and isinstance(data["leaders"], list):
            found.append(data)

        for value in data.values():
            found.extend(find_leader_categories(value))

    elif isinstance(data, list):
        for item in data:
            found.extend(find_leader_categories(item))

    return found


def extract_athlete_name(leader: Dict[str, Any]) -> str:
    athlete = resolve_ref(
        leader.get("athlete")
        or leader.get("player")
        or leader.get("person")
        or {}
    )

    return normalize_spaces(
        athlete.get("displayName")
        or athlete.get("fullName")
        or athlete.get("shortName")
        or leader.get("displayName")
        or leader.get("name")
        or "Unknown Player"
    )


def extract_team_abbr(leader: Dict[str, Any]) -> str:
    team_obj = leader.get("team") or leader.get("teamAthlete") or {}

    if isinstance(team_obj, dict) and "$ref" in team_obj:
        team_obj = resolve_ref(team_obj)

    abbr = normalize_spaces(
        team_obj.get("abbreviation")
        or team_obj.get("shortDisplayName")
        or team_obj.get("name")
        or ""
    ).upper()

    if abbr == "WAS":
        abbr = "WSH"

    if abbr in TEAM_ABBRS:
        return abbr

    return ""


def extract_leader_value(leader: Dict[str, Any]) -> Optional[float]:
    for key in ["value", "displayValue", "stat", "score"]:
        if key in leader:
            val = safe_float(leader.get(key))
            if val is not None:
                return val

    statistics = leader.get("statistics") or leader.get("stats") or []
    if isinstance(statistics, list):
        for s in statistics:
            if isinstance(s, dict):
                val = safe_float(s.get("value") or s.get("displayValue"))
                if val is not None:
                    return val

    return None


def fetch_top_from_leaders_api(
    season: int,
    seasontype: int,
    slug: str,
    aliases: List[str],
    mode: str,
) -> List[Tuple[int, str, Number]]:
    url = CORE_LEADERS_URL.format(season=season, seasontype=seasontype)
    data = fetch_json(url)

    categories = find_leader_categories(data)

    matched = None
    for category in categories:
        if category_matches(category, aliases, slug):
            matched = category
            break

    if not matched:
        available = []
        for c in categories[:40]:
            available.append({
                "name": c.get("name"),
                "displayName": c.get("displayName"),
                "shortDisplayName": c.get("shortDisplayName"),
            })
        raise RuntimeError(f"No matching category for {slug}. Available sample: {available}")

    rows = []
    for leader in matched.get("leaders", []):
        if not isinstance(leader, dict):
            continue

        name = extract_athlete_name(leader)
        team = extract_team_abbr(leader)
        value = extract_leader_value(leader)

        if value is None:
            continue

        display_name = f"{name} {team}".strip()
        rows.append((display_name, value))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)[:TOP_N]

    output = []
    for i, (name, val) in enumerate(rows, start=1):
        if mode == "float1":
            output.append((i, name, float(val)))
        else:
            output.append((i, name, int(round(val))))

    if not output:
        raise RuntimeError(f"No usable leaders for {slug}")

    return output


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


def fmt_value(val: Number, mode: str) -> str:
    if mode == "float1":
        return f"{float(val):.1f}"
    return f"{int(val):,}"


def fit_text(draw, text: str, font, max_width: int) -> str:
    text = str(text)
    if draw.textlength(text, font=font) <= max_width:
        return text

    while len(text) > 3 and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]

    return text.rstrip() + "…"


def split_name_team(display_name: str) -> Tuple[str, str]:
    display_name = normalize_spaces(display_name)
    parts = display_name.split()

    if parts and parts[-1].upper() in TEAM_ABBRS:
        team = parts[-1].upper()
        if team == "WAS":
            team = "WSH"
        return " ".join(parts[:-1]), team

    return display_name, ""


def draw_single_stat_poster(
    out_path: str,
    poster_title: str,
    stat_title: str,
    subtitle: str,
    items: List[Tuple[int, str, Number]],
    mode: str,
):
    width, height = 1080, 1920

    img = Image.new("RGB", (width, height), (10, 14, 24))
    draw = ImageDraw.Draw(img)

    title_font = load_font(48, bold=True)
    stat_font = load_font(68, bold=True)
    sub_font = load_font(22, bold=False)

    rank_font = load_font(31, bold=True)
    name_font = load_font(43, bold=True)
    team_font = load_font(24, bold=True)
    value_font = load_font(45, bold=True)

    white = (246, 248, 252)
    muted = (208, 218, 238)
    blue = (128, 183, 255)
    dark = (24, 29, 42)
    border = (64, 74, 98)

    draw.rectangle((0, 0, width, 178), fill=(22, 38, 74))
    draw.rectangle((0, 178, width, 187), fill=blue)

    for y in range(198, height, 30):
        color = (14, 18, 28) if (y // 30) % 2 == 0 else (12, 16, 26)
        draw.rectangle((0, y, width, y + 15), fill=color)

    title = fit_text(draw, poster_title.upper(), title_font, width - 90)
    stat = fit_text(draw, stat_title.upper(), stat_font, width - 90)
    sub = fit_text(draw, subtitle, sub_font, width - 90)

    draw.text(
        ((width - draw.textlength(title, font=title_font)) / 2, 24),
        title,
        font=title_font,
        fill=white,
    )

    draw.text(
        ((width - draw.textlength(stat, font=stat_font)) / 2, 78),
        stat,
        font=stat_font,
        fill=white,
    )

    draw.text(
        ((width - draw.textlength(sub, font=sub_font)) / 2, 143),
        sub,
        font=sub_font,
        fill=muted,
    )

    x0, x1 = 42, width - 42
    top = 220
    bottom = height - 42
    gap = 14
    row_h = int((bottom - top - gap * (TOP_N - 1)) / TOP_N)

    for rank, display_name, val in items:
        y0 = top + (rank - 1) * (row_h + gap)
        y1 = y0 + row_h

        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=26,
            fill=dark,
            outline=border,
            width=3,
        )

        pill = (x0 + 18, y0 + 22, x0 + 92, y0 + 84)
        draw.rounded_rectangle(pill, radius=18, fill=blue)

        rank_text = str(rank)
        rank_w = draw.textlength(rank_text, font=rank_font)

        draw.text(
            (
                pill[0] + (pill[2] - pill[0] - rank_w) / 2,
                pill[1] + 12,
            ),
            rank_text,
            font=rank_font,
            fill=(15, 20, 28),
        )

        player_name, team = split_name_team(display_name)
        value_text = fmt_value(val, mode)
        value_w = draw.textlength(value_text, font=value_font)

        draw.text(
            (x1 - 30 - value_w, y0 + 31),
            value_text,
            font=value_font,
            fill=white,
        )

        max_name_w = x1 - x0 - 150 - value_w - 55
        name_text = fit_text(draw, player_name.upper(), name_font, max_name_w)

        draw.text(
            (x0 + 112, y0 + 24),
            name_text,
            font=name_font,
            fill=white,
        )

        if team:
            tag_x1 = x0 + 114
            tag_y1 = y0 + 83
            tag_x2 = tag_x1 + 92
            tag_y2 = tag_y1 + 38

            draw.rounded_rectangle(
                (tag_x1, tag_y1, tag_x2, tag_y2),
                radius=13,
                fill=(16, 28, 54),
                outline=(86, 104, 140),
                width=1,
            )

            team_w = draw.textlength(team, font=team_font)

            draw.text(
                (tag_x1 + (tag_x2 - tag_x1 - team_w) / 2, tag_y1 + 5),
                team,
                font=team_font,
                fill=blue,
            )

        draw.rectangle((x0 + 18, y1 - 12, x1 - 18, y1 - 8), fill=blue)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")


def stat_mode_for_slug(slug: str) -> str:
    if slug == "sacks":
        return "float1"
    return "int"


def build_stat_sections(season: int, seasontype: int) -> Dict[str, Tuple[str, List[Tuple[int, str, Number]], str]]:
    output = {}

    for slug, _espn_title, short_title, aliases in STAT_CONFIG:
        mode = stat_mode_for_slug(slug)

        try:
            items = fetch_top_from_leaders_api(
                season=season,
                seasontype=seasontype,
                slug=slug,
                aliases=aliases,
                mode=mode,
            )
            output[slug] = (short_title, items, mode)
            print(f"Generated data for {slug}: {len(items)} rows")
        except Exception as e:
            print(f"WARNING: Failed to fetch {slug}: {e}")

    return output


def generate_all_stat_leader_posters(season: int, seasontype: int, outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)

    phase = "Regular Season" if seasontype == 2 else "Postseason"
    updated = datetime.utcnow().strftime("%b %d, %Y • %I:%M %p UTC")
    subtitle = f"Season {season} • {phase} • Updated {updated}"

    sections = build_stat_sections(season, seasontype)
    outputs = {}

    for slug, _espn_title, short_title, _aliases in STAT_CONFIG:
        if slug not in sections:
            print(f"WARNING: Skipping poster for {slug}; no data available")
            continue

        stat_title, items, mode = sections[slug]
        out_path = os.path.join(outdir, f"{slug}_s{season}_t{seasontype}.png")

        draw_single_stat_poster(
            out_path=out_path,
            poster_title="NFL Statistical Leaders",
            stat_title=stat_title,
            subtitle=subtitle,
            items=items,
            mode=mode,
        )

        outputs[slug] = out_path

    if not outputs:
        raise RuntimeError(f"No stat leader posters generated for season={season}, seasontype={seasontype}")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--seasontype", type=int, default=2, choices=[2, 3])
    parser.add_argument("--outdir", type=str, default=os.path.join(os.path.expanduser("~"), "Desktop"))
    args = parser.parse_args()

    outputs = generate_all_stat_leader_posters(
        season=args.season,
        seasontype=args.seasontype,
        outdir=args.outdir,
    )

    print("\nDONE")
    for slug, path in outputs.items():
        print(slug, "->", path)


if __name__ == "__main__":
    main()
