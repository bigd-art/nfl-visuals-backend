# app/scripts/nfl_team_stat_leaders_generate.py
# ==========================================================
# ESPN Team Stat Leaders (REGULAR + POSTSEASON) — ROBUST
#
# Key fix:
# - DO NOT pair separate "names" + "stats" tables (ESPN order shifts)
# - Instead fetch ESPN "table=..." pages where NAME + STAT are in the SAME table.
#
# Public API used by router:
#   leaders = extract_team_leaders(team, season, seasontype)
#   team_gen.draw_leaders_grid_poster(...)
#
# CLI local test:
#   python3 -m app.scripts.nfl_team_stat_leaders_generate --team SEA --season 2025 --seasontype 3 --outdir .
# ==========================================================

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

Number = Union[int, float]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

TEAM_TO_ESPN = {
    "ARI": "ari/arizona-cardinals",
    "ATL": "atl/atlanta-falcons",
    "BAL": "bal/baltimore-ravens",
    "BUF": "buf/buffalo-bills",
    "CAR": "car/carolina-panthers",
    "CHI": "chi/chicago-bears",
    "CIN": "cin/cincinnati-bengals",
    "CLE": "cle/cleveland-browns",
    "DAL": "dal/dallas-cowboys",
    "DEN": "den/denver-broncos",
    "DET": "det/detroit-lions",
    "GB": "gb/green-bay-packers",
    "HOU": "hou/houston-texans",
    "IND": "ind/indianapolis-colts",
    "JAX": "jax/jacksonville-jaguars",
    "KC": "kc/kansas-city-chiefs",
    "LAC": "lac/los-angeles-chargers",
    "LAR": "lar/los-angeles-rams",
    "LV": "lv/las-vegas-raiders",
    "MIA": "mia/miami-dolphins",
    "MIN": "min/minnesota-vikings",
    "NE": "ne/new-england-patriots",
    "NO": "no/new-orleans-saints",
    "NYG": "nyg/new-york-giants",
    "NYJ": "nyj/new-york-jets",
    "PHI": "phi/philadelphia-eagles",
    "PIT": "pit/pittsburgh-steelers",
    "SEA": "sea/seattle-seahawks",
    "SF": "sf/san-francisco-49ers",
    "TB": "tb/tampa-bay-buccaneers",
    "TEN": "ten/tennessee-titans",
    "WAS": "wsh/washington-commanders",
    "WSH": "wsh/washington-commanders",
}

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    s = s.replace(",", "")
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

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c

def _fetch_html(url: str, timeout: int = 30, retries: int = 4) -> str:
    last = None
    s = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            r = s.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            html = r.text or ""
            if "<table" not in html.lower():
                snippet = normalize_spaces(html[:400])
                raise RuntimeError(f"ESPN HTML has no <table>. Snippet: {snippet}")
            return html
        except Exception as e:
            last = e
            time.sleep(1.4 * attempt)
    raise RuntimeError(f"Failed to fetch ESPN after {retries} tries: {last}")

def _read_tables(url: str) -> List[pd.DataFrame]:
    html = _fetch_html(url)
    try:
        tabs = pd.read_html(StringIO(html))
    except Exception as e:
        raise RuntimeError(f"pd.read_html failed for url={url}. Error: {e}")
    if not tabs:
        raise RuntimeError(f"pd.read_html returned 0 tables for url={url}")
    return tabs

def _pick_best_player_table(tables: List[pd.DataFrame], stat_col_wanted: str) -> pd.DataFrame:
    """
    ESPN 'table=' pages typically have ONE main table with NAME + stats.
    We select the table that contains the wanted stat col and has a player-ish column.
    """
    wanted = _norm_col(stat_col_wanted)

    def has_player_col(df: pd.DataFrame) -> bool:
        df = flatten_columns(df.copy())
        cols = [_norm_col(c) for c in df.columns]
        # Common ESPN player columns:
        return any(c in {"name", "player"} for c in cols) or any("name" in c or "player" in c for c in cols)

    def has_stat_col(df: pd.DataFrame) -> bool:
        df = flatten_columns(df.copy())
        cols = [_norm_col(c) for c in df.columns]
        if wanted in cols:
            return True
        # allow partial match if ESPN uses something like "total tackles" etc
        return any(wanted == c or wanted in c for c in cols)

    scored: List[Tuple[int, pd.DataFrame]] = []
    for t in tables:
        try:
            tt = flatten_columns(t.copy())
            score = 0
            if has_player_col(tt):
                score += 2
            if has_stat_col(tt):
                score += 3
            # bigger table usually the main one
            score += min(len(tt), 50) // 10
            scored.append((score, tt))
        except Exception:
            continue

    if not scored:
        raise RuntimeError("No readable tables found.")

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return best

def _get_player_name(series_row: pd.Series, df_cols_norm: List[str]) -> str:
    """
    Build a clean player label.
    Prefer Name/Player columns; append POS if present.
    """
    # find name col
    name_idx = None
    for i, c in enumerate(df_cols_norm):
        if c in {"name", "player"} or ("name" in c) or ("player" in c):
            name_idx = i
            break

    if name_idx is None:
        # fallback: first column
        name_idx = 0

    name = normalize_spaces(series_row.iloc[name_idx])

    # append position if present
    pos = ""
    for i, c in enumerate(df_cols_norm):
        if c in {"pos", "position"}:
            pos = normalize_spaces(series_row.iloc[i])
            break

    if pos and pos.lower() not in {"nan", "none", "-"}:
        # avoid duplicating if ESPN already includes position in name string
        if re.search(r"\b" + re.escape(pos) + r"\b", name):
            return name
        return f"{name} {pos}".strip()

    return name

def _find_stat_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]
    for cand in candidates:
        c0 = _norm_col(cand)
        # exact first
        for i, c in enumerate(low):
            if c == c0:
                return cols[i]
        # contains fallback
        for i, c in enumerate(low):
            if c0 in c:
                return cols[i]
    raise RuntimeError(f"Could not find any of stat candidates {candidates} in columns={cols}")

def _leader_from_single_table(df: pd.DataFrame, stat_col: str, mode: str) -> Tuple[str, Number]:
    df = flatten_columns(df.copy())

    cols_norm = [_norm_col(c) for c in df.columns]
    stat_series = df[stat_col]

    if mode == "float1":
        vals = stat_series.map(safe_float)
    else:
        vals = stat_series.map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")

    # Drop empty rows but keep alignment with df
    good = vals_num.notna()
    df2 = df.loc[good].reset_index(drop=True)
    vals2 = vals_num.loc[good].reset_index(drop=True)

    if df2.empty:
        raise RuntimeError(f"All values empty/NaN for stat_col={stat_col}")

    best_pos = int(vals2.values.argmax())
    who = _get_player_name(df2.iloc[best_pos], cols_norm)
    best_val = vals2.iloc[best_pos]
    return who, best_val

def build_team_stats_url(team: str, season: int, seasontype: int, table: str, sort: str) -> str:
    """
    Reliable format for ESPN team stats table pages:
    https://www.espn.com/nfl/team/stats/_/season/{season}/seasontype/{seasontype}/name/{slug}/table/{table}/sort/{sort}/dir/desc
    """
    t = team.strip().upper()
    if t not in TEAM_TO_ESPN:
        raise ValueError(f"Unknown team: {t}")
    slug = TEAM_TO_ESPN[t]
    return (
        "https://www.espn.com/nfl/team/stats/_"
        f"/season/{int(season)}"
        f"/seasontype/{int(seasontype)}"
        f"/name/{slug}"
        f"/table/{table}"
        f"/sort/{sort}"
        "/dir/desc"
    )

@dataclass(frozen=True)
class StatSpec:
    label: str
    table: str
    sort: str
    stat_candidates: List[str]
    mode: str  # "int" or "float1"

STAT_SPECS: List[StatSpec] = [
    # offense
    StatSpec("Passing Yards", "passing", "passingYards", ["YDS", "YDS/G", "PASS YDS", "PASSING YDS", "passing yards"], "int"),
    StatSpec("Passing TDs", "passing", "passingTouchdowns", ["TD", "PASS TD", "PASSING TD", "passing touchdowns"], "int"),
    StatSpec("Interceptions Thrown", "passing", "interceptions", ["INT", "INTS", "INTERCEPTIONS"], "int"),

    StatSpec("Rushing Yards", "rushing", "rushingYards", ["YDS", "RUSH YDS", "RUSHING YDS", "rushing yards"], "int"),
    StatSpec("Rushing TDs", "rushing", "rushingTouchdowns", ["TD", "RUSH TD", "RUSHING TD", "rushing touchdowns"], "int"),

    StatSpec("Receiving Yards", "receiving", "receivingYards", ["YDS", "REC YDS", "RECEIVING YDS", "receiving yards"], "int"),
    StatSpec("Receiving TDs", "receiving", "receivingTouchdowns", ["TD", "REC TD", "RECEIVING TD", "receiving touchdowns"], "int"),

    # defense
    StatSpec("Sacks", "defensive", "sacks", ["SACK", "SACKS"], "float1"),
    StatSpec("Tackles", "defensive", "totalTackles", ["TOT", "TKL", "TACK", "TOTAL", "TOTAL TACKLES"], "int"),
    StatSpec("Interceptions", "defensive", "interceptions", ["INT", "INTS", "INTERCEPTIONS"], "int"),
]

def extract_team_leaders(team: str, season: int, seasontype: int) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Returns dict:
      leaders["Sacks"] -> ("1", "Player Name POS", "", "2.0")
    """
    out: Dict[str, Tuple[str, str, str, str]] = {}

    for spec in STAT_SPECS:
        url = build_team_stats_url(team, season, seasontype, spec.table, spec.sort)
        tables = _read_tables(url)

        # pick main table then locate correct stat column
        # (for tackles, ESPN sometimes uses TOT; we pick from candidates)
        df = _pick_best_player_table(tables, spec.stat_candidates[0])
        stat_col = _find_stat_col(df, spec.stat_candidates)

        who, val = _leader_from_single_table(df, stat_col, spec.mode)

        if spec.mode == "float1":
            val_str = f"{float(val):.1f}"
        else:
            val_str = str(int(val))

        out[spec.label] = ("1", who, "", val_str)

    return out

def generate_posters(team: str, season: int, seasontype: int, outdir: str) -> Tuple[str, str]:
    """
    Generates 2 PNGs (offense/defense) and returns (out_off, out_def).
    """
    team = team.strip().upper()
    os.makedirs(outdir, exist_ok=True)

    leaders = extract_team_leaders(team, season, seasontype)

    season_label = "Postseason" if int(seasontype) == 3 else "Regular Season"
    updated = time.strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"{team} • {season_label} • Updated {updated}"

    offense_order = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
    ]
    defense_order = ["Sacks", "Tackles", "Interceptions"]

    # build sections (label, rank, name, value)
    offense_sections = [(cat, leaders[cat][0], leaders[cat][1], leaders[cat][3]) for cat in offense_order]
    defense_sections = [(cat, leaders[cat][0], leaders[cat][1], leaders[cat][3]) for cat in defense_order]

    out_off = os.path.join(outdir, f"{team.lower()}_{season}_{seasontype}_offense.png")
    out_def = os.path.join(outdir, f"{team.lower()}_{season}_{seasontype}_defense.png")

    draw_leaders_grid_poster(
        out_off,
        "Offensive Statistical Leaders",
        subtitle,
        offense_sections,
        cols=2,
        rows=4,
    )

    draw_leaders_grid_poster(
        out_def,
        "Defensive Statistical Leaders",
        subtitle,
        defense_sections,
        cols=1,
        rows=3,
    )

    return out_off, out_def

# ==========================================================
# POSTER RENDERER — KEEP STYLE IDENTICAL
# ==========================================================

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_leaders_grid_poster(
    out_path: str,
    title: str,
    subtitle: str,
    sections,
    cols: int = 2,
    rows: int = 4,
):
    W, H = 1400, 2400
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = _load_font(60, bold=True)
    sub_font   = _load_font(28, bold=False)
    head_font  = _load_font(30, bold=True)
    name_font  = _load_font(26, bold=False)
    val_font   = _load_font(34, bold=True)

    d.text((70, 60), title, font=title_font, fill=(245, 245, 245))
    d.text((70, 135), subtitle, font=sub_font, fill=(170, 170, 185))

    pad = 60
    top = 200
    grid_x0, grid_y0 = pad, top
    grid_x1, grid_y1 = W - pad, H - 120

    d.rounded_rectangle(
        [grid_x0, grid_y0, grid_x1, grid_y1],
        radius=26,
        fill=(20, 20, 28),
        outline=(45, 45, 60),
        width=2,
    )

    cell_w = (grid_x1 - grid_x0) / cols
    cell_h = (grid_y1 - grid_y0) / rows

    def _fit_text(text: str, max_w: float, font: ImageFont.FreeTypeFont) -> str:
        t = str(text)
        if d.textlength(t, font=font) <= max_w:
            return t
        while len(t) > 3 and d.textlength(t + "…", font=font) > max_w:
            t = t[:-1]
        return t + "…"

    for i, sec in enumerate(sections):
        if i >= cols * rows:
            break

        label = sec[0]
        if len(sec) == 4:
            _, rank, name, value = sec
            team = ""
        else:
            _, rank, name, team, value = sec

        cx = grid_x0 + (i % cols) * cell_w
        cy = grid_y0 + (i // cols) * cell_h

        px = 26
        py = 22

        if (i % cols) != 0:
            d.line([(cx, cy + 18), (cx, cy + cell_h - 18)], fill=(35, 35, 48), width=2)
        if (i // cols) != 0:
            d.line([(cx + 18, cy), (cx + cell_w - 18, cy)], fill=(35, 35, 48), width=2)

        label_text = _fit_text(label, cell_w - 2 * px, head_font)
        d.text((cx + px, cy + py), label_text, font=head_font, fill=(235, 235, 245))

        who = f"{name}"
        if team:
            who = f"{name} • {team}"
        who = _fit_text(who, cell_w - 2 * px, name_font)
        d.text((cx + px, cy + py + 48), who, font=name_font, fill=(170, 170, 185))

        val = str(value)
        tw = d.textlength(val, font=val_font)
        d.text((cx + cell_w - px - tw, cy + py + 10), val, font=val_font, fill=(235, 235, 245))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")

# ==========================================================
# CLI (LOCAL TESTING)
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="SEA, DAL, KC, etc.")
    ap.add_argument("--season", required=True, type=int, help="Season year, e.g. 2025")
    ap.add_argument("--seasontype", required=True, type=int, choices=[2, 3], help="2=Regular, 3=Postseason")
    ap.add_argument("--outdir", default=".", help="Output folder")
    args = ap.parse_args()

    off, deff = generate_posters(args.team, args.season, args.seasontype, args.outdir)
    print("Wrote:")
    print(" -", off)
    print(" -", deff)

if __name__ == "__main__":
    main()
