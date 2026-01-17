# app/scripts/nfl_stat_leaders_generate.py
#
# Based on your exact script:
# - Same ESPN scraping / pandas.read_html logic
# - Same name normalization (fix glued TEAM)
# - CHANGED OUTPUT: 10 separate posters (one per category)
# - Phone-sized posters: 1440 x 2560 (big + readable like earlier)
# - Designed for GitHub/Render Linux (font fallback included)
#
# Output filenames (and intended order):
# 01_passing_yards.png
# 02_passing_tds.png
# 03_interceptions_thrown.png
# 04_rushing_yards.png
# 05_rushing_tds.png
# 06_receiving_yards.png
# 07_receiving_tds.png
# 08_sacks.png
# 09_tackles.png
# 10_interceptions.png

import os
import re
import argparse
from io import StringIO
from datetime import datetime
from typing import List, Optional, Tuple, Union

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TOP_N = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )
}

Number = Union[int, float]

TEAM_ABBRS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND",
    "JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA",
    "SF","TB","TEN","WAS","WSH"
}

TEAM_ALT = "|".join(sorted(TEAM_ABBRS, key=len, reverse=True))
TEAM_END_RE = re.compile(rf"^(?P<name>.*?)(?P<team>{TEAM_ALT})(?P<trail>[\s\W]*)$")


# ----------------------------
# String normalization (fix glued TEAM)
# ----------------------------
def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)  # zero-width chars
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def enforce_space_before_team(s: str) -> str:
    s = normalize_spaces(s)
    m = TEAM_END_RE.match(s)
    if not m:
        return s
    name_part = m.group("name").strip()
    team = m.group("team").strip()
    name_part = re.sub(r"[^\w\.\-'\s]+$", "", name_part).strip()
    return f"{name_part} {team}".strip()


# ----------------------------
# ESPN URLs (seasontype: 2 regular, 3 postseason)
# ----------------------------
def build_urls(season: int, seasontype: int):
    base_player = f"https://www.espn.com/nfl/stats/player/_/season/{season}/seasontype/{seasontype}"
    base_rush = f"https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{season}/seasontype/{seasontype}"
    base_rec = f"https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{season}/seasontype/{seasontype}"
    base_def = f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{seasontype}"

    return {
        "Passing Yards": (
            base_player,
            ["YDS", "PASS YDS", "Pass YDS"],
            "int",
        ),
        "Passing TDs": (
            f"{base_player}/table/passing/sort/passingTouchdowns/dir/desc",
            ["TD", "PASS TD", "Pass TD"],
            "int",
        ),
        "Interceptions Thrown": (
            f"{base_player}/table/passing/sort/interceptions/dir/desc",
            ["INT", "Interceptions"],
            "int",
        ),
        "Rushing Yards": (
            base_rush,
            ["YDS", "RUSH YDS", "Rush YDS"],
            "int",
        ),
        "Rushing TDs": (
            f"{base_rush}/table/rushing/sort/rushingTouchdowns/dir/desc",
            ["TD", "RUSH TD", "Rush TD"],
            "int",
        ),
        "Receiving Yards": (
            base_rec,
            ["YDS", "REC YDS", "Rec YDS"],
            "int",
        ),
        "Receiving TDs": (
            f"{base_rec}/table/receiving/sort/receivingTouchdowns/dir/desc",
            ["TD", "REC TD", "Rec TD"],
            "int",
        ),
        "Sacks": (
            f"{base_def}/table/defensive/sort/sacks/dir/desc",
            ["SACK", "Sacks SACK", "Sacks"],
            "float1",
        ),
        "Tackles": (
            f"{base_def}/table/defensive/sort/totalTackles/dir/desc",
            ["TOT", "Tackles TOT", "Total", "Tackles"],
            "int",
        ),
        "Interceptions (Defense)": (
            f"{base_def}/table/defensiveInterceptions/sort/interceptions/dir/desc",
            ["INT", "Interceptions INT", "Interceptions"],
            "int",
        ),
    }


# ----------------------------
# Table parsing helpers
# ----------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip()]).strip()
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
    except ValueError:
        return None


def safe_int(x) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    return int(round(f))


def col_lookup(cols: List[str]) -> dict:
    return {str(c).strip().lower(): c for c in cols}


def fetch_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    return [flatten_columns(t.copy()) for t in tables]


def pick_name_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    for t in tables:
        cmap = col_lookup(list(t.columns))
        if "name" in cmap and t.shape[1] <= 4:
            return t
    for t in tables:
        cmap = col_lookup(list(t.columns))
        if "name" in cmap:
            return t
    return None


def find_stat_table(tables: List[pd.DataFrame], candidates: List[str]) -> Optional[pd.DataFrame]:
    cand_l = {c.strip().lower() for c in candidates}
    for t in tables:
        cols_l = {c.strip().lower() for c in t.columns}
        for cand in cand_l:
            for col in cols_l:
                if cand == col or cand in col:
                    return t
    return None


def pick_widest_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not tables:
        return None
    return sorted(tables, key=lambda d: d.shape[1], reverse=True)[0]


def stitch_tables_if_needed(name_t: Optional[pd.DataFrame], stat_t: pd.DataFrame) -> pd.DataFrame:
    cmap = col_lookup(list(stat_t.columns))
    if "name" in cmap:
        return stat_t

    if name_t is None:
        return stat_t

    if len(name_t) == len(stat_t):
        st = stat_t.copy()
        if any(str(c).strip().lower() == "rk" for c in st.columns):
            st = st.drop(columns=[c for c in st.columns if str(c).strip().lower() == "rk"])
        return pd.concat([name_t.reset_index(drop=True), st.reset_index(drop=True)], axis=1)

    return stat_t


def choose_stat_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    cols_l = [str(c).strip().lower() for c in cols]
    cand_l = [c.strip().lower() for c in candidates]

    for ck in cand_l:
        for i, colk in enumerate(cols_l):
            if colk == ck:
                return cols[i]
    for ck in cand_l:
        for i, colk in enumerate(cols_l):
            if ck in colk:
                return cols[i]

    raise RuntimeError(f"Stat column not found. Candidates={candidates}. Columns={list(df.columns)}")


def topN_from_url(url: str, stat_candidates: List[str], mode: str) -> List[Tuple[int, str, Number]]:
    tables = fetch_tables(url)
    if not tables:
        raise RuntimeError(f"No tables found at: {url}")

    name_t = pick_name_table(tables)
    stat_t = find_stat_table(tables, stat_candidates)
    if stat_t is None:
        stat_t = pick_widest_table(tables)
    if stat_t is None:
        raise RuntimeError(f"Could not choose a stat table at: {url}")

    df = stitch_tables_if_needed(name_t, stat_t)

    cmap = col_lookup(list(df.columns))
    if "name" not in cmap:
        raise RuntimeError(f"No Name column after stitching. Columns={list(df.columns)}")
    name_col = cmap["name"]

    stat_col = choose_stat_col(df, stat_candidates)

    work = df[[name_col, stat_col]].copy()
    if mode == "float1":
        work["__val__"] = work[stat_col].map(safe_float)
    else:
        work["__val__"] = work[stat_col].map(safe_int)

    work = work.dropna(subset=["__val__"]).sort_values("__val__", ascending=False).head(TOP_N)

    out = []
    for i, rec in enumerate(work.to_dict("records"), start=1):
        raw = normalize_spaces(rec[name_col])
        display_name = enforce_space_before_team(raw)
        out.append((i, display_name, rec["__val__"]))
    return out


# ----------------------------
# Poster drawing (PHONE BIG)
# ----------------------------
def load_font(size: int, bold: bool = False):
    """
    Works on Linux (GitHub Actions/Render) and Mac locally.
    """
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

    # Mac fallbacks
    candidates += [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def fmt_value(val: Number, mode: str) -> str:
    if mode == "float1":
        return f"{float(val):.1f}"
    return str(int(val))


def draw_single_category_poster(
    out_path: str,
    category_title: str,
    subtitle: str,
    items: List[Tuple[int, str, Number]],
    mode: str,
):
    # Phone poster size
    W, H = 1440, 2560
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    # BIG typography (readable on phone)
    title_font = load_font(74, bold=True)
    sub_font = load_font(42, bold=False)
    row_name_font = load_font(52, bold=True)
    row_team_font = load_font(42, bold=False)
    val_font = load_font(64, bold=True)
    rank_font = load_font(46, bold=True)

    # Header
    margin_x = 90
    top_y = 140
    d.text((margin_x, top_y), category_title, font=title_font, fill=(245, 245, 245))
    d.text((margin_x, top_y + 92), subtitle, font=sub_font, fill=(180, 180, 190))

    # Divider
    div_y = top_y + 240 - 35
    d.line((margin_x, div_y, W - margin_x, div_y), fill=(85, 85, 85), width=3)

    # Rows
    row_h = 175
    start_y = top_y + 240

    for i, (rank, display_name, val) in enumerate(items, start=1):
        y = start_y + (i - 1) * row_h

        # rank
        d.text((margin_x, y + 52), f"{rank}", font=rank_font, fill=(170, 170, 170))

        # split name/team for nicer layout
        dn = str(display_name).strip()
        m = TEAM_END_RE.match(dn)
        if m:
            nm = m.group("name").strip()
            tm = m.group("team").strip()
        else:
            nm = dn
            tm = "--"

        name_x = margin_x + 90
        d.text((name_x, y + 32), nm, font=row_name_font, fill=(255, 255, 255))
        d.text((name_x, y + 98), tm, font=row_team_font, fill=(170, 170, 170))

        # value right aligned
        val_txt = fmt_value(val, mode)
        tw = d.textlength(val_txt, font=val_font)
        d.text((W - margin_x - tw, y + 45), val_txt, font=val_font, fill=(255, 255, 255))

        # row divider
        d.line((margin_x, y + row_h - 18, W - margin_x, y + row_h - 18), fill=(55, 55, 55), width=2)

    img.save(out_path, "PNG", optimize=True)


def safe_slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "category"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seasontype", type=int, default=2, choices=[2, 3], help="2=Regular, 3=Postseason")
    ap.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "Desktop"),
        help="Where to write PNGs (Render/GHA can point this to a temp folder).",
    )
    args = ap.parse_args()

    season = args.season
    seasontype = args.seasontype
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    phase = "Regular Season" if seasontype == 2 else "Postseason"
    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"Season {season} • {phase} • Updated {updated}"

    URLS = build_urls(season, seasontype)

    # Exact desired order
    ordered = [
        ("Passing Yards", "01_passing_yards"),
        ("Passing TDs", "02_passing_tds"),
        ("Interceptions Thrown", "03_interceptions_thrown"),
        ("Rushing Yards", "04_rushing_yards"),
        ("Rushing TDs", "05_rushing_tds"),
        ("Receiving Yards", "06_receiving_yards"),
        ("Receiving TDs", "07_receiving_tds"),
        ("Sacks", "08_sacks"),
        ("Tackles", "09_tackles"),
        ("Interceptions (Defense)", "10_interceptions"),
    ]

    print("")
    print(f"Generating {len(ordered)} stat leader posters for season={season} seasontype={seasontype}…")
    print("")

    for title, outbase in ordered:
        url, cand, mode = URLS[title]
        items = topN_from_url(url, cand, mode)

        # Display title for defense interceptions should be "Interceptions"
        display_title = "Interceptions" if title == "Interceptions (Defense)" else title

        out_path = os.path.join(outdir, f"{outbase}.png")
        draw_single_category_poster(out_path, display_title, subtitle, items, mode)
        print(f"✅ {display_title} -> {out_path}")

    print("\nDONE ✅\n")


if __name__ == "__main__":
    main()
