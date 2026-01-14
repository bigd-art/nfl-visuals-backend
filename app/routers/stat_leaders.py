import os
import re
import argparse
from io import StringIO
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# Config
# ----------------------------
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
TEAM_PATTERN = re.compile(r"(" + "|".join(sorted(TEAM_ABBRS, key=len, reverse=True)) + r")$")

# These keys MUST stay stable (Expo uses them to match posters)
CATEGORIES = [
    ("passing_yards", "Passing Yards", "passing_yards", ["YDS", "PASS YDS", "Pass YDS"], "int"),
    ("passing_tds", "Passing TDs", "passing_tds", ["TD", "PASS TD", "Pass TD"], "int"),
    ("interceptions_thrown", "Interceptions Thrown", "interceptions", ["INT", "Interceptions"], "int"),
    ("rushing_yards", "Rushing Yards", "rushing_yards", ["YDS", "RUSH YDS", "Rush YDS"], "int"),
    ("rushing_tds", "Rushing TDs", "rushing_tds", ["TD", "RUSH TD", "Rush TD"], "int"),
    ("receiving_yards", "Receiving Yards", "receiving_yards", ["YDS", "REC YDS", "Rec YDS"], "int"),
    ("receiving_tds", "Receiving TDs", "receiving_tds", ["TD", "REC TD", "Rec TD"], "int"),
    ("sacks", "Sacks", "sacks", ["SACK", "Sacks SACK", "Sacks"], "float1"),
    ("tackles", "Tackles", "totalTackles", ["TOT", "Tackles TOT", "Total", "Tackles"], "int"),
    ("interceptions", "Interceptions (Defense)", "defensive_interceptions", ["INT", "Interceptions INT", "Interceptions"], "int"),
]

# ----------------------------
# String normalization (fix glued TEAM)
# ----------------------------
def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def enforce_space_before_team(s: str) -> str:
    s = normalize_spaces(s)
    m = TEAM_PATTERN.search(s)
    if not m:
        return s
    team = m.group(1)
    name_part = s[: -len(team)].strip()
    name_part = re.sub(r"[^\w\.\-'\s]+$", "", name_part).strip()
    return f"{name_part} {team}".strip()

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

def topN_from_url(url: str, stat_candidates: List[str], mode: str, topn: int) -> List[Tuple[int, str, Number]]:
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

    work = work.dropna(subset=["__val__"]).sort_values("__val__", ascending=False).head(topn)

    out = []
    for i, rec in enumerate(work.to_dict("records"), start=1):
        raw = normalize_spaces(rec[name_col])
        display_name = enforce_space_before_team(raw)
        out.append((i, display_name, rec["__val__"]))
    return out

# ----------------------------
# URL builder (regular vs playoffs)
# seasontype: 2=regular, 3=postseason
# ----------------------------
def build_urls(season: int, seasontype: int) -> Dict[str, Tuple[str, List[str], str]]:
    st = int(seasontype)

    urls = {
        "passing_yards": (
            f"https://www.espn.com/nfl/stats/player/_/season/{season}/seasontype/{st}",
            ["YDS", "PASS YDS", "Pass YDS"],
            "int",
        ),
        "passing_tds": (
            f"https://www.espn.com/nfl/stats/player/_/season/{season}/seasontype/{st}/table/passing/sort/passingTouchdowns/dir/desc",
            ["TD", "PASS TD", "Pass TD"],
            "int",
        ),
        "interceptions": (
            f"https://www.espn.com/nfl/stats/player/_/season/{season}/seasontype/{st}/table/passing/sort/interceptions/dir/desc",
            ["INT", "Interceptions"],
            "int",
        ),
        "rushing_yards": (
            f"https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{season}/seasontype/{st}",
            ["YDS", "RUSH YDS", "Rush YDS"],
            "int",
        ),
        "rushing_tds": (
            f"https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{season}/seasontype/{st}/table/rushing/sort/rushingTouchdowns/dir/desc",
            ["TD", "RUSH TD", "Rush TD"],
            "int",
        ),
        "receiving_yards": (
            f"https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{season}/seasontype/{st}",
            ["YDS", "REC YDS", "Rec YDS"],
            "int",
        ),
        "receiving_tds": (
            f"https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{season}/seasontype/{st}/table/receiving/sort/receivingTouchdowns/dir/desc",
            ["TD", "REC TD", "Rec TD"],
            "int",
        ),
        "sacks": (
            f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{st}/table/defensive/sort/sacks/dir/desc",
            ["SACK", "Sacks SACK", "Sacks"],
            "float1",
        ),
        "totalTackles": (
            f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{st}/table/defensive/sort/totalTackles/dir/desc",
            ["TOT", "Tackles TOT", "Total", "Tackles"],
            "int",
        ),
        "defensive_interceptions": (
            f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{st}/table/defensiveInterceptions/sort/interceptions/dir/desc",
            ["INT", "Interceptions INT", "Interceptions"],
            "int",
        ),
    }
    return urls

# ----------------------------
# Phone-friendly poster drawing
# ----------------------------
def load_font(size: int, bold: bool = False):
    candidates = [
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

def draw_phone_poster(
    out_path: str,
    title: str,
    subtitle: str,
    items: List[Tuple[int, str, Number]],
    mode: str,
):
    # Phone-first canvas
    W, H = 1080, 1920
    img = Image.new("RGB", (W, H), (10, 10, 14))
    d = ImageDraw.Draw(img)

    # BIG readable fonts
    title_font = load_font(72, bold=True)
    sub_font = load_font(34, bold=False)

    card_title_font = load_font(46, bold=True)
    row_font = load_font(40, bold=False)
    rank_font = load_font(40, bold=True)

    # Header
    d.text((70, 60), "NFL Statistical Leaders", font=title_font, fill=(245, 245, 245))
    d.text((70, 145), subtitle, font=sub_font, fill=(170, 170, 185))

    # Card (big, minimal padding, fills screen)
    x0, y0 = 60, 240
    x1, y1 = W - 60, H - 120

    d.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=34,
        fill=(20, 20, 28),
        outline=(60, 60, 85),
        width=3,
    )

    d.text((x0 + 40, y0 + 30), title, font=card_title_font, fill=(240, 240, 245))

    # Rows
    top_y = y0 + 110
    line_h = 86  # big spacing so it doesn't blur when scaled
    left_pad = x0 + 40
    right_pad = x1 - 40

    for idx, (rank, display_name, val) in enumerate(items):
        y = top_y + idx * line_h
        if y > y1 - 80:
            break

        # rank
        d.text((left_pad, y), f"{rank}.", font=rank_font, fill=(230, 230, 238))

        # name
        name_x = left_pad + 70
        d.text((name_x, y), display_name, font=row_font, fill=(220, 220, 230))

        # value right aligned
        val_txt = fmt_value(val, mode)
        tw = d.textlength(val_txt, font=row_font)
        d.text((right_pad - tw, y), val_txt, font=row_font, fill=(220, 220, 230))

    img.save(out_path, "PNG")

# ----------------------------
# Main generation
# ----------------------------
def generate_all(season: int, seasontype: int, outdir: str, topn: int = TOP_N):
    os.makedirs(outdir, exist_ok=True)

    phase = "Regular Season" if int(seasontype) == 2 else "Postseason"
    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"Season {season} • {phase} • Updated {updated}"

    urls = build_urls(season, seasontype)

    for key, label, url_key, candidates, mode in CATEGORIES:
        url, stat_cands, mode2 = urls[url_key]
        # (use candidates passed here if you want, but url map is authoritative)
        items = topN_from_url(url, stat_cands, mode2, topn)

        # IMPORTANT: version bump so cache doesn't serve old posters
        tag = "reg" if int(seasontype) == 2 else "post"
        out_path = os.path.join(outdir, f"stat_leader_{key}_{season}_{tag}_v2.png")

        draw_phone_poster(out_path, label, subtitle, items, mode2)

    return outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seasontype", type=int, default=2, help="2=regular, 3=postseason")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.expanduser("~"), "Desktop"))
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    print("Generating phone-friendly stat leader posters...")
    outdir = generate_all(args.season, args.seasontype, args.outdir, args.topn)

    print("\nDONE ✅")
    # Print paths so you can confirm in terminal / logs
    tag = "reg" if int(args.seasontype) == 2 else "post"
    for key, *_ in CATEGORIES:
        p = os.path.join(outdir, f"stat_leader_{key}_{args.season}_{tag}_v2.png")
        print(p)

if __name__ == "__main__":
    main()
