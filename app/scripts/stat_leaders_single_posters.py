# nfl_stat_leaders_single_posters.py
# Creates ONE poster PER category (10 total) and saves them to Desktop (or --outdir)

import os
import re
import argparse
from io import StringIO
from datetime import datetime
from typing import List, Optional, Tuple, Union

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Number = Union[int, float]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )
}

TOP_N_DEFAULT = 10

TEAM_ABBRS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND",
    "JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA",
    "SF","TB","TEN","WAS","WSH"
}
TEAM_PATTERN = re.compile(r"(" + "|".join(sorted(TEAM_ABBRS, key=len, reverse=True)) + r")$")

CATEGORY_SPECS = [
    ("Passing Yards", "passingYards", ["YDS", "PASS YDS", "Pass YDS"], "int"),
    ("Passing TDs", "passingTouchdowns", ["TD", "PASS TD", "Pass TD"], "int"),
    ("Interceptions Thrown", "interceptions", ["INT", "Interceptions"], "int"),
    ("Rushing Yards", "rushingYards", ["YDS", "RUSH YDS", "Rush YDS"], "int"),
    ("Rushing TDs", "rushingTouchdowns", ["TD", "RUSH TD", "Rush TD"], "int"),
    ("Receiving Yards", "receivingYards", ["YDS", "REC YDS", "Rec YDS"], "int"),
    ("Receiving TDs", "receivingTouchdowns", ["TD", "REC TD", "Rec TD"], "int"),
    ("Sacks", "sacks", ["SACK", "Sacks SACK", "Sacks"], "float1"),
    ("Tackles", "totalTackles", ["TOT", "Tackles TOT", "Total", "Tackles"], "int"),
    ("Interceptions (Defense)", "defInterceptions", ["INT", "Interceptions INT", "Interceptions"], "int"),
]


def build_urls(season: int, seasontype: int) -> dict:
    base = f"https://www.espn.com/nfl/stats/player/_/season/{season}/seasontype/{seasontype}"
    return {
        "passingYards": (base, ["YDS", "PASS YDS", "Pass YDS"], "int"),
        "passingTouchdowns": (
            f"{base}/table/passing/sort/passingTouchdowns/dir/desc",
            ["TD", "PASS TD", "Pass TD"],
            "int",
        ),
        "interceptions": (
            f"{base}/table/passing/sort/interceptions/dir/desc",
            ["INT", "Interceptions"],
            "int",
        ),

        "rushingYards": (
            f"https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{season}/seasontype/{seasontype}",
            ["YDS", "RUSH YDS", "Rush YDS"],
            "int",
        ),
        "rushingTouchdowns": (
            f"https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{season}/seasontype/{seasontype}/table/rushing/sort/rushingTouchdowns/dir/desc",
            ["TD", "RUSH TD", "Rush TD"],
            "int",
        ),

        "receivingYards": (
            f"https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{season}/seasontype/{seasontype}",
            ["YDS", "REC YDS", "Rec YDS"],
            "int",
        ),
        "receivingTouchdowns": (
            f"https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{season}/seasontype/{seasontype}/table/receiving/sort/receivingTouchdowns/dir/desc",
            ["TD", "REC TD", "Rec TD"],
            "int",
        ),

        "sacks": (
            f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{seasontype}/table/defensive/sort/sacks/dir/desc",
            ["SACK", "Sacks SACK", "Sacks"],
            "float1",
        ),
        "totalTackles": (
            f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{seasontype}/table/defensive/sort/totalTackles/dir/desc",
            ["TOT", "Tackles TOT", "Total", "Tackles"],
            "int",
        ),
        "defInterceptions": (
            f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{seasontype}/table/defensiveInterceptions/sort/interceptions/dir/desc",
            ["INT", "Interceptions INT", "Interceptions"],
            "int",
        ),
    }


# ----------------------------
# String normalization (team spacing fix)
# ----------------------------
def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)  # zero-width chars
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


def topN_from_url(url: str, stat_candidates: List[str], mode: str, top_n: int) -> List[Tuple[int, str, Number]]:
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

    work = work.dropna(subset=["__val__"]).sort_values("__val__", ascending=False).head(top_n)

    out = []
    for i, rec in enumerate(work.to_dict("records"), start=1):
        raw = normalize_spaces(rec[name_col])
        display_name = enforce_space_before_team(raw)
        out.append((i, display_name, rec["__val__"]))
    return out


# ----------------------------
# Poster drawing: single category poster (BIG text for phones)
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


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def draw_single_category_poster(
    out_path: str,
    big_title: str,
    subtitle: str,
    category_title: str,
    items: List[Tuple[int, str, Number]],
    mode: str,
):
    # 1400 wide, tall, BIG fonts
    W, H = 1400, 1800
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = load_font(68, bold=True)
    sub_font = load_font(30, bold=False)
    cat_font = load_font(54, bold=True)
    row_font = load_font(40, bold=False)

    d.text((70, 50), big_title, font=title_font, fill=(245, 245, 245))
    d.text((70, 130), subtitle, font=sub_font, fill=(185, 185, 195))

    x0, y0, x1, y1 = 70, 220, W - 70, H - 90
    d.rounded_rectangle([x0, y0, x1, y1], radius=26, fill=(20, 20, 28), outline=(45, 45, 60), width=2)

    d.text((x0 + 26, y0 + 22), category_title, font=cat_font, fill=(240, 240, 245))

    y = y0 + 110
    line_h = 62

    for rank, name, val in items:
        left_txt = f"{rank:>2}. {name}"
        d.text((x0 + 26, y), left_txt, font=row_font, fill=(220, 220, 230))

        val_txt = fmt_value(val, mode)
        tw = d.textlength(val_txt, font=row_font)
        d.text((x1 - 26 - tw, y), val_txt, font=row_font, fill=(220, 220, 230))

        y += line_h
        if y > y1 - 40:
            break

    img.save(out_path, "PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seasontype", type=int, default=2, help="2=regular, 3=postseason")
    ap.add_argument("--topn", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    season = int(args.season)
    seasontype = int(args.seasontype)
    topn = int(args.topn)

    phase = "Regular Season" if seasontype == 2 else "Postseason"
    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"Season {season} • {phase} • Updated {updated}"

    outdir = args.outdir.strip() or os.path.join(os.path.expanduser("~"), "Desktop")
    os.makedirs(outdir, exist_ok=True)

    urls = build_urls(season, seasontype)

    print("\nGenerating single-category stat leader posters…\n")

    outputs = []
    for pretty_name, key, candidates, mode in CATEGORY_SPECS:
        url, _, _ = urls[key]
        items = topN_from_url(url, candidates, mode, topn)

        safe_cat = slugify(pretty_name.replace("(Defense)", "").strip())
        tag = "reg" if seasontype == 2 else "post"
        out_path = os.path.join(outdir, f"stat_leader_{safe_cat}_{season}_{tag}.png")

        draw_single_category_poster(
            out_path=out_path,
            big_title="NFL Statistical Leaders",
            subtitle=subtitle,
            category_title=pretty_name.replace(" (Defense)", ""),
            items=items,
            mode=mode,
        )

        outputs.append(out_path)
        print("✅", out_path)

    print("\nDONE ✅")
    for p in outputs:
        print(p)
    print("")


if __name__ == "__main__":
    main()

