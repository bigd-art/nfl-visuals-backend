# app/scripts/nfl_stat_leaders_generate.py
import os
import re
import argparse
from io import StringIO
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict

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

# slug, display title, short title
STAT_CONFIG = [
    ("passing_yards", "Passing Yards", "Passing Yards"),
    ("passing_tds", "Passing TDs", "Passing TDs"),
    ("interceptions_thrown", "Interceptions Thrown", "Interceptions Thrown"),
    ("rushing_yards", "Rushing Yards", "Rushing Yards"),
    ("rushing_tds", "Rushing TDs", "Rushing TDs"),
    ("receiving_yards", "Receiving Yards", "Receiving Yards"),
    ("receiving_tds", "Receiving TDs", "Receiving TDs"),
    ("sacks", "Sacks", "Sacks"),
    ("tackles", "Tackles", "Tackles"),
    ("interceptions_defense", "Interceptions (Defense)", "Interceptions"),
]


# ----------------------------
# String normalization
# ----------------------------
def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u00ad", "")
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
# ESPN URLs
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
# Drawing
# ----------------------------
def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def draw_single_stat_poster(
    out_path: str,
    poster_title: str,
    stat_title: str,
    subtitle: str,
    items: List[Tuple[int, str, Number]],
    mode: str,
):
    W, H = 1080, 1920
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = load_font(52, bold=True)
    stat_font = load_font(34, bold=True)
    sub_font = load_font(24, bold=False)
    header_font = load_font(22, bold=True)
    row_font = load_font(28, bold=False)

    text = (245, 245, 245)
    muted = (180, 180, 190)
    card_fill = (20, 20, 28)
    outline = (45, 45, 60)
    grid = (40, 48, 66)

    d.text((50, 40), poster_title, font=title_font, fill=text)
    d.text((50, 110), stat_title, font=stat_font, fill=(220, 220, 230))
    d.text((50, 160), subtitle, font=sub_font, fill=muted)

    x0, y0, x1, y1 = 50, 240, W - 50, H - 60
    d.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=card_fill, outline=outline, width=2)

    header_y = y0 + 16
    d.text((x0 + 24, header_y), "RK", font=header_font, fill=muted)
    d.text((x0 + 110, header_y), "PLAYER", font=header_font, fill=muted)
    value_label = stat_title.upper()
    tw = d.textlength(value_label, font=header_font)
    d.text((x1 - 24 - tw, header_y), value_label, font=header_font, fill=muted)

    d.line((x0 + 20, header_y + 34, x1 - 20, header_y + 34), fill=grid, width=2)

    y = header_y + 54
    row_h = 135

    for rank, display_name, val in items:
        d.rounded_rectangle(
            [x0 + 16, y, x1 - 16, y + row_h - 10],
            radius=16,
            fill=(14, 18, 26),
            outline=(35, 42, 58),
            width=1,
        )

        d.text((x0 + 28, y + 18), str(rank), font=row_font, fill=text)

        player_text = display_name
        d.text((x0 + 110, y + 18), player_text, font=row_font, fill=text)

        val_txt = fmt_value(val, mode)
        val_w = d.textlength(val_txt, font=row_font)
        d.text((x1 - 30 - val_w, y + 18), val_txt, font=row_font, fill=text)

        y += row_h
        if y > y1 - 40:
            break

    img.save(out_path, "PNG")


# ----------------------------
# Generation helpers
# ----------------------------
def build_stat_sections(season: int, seasontype: int) -> Dict[str, Tuple[str, List[Tuple[int, str, Number]], str]]:
    urls = build_urls(season, seasontype)
    out: Dict[str, Tuple[str, List[Tuple[int, str, Number]], str]] = {}

    for slug, espn_title, short_title in STAT_CONFIG:
        url, cand, mode = urls[espn_title]
        items = topN_from_url(url, cand, mode)
        out[slug] = (short_title, items, mode)

    return out


def generate_all_stat_leader_posters(season: int, seasontype: int, outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)

    phase = "Regular Season" if seasontype == 2 else "Postseason"
    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"Season {season} • {phase} • Updated {updated}"

    sections = build_stat_sections(season, seasontype)
    outputs: Dict[str, str] = {}

    for slug, _espn_title, short_title in STAT_CONFIG:
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

    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--seasontype", type=int, default=2, choices=[2, 3], help="2=Regular, 3=Postseason")
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.expanduser("~"), "Desktop"))
    args = ap.parse_args()

    outputs = generate_all_stat_leader_posters(
        season=args.season,
        seasontype=args.seasontype,
        outdir=args.outdir,
    )

    print("\nDONE ✅")
    for slug, path in outputs.items():
        print(slug, "->", path)
    print("")


if __name__ == "__main__":
    main()
