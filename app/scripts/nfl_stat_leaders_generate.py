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


def build_urls(season: int, seasontype: int):
    base_player = f"https://www.espn.com/nfl/stats/player/_/season/{season}/seasontype/{seasontype}"
    base_rush = f"https://www.espn.com/nfl/stats/player/_/stat/rushing/season/{season}/seasontype/{seasontype}"
    base_rec = f"https://www.espn.com/nfl/stats/player/_/stat/receiving/season/{season}/seasontype/{seasontype}"
    base_def = f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{season}/seasontype/{seasontype}"

    return {
        "Passing Yards": (base_player, ["YDS", "PASS YDS", "Pass YDS"], "int"),
        "Passing TDs": (f"{base_player}/table/passing/sort/passingTouchdowns/dir/desc", ["TD", "PASS TD", "Pass TD"], "int"),
        "Interceptions Thrown": (f"{base_player}/table/passing/sort/interceptions/dir/desc", ["INT", "Interceptions"], "int"),
        "Rushing Yards": (base_rush, ["YDS", "RUSH YDS", "Rush YDS"], "int"),
        "Rushing TDs": (f"{base_rush}/table/rushing/sort/rushingTouchdowns/dir/desc", ["TD", "RUSH TD", "Rush TD"], "int"),
        "Receiving Yards": (base_rec, ["YDS", "REC YDS", "Rec YDS"], "int"),
        "Receiving TDs": (f"{base_rec}/table/receiving/sort/receivingTouchdowns/dir/desc", ["TD", "REC TD", "Rec TD"], "int"),
        "Sacks": (f"{base_def}/table/defensive/sort/sacks/dir/desc", ["SACK", "Sacks SACK", "Sacks"], "float1"),
        "Tackles": (f"{base_def}/table/defensive/sort/totalTackles/dir/desc", ["TOT", "Tackles TOT", "Total", "Tackles"], "int"),
        "Interceptions (Defense)": (f"{base_def}/table/defensiveInterceptions/sort/interceptions/dir/desc", ["INT", "Interceptions INT", "Interceptions"], "int"),
    }


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
        return " ".join(parts[:-1]), parts[-1].upper()

    return display_name, ""


def draw_single_stat_poster(
    out_path: str,
    poster_title: str,
    stat_title: str,
    subtitle: str,
    items: List[Tuple[int, str, Number]],
    mode: str,
):
    W, H = 1080, 1920
    img = Image.new("RGB", (W, H), (10, 14, 24))
    d = ImageDraw.Draw(img)

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

    d.rectangle((0, 0, W, 178), fill=(22, 38, 74))
    d.rectangle((0, 178, W, 187), fill=blue)

    for y in range(198, H, 30):
        color = (14, 18, 28) if (y // 30) % 2 == 0 else (12, 16, 26)
        d.rectangle((0, y, W, y + 15), fill=color)

    title = fit_text(d, poster_title.upper(), title_font, W - 90)
    stat = fit_text(d, stat_title.upper(), stat_font, W - 90)
    sub = fit_text(d, subtitle, sub_font, W - 90)

    d.text(((W - d.textlength(title, font=title_font)) / 2, 24), title, font=title_font, fill=white)
    d.text(((W - d.textlength(stat, font=stat_font)) / 2, 78), stat, font=stat_font, fill=white)
    d.text(((W - d.textlength(sub, font=sub_font)) / 2, 143), sub, font=sub_font, fill=muted)

    x0, x1 = 42, W - 42
    top = 220
    bottom = H - 42
    gap = 14
    row_h = int((bottom - top - gap * (TOP_N - 1)) / TOP_N)

    for rank, display_name, val in items:
        y0 = top + (rank - 1) * (row_h + gap)
        y1 = y0 + row_h

        d.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=26,
            fill=dark,
            outline=border,
            width=3,
        )

        pill = (x0 + 18, y0 + 22, x0 + 92, y0 + 84)
        d.rounded_rectangle(pill, radius=18, fill=blue)

        rank_text = str(rank)
        d.text(
            (
                pill[0] + (pill[2] - pill[0] - d.textlength(rank_text, font=rank_font)) / 2,
                pill[1] + 12,
            ),
            rank_text,
            font=rank_font,
            fill=(15, 20, 28),
        )

        player_name, team = split_name_team(display_name)
        value_text = fmt_value(val, mode)
        value_w = d.textlength(value_text, font=value_font)

        d.text(
            (x1 - 30 - value_w, y0 + 31),
            value_text,
            font=value_font,
            fill=white,
        )

        max_name_w = x1 - x0 - 150 - value_w - 55
        name_text = fit_text(d, player_name.upper(), name_font, max_name_w)

        d.text(
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

            d.rounded_rectangle(
                (tag_x1, tag_y1, tag_x2, tag_y2),
                radius=13,
                fill=(16, 28, 54),
                outline=(86, 104, 140),
                width=1,
            )

            team_w = d.textlength(team, font=team_font)
            d.text(
                (tag_x1 + (tag_x2 - tag_x1 - team_w) / 2, tag_y1 + 5),
                team,
                font=team_font,
                fill=blue,
            )

        d.rectangle((x0 + 18, y1 - 12, x1 - 18, y1 - 8), fill=blue)

    img.save(out_path, "PNG")


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
