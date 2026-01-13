import os
import re
from io import StringIO
from typing import List, Optional, Tuple, Union

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEASON = 2025
WEEK_LABEL = "Week 18"
TOP_N = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )
}

Number = Union[int, float]

# Real NFL team abbreviations (ESPN style)
TEAM_ABBRS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND",
    "JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA",
    "SF","TB","TEN","WAS"
}

URLS = {
    "Passing Yards": (
        "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2",
        ["YDS", "PASS YDS", "Pass YDS"],
        "int",
    ),
    "Passing TDs": (
        "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/passing/sort/passingTouchdowns/dir/desc",
        ["TD", "PASS TD", "Pass TD"],
        "int",
    ),
    "Interceptions Thrown": (
        "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/passing/sort/interceptions/dir/desc",
        ["INT", "Interceptions"],
        "int",
    ),
    "Rushing Yards": (
        "https://www.espn.com/nfl/stats/player/_/stat/rushing/season/2025/seasontype/2",
        ["YDS", "RUSH YDS", "Rush YDS"],
        "int",
    ),
    "Rushing TDs": (
        "https://www.espn.com/nfl/stats/player/_/stat/rushing/season/2025/seasontype/2/table/rushing/sort/rushingTouchdowns/dir/desc",
        ["TD", "RUSH TD", "Rush TD"],
        "int",
    ),
    "Receiving Yards": (
        "https://www.espn.com/nfl/stats/player/_/stat/receiving/season/2025/seasontype/2",
        ["YDS", "REC YDS", "Rec YDS"],
        "int",
    ),
    "Receiving TDs": (
        "https://www.espn.com/nfl/stats/player/_/stat/receiving/season/2025/seasontype/2/table/receiving/sort/receivingTouchdowns/dir/desc",
        ["TD", "REC TD", "Rec TD"],
        "int",
    ),
    "Sacks": (
        "https://www.espn.com/nfl/stats/player/_/view/defense/season/2025/seasontype/2/table/defensive/sort/sacks/dir/desc",
        ["SACK", "Sacks SACK", "Sacks"],
        "float1",
    ),
    "Tackles": (
        "https://www.espn.com/nfl/stats/player/_/view/defense/season/2025/seasontype/2/table/defensive/sort/totalTackles/dir/desc",
        ["TOT", "Tackles TOT", "Total", "Tackles"],
        "int",
    ),
    "Interceptions (Defense)": (
        "https://www.espn.com/nfl/stats/player/_/view/defense/season/2025/seasontype/2/table/defensiveInterceptions/sort/interceptions/dir/desc",
        ["INT", "Interceptions INT", "Interceptions"],
        "int",
    ),
}


# ----------------------------
# parsing helpers
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


def parse_name_team(name_cell: str) -> Tuple[str, str]:
    """
    Correctly handles suffixes like Jr., II, III, IV, V.
    We ONLY treat the last token as a team if it's a real NFL abbreviation.
    Also handles glued team: 'James Cook IIIBUF' -> ('James Cook III', 'BUF')
    """
    s = str(name_cell).strip()
    # normalize weird spaces (including non-breaking)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = re.sub(r"\s+", " ", s)

    # Case 1: spaced team at end
    parts = s.split(" ")
    if len(parts) >= 2:
        last = parts[-1].strip()
        if last in TEAM_ABBRS:
            return " ".join(parts[:-1]).strip(), last

    # Case 2: glued team at end (match only real team abbrs)
    # Find a team abbr at the very end
    for abbr in sorted(TEAM_ABBRS, key=len, reverse=True):
        if s.endswith(abbr) and len(s) > len(abbr) + 1:
            name = s[:-len(abbr)].strip()
            # avoid false matches like just "BUF" alone
            if " " in name:
                return name, abbr

    return s, ""


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

    for ck in cand_l:  # exact
        for i, colk in enumerate(cols_l):
            if colk == ck:
                return cols[i]

    for ck in cand_l:  # substring
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
        player, team = parse_name_team(rec[name_col])
        display_name = player if not team else f"{player} {team}"
        out.append((i, display_name, rec["__val__"]))
    return out


# ----------------------------
# poster drawing
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


def draw_poster(out_path: str, title: str, subtitle: str,
                sections: List[Tuple[str, List[Tuple[int, str, Number]], str]],
                cols: int, rows: int, height: int,
                # styling knobs (so we can make defense “denser”)
                sub_size: int, head_size: int, row_size: int, line_h: int, y_start_offset: int):
    W, H = 1400, height
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = load_font(56, bold=True)
    sub_font = load_font(sub_size, bold=False)
    head_font = load_font(head_size, bold=True)
    row_font = load_font(row_size, bold=False)
    foot_font = load_font(18, bold=False)

    d.text((60, 40), title, font=title_font, fill=(245, 245, 245))
    d.text((60, 112), subtitle, font=sub_font, fill=(180, 180, 190))

    left = 60
    top = 180
    gap_x = 40
    gap_y = 35
    box_w = (W - left * 2 - gap_x * (cols - 1)) // cols
    box_h = (H - top - 80 - gap_y * (rows - 1)) // rows

    for idx, (sec_title, items, mode) in enumerate(sections):
        c = idx % cols
        r = idx // cols
        x0 = left + c * (box_w + gap_x)
        y0 = top + r * (box_h + gap_y)
        x1 = x0 + box_w
        y1 = y0 + box_h

        d.rounded_rectangle([x0, y0, x1, y1], radius=22,
                            fill=(20, 20, 28), outline=(45, 45, 60), width=2)
        d.text((x0 + 18, y0 + 14), sec_title, font=head_font, fill=(240, 240, 245))

        y = y0 + y_start_offset
        for rank, display_name, val in items:
            left_txt = f"{rank:>2}. {display_name}"
            d.text((x0 + 18, y), left_txt, font=row_font, fill=(210, 210, 220))

            val_txt = fmt_value(val, mode)
            tw = d.textlength(val_txt, font=row_font)
            d.text((x1 - 18 - tw, y), val_txt, font=row_font, fill=(210, 210, 220))

            y += line_h
            if y > y1 - 18:
                break

    d.text((60, H - 55), "Data source: ESPN • Generated locally", font=foot_font, fill=(140, 140, 150))
    img.save(out_path, "PNG")


def main():
    subtitle = f"Season {SEASON} • Regular Season • Through {WEEK_LABEL}"

    offense_titles = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
    ]
    offense_sections = []
    for t in offense_titles:
        url, cand, mode = URLS[t]
        offense_sections.append((t, topN_from_url(url, cand, mode), mode))

    defense_titles = ["Sacks", "Tackles", "Interceptions (Defense)"]
    defense_sections = []
    for t in defense_titles:
        url, cand, mode = URLS[t]
        label = t.replace(" (Defense)", "")
        defense_sections.append((label, topN_from_url(url, cand, mode), mode))

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    out_off = os.path.join(desktop, f"offense_stat_leaders_{SEASON}_wk18_v8.png")
    out_def = os.path.join(desktop, f"defense_stat_leaders_{SEASON}_wk18_v8.png")

    # Offense styling (keep as your “perfect” version)
    draw_poster(
        out_off, "Offensive Statistical Leaders", subtitle,
        offense_sections, cols=2, rows=4, height=2000,
        sub_size=28, head_size=32, row_size=26, line_h=33, y_start_offset=64
    )

    # Defense styling: denser + bigger to fill the tall boxes
    draw_poster(
        out_def, "Defensive Statistical Leaders", subtitle,
        defense_sections, cols=1, rows=3, height=1800,
        sub_size=30, head_size=36, row_size=30, line_h=38, y_start_offset=72
    )

    print("\nDONE ✅")
    print(out_off)
    print(out_def)
    print("")


if __name__ == "__main__":
    main()

