import os
import re
from io import StringIO
from typing import List, Optional, Tuple

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEASON = 2025
SEASONTYPE = 2  # regular season
WEEK_LABEL = "Week 18"
TOP_N = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )
}

URL_PASSING = f"https://www.espn.com/nfl/stats/player/_/season/{SEASON}/seasontype/{SEASONTYPE}"
URL_RUSHING = f"https://www.espn.com/nfl/stats/player/_/season/{SEASON}/seasontype/{SEASONTYPE}/table/rushing"
URL_RECEIVING = f"https://www.espn.com/nfl/stats/player/_/season/{SEASON}/seasontype/{SEASONTYPE}/table/receiving"
URL_DEFENSE = f"https://www.espn.com/nfl/stats/player/_/view/defense/season/{SEASON}/seasontype/{SEASONTYPE}"


# ---------- parsing helpers ----------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() != ""]).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_int(x) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    s = s.replace(",", "")
    s = re.sub(r"[^\d\-]", "", s)
    if s == "" or s == "-":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_name_team(name_cell: str) -> Tuple[str, str]:
    s = str(name_cell).strip()
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if len(parts) >= 2 and re.fullmatch(r"[A-Z]{2,4}", parts[-1]):
        return " ".join(parts[:-1]).strip(), parts[-1].strip()
    return s, ""


def fetch_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    # avoid the FutureWarning you saw
    tables = pd.read_html(StringIO(r.text))
    return [flatten_columns(t.copy()) for t in tables]


def normalize_col_lookup(cols: List[str]) -> dict:
    return {c.strip().lower(): c for c in cols}


def stitch_name_and_stats(tables: List[pd.DataFrame], stat_col: str) -> pd.DataFrame:
    """
    ESPN often renders:
      table1: RK, Name
      table2: stat columns (YDS/TD/...)
    We find the table with the stat_col and stitch it with the Name table
    that has the same number of rows.
    """
    stat_col_l = stat_col.strip().lower()

    # candidates
    name_tables = []
    stat_tables = []

    for t in tables:
        cols_l = [c.lower() for c in t.columns]
        colmap = normalize_col_lookup(list(t.columns))

        has_name = ("name" in colmap) or any(c == "name" for c in cols_l)
        has_stat = any(c.strip().lower() == stat_col_l for c in t.columns)

        # Name table is typically small: RK + Name
        if has_name and t.shape[1] <= 3:
            name_tables.append(t)

        if has_stat:
            stat_tables.append(t)

    if not stat_tables:
        raise RuntimeError(
            f"Could not find stat column '{stat_col}'. "
            f"Tables had columns like: {[list(t.columns) for t in tables[:6]]}"
        )

    # If the stat table already includes Name, return it
    for st in stat_tables:
        colmap = normalize_col_lookup(list(st.columns))
        if "name" in colmap:
            return st

    # Otherwise stitch with matching name table by row count
    for st in stat_tables:
        for nt in name_tables:
            if len(st) == len(nt):
                # drop duplicate RK if present on right side
                st_cols_l = [c.lower() for c in st.columns]
                if "rk" in st_cols_l:
                    st = st.drop(columns=[c for c in st.columns if c.lower() == "rk"])
                merged = pd.concat([nt.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
                return merged

    # Last resort: pick the widest table that has the stat column
    stat_tables = sorted(stat_tables, key=lambda d: d.shape[1], reverse=True)
    return stat_tables[0]


def top_n(url: str, stat_col: str, n: int = TOP_N) -> List[Tuple[int, str, str, int]]:
    tables = fetch_tables(url)
    df = stitch_name_and_stats(tables, stat_col)

    colmap = normalize_col_lookup(list(df.columns))
    name_col = colmap.get("name")
    if not name_col:
        raise RuntimeError(f"No Name column after stitching. Columns: {list(df.columns)}")

    # exact stat col match (case-insensitive)
    stat_actual = None
    for c in df.columns:
        if c.strip().lower() == stat_col.strip().lower():
            stat_actual = c
            break
    if not stat_actual:
        raise RuntimeError(f"Stat col '{stat_col}' not found. Columns: {list(df.columns)}")

    work = df[[name_col, stat_actual]].copy()
    work["__val__"] = work[stat_actual].apply(safe_int)
    work = work.dropna(subset=["__val__"]).sort_values("__val__", ascending=False).head(n)

    out = []
    for i, row in enumerate(work.itertuples(index=False), start=1):
        nm = getattr(row, name_col)
        val = int(getattr(row, "__val__"))
        name, team = parse_name_team(nm)
        out.append((i, name, team, val))
    return out


# ---------- poster drawing ----------
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


def draw_poster(out_path: str, title: str, subtitle: str,
                sections: List[Tuple[str, List[Tuple[int, str, str, int]]]],
                cols: int, rows: int):
    W, H = 1400, 1800
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = load_font(56, bold=True)
    sub_font = load_font(26, bold=False)
    head_font = load_font(28, bold=True)
    row_font = load_font(22, bold=False)
    foot_font = load_font(18, bold=False)

    d.text((60, 40), title, font=title_font, fill=(245, 245, 245))
    d.text((60, 110), subtitle, font=sub_font, fill=(180, 180, 190))

    left = 60
    top = 170
    gap_x = 40
    gap_y = 35
    box_w = (W - left * 2 - gap_x * (cols - 1)) // cols
    box_h = (H - top - 80 - gap_y * (rows - 1)) // rows

    for idx, (sec_title, items) in enumerate(sections):
        c = idx % cols
        r = idx // cols
        x0 = left + c * (box_w + gap_x)
        y0 = top + r * (box_h + gap_y)
        x1 = x0 + box_w
        y1 = y0 + box_h

        d.rounded_rectangle([x0, y0, x1, y1], radius=22,
                            fill=(20, 20, 28), outline=(45, 45, 60), width=2)
        d.text((x0 + 18, y0 + 14), sec_title, font=head_font, fill=(240, 240, 245))

        y = y0 + 60
        line_h = 30
        for rank, name, team, val in items:
            left_txt = f"{rank:>2}. {name}" + (f"  ({team})" if team else "")
            d.text((x0 + 18, y), left_txt, font=row_font, fill=(210, 210, 220))
            val_txt = str(val)
            tw = d.textlength(val_txt, font=row_font)
            d.text((x1 - 18 - tw, y), val_txt, font=row_font, fill=(210, 210, 220))
            y += line_h
            if y > y1 - 18:
                break

    d.text((60, H - 55), "Data source: ESPN • Generated locally", font=foot_font, fill=(140, 140, 150))
    img.save(out_path, "PNG")


def main():
    subtitle = f"Season {SEASON} • Regular Season • Through {WEEK_LABEL}"

    # offense
    passing_yds = top_n(URL_PASSING, "YDS", TOP_N)
    passing_tds = top_n(URL_PASSING, "TD", TOP_N)

    rushing_yds = top_n(URL_RUSHING, "YDS", TOP_N)
    rushing_tds = top_n(URL_RUSHING, "TD", TOP_N)

    receiving_yds = top_n(URL_RECEIVING, "YDS", TOP_N)
    receiving_tds = top_n(URL_RECEIVING, "TD", TOP_N)

    offense_sections = [
        ("Passing Yards", passing_yds),
        ("Passing TDs", passing_tds),
        ("Rushing Yards", rushing_yds),
        ("Rushing TDs", rushing_tds),
        ("Receiving Yards", receiving_yds),
        ("Receiving TDs", receiving_tds),
    ]

    # defense
    sacks = top_n(URL_DEFENSE, "SACK", TOP_N)
    tackles = top_n(URL_DEFENSE, "TOT", TOP_N)
    interceptions = top_n(URL_DEFENSE, "INT", TOP_N)

    defense_sections = [
        ("Sacks", sacks),
        ("Tackles", tackles),
        ("Interceptions", interceptions),
    ]

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    out_off = os.path.join(desktop, f"offense_stat_leaders_{SEASON}_wk18_v3.png")
    out_def = os.path.join(desktop, f"defense_stat_leaders_{SEASON}_wk18_v3.png")

    draw_poster(out_off, "Offensive Statistical Leaders", subtitle, offense_sections, cols=2, rows=3)
    draw_poster(out_def, "Defensive Statistical Leaders", subtitle, defense_sections, cols=1, rows=3)

    print("\nDONE ✅")
    print(out_off)
    print(out_def)
    print("")


if __name__ == "__main__":
    main()

