import os
import re
from typing import List, Tuple, Optional

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEASON = 2025
SEASONTYPE = 2  # 2 = Regular Season
WEEK_LABEL = "Week 18"

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


# ----------------------------
# Helpers: ESPN table parsing
# ----------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # ESPN sometimes gives MultiIndex columns; flatten them cleanly
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


def pick_best_table(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    ESPN pages have multiple tables. We want the main stats table.
    Most reliable: must contain 'RK' and 'Name' (or a 'Name' column).
    """
    for t in tables:
        t = flatten_columns(t.copy())
        cols = [c.lower() for c in t.columns]
        if any(c == "rk" for c in cols) and any("name" == c for c in cols):
            return t
    # fallback: first reasonably wide table
    tables = [flatten_columns(t.copy()) for t in tables]
    tables = sorted(tables, key=lambda d: d.shape[1], reverse=True)
    return tables[0]


def fetch_main_table(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    if not tables:
        raise RuntimeError(f"No tables found at {url}")
    return pick_best_table(tables)


def parse_name_team(name_cell: str) -> Tuple[str, str]:
    """
    ESPN 'Name' column usually looks like:
      'Matthew Stafford LAR' or 'Jordyn Brooks MIA'
    We'll treat last token as team if it's 2-4 uppercase letters.
    """
    s = str(name_cell).strip()
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if len(parts) >= 2 and re.fullmatch(r"[A-Z]{2,4}", parts[-1]):
        team = parts[-1]
        name = " ".join(parts[:-1]).strip()
        return name, team
    return s, ""


def top_n_by_stat(url: str, stat_col: str, n: int = 10) -> List[Tuple[int, str, str, int]]:
    df = fetch_main_table(url)

    # Normalize columns (some pages put 'Name' vs 'NAME')
    cols_map = {c.lower(): c for c in df.columns}
    name_col = cols_map.get("name", None)
    if name_col is None:
        # fallback: first column that contains 'name'
        for c in df.columns:
            if "name" in c.lower():
                name_col = c
                break
    if name_col is None:
        raise RuntimeError(f"Couldn't find a Name column. Columns: {list(df.columns)}")

    # Find stat column exactly (case-insensitive)
    stat_actual = None
    for c in df.columns:
        if c.strip().lower() == stat_col.strip().lower():
            stat_actual = c
            break
    if stat_actual is None:
        raise RuntimeError(
            f"Couldn't find stat column '{stat_col}' on {url}\n"
            f"Available columns: {list(df.columns)}"
        )

    work = df[[name_col, stat_actual]].copy()
    work["__val__"] = work[stat_actual].apply(safe_int)
    work = work.dropna(subset=["__val__"])
    work = work.sort_values("__val__", ascending=False).head(n)

    out = []
    for i, row in enumerate(work.itertuples(index=False), start=1):
        nm = getattr(row, name_col)
        val = int(getattr(row, "__val__"))
        name, team = parse_name_team(nm)
        out.append((i, name, team, val))
    return out


# ----------------------------
# Poster drawing
# ----------------------------
def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_poster(
    out_path: str,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, List[Tuple[int, str, str, int]]]],
    layout_cols: int,
    layout_rows: int,
):
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

    box_w = (W - left * 2 - gap_x * (layout_cols - 1)) // layout_cols
    box_h = (H - top - 80 - gap_y * (layout_rows - 1)) // layout_rows

    for idx, (sec_title, items) in enumerate(sections):
        c = idx % layout_cols
        r = idx // layout_cols

        x0 = left + c * (box_w + gap_x)
        y0 = top + r * (box_h + gap_y)
        x1 = x0 + box_w
        y1 = y0 + box_h

        d.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(20, 20, 28), outline=(45, 45, 60), width=2)
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

    # OFFENSE (sort by YDS + TD in each relevant table)
    passing_yds = top_n_by_stat(URL_PASSING, "YDS", 10)
    passing_tds = top_n_by_stat(URL_PASSING, "TD", 10)

    rushing_yds = top_n_by_stat(URL_RUSHING, "YDS", 10)
    rushing_tds = top_n_by_stat(URL_RUSHING, "TD", 10)

    receiving_yds = top_n_by_stat(URL_RECEIVING, "YDS", 10)
    receiving_tds = top_n_by_stat(URL_RECEIVING, "TD", 10)

    offense_sections = [
        ("Passing Yards", passing_yds),
        ("Passing TDs", passing_tds),
        ("Rushing Yards", rushing_yds),
        ("Rushing TDs", rushing_tds),
        ("Receiving Yards", receiving_yds),
        ("Receiving TDs", receiving_tds),
    ]

    # DEFENSE (same defense table contains TOT, SACK, INT)
    tackles = top_n_by_stat(URL_DEFENSE, "TOT", 10)
    sacks = top_n_by_stat(URL_DEFENSE, "SACK", 10)
    interceptions = top_n_by_stat(URL_DEFENSE, "INT", 10)

    defense_sections = [
        ("Sacks", sacks),
        ("Tackles", tackles),
        ("Interceptions", interceptions),
    ]

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    out_off = os.path.join(desktop, f"offense_stat_leaders_{SEASON}_wk18_v2.png")
    out_def = os.path.join(desktop, f"defense_stat_leaders_{SEASON}_wk18_v2.png")

    draw_poster(out_off, "Offensive Statistical Leaders", subtitle, offense_sections, layout_cols=2, layout_rows=3)
    draw_poster(out_def, "Defensive Statistical Leaders", subtitle, defense_sections, layout_cols=1, layout_rows=3)

    print("\nDONE ✅")
    print("Saved:")
    print(out_off)
    print(out_def)
    print("")


if __name__ == "__main__":
    main()

