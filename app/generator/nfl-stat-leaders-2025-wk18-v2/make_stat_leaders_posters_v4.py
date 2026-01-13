import os
import re
from io import StringIO
from typing import List, Optional, Tuple

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEASON = 2025
SEASONTYPE = 2
WEEK_LABEL = "Week 18"
TOP_N = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )
}

# --- YOUR URLs (already sorted) ---
URLS = {
    "Passing Yards": (
        "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2",
        ["YDS", "Pass YDS", "PASS YDS"],
    ),
    "Passing TDs": (
        "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/passing/sort/passingTouchdowns/dir/desc",
        ["TD", "Pass TD", "PASS TD"],
    ),
    "Interceptions Thrown": (
        "https://www.espn.com/nfl/stats/player/_/season/2025/seasontype/2/table/passing/sort/interceptions/dir/desc",
        ["INT", "Interceptions", "INTS"],
    ),
    "Rushing Yards": (
        "https://www.espn.com/nfl/stats/player/_/stat/rushing/season/2025/seasontype/2",
        ["YDS", "Rush YDS", "RUSH YDS"],
    ),
    "Rushing TDs": (
        "https://www.espn.com/nfl/stats/player/_/stat/rushing/season/2025/seasontype/2/table/rushing/sort/rushingTouchdowns/dir/desc",
        ["TD", "Rush TD", "RUSH TD"],
    ),
    "Receiving Yards": (
        "https://www.espn.com/nfl/stats/player/_/stat/receiving/season/2025/seasontype/2",
        ["YDS", "Rec YDS", "REC YDS"],
    ),
    "Receiving TDs": (
        "https://www.espn.com/nfl/stats/player/_/stat/receiving/season/2025/seasontype/2/table/receiving/sort/receivingTouchdowns/dir/desc",
        ["TD", "Rec TD", "REC TD"],
    ),
    "Sacks": (
        "https://www.espn.com/nfl/stats/player/_/view/defense/season/2025/seasontype/2/table/defensive/sort/sacks/dir/desc",
        ["SACK", "SCK", "Sacks"],
    ),
    "Tackles": (
        "https://www.espn.com/nfl/stats/player/_/view/defense/season/2025/seasontype/2/table/defensive/sort/totalTackles/dir/desc",
        ["TOT", "Total", "TACK", "Tackles", "TOTAL"],
    ),
    "Interceptions (Defense)": (
        "https://www.espn.com/nfl/stats/player/_/view/defense/season/2025/seasontype/2/table/defensiveInterceptions/sort/interceptions/dir/desc",
        ["INT", "Interceptions", "INTS"],
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


def safe_int(x) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "-"}:
        return None
    s = s.replace(",", "")
    s = re.sub(r"[^\d\-]", "", s)
    if s in {"", "-"}:
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
    tables = pd.read_html(StringIO(r.text))
    return [flatten_columns(t.copy()) for t in tables]


def col_lookup(cols: List[str]) -> dict:
    return {str(c).strip().lower(): c for c in cols}


def pick_name_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    # Name table usually small and contains Name
    for t in tables:
        cmap = col_lookup(list(t.columns))
        if "name" in cmap and t.shape[1] <= 4:
            return t
    # fallback: any table containing Name
    for t in tables:
        cmap = col_lookup(list(t.columns))
        if "name" in cmap:
            return t
    return None


def find_stat_table(tables: List[pd.DataFrame], candidates: List[str]) -> Optional[pd.DataFrame]:
    cand_l = [c.strip().lower() for c in candidates]
    for t in tables:
        cols_l = [c.strip().lower() for c in t.columns]
        if any(c in cols_l for c in cand_l):
            return t
    return None


def stitch_tables_if_needed(name_t: Optional[pd.DataFrame], stat_t: pd.DataFrame) -> pd.DataFrame:
    # If stat table already includes Name, we're good
    cmap = col_lookup(list(stat_t.columns))
    if "name" in cmap:
        return stat_t

    if name_t is None:
        return stat_t

    # ESPN often splits: Name table and Stats table have same row count
    if len(name_t) == len(stat_t):
        # avoid duplicate RK column if present in both
        st = stat_t.copy()
        if any(c.lower() == "rk" for c in st.columns):
            st = st.drop(columns=[c for c in st.columns if c.lower() == "rk"])
        merged = pd.concat([name_t.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
        return merged

    return stat_t


def choose_stat_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cmap = col_lookup(list(df.columns))
    for c in candidates:
        key = c.strip().lower()
        if key in cmap:
            return cmap[key]

    # fallback: pick the *first* numeric-looking column not in ignore list
    ignore = {"rk", "name", "team", "pos", "gp", "gs", "att", "cmp", "pct", "lng", "avg"}
    numeric_cols = []
    for col in df.columns:
        if str(col).strip().lower() in ignore:
            continue
        vals = df[col].map(safe_int)
        if vals.notna().sum() >= 5:
            numeric_cols.append(col)
    if numeric_cols:
        return numeric_cols[0]

    raise RuntimeError(f"Could not determine stat column. Columns: {list(df.columns)}")


def top10_from_url(url: str, stat_candidates: List[str]) -> List[Tuple[int, str, str, int]]:
    tables = fetch_tables(url)
    name_t = pick_name_table(tables)
    stat_t = find_stat_table(tables, stat_candidates) or (tables[0] if tables else None)
    if stat_t is None:
        raise RuntimeError(f"No tables found for URL: {url}")

    df = stitch_tables_if_needed(name_t, stat_t)

    cmap = col_lookup(list(df.columns))
    if "name" not in cmap:
        raise RuntimeError(f"No Name column after stitching. Columns: {list(df.columns)}")

    name_col = cmap["name"]
    stat_col = choose_stat_col(df, stat_candidates)

    work = df[[name_col, stat_col]].copy()
    work["__val__"] = work[stat_col].map(safe_int)
    work = work.dropna(subset=["__val__"]).sort_values("__val__", ascending=False).head(TOP_N)

    out = []
    for idx, row in enumerate(work.to_dict("records"), start=1):
        nm = row[name_col]
        val = int(row["__val__"])
        name, team = parse_name_team(nm)
        out.append((idx, name, team, val))
    return out


# ----------------------------
# Poster drawing
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


def draw_poster(out_path: str, title: str, subtitle: str,
                sections: List[Tuple[str, List[Tuple[int, str, str, int]]]],
                cols: int, rows: int, height: int = 1800):
    W, H = 1400, height
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

    # OFFENSE (now includes Interceptions Thrown)
    offense_order = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
    ]
    offense_sections = []
    for title in offense_order:
        url, cand = URLS[title]
        offense_sections.append((title, top10_from_url(url, cand)))

    # DEFENSE
    defense_order = ["Sacks", "Tackles", "Interceptions (Defense)"]
    defense_sections = []
    for title in defense_order:
        url, cand = URLS[title]
        defense_sections.append((title.replace(" (Defense)", ""), top10_from_url(url, cand)))

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    out_off = os.path.join(desktop, f"offense_stat_leaders_{SEASON}_wk18_v4.png")
    out_def = os.path.join(desktop, f"defense_stat_leaders_{SEASON}_wk18_v4.png")

    # 7 sections => 2 cols x 4 rows (one empty slot) -> make poster a bit taller
    draw_poster(out_off, "Offensive Statistical Leaders", subtitle, offense_sections, cols=2, rows=4, height=2000)
    draw_poster(out_def, "Defensive Statistical Leaders", subtitle, defense_sections, cols=1, rows=3, height=1800)

    print("\nDONE ✅")
    print(out_off)
    print(out_def)
    print("")


if __name__ == "__main__":
    main()

