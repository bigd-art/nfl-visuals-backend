import re
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HEADERS = {"User-Agent": "Mozilla/5.0"}
Number = Union[int, float]


# -------------------------
# Text + number helpers
# -------------------------
def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    except:
        return None


def safe_int(x) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    return int(round(f))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]).strip()
            for tup in df.columns
        ]
    df.columns = [normalize_spaces(c) for c in df.columns]
    return df


def fetch_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    return [flatten_columns(t.copy()) for t in tables]


def looks_like_name_table(df: pd.DataFrame) -> bool:
    # ESPN "name" tables usually have 1 column and are mostly strings (player names)
    if df is None or df.empty:
        return False
    if len(df.columns) > 2:
        return False
    # first col tends to be name-ish
    col0 = normalize_spaces(df.columns[0]).lower()
    return ("name" in col0) or (len(df.columns) == 1)


def strip_total_rows(name_df: pd.DataFrame) -> pd.DataFrame:
    if name_df is None or name_df.empty:
        return name_df
    col0 = name_df.columns[0]
    s = name_df[col0].astype(str).map(normalize_spaces)
    mask_total = s.str.lower().str.contains(r"\btotal\b", na=False)
    return name_df.loc[~mask_total].reset_index(drop=True)


def pick_col_fuzzy(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """
    Return the first matching column name given a list of patterns.
    Patterns are checked as:
      - exact match
      - substring match
      - regex search
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    cols_low = [normalize_spaces(c).lower() for c in cols]

    for pat in patterns:
        p = pat.lower().strip()

        # exact
        for i, c in enumerate(cols_low):
            if c == p:
                return cols[i]

        # substring
        for i, c in enumerate(cols_low):
            if p in c:
                return cols[i]

        # regex
        try:
            rx = re.compile(p, re.IGNORECASE)
            for i, c in enumerate(cols):
                if rx.search(str(c)):
                    return cols[i]
        except:
            pass

    return None


def leader_from_two_tables(
    name_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    stat_col: str,
    mode: str = "int",
) -> Tuple[str, Number]:
    """
    Returns (player_name, best_value).
    Never crashes if there are missing values; will default to ('—', 0).
    """
    if name_df is None or stat_df is None or name_df.empty or stat_df.empty:
        return "—", 0

    # Align by row index
    n = min(len(name_df), len(stat_df))
    name_df = name_df.iloc[:n].reset_index(drop=True).copy()
    stat_df = stat_df.iloc[:n].reset_index(drop=True).copy()

    # Normalize + drop Total
    name_df.iloc[:, 0] = name_df.iloc[:, 0].map(normalize_spaces)
    name_df = strip_total_rows(name_df)
    stat_df = stat_df.iloc[: len(name_df)].reset_index(drop=True)

    if stat_col not in stat_df.columns:
        return "—", 0

    if mode == "float1":
        vals = stat_df[stat_col].map(safe_float)
    else:
        vals = stat_df[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")

    # If all NaN, return default
    if vals_num.isna().all():
        return normalize_spaces(name_df.iloc[0, 0]) if len(name_df) else "—", 0

    # idxmax picks first max, good enough
    best_i = int(vals_num.fillna(-1e18).idxmax())
    best_val = vals_num.loc[best_i]
    if pd.isna(best_val):
        best_val = 0

    who = normalize_spaces(name_df.iloc[best_i, 0]) if best_i < len(name_df) else "—"
    return who, best_val


# -------------------------
# ESPN table detection
# -------------------------
def find_stat_table(tables: List[pd.DataFrame], want_cols_any: List[str], want_cols_all: List[str]) -> Optional[int]:
    """
    Find index of table that contains:
      - ALL of want_cols_all (fuzzy)
      - ANY of want_cols_any (fuzzy)
    """
    for i, df in enumerate(tables):
        cols_low = [normalize_spaces(c).lower() for c in df.columns]

        # must contain all required
        ok_all = True
        for req in want_cols_all:
            req = req.lower()
            if not any(req == c or req in c for c in cols_low):
                ok_all = False
                break
        if not ok_all:
            continue

        # must contain any optional hint
        if want_cols_any:
            ok_any = False
            for hint in want_cols_any:
                hint = hint.lower()
                if any(hint == c or hint in c for c in cols_low):
                    ok_any = True
                    break
            if not ok_any:
                continue

        return i

    return None


def get_name_table_for(tables: List[pd.DataFrame], stat_idx: int) -> Optional[pd.DataFrame]:
    if stat_idx is None or stat_idx <= 0:
        return None
    cand = tables[stat_idx - 1]
    if looks_like_name_table(cand):
        # normalize first col name to "Name"
        cand = cand.copy()
        cand.columns = ["Name"] + list(cand.columns[1:])
        return cand[["Name"]]
    return None


# -------------------------
# Public API for router
# -------------------------
def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Returns dict:
      category -> (player_name, team_abbr_placeholder, value_str)

    Router expects leaders[cat][0], [1], [2]
    """
    tables = fetch_tables(team_url)

    # Passing: needs YDS, TD, INT
    pass_idx = find_stat_table(
        tables,
        want_cols_any=["cmp", "att", "yds", "td"],
        want_cols_all=["yds", "td", "int"],
    )

    # Rushing: needs YDS, TD (and often ATT)
    rush_idx = find_stat_table(
        tables,
        want_cols_any=["att", "car", "yds", "td"],
        want_cols_all=["yds", "td"],
    )

    # Receiving: needs YDS, TD (and often REC)
    rec_idx = find_stat_table(
        tables,
        want_cols_any=["rec", "tgts", "yds", "td"],
        want_cols_all=["yds", "td"],
    )

    # Defense: needs some form of sacks/tackles/int
    def_idx = find_stat_table(
        tables,
        want_cols_any=["tack", "sack", "int"],
        want_cols_all=[],
    )

    pass_names = get_name_table_for(tables, pass_idx) if pass_idx is not None else None
    rush_names = get_name_table_for(tables, rush_idx) if rush_idx is not None else None
    rec_names = get_name_table_for(tables, rec_idx) if rec_idx is not None else None
    def_names = get_name_table_for(tables, def_idx) if def_idx is not None else None

    pass_stats = tables[pass_idx] if pass_idx is not None else None
    rush_stats = tables[rush_idx] if rush_idx is not None else None
    rec_stats = tables[rec_idx] if rec_idx is not None else None
    def_stats = tables[def_idx] if def_idx is not None else None

    # offense col selection (these are usually exactly named, but still keep fuzzy)
    col_pass_yds = pick_col_fuzzy(pass_stats, ["yds", r"\byds\b"]) if pass_stats is not None else None
    col_pass_td = pick_col_fuzzy(pass_stats, ["td", r"\btd\b"]) if pass_stats is not None else None
    col_pass_int = pick_col_fuzzy(pass_stats, ["int", r"\bint\b"]) if pass_stats is not None else None

    col_rush_yds = pick_col_fuzzy(rush_stats, ["yds", r"\byds\b"]) if rush_stats is not None else None
    col_rush_td = pick_col_fuzzy(rush_stats, ["td", r"\btd\b"]) if rush_stats is not None else None

    col_rec_yds = pick_col_fuzzy(rec_stats, ["yds", r"\byds\b"]) if rec_stats is not None else None
    col_rec_td = pick_col_fuzzy(rec_stats, ["td", r"\btd\b"]) if rec_stats is not None else None

    # defense fuzzy cols (THIS IS THE MAIN FIX)
    # try many variations so it never fails
    col_sack = pick_col_fuzzy(def_stats, [
        "sack", "sacks", "sacks sack", "sacks\s+sack", r"\bsack(s)?\b", "sk",
    ]) if def_stats is not None else None

    col_tack = pick_col_fuzzy(def_stats, [
        "tot", "total", "tackles", "tackles tot", "tackles\s+tot", r"\btot(al)?\b",
    ]) if def_stats is not None else None

    col_def_int = pick_col_fuzzy(def_stats, [
        "interceptions", "interceptions int", r"\bint\b", "ints",
    ]) if def_stats is not None else None

    # compute leaders (never throws)
    pass_yds_who, pass_yds = ("—", 0) if not (pass_names is not None and pass_stats is not None and col_pass_yds) else leader_from_two_tables(pass_names, pass_stats, col_pass_yds, "int")
    pass_td_who, pass_td = ("—", 0) if not (pass_names is not None and pass_stats is not None and col_pass_td) else leader_from_two_tables(pass_names, pass_stats, col_pass_td, "int")
    pass_int_who, pass_int = ("—", 0) if not (pass_names is not None and pass_stats is not None and col_pass_int) else leader_from_two_tables(pass_names, pass_stats, col_pass_int, "int")

    rush_yds_who, rush_yds = ("—", 0) if not (rush_names is not None and rush_stats is not None and col_rush_yds) else leader_from_two_tables(rush_names, rush_stats, col_rush_yds, "int")
    rush_td_who, rush_td = ("—", 0) if not (rush_names is not None and rush_stats is not None and col_rush_td) else leader_from_two_tables(rush_names, rush_stats, col_rush_td, "int")

    rec_yds_who, rec_yds = ("—", 0) if not (rec_names is not None and rec_stats is not None and col_rec_yds) else leader_from_two_tables(rec_names, rec_stats, col_rec_yds, "int")
    rec_td_who, rec_td = ("—", 0) if not (rec_names is not None and rec_stats is not None and col_rec_td) else leader_from_two_tables(rec_names, rec_stats, col_rec_td, "int")

    sack_who, sack_val = ("—", 0) if not (def_names is not None and def_stats is not None and col_sack) else leader_from_two_tables(def_names, def_stats, col_sack, "float1")
    tack_who, tack_val = ("—", 0) if not (def_names is not None and def_stats is not None and col_tack) else leader_from_two_tables(def_names, def_stats, col_tack, "int")
    int_who, int_val = ("—", 0) if not (def_names is not None and def_stats is not None and col_def_int) else leader_from_two_tables(def_names, def_stats, col_def_int, "int")

    # Router expects 3 values; we keep "team" placeholder as "" (you can display or ignore)
    leaders: Dict[str, Tuple[str, str, str]] = {
        "Passing Yards": (pass_yds_who, "", str(int(pass_yds))),
        "Passing TDs": (pass_td_who, "", str(int(pass_td))),
        "Interceptions Thrown": (pass_int_who, "", str(int(pass_int))),
        "Rushing Yards": (rush_yds_who, "", str(int(rush_yds))),
        "Rushing TDs": (rush_td_who, "", str(int(rush_td))),
        "Receiving Yards": (rec_yds_who, "", str(int(rec_yds))),
        "Receiving TDs": (rec_td_who, "", str(int(rec_td))),
        "Sacks": (sack_who, "", f"{float(sack_val):.1f}"),
        "Tackles": (tack_who, "", str(int(tack_val))),
        "Interceptions": (int_who, "", str(int(int_val))),
    }

    return leaders


# -------------------------
# Poster drawing (grid)
# -------------------------
def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except:
            pass
    return ImageFont.load_default()


def draw_leaders_grid_poster(
    out_path: str,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, str, str, str]],
    cols: int = 2,
    rows: int = 4,
):
    """
    sections: list of tuples: (category, player, team, value)
    """
    W, H = 1440, 2560
    bg = (12, 12, 16)
    card = (20, 20, 28)
    border = (45, 45, 60)
    txt = (235, 235, 245)
    sub = (170, 170, 185)

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    title_font = load_font(66, bold=True)
    sub_font = load_font(28, bold=False)
    cat_font = load_font(34, bold=True)
    name_font = load_font(30, bold=False)
    val_font = load_font(46, bold=True)

    d.text((70, 60), title, font=title_font, fill=txt)
    d.text((70, 145), subtitle, font=sub_font, fill=sub)

    # grid area
    grid_x0, grid_y0 = 70, 230
    grid_x1, grid_y1 = W - 70, H - 120

    d.rounded_rectangle([grid_x0, grid_y0, grid_x1, grid_y1], radius=26, fill=card, outline=border, width=2)

    inner_pad = 30
    gx0 = grid_x0 + inner_pad
    gy0 = grid_y0 + inner_pad
    gx1 = grid_x1 - inner_pad
    gy1 = grid_y1 - inner_pad

    cell_w = (gx1 - gx0) / cols
    cell_h = (gy1 - gy0) / rows

    for idx, (cat, player, team, value) in enumerate(sections):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break

        x0 = gx0 + c * cell_w
        y0 = gy0 + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        # separators
        if c > 0:
            d.line([(x0, y0), (x0, y1)], fill=(35, 35, 48), width=2)
        if r > 0:
            d.line([(x0, y0), (x1, y0)], fill=(35, 35, 48), width=2)

        pad = 26
        cat = normalize_spaces(cat)
        player = normalize_spaces(player)
        value = normalize_spaces(value)

        d.text((x0 + pad, y0 + 18), cat, font=cat_font, fill=txt)
        d.text((x0 + pad, y0 + 68), player, font=name_font, fill=sub)

        tw = d.textlength(value, font=val_font)
        d.text((x1 - pad - tw, y0 + 28), value, font=val_font, fill=txt)

    img.save(out_path, "PNG")
