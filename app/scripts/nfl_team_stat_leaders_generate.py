# app/scripts/nfl_team_stat_leaders_generate.py

import os
import re
from io import StringIO
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

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
    except Exception:
        return None


def safe_int(x) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    return int(round(f))


def fetch_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(
                [str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]
            ).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _colnorm(c: str) -> str:
    c = normalize_spaces(c).lower()
    c = re.sub(r"[^a-z0-9]+", " ", c).strip()
    return c


def pick_col(df: pd.DataFrame, want_any: List[str], *, must_not_contain: List[str] = None) -> str:
    """
    Robust column picker:
    - want_any: list of keywords/phrases. If ANY matches, we accept.
    - must_not_contain: optional blacklist keywords.
    """
    if must_not_contain is None:
        must_not_contain = []

    cols = list(df.columns)
    norm_map = {_colnorm(c): c for c in cols}

    want_norm = [_colnorm(w) for w in want_any]
    bad_norm = [_colnorm(b) for b in must_not_contain]

    # 1) exact normalized match
    for wn in want_norm:
        for kn, orig in norm_map.items():
            if kn == wn:
                if any(b in kn for b in bad_norm):
                    continue
                return orig

    # 2) substring match (keyword inside col)
    for wn in want_norm:
        for kn, orig in norm_map.items():
            if wn and wn in kn:
                if any(b in kn for b in bad_norm):
                    continue
                return orig

    # 3) token overlap match (for weird ESPN headers)
    want_tokens = [set(wn.split()) for wn in want_norm if wn]
    for kn, orig in norm_map.items():
        kt = set(kn.split())
        for wt in want_tokens:
            if wt and wt.issubset(kt):
                if any(b in kn for b in bad_norm):
                    continue
                return orig

    raise RuntimeError(f"Could not find column for {want_any}. Have: {list(df.columns)}")


def split_name_and_stats(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    ESPN sometimes provides:
      - A "Name" table + a separate stats table
      - OR a single table that includes the player name as the first column
    This tries to extract a name series and a stats df.
    """
    df = df.copy()
    df = flatten_columns(df)

    # common name columns
    for cand in ["Name", "PLAYER", "Player", "Athlete"]:
        if cand in df.columns:
            names = df[cand].astype(str).map(normalize_spaces)
            stats = df.drop(columns=[cand])
            return names, stats

    # fallback: first column as name if it looks like names
    first = df.columns[0]
    col0 = df[first].astype(str).map(normalize_spaces)

    # Heuristic: if many rows have spaces/letters and NOT mostly numeric -> treat as names
    nonempty = col0[col0 != ""]
    if len(nonempty) > 0:
        numeric_like = nonempty.apply(lambda x: bool(re.fullmatch(r"[\d\.\-]+", x)))
        # if < 30% numeric-like -> names
        if numeric_like.mean() < 0.30:
            names = col0
            stats = df.drop(columns=[first])
            return names, stats

    # If we cannot find names, still return something (names blank)
    names = pd.Series([""] * len(df))
    return names, df


def leader_from_table(names: pd.Series, stats: pd.DataFrame, stat_col: str, mode: str) -> Tuple[str, Number]:
    """
    Find the max leader for stat_col, returning (player_name, value).
    """
    names = names.reset_index(drop=True).astype(str).map(normalize_spaces)
    stats = stats.reset_index(drop=True)

    # Remove "Total" rows
    mask_total = names.str.lower().str.contains(r"\btotal\b", na=False)
    if mask_total.any():
        names = names.loc[~mask_total].reset_index(drop=True)
        stats = stats.loc[~mask_total].reset_index(drop=True)

    if stat_col not in stats.columns:
        raise RuntimeError(f"Missing stat col {stat_col} in {list(stats.columns)}")

    if mode == "float1":
        vals = stats[stat_col].map(safe_float)
    else:
        vals = stats[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")
    if vals_num.isna().all():
        raise RuntimeError(f"All values NaN for {stat_col}")

    best_i = int(vals_num.idxmax())
    best_val = float(vals_num.loc[best_i]) if mode == "float1" else int(vals_num.loc[best_i])

    who = names.iloc[best_i] if best_i < len(names) else ""
    who = normalize_spaces(who)
    return who, best_val


# -------------------------
# Public API: scraping
# -------------------------
def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Returns:
      leaders[category] = (player_name, value_string, extra_string)
    Router expects leaders[cat][0], [1], [2].
    """
    tables_raw = fetch_tables(team_url)
    tables = [flatten_columns(t) for t in tables_raw]

    # Identify candidate tables by columns
    def has_cols(df: pd.DataFrame, keys: List[str]) -> bool:
        cols = [_colnorm(c) for c in df.columns]
        return all(any(_colnorm(k) == c or _colnorm(k) in c for c in cols) for k in keys)

    passing_idx = None
    rushing_idx = None
    receiving_idx = None
    defense_idx = None

    # We search for stats tables first; names might be in same table or previous table
    for i, df in enumerate(tables):
        cols_norm = " ".join([_colnorm(c) for c in df.columns])

        # passing: needs yds td int
        if passing_idx is None and ("yds" in cols_norm and "td" in cols_norm and re.search(r"\bint\b", cols_norm)):
            passing_idx = i

        # rushing: has car/att + yds + td
        if rushing_idx is None and ("yds" in cols_norm and "td" in cols_norm and ("car" in cols_norm or "att" in cols_norm)):
            # avoid picking passing again
            if i != passing_idx:
                rushing_idx = i

        # receiving: has rec + yds + td
        if receiving_idx is None and ("rec" in cols_norm and "yds" in cols_norm and "td" in cols_norm):
            # avoid picking rushing/passing
            if i not in {passing_idx, rushing_idx}:
                receiving_idx = i

        # defense: sacks + int + some tackle-ish column
        if defense_idx is None and ("sack" in cols_norm and re.search(r"\bint\b", cols_norm)):
            defense_idx = i

    if passing_idx is None:
        raise RuntimeError("Could not locate passing stats table from ESPN tables.")
    if rushing_idx is None:
        raise RuntimeError("Could not locate rushing stats table from ESPN tables.")
    if receiving_idx is None:
        raise RuntimeError("Could not locate receiving stats table from ESPN tables.")
    if defense_idx is None:
        raise RuntimeError("Could not locate defense stats table from ESPN tables.")

    def get_name_stats(i: int) -> Tuple[pd.Series, pd.DataFrame]:
        # best case: names are inside the same table
        names, stats = split_name_and_stats(tables[i])
        if names.str.strip().eq("").all():
            # fallback: try previous table as the names table
            if i - 1 >= 0:
                n2, _ = split_name_and_stats(tables[i - 1])
                if not n2.str.strip().eq("").all():
                    names = n2
        return names, stats

    # Passing
    pass_names, pass_stats = get_name_stats(passing_idx)
    col_pass_yds = pick_col(pass_stats, ["YDS", "Yards", "Pass Yds", "Passing Yds"])
    col_pass_td = pick_col(pass_stats, ["TD", "Pass TD", "Passing TD"])
    # Avoid grabbing "INT%" or other weird ones
    col_pass_int = pick_col(pass_stats, ["INT", "Interceptions"], must_not_contain=["int pct", "int%"])

    pass_yds_who, pass_yds = leader_from_table(pass_names, pass_stats, col_pass_yds, "int")
    pass_td_who, pass_td = leader_from_table(pass_names, pass_stats, col_pass_td, "int")
    pass_int_who, pass_int = leader_from_table(pass_names, pass_stats, col_pass_int, "int")

    # Rushing
    rush_names, rush_stats = get_name_stats(rushing_idx)
    col_rush_yds = pick_col(rush_stats, ["YDS", "Rush Yds", "Rushing Yds", "Yards"])
    col_rush_td = pick_col(rush_stats, ["TD", "Rush TD", "Rushing TD"])
    rush_yds_who, rush_yds = leader_from_table(rush_names, rush_stats, col_rush_yds, "int")
    rush_td_who, rush_td = leader_from_table(rush_names, rush_stats, col_rush_td, "int")

    # Receiving
    rec_names, rec_stats = get_name_stats(receiving_idx)
    col_rec_yds = pick_col(rec_stats, ["YDS", "Rec Yds", "Receiving Yds", "Yards"])
    col_rec_td = pick_col(rec_stats, ["TD", "Rec TD", "Receiving TD"])
    rec_yds_who, rec_yds = leader_from_table(rec_names, rec_stats, col_rec_yds, "int")
    rec_td_who, rec_td = leader_from_table(rec_names, rec_stats, col_rec_td, "int")

    # Defense
    def_names, def_stats = get_name_stats(defense_idx)

    # sacks
    col_sack = pick_col(def_stats, ["SACK", "SACKS", "Sacks", "Sack"])
    # tackles (THIS is what was failing)
    col_tackles = pick_col(
        def_stats,
        [
            "Tackles TOT",
            "Tackles Total",
            "Tackles",
            "TOT",
            "TOTL",
            "TOTAL",
            "COMB",
            "COMBINED",
            "TKL",
            "TCK",
            "TACK",
        ],
    )
    # interceptions
    col_int = pick_col(def_stats, ["INT", "Interceptions", "INTS"], must_not_contain=["int pct", "int%"])

    sack_who, sack_val = leader_from_table(def_names, def_stats, col_sack, "float1")
    tackles_who, tackles_val = leader_from_table(def_names, def_stats, col_tackles, "int")
    int_who, int_val = leader_from_table(def_names, def_stats, col_int, "int")

    leaders: Dict[str, Tuple[str, str, str]] = {
        "Passing Yards": (pass_yds_who, str(int(pass_yds)), ""),
        "Passing TDs": (pass_td_who, str(int(pass_td)), ""),
        "Interceptions Thrown": (pass_int_who, str(int(pass_int)), ""),
        "Rushing Yards": (rush_yds_who, str(int(rush_yds)), ""),
        "Rushing TDs": (rush_td_who, str(int(rush_td)), ""),
        "Receiving Yards": (rec_yds_who, str(int(rec_yds)), ""),
        "Receiving TDs": (rec_td_who, str(int(rec_td)), ""),
        "Sacks": (sack_who, f"{float(sack_val):.1f}", ""),
        "Tackles": (tackles_who, str(int(tackles_val)), ""),
        "Interceptions": (int_who, str(int(int_val)), ""),
    }
    return leaders


# -------------------------
# Poster drawing (grid)
# -------------------------
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


def draw_leaders_grid_poster(
    out_path: str,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, str, str, str]],
    cols: int = 2,
    rows: int = 4,
) -> None:
    """
    sections: list of (category, player, value, extra)
    """
    W, H = 1440, 2560
    bg = (12, 12, 16)
    card = (20, 20, 28)
    outline = (45, 45, 60)
    sub = (170, 170, 185)
    white = (235, 235, 245)

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    title_font = load_font(64, bold=True)
    sub_font = load_font(28, bold=False)
    head_font = load_font(34, bold=True)
    name_font = load_font(28, bold=False)
    val_font = load_font(36, bold=True)

    d.text((70, 60), title, font=title_font, fill=white)
    d.text((70, 140), subtitle, font=sub_font, fill=sub)

    pad = 70
    top = 220
    bottom = 120
    grid_x0 = pad
    grid_y0 = top
    grid_x1 = W - pad
    grid_y1 = H - bottom

    d.rounded_rectangle([grid_x0, grid_y0, grid_x1, grid_y1], radius=26, fill=card, outline=outline, width=2)

    # inner grid padding
    inner_pad = 30
    gx0 = grid_x0 + inner_pad
    gy0 = grid_y0 + inner_pad
    gx1 = grid_x1 - inner_pad
    gy1 = grid_y1 - inner_pad

    cell_w = (gx1 - gx0) / cols
    cell_h = (gy1 - gy0) / rows

    # draw cells + content
    for idx, (cat, player, value, extra) in enumerate(sections):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break

        x0 = gx0 + c * cell_w
        y0 = gy0 + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        # cell border line
        d.rounded_rectangle([x0, y0, x1, y1], radius=18, outline=(35, 35, 48), width=2)

        cat = normalize_spaces(cat)
        player = normalize_spaces(player)
        value = normalize_spaces(value)
        extra = normalize_spaces(extra)

        d.text((x0 + 22, y0 + 18), cat, font=head_font, fill=white)
        d.text((x0 + 22, y0 + 70), player, font=name_font, fill=sub)

        tw = d.textlength(value, font=val_font)
        d.text((x1 - 22 - tw, y0 + 52), value, font=val_font, fill=white)

        if extra:
            d.text((x0 + 22, y1 - 38), extra, font=sub_font, fill=sub)

    img.save(out_path, "PNG")
