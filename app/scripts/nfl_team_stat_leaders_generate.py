# app/scripts/nfl_team_stat_leaders_generate.py
# ==========================================================
# Render-safe ESPN Team Leaders scraper (FIXES "could not find tot")
#
# Keeps your router API:
#   leaders = extract_team_leaders(team_url)
#   leaders["Sacks"] -> (rank, player, team, value)
#
# Keeps your existing poster renderer:
#   draw_leaders_grid_poster(...)  <-- keep your old one below
# ==========================================================

import re
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

Number = Union[int, float]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

# -----------------------------
# basic helpers
# -----------------------------
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
    # keep digits, dot, minus
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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_tables(team_url: str) -> List[pd.DataFrame]:
    r = requests.get(team_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def is_name_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    df = flatten_columns(df.copy())
    if df.shape[1] != 1:
        return False
    col = df.columns[0]
    sample = df[col].astype(str).head(8).tolist()
    alpha = sum(bool(re.search(r"[A-Za-z]", s)) for s in sample)
    return alpha >= max(1, len(sample) // 2)


def clean_name_series(name_s: pd.Series) -> pd.Series:
    s = name_s.astype(str).map(normalize_spaces)
    bad = s.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False)
    s = s[~bad].copy()
    s = s[s.str.len() > 0]
    return s.reset_index(drop=True)


def classify_stats_table(df: pd.DataFrame) -> Optional[str]:
    """
    Returns: "pass" | "rush" | "rec" | "def" | None
    """
    if df is None or df.empty:
        return None
    df = flatten_columns(df.copy())
    cols = [str(c).strip().lower() for c in df.columns]

    # defense signature: tackles + sacks + ints often show up together
    def_hits = 0
    for k in ["sack", "int", "solo", "ast", "tot", "tkl", "tack", "ff"]:
        def_hits += 1 if any(k in c for c in cols) else 0
    if def_hits >= 3 and any("sack" in c for c in cols):
        return "def"

    # passing: cmp/att/qbr/rtg
    pass_hits = 0
    for k in ["cmp", "att", "pct", "yds", "td", "int", "rtg", "qbr"]:
        pass_hits += 1 if any(k == c or k in c for c in cols) else 0
    if pass_hits >= 5 and (any("cmp" in c for c in cols) or any("rtg" in c or "qbr" in c for c in cols)):
        return "pass"

    # receiving: rec/tgts
    rec_hits = 0
    for k in ["rec", "tgts", "yds", "avg", "td", "lng"]:
        rec_hits += 1 if any(k == c or k in c for c in cols) else 0
    if rec_hits >= 4 and any("rec" == c or "rec" in c for c in cols):
        return "rec"

    # rushing: att/yds/avg/td/lng (without passing-only cols)
    rush_hits = 0
    for k in ["att", "yds", "avg", "td", "lng"]:
        rush_hits += 1 if any(k == c or k in c for c in cols) else 0
    if rush_hits >= 4 and not any("cmp" in c for c in cols) and not any("rtg" in c or "qbr" in c for c in cols):
        return "rush"

    return None


def find_named_table_pairs(tables: List[pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    ESPN often renders: [names] [stats] [names] [stats] ...
    We detect stat tables by signature and then grab the closest name table right before it.
    """
    pairs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    name_idx = set()
    cats: Dict[int, str] = {}

    for i, t in enumerate(tables):
        try:
            if is_name_table(t):
                name_idx.add(i)
            cat = classify_stats_table(t)
            if cat:
                cats[i] = cat
        except Exception:
            continue

    for i, cat in cats.items():
        if cat in pairs:
            continue
        j = i - 1
        while j >= 0:
            if j in name_idx:
                if len(tables[j]) >= 2 and len(tables[i]) >= 2:
                    pairs[cat] = (tables[j], tables[i])
                break
            j -= 1

    return pairs


# -----------------------------
# strict column picking
# -----------------------------
def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def pick_td_col(df: pd.DataFrame) -> str:
    # TD must be EXACT token, not LTD / TD%
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]
    for i, c in enumerate(low):
        if c == "td":
            return cols[i]
    # token fallback (avoid ltd/td%)
    for i, c in enumerate(low):
        if re.search(r"\btd\b", c) and "td%" not in c and "ltd" not in c:
            return cols[i]
    raise RuntimeError(f"Could not find TD column. Columns={cols}")


def pick_int_col(df: pd.DataFrame) -> str:
    # INT must be exact token, not INT%
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]
    for i, c in enumerate(low):
        if c == "int":
            return cols[i]
    for i, c in enumerate(low):
        if re.search(r"\bint\b", c) and "int%" not in c:
            return cols[i]
    raise RuntimeError(f"Could not find INT column. Columns={cols}")


def pick_yds_col(df: pd.DataFrame) -> str:
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]
    for i, c in enumerate(low):
        if c == "yds" or c.endswith(" yds") or re.fullmatch(r"yds", c):
            return cols[i]
    for i, c in enumerate(low):
        if "yds" in c:
            return cols[i]
    raise RuntimeError(f"Could not find YDS column. Columns={cols}")


def pick_sack_col(df: pd.DataFrame) -> str:
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]
    for i, c in enumerate(low):
        if "sack" in c:  # sack, sacks
            return cols[i]
    raise RuntimeError(f"Could not find SACK column. Columns={cols}")


def pick_tackles_col(df: pd.DataFrame) -> str:
    """
    ESPN changes this a lot. We try:
    TOT -> TKL -> TACK -> COMB -> anything containing tack/tkl/total.
    """
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]

    # preferred exacts
    for target in ["tot", "tkl", "tack", "combined", "comb", "total tackles", "tot tackles"]:
        for i, c in enumerate(low):
            if c == target:
                return cols[i]

    # token-based (safe-ish)
    for i, c in enumerate(low):
        if re.search(r"\b(tot|tkl|tack)\b", c):
            return cols[i]

    # broad contains
    for i, c in enumerate(low):
        if "tack" in c or "tkl" in c or "total" in c:
            return cols[i]

    raise RuntimeError(f"Could not find tackles column (TOT/TKL/etc). Columns={cols}")


def leader_from_name_and_stats(
    name_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    stat_col: str,
    mode: str,
) -> Tuple[str, Number]:
    name_df = flatten_columns(name_df.copy()).reset_index(drop=True)
    stat_df = flatten_columns(stat_df.copy()).reset_index(drop=True)

    # ESPN sometimes gives slightly different row counts between the name table and stat table.
    # Force them to the same length BEFORE any masking.
    n0 = min(len(name_df), len(stat_df))
    if n0 <= 0:
        raise RuntimeError("Empty name/stats table.")

    name_df = name_df.iloc[:n0].reset_index(drop=True)
    stat_df = stat_df.iloc[:n0].reset_index(drop=True)

    name_col = name_df.columns[0]

    # Build a POSITIONAL bad-mask (no index alignment issues)
    raw_names = name_df[name_col].astype(str).map(normalize_spaces)
    bad_mask = (
        raw_names.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False)
        | (raw_names.str.len() == 0)
    ).to_numpy()  # <-- critical

    # Apply mask positionally on BOTH
    keep = ~bad_mask
    name_keep = raw_names.iloc[keep].reset_index(drop=True)
    stat_keep = stat_df.iloc[keep].reset_index(drop=True)

    # Clean names (after positional filtering)
    name_keep = name_keep[name_keep.str.len() > 0].reset_index(drop=True)

    # Now re-align again just in case cleaning changed counts
    n = min(len(name_keep), len(stat_keep))
    name_keep = name_keep.iloc[:n].reset_index(drop=True)
    stat_keep = stat_keep.iloc[:n].reset_index(drop=True)

    if stat_col not in stat_keep.columns:
        raise RuntimeError(f"Expected stat col '{stat_col}' not found. Columns={list(stat_keep.columns)}")

    if mode == "float1":
        vals = stat_keep[stat_col].map(safe_float)
    else:
        vals = stat_keep[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")
    if vals_num.isna().all():
        raise RuntimeError(f"All values are NaN for '{stat_col}'")

    best_i = int(vals_num.idxmax())
    who = normalize_spaces(name_keep.iloc[best_i])
    best_val = vals_num.iloc[best_i]
    return who, best_val



# -----------------------------
# PUBLIC API (router uses this)
# -----------------------------
def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, str, str]]:
    tables = fetch_tables(team_url)
    pairs = find_named_table_pairs(tables)

    missing = [k for k in ["pass", "rush", "rec", "def"] if k not in pairs]
    if missing:
        raise RuntimeError(f"Could not detect ESPN tables: {missing}")

    name_pass, pass_stats = pairs["pass"]
    name_rush, rush_stats = pairs["rush"]
    name_rec, rec_stats = pairs["rec"]
    name_def, def_stats = pairs["def"]

    pass_stats = flatten_columns(pass_stats)
    rush_stats = flatten_columns(rush_stats)
    rec_stats = flatten_columns(rec_stats)
    def_stats = flatten_columns(def_stats)

    # offense columns (strict)
    col_pass_yds = pick_yds_col(pass_stats)
    col_pass_td = pick_td_col(pass_stats)
    col_pass_int = pick_int_col(pass_stats)

    col_rush_yds = pick_yds_col(rush_stats)
    col_rush_td = pick_td_col(rush_stats)

    col_rec_yds = pick_yds_col(rec_stats)
    col_rec_td = pick_td_col(rec_stats)

    # defense columns (robust)
    col_sack = pick_sack_col(def_stats)
    col_tackles = pick_tackles_col(def_stats)
    col_def_int = pick_int_col(def_stats)

    # leaders
    pass_yds_who, pass_yds = leader_from_name_and_stats(name_pass, pass_stats, col_pass_yds, "int")
    pass_td_who, pass_td = leader_from_name_and_stats(name_pass, pass_stats, col_pass_td, "int")
    pass_int_who, pass_int = leader_from_name_and_stats(name_pass, pass_stats, col_pass_int, "int")

    rush_yds_who, rush_yds = leader_from_name_and_stats(name_rush, rush_stats, col_rush_yds, "int")
    rush_td_who, rush_td = leader_from_name_and_stats(name_rush, rush_stats, col_rush_td, "int")

    rec_yds_who, rec_yds = leader_from_name_and_stats(name_rec, rec_stats, col_rec_yds, "int")
    rec_td_who, rec_td = leader_from_name_and_stats(name_rec, rec_stats, col_rec_td, "int")

    sack_who, sack_val = leader_from_name_and_stats(name_def, def_stats, col_sack, "float1")
    tackles_who, tackles_val = leader_from_name_and_stats(name_def, def_stats, col_tackles, "int")
    int_who, int_val = leader_from_name_and_stats(name_def, def_stats, col_def_int, "int")

    # router expects: (rank, name, team, value)
    leaders: Dict[str, Tuple[str, str, str, str]] = {
        "Passing Yards": ("1", pass_yds_who, "", str(int(pass_yds))),
        "Passing TDs": ("1", pass_td_who, "", str(int(pass_td))),
        "Interceptions Thrown": ("1", pass_int_who, "", str(int(pass_int))),
        "Rushing Yards": ("1", rush_yds_who, "", str(int(rush_yds))),
        "Rushing TDs": ("1", rush_td_who, "", str(int(rush_td))),
        "Receiving Yards": ("1", rec_yds_who, "", str(int(rec_yds))),
        "Receiving TDs": ("1", rec_td_who, "", str(int(rec_td))),
        "Sacks": ("1", sack_who, "", f"{float(sack_val):.1f}"),
        "Tackles": ("1", tackles_who, "", str(int(tackles_val))),
        "Interceptions": ("1", int_who, "", str(int(int_val))),
    }
    return leaders


# ----------------------------------------------------------
# IMPORTANT:
# KEEP YOUR EXISTING draw_leaders_grid_poster BELOW THIS LINE
# Do NOT change it if you like the style.
# ==========================================================
# Compatibility wrapper
# Your router expects team_gen.draw_leaders_grid_poster(...)
# ==========================================================

from PIL import Image, ImageDraw, ImageFont
import os

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
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
    sections,
    cols: int = 2,
    rows: int = 4,
):
    """
    sections is what your router builds:
      (label, leaders[cat][0], leaders[cat][1], leaders[cat][2])

    We accept either:
      - (label, rank, name, value)  OR
      - (label, rank, name, team, value)
    """
    W, H = 1400, 2400
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = _load_font(60, bold=True)
    sub_font   = _load_font(28, bold=False)
    head_font  = _load_font(30, bold=True)
    name_font  = _load_font(26, bold=False)
    val_font   = _load_font(34, bold=True)

    d.text((70, 60), title, font=title_font, fill=(245, 245, 245))
    d.text((70, 135), subtitle, font=sub_font, fill=(170, 170, 185))

    pad = 60
    top = 200
    grid_x0, grid_y0 = pad, top
    grid_x1, grid_y1 = W - pad, H - 120

    d.rounded_rectangle([grid_x0, grid_y0, grid_x1, grid_y1], radius=26, fill=(20, 20, 28), outline=(45, 45, 60), width=2)

    cell_w = (grid_x1 - grid_x0) / cols
    cell_h = (grid_y1 - grid_y0) / rows

    def _fit_text(text: str, max_w: float, font: ImageFont.FreeTypeFont) -> str:
        t = str(text)
        if d.textlength(t, font=font) <= max_w:
            return t
        # simple ellipsis trim
        while len(t) > 3 and d.textlength(t + "…", font=font) > max_w:
            t = t[:-1]
        return t + "…"

    for i, sec in enumerate(sections):
        if i >= cols * rows:
            break

        # normalize section tuple
        label = sec[0]
        # accept 4-tuple or 5-tuple
        if len(sec) == 4:
            _, rank, name, value = sec
            team = ""
        else:
            _, rank, name, team, value = sec

        cx = grid_x0 + (i % cols) * cell_w
        cy = grid_y0 + (i // cols) * cell_h

        # cell padding
        px = 26
        py = 22

        # divider lines
        if (i % cols) != 0:
            d.line([(cx, cy + 18), (cx, cy + cell_h - 18)], fill=(35, 35, 48), width=2)
        if (i // cols) != 0:
            d.line([(cx + 18, cy), (cx + cell_w - 18, cy)], fill=(35, 35, 48), width=2)

        label_text = _fit_text(label, cell_w - 2 * px, head_font)
        d.text((cx + px, cy + py), label_text, font=head_font, fill=(235, 235, 245))

        who = f"{name}"
        if team:
            who = f"{name} • {team}"
        who = _fit_text(who, cell_w - 2 * px, name_font)
        d.text((cx + px, cy + py + 48), who, font=name_font, fill=(170, 170, 185))

        val = str(value)
        tw = d.textlength(val, font=val_font)
        d.text((cx + cell_w - px - tw, cy + py + 10), val, font=val_font, fill=(235, 235, 245))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")

# ----------------------------------------------------------

# def draw_leaders_grid_poster(...):
#     (leave your current implementation here unchanged)
