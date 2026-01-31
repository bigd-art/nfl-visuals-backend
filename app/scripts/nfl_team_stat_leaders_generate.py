# app/scripts/nfl_team_stat_leaders_generate.py
# ==========================================================
# Render-safe ESPN Team Leaders scraper
#
# Public API used by router:
#   leaders = extract_team_leaders(team_url)
#   leaders["Sacks"] -> (rank, player, team, value)
#
# Poster renderer must remain identical:
#   draw_leaders_grid_poster(...)
# ==========================================================

import os
import re
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

Number = Union[int, float]

# ESPN can be touchy in CI. These headers + retries help a lot.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
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
            " ".join(
                [str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]
            ).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -----------------------------
# fetch + read_html with retries
# -----------------------------
def fetch_html(team_url: str, timeout: int = 30, retries: int = 3) -> str:
    """
    ESPN may intermittently return blocked pages / partial HTML in CI.
    We retry with backoff and validate we got something table-like.
    """
    last_err = None
    session = requests.Session()

    for attempt in range(1, retries + 1):
        try:
            r = session.get(team_url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            html = r.text or ""

            # Basic sanity checks: ESPN stat pages should contain tables somewhere.
            # If we don't see any table markers, we may have been served a blocked/redirect page.
            if "<table" not in html.lower():
                # Sometimes ESPN loads via scripts; but team stats pages typically still include tables.
                # Provide a helpful error snippet.
                snippet = normalize_spaces(html[:400])
                raise RuntimeError(f"ESPN HTML contained no <table>. Snippet: {snippet}")

            return html

        except Exception as e:
            last_err = e
            # exponential-ish backoff
            sleep_s = 1.5 * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch ESPN page after {retries} tries: {last_err}")


def fetch_tables(team_url: str) -> List[pd.DataFrame]:
    html = fetch_html(team_url)
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        # Add context to make CI failures obvious
        raise RuntimeError(f"pd.read_html failed. Ensure lxml/html5lib are installed. Error: {e}")
    if not tables:
        raise RuntimeError("pd.read_html returned 0 tables (unexpected for ESPN team stats page).")
    return tables


# -----------------------------
# detect name tables + classify stat tables
# -----------------------------
def is_name_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    df = flatten_columns(df.copy())
    if df.shape[1] != 1:
        return False
    col = df.columns[0]
    sample = df[col].astype(str).head(10).tolist()
    alpha = sum(bool(re.search(r"[A-Za-z]", s)) for s in sample)
    # Must have enough alpha strings to be a name list
    return alpha >= max(2, len(sample) // 2)


def classify_stats_table(df: pd.DataFrame) -> Optional[str]:
    """
    Returns: "pass" | "rush" | "rec" | "def" | None
    """
    if df is None or df.empty:
        return None

    df = flatten_columns(df.copy())
    cols = [str(c).strip().lower() for c in df.columns]
    cols_norm = [re.sub(r"\s+", " ", c) for c in cols]

    def has_any(substrs: List[str]) -> bool:
        return any(any(s in c for c in cols_norm) for s in substrs)

    # Defense signature
    # ESPN defense tables often have: SOLO, AST, TOT, SACK, INT, FF ...
    def_hits = 0
    for k in ["solo", "ast", "tot", "tkl", "tack", "sack", "int", "ff"]:
        def_hits += 1 if has_any([k]) else 0
    if def_hits >= 3 and has_any(["sack"]):
        return "def"

    # Passing signature: CMP, ATT, YDS, TD, INT, RTG/QBR
    pass_hits = 0
    for k in ["cmp", "att", "yds", "td", "int", "rtg", "qbr", "pct"]:
        pass_hits += 1 if has_any([k]) else 0
    if pass_hits >= 5 and (has_any(["cmp"]) or has_any(["rtg", "qbr"])):
        return "pass"

    # Receiving signature: REC, TGTS, YDS, TD
    rec_hits = 0
    for k in ["rec", "tgts", "yds", "td", "avg", "lng"]:
        rec_hits += 1 if has_any([k]) else 0
    if rec_hits >= 4 and has_any(["rec"]):
        return "rec"

    # Rushing signature: ATT, YDS, AVG, TD, LNG (but not CMP/RTG)
    rush_hits = 0
    for k in ["att", "yds", "avg", "td", "lng"]:
        rush_hits += 1 if has_any([k]) else 0
    if rush_hits >= 4 and not has_any(["cmp", "rtg", "qbr"]):
        return "rush"

    return None


def find_named_table_pairs(tables: List[pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    ESPN often renders: [names] [stats] [names] [stats] ...
    We detect stat tables by signature and then grab the closest name table before it.
    """
    pairs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    name_idx = set()
    cat_at: Dict[int, str] = {}

    for i, t in enumerate(tables):
        try:
            if is_name_table(t):
                name_idx.add(i)
            cat = classify_stats_table(t)
            if cat:
                cat_at[i] = cat
        except Exception:
            continue

    for stat_i, cat in cat_at.items():
        if cat in pairs:
            continue

        # find nearest preceding name table
        j = stat_i - 1
        while j >= 0:
            if j in name_idx:
                # Both should have at least 2 rows to be meaningful
                if len(tables[j]) >= 2 and len(tables[stat_i]) >= 2:
                    pairs[cat] = (tables[j], tables[stat_i])
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
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]

    for i, c in enumerate(low):
        if c == "td":
            return cols[i]
    for i, c in enumerate(low):
        if re.search(r"\btd\b", c) and "td%" not in c and "ltd" not in c:
            return cols[i]
    raise RuntimeError(f"Could not find TD column. Columns={cols}")


def pick_int_col(df: pd.DataFrame) -> str:
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
        if c == "yds":
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
        if "sack" in c:
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

    for target in ["tot", "tkl", "tack", "combined", "comb", "total tackles", "tot tackles"]:
        for i, c in enumerate(low):
            if c == target:
                return cols[i]

    for i, c in enumerate(low):
        if re.search(r"\b(tot|tkl|tack)\b", c):
            return cols[i]

    for i, c in enumerate(low):
        if "tack" in c or "tkl" in c or "total" in c:
            return cols[i]

    raise RuntimeError(f"Could not find tackles column (TOT/TKL/etc). Columns={cols}")


# -----------------------------
# leader extraction (fixes missing values)
# -----------------------------
def leader_from_name_and_stats(
    name_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    stat_col: str,
    mode: str,  # "int" or "float1"
) -> Tuple[str, Number]:
    """
    Returns: (player_name, best_value)
    Ensures name/stat alignment is purely positional (CI-safe).
    """
    name_df = flatten_columns(name_df.copy()).reset_index(drop=True)
    stat_df = flatten_columns(stat_df.copy()).reset_index(drop=True)

    if name_df.empty or stat_df.empty:
        raise RuntimeError("Empty name or stats table.")

    # Trim to same length first (positional)
    n0 = min(len(name_df), len(stat_df))
    name_df = name_df.iloc[:n0].reset_index(drop=True)
    stat_df = stat_df.iloc[:n0].reset_index(drop=True)

    name_col = name_df.columns[0]
    raw_names = name_df[name_col].astype(str).map(normalize_spaces)

    # Build a boolean keep mask positionally
    bad = (
        raw_names.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False)
        | (raw_names.str.len() == 0)
    ).to_numpy()

    keep_mask = ~bad

    # Apply mask positionally to BOTH frames
    names = raw_names.iloc[keep_mask].reset_index(drop=True)
    stats = stat_df.iloc[keep_mask].reset_index(drop=True)

    if stat_col not in stats.columns:
        raise RuntimeError(f"Expected stat col '{stat_col}' not found. Columns={list(stats.columns)}")

    # Convert values
    if mode == "float1":
        vals = stats[stat_col].map(safe_float)
    else:
        vals = stats[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")

    # Remove NaNs while keeping positional linkage
    good_idx = vals_num.notna().to_numpy()
    names = names.iloc[good_idx].reset_index(drop=True)
    vals_num = vals_num.iloc[good_idx].reset_index(drop=True)

    if vals_num.empty:
        raise RuntimeError(f"All values are NaN/empty for '{stat_col}' after cleaning.")

    # Use argmax (position-based) — avoids idxmax index label weirdness
    best_pos = int(vals_num.values.argmax())
    who = normalize_spaces(names.iloc[best_pos])
    best_val = vals_num.iloc[best_pos]

    return who, best_val


# -----------------------------
# PUBLIC API (router uses this)
# -----------------------------
def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, str, str]]:
    tables = fetch_tables(team_url)
    pairs = find_named_table_pairs(tables)

    missing = [k for k in ["pass", "rush", "rec", "def"] if k not in pairs]
    if missing:
        # Provide some debugging help: count tables and show first few column sets
        sample_cols = []
        for i, t in enumerate(tables[:8]):
            try:
                tt = flatten_columns(t.copy())
                sample_cols.append((i, list(tt.columns)[:8]))
            except Exception:
                sample_cols.append((i, ["<unreadable>"]))
        raise RuntimeError(f"Could not detect ESPN tables: {missing}. Table samples: {sample_cols}")

    name_pass, pass_stats = pairs["pass"]
    name_rush, rush_stats = pairs["rush"]
    name_rec, rec_stats = pairs["rec"]
    name_def, def_stats = pairs["def"]

    pass_stats = flatten_columns(pass_stats)
    rush_stats = flatten_columns(rush_stats)
    rec_stats = flatten_columns(rec_stats)
    def_stats = flatten_columns(def_stats)

    # offense columns
    col_pass_yds = pick_yds_col(pass_stats)
    col_pass_td = pick_td_col(pass_stats)
    col_pass_int = pick_int_col(pass_stats)

    col_rush_yds = pick_yds_col(rush_stats)
    col_rush_td = pick_td_col(rush_stats)

    col_rec_yds = pick_yds_col(rec_stats)
    col_rec_td = pick_td_col(rec_stats)

    # defense columns
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

    # router expects: (rank, name, team, value) — team blank is fine
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


# ==========================================================
# KEEP YOUR EXISTING draw_leaders_grid_poster BELOW THIS LINE
# (Unchanged poster design)
# ==========================================================

from PIL import Image, ImageDraw, ImageFont

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
        while len(t) > 3 and d.textlength(t + "…", font=font) > max_w:
            t = t[:-1]
        return t + "…"

    for i, sec in enumerate(sections):
        if i >= cols * rows:
            break

        label = sec[0]
        if len(sec) == 4:
            _, rank, name, value = sec
            team = ""
        else:
            _, rank, name, team, value = sec

        cx = grid_x0 + (i % cols) * cell_w
        cy = grid_y0 + (i // cols) * cell_h

        px = 26
        py = 22

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
