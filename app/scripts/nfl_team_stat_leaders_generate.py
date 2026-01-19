# app/generator/team_stat_leaders.py
# ✅ Render-safe + ESPN-robust team leader posters
# Fixes the “wrong data” issue by:
# 1) NOT relying on hard-coded table indexes (ESPN table order shifts).
# 2) Auto-detecting Passing/Rushing/Receiving/Defense tables by column signatures.
# 3) Pairing the correct NAME table with the correct STAT table (by adjacency + validation).
# 4) Dropping “Total/TEAM” rows and non-player rows defensively.
# 5) Producing consistent leaders even if ESPN adds/removes columns.

import os
import re
from io import StringIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Number = Union[int, float]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

TEAM_TO_ESPN = {
    "ARI": "ari/arizona-cardinals",
    "ATL": "atl/atlanta-falcons",
    "BAL": "bal/baltimore-ravens",
    "BUF": "buf/buffalo-bills",
    "CAR": "car/carolina-panthers",
    "CHI": "chi/chicago-bears",
    "CIN": "cin/cincinnati-bengals",
    "CLE": "cle/cleveland-browns",
    "DAL": "dal/dallas-cowboys",
    "DEN": "den/denver-broncos",
    "DET": "det/detroit-lions",
    "GB": "gb/green-bay-packers",
    "HOU": "hou/houston-texans",
    "IND": "ind/indianapolis-colts",
    "JAX": "jax/jacksonville-jaguars",
    "KC": "kc/kansas-city-chiefs",
    "LAC": "lac/los-angeles-chargers",
    "LAR": "lar/los-angeles-rams",
    "LV": "lv/las-vegas-raiders",
    "MIA": "mia/miami-dolphins",
    "MIN": "min/minnesota-vikings",
    "NE": "ne/new-england-patriots",
    "NO": "no/new-orleans-saints",
    "NYG": "nyg/new-york-giants",
    "NYJ": "nyj/new-york-jets",
    "PHI": "phi/philadelphia-eagles",
    "PIT": "pit/pittsburgh-steelers",
    "SEA": "sea/seattle-seahawks",
    "SF": "sf/san-francisco-49ers",
    "TB": "tb/tampa-bay-buccaneers",
    "TEN": "ten/tennessee-titans",
    "WAS": "wsh/washington-commanders",
    "WSH": "wsh/washington-commanders",
}

# -----------------------------
# text + parsing helpers
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
    # ESPN defense tables sometimes come as MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def fetch_all_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))

def is_name_table(df: pd.DataFrame) -> bool:
    # ESPN often renders a separate 1-col table containing player names.
    if df is None or df.empty:
        return False
    df = flatten_columns(df.copy())
    # Name tables are usually 1 column and have mostly non-numeric strings
    if df.shape[1] != 1:
        return False
    col = df.columns[0]
    sample = df[col].astype(str).head(8).tolist()
    # Avoid picking numeric-only tables
    alpha_count = sum(bool(re.search(r"[A-Za-z]", s)) for s in sample)
    return alpha_count >= max(1, len(sample) // 2)

def colset(df: pd.DataFrame) -> set:
    df = flatten_columns(df.copy())
    return {str(c).strip().lower() for c in df.columns}

def classify_stats_table(df: pd.DataFrame) -> Optional[str]:
    """
    Returns: "pass" | "rush" | "rec" | "def" | None
    Classification uses column signatures that ESPN keeps fairly stable.
    """
    if df is None or df.empty:
        return None
    cols = colset(df)

    # Passing tables usually include CMP/ATT/INT/RTG/QBR etc
    pass_hits = 0
    for k in ["cmp", "att", "pct", "yds", "td", "int", "rtg", "qbr"]:
        pass_hits += 1 if any(k == c or c.endswith(f" {k}") or k in c for c in cols) else 0

    # Rushing tables often include ATT, YDS, AVG, TD, LNG (but not CMP/RTG/QBR)
    rush_hits = 0
    for k in ["att", "yds", "avg", "td", "lng"]:
        rush_hits += 1 if any(k == c or k in c for c in cols) else 0
    has_cmp = any("cmp" == c or "cmp" in c for c in cols)
    has_rtg = any("rtg" == c or "rtg" in c for c in cols) or any("qbr" == c or "qbr" in c for c in cols)

    # Receiving tables often include REC, TGTS, YDS, AVG, TD, LNG
    rec_hits = 0
    for k in ["rec", "tgts", "yds", "avg", "td", "lng"]:
        rec_hits += 1 if any(k == c or k in c for c in cols) else 0

    # Defense tables often include TOT, SOLO, AST, SACK, INT, FF
    def_hits = 0
    for k in ["tot", "solo", "ast", "sack", "int", "ff"]:
        def_hits += 1 if any(k == c or k in c for c in cols) else 0

    # Decide with priority (def is most distinct)
    if def_hits >= 3 and ("sack" in " ".join(cols) or "tot" in " ".join(cols)):
        return "def"

    # Passing should strongly contain QB-only columns
    if pass_hits >= 5 and (has_cmp or has_rtg):
        return "pass"

    # Receiving should contain rec/tgts
    if rec_hits >= 4 and any("rec" == c or c.startswith("rec") or "rec" in c for c in cols):
        return "rec"

    # Rushing is often the remaining YDS/TD/ATT/AVG/LNG table without passing-only cols
    if rush_hits >= 4 and not has_cmp and not has_rtg:
        return "rush"

    return None

def clean_name_series(name_s: pd.Series) -> pd.Series:
    s = name_s.astype(str).map(normalize_spaces)

    # Drop obvious non-player / summary rows
    bad = s.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False)
    s = s[~bad].copy()

    # Some ESPN rows can be empty
    s = s[s.str.len() > 0]
    return s.reset_index(drop=True)

def pick_col_by_keywords(df: pd.DataFrame, keywords: List[str]) -> str:
    cols = list(flatten_columns(df.copy()).columns)
    lowmap = {c.lower(): c for c in cols}

    # Exact / contains match across all keywords in order
    for kw in keywords:
        kw = kw.lower()
        for low, orig in lowmap.items():
            if low == kw or kw in low:
                return orig

    # As fallback, try "contains all tokens"
    for low, orig in lowmap.items():
        ok = True
        for token in keywords:
            token = token.lower()
            if token not in low:
                ok = False
                break
        if ok:
            return orig

    raise RuntimeError(f"Missing expected column for keywords={keywords}. Have columns={cols}")

def leader_from_name_and_stats(
    name_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    stat_col: str,
    mode: str,
) -> Tuple[str, Number]:
    name_df = flatten_columns(name_df.copy())
    stat_df = flatten_columns(stat_df.copy())

    # Ensure first col of name_df is name
    name_col = name_df.columns[0]
    names = clean_name_series(name_df[name_col])

    # Align stats to the *same length* as names AFTER cleaning by dropping corresponding stat rows
    # We do it by applying the same "bad row" mask on the original name series.
    raw_names = name_df[name_col].astype(str).map(normalize_spaces)
    bad_mask = raw_names.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False) | (raw_names.str.len() == 0)

    stat_df2 = stat_df.loc[~bad_mask].reset_index(drop=True).copy()
    n = min(len(names), len(stat_df2))
    names = names.iloc[:n].reset_index(drop=True)
    stat_df2 = stat_df2.iloc[:n].reset_index(drop=True)

    if stat_col not in stat_df2.columns:
        raise RuntimeError(f"Expected stat col '{stat_col}' not found. Columns={list(stat_df2.columns)}")

    if mode == "float1":
        vals = stat_df2[stat_col].map(safe_float)
    else:
        vals = stat_df2[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")
    if vals_num.isna().all():
        raise RuntimeError(f"All values are NaN for '{stat_col}'")

    best_i = int(vals_num.idxmax())
    who = normalize_spaces(names.iloc[best_i])
    best_val = vals_num.iloc[best_i]
    return who, best_val

# -----------------------------
# poster rendering (Render-safe fonts)
# -----------------------------
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        # Render / Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # mac fallback (dev)
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_team_poster(out_path: str, title: str, subtitle: str, rows: List[Tuple[str, str, str]]):
    W, H = 1400, 2400
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = load_font(64, bold=True)
    sub_font   = load_font(28, bold=False)
    head_font  = load_font(34, bold=True)
    name_font  = load_font(28, bold=False)
    val_font   = load_font(30, bold=True)

    d.text((70, 60), title, font=title_font, fill=(245,245,245))
    d.text((70, 140), subtitle, font=sub_font, fill=(170,170,185))

    x0, y0 = 70, 220
    x1, y1 = W - 70, H - 120
    d.rounded_rectangle([x0, y0, x1, y1], radius=26, fill=(20,20,28), outline=(45,45,60), width=2)

    row_h = 240
    pad_x = 40
    for i, (label, who, val) in enumerate(rows):
        ry = y0 + 30 + i * row_h

        d.text((x0 + pad_x, ry), label, font=head_font, fill=(235,235,245))
        d.text((x0 + pad_x, ry + 55), who, font=name_font, fill=(170,170,185))

        tw = d.textlength(val, font=val_font)
        d.text((x1 - pad_x - tw, ry + 25), val, font=val_font, fill=(235,235,245))

        if i < len(rows) - 1:
            d.line([(x0 + 25, ry + row_h - 25), (x1 - 25, ry + row_h - 25)], fill=(35,35,48), width=2)

    img.save(out_path, "PNG")

# -----------------------------
# core extraction (NO hard-coded table indexes)
# -----------------------------
def find_named_table_pairs(tables: List[pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk through all ESPN tables and build pairs (name_table, stats_table) for:
    pass, rush, rec, def

    ESPN often renders: [names] [stats] [names] [stats] ...
    but can insert extra tables, so we classify the stats tables and then find
    the nearest valid name-table directly preceding it.
    """
    pairs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    # Precompute which indexes are name tables and stats categories
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

    # For each stats table, find the closest name table right before it.
    for i, cat in cat_at.items():
        if cat in pairs:
            continue  # keep first valid
        # walk backwards to find nearest name table
        j = i - 1
        while j >= 0:
            if j in name_idx:
                # validate that lengths are reasonably aligned
                nlen = len(tables[j])
                slen = len(tables[i])
                if nlen >= 2 and slen >= 2:
                    pairs[cat] = (tables[j], tables[i])
                break
            j -= 1

    return pairs

def build_team_leaders(team: str, team_url: Optional[str] = None) -> Tuple[List[Tuple[str,str,str]], List[Tuple[str,str,str]], str]:
    """
    Returns:
      offense_rows, defense_rows, subtitle
    """
    t = team.strip().upper()
    if team_url:
        url = team_url
    else:
        if t not in TEAM_TO_ESPN:
            raise ValueError(f"Unknown team '{t}'")
        url = f"https://www.espn.com/nfl/team/stats/_/name/{TEAM_TO_ESPN[t]}"

    tables = fetch_all_tables(url)
    pairs = find_named_table_pairs(tables)

    missing = [k for k in ["pass", "rush", "rec", "def"] if k not in pairs]
    if missing:
        # If ESPN changes markup, this message will tell you what's missing
        raise RuntimeError(f"Could not detect required ESPN table(s): {missing}")

    name_pass, pass_stats = pairs["pass"]
    name_rush, rush_stats = pairs["rush"]
    name_rec,  rec_stats  = pairs["rec"]
    name_def,  def_stats  = pairs["def"]

    pass_stats = flatten_columns(pass_stats)
    rush_stats = flatten_columns(rush_stats)
    rec_stats  = flatten_columns(rec_stats)
    def_stats  = flatten_columns(def_stats)

    # OFFENSE leaders (choose columns robustly)
    col_pass_yds = pick_col_by_keywords(pass_stats, ["yds"])
    col_pass_td  = pick_col_by_keywords(pass_stats, ["td"])
    col_pass_int = pick_col_by_keywords(pass_stats, ["int"])

    col_rush_yds = pick_col_by_keywords(rush_stats, ["yds"])
    col_rush_td  = pick_col_by_keywords(rush_stats, ["td"])

    col_rec_yds  = pick_col_by_keywords(rec_stats, ["yds"])
    col_rec_td   = pick_col_by_keywords(rec_stats, ["td"])

    pass_yds_who, pass_yds = leader_from_name_and_stats(name_pass, pass_stats, col_pass_yds, "int")
    pass_td_who,  pass_td  = leader_from_name_and_stats(name_pass, pass_stats, col_pass_td,  "int")
    pass_int_who, pass_int = leader_from_name_and_stats(name_pass, pass_stats, col_pass_int, "int")

    rush_yds_who, rush_yds = leader_from_name_and_stats(name_rush, rush_stats, col_rush_yds, "int")
    rush_td_who,  rush_td  = leader_from_name_and_stats(name_rush, rush_stats, col_rush_td,  "int")

    rec_yds_who,  rec_yds  = leader_from_name_and_stats(name_rec,  rec_stats,  col_rec_yds,  "int")
    rec_td_who,   rec_td   = leader_from_name_and_stats(name_rec,  rec_stats,  col_rec_td,   "int")

    # DEFENSE leaders
    # ESPN labels vary, so search by keywords
    col_sack = pick_col_by_keywords(def_stats, ["sack"])
    col_tot  = pick_col_by_keywords(def_stats, ["tot"])   # total tackles
    col_int  = pick_col_by_keywords(def_stats, ["int"])   # interceptions

    sack_who, sack_val = leader_from_name_and_stats(name_def, def_stats, col_sack, "float1")
    tot_who,  tot_val  = leader_from_name_and_stats(name_def, def_stats, col_tot,  "int")
    int_who,  int_val  = leader_from_name_and_stats(name_def, def_stats, col_int,  "int")

    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"{t} • Updated {updated}"

    offense_rows = [
        ("Passing Yards",          pass_yds_who, str(int(pass_yds))),
        ("Passing TDs",            pass_td_who,  str(int(pass_td))),
        ("Interceptions Thrown",   pass_int_who, str(int(pass_int))),
        ("Rushing Yards",          rush_yds_who, str(int(rush_yds))),
        ("Rushing TDs",            rush_td_who,  str(int(rush_td))),
        ("Receiving Yards",        rec_yds_who,  str(int(rec_yds))),
        ("Receiving TDs",          rec_td_who,   str(int(rec_td))),
    ]

    defense_rows = [
        ("Sacks",          sack_who, f"{float(sack_val):.1f}"),
        ("Tackles",        tot_who,  str(int(tot_val))),
        ("Interceptions",  int_who,  str(int(int_val))),
    ]

    return offense_rows, defense_rows, subtitle

# -----------------------------
# public generator API (for your FastAPI route)
# -----------------------------
def generate_team_stat_leader_posters(
    team: str,
    team_url: Optional[str] = None,
    out_dir: str = "/tmp",
) -> List[str]:
    """
    Creates 2 PNGs in out_dir and returns their file paths.
    Your router should upload these to Supabase and return public URLs.
    """
    offense_rows, defense_rows, subtitle = build_team_leaders(team, team_url=team_url)

    os.makedirs(out_dir, exist_ok=True)
    t = team.strip().upper()

    out_off = os.path.join(out_dir, f"team_offense_leaders_{t}.png")
    out_def = os.path.join(out_dir, f"team_defense_leaders_{t}.png")

    draw_team_poster(out_off, "Offensive Team Leaders", subtitle, offense_rows)
    draw_team_poster(out_def, "Defensive Team Leaders", subtitle, defense_rows)

    return [out_off, out_def]
