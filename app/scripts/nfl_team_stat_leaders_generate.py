# app/scripts/nfl_team_stat_leaders_generate.py
# ✅ Keeps OLD poster style (draw_leaders_grid_poster)
# ✅ Fixes WRONG ESPN DATA by robust table pairing + strict column selection
#
# Your router expects:
#   leaders = extract_team_leaders(team_url)
#   leaders["Passing Yards"] => (rank, name, team, value)
#
# This file provides that API.

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
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
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
            " ".join(
                [str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]
            ).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_all_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def is_name_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    df = flatten_columns(df.copy())

    # ESPN "name" tables are usually 1 column containing player names
    if df.shape[1] != 1:
        return False

    col = df.columns[0]
    sample = df[col].astype(str).head(8).tolist()
    alpha_count = sum(bool(re.search(r"[A-Za-z]", s)) for s in sample)
    return alpha_count >= max(1, len(sample) // 2)


def colset(df: pd.DataFrame) -> set:
    df = flatten_columns(df.copy())
    return {str(c).strip().lower() for c in df.columns}


def classify_stats_table(df: pd.DataFrame) -> Optional[str]:
    """
    Returns: "pass" | "rush" | "rec" | "def" | None
    """
    if df is None or df.empty:
        return None
    cols = colset(df)

    # Passing signatures
    pass_hits = 0
    for k in ["cmp", "att", "pct", "yds", "td", "int", "rtg", "qbr"]:
        pass_hits += 1 if any(k == c or k in c for c in cols) else 0

    # Rushing signatures (no cmp/rtg/qbr)
    rush_hits = 0
    for k in ["att", "yds", "avg", "td", "lng"]:
        rush_hits += 1 if any(k == c or k in c for c in cols) else 0
    has_cmp = any("cmp" == c or "cmp" in c for c in cols)
    has_rtg = any("rtg" == c or "rtg" in c for c in cols) or any(
        "qbr" == c or "qbr" in c for c in cols
    )

    # Receiving signatures
    rec_hits = 0
    for k in ["rec", "tgts", "yds", "avg", "td", "lng"]:
        rec_hits += 1 if any(k == c or k in c for c in cols) else 0

    # Defense signatures
    def_hits = 0
    for k in ["tot", "solo", "ast", "sack", "int", "ff"]:
        def_hits += 1 if any(k == c or k in c for c in cols) else 0

    if def_hits >= 3 and ("sack" in " ".join(cols) or "tot" in " ".join(cols)):
        return "def"

    if pass_hits >= 5 and (has_cmp or has_rtg):
        return "pass"

    if rec_hits >= 4 and any("rec" == c or "rec" in c for c in cols):
        return "rec"

    if rush_hits >= 4 and not has_cmp and not has_rtg:
        return "rush"

    return None


def clean_name_series(name_s: pd.Series) -> pd.Series:
    s = name_s.astype(str).map(normalize_spaces)
    bad = s.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False)
    s = s[~bad].copy()
    s = s[s.str.len() > 0]
    return s.reset_index(drop=True)


# -----------------------------
# ✅ STRICT column selection (THIS FIXES WRONG DATA)
# -----------------------------
def _normalize_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def pick_col_strict(df: pd.DataFrame, want: str) -> str:
    """
    Strict match for columns like:
      want="yds" => exact "yds" or endswith " yds"
      want="td"  => exact "td" only (NOT "ltd", "td%", etc.)
      want="int" => exact "int" only (NOT "int%", etc.)
      want="sack" => contains sack
      want="tot" => exact tot or contains "total"
    """
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_normalize_col(c) for c in cols]

    want = want.lower().strip()

    # exact match first
    for i, c in enumerate(low):
        if c == want:
            return cols[i]

    # special cases
    if want == "yds":
        for i, c in enumerate(low):
            if c.endswith(" yds") or c == "yds" or "yds" == c:
                return cols[i]
        # fallback: contains yds
        for i, c in enumerate(low):
            if "yds" in c:
                return cols[i]

    if want == "td":
        # TD must be a standalone token
        for i, c in enumerate(low):
            if c == "td":
                return cols[i]
        for i, c in enumerate(low):
            if re.fullmatch(r".*\btd\b.*", c) and "td%" not in c and "ltd" not in c:
                # still risky; but ok as fallback
                return cols[i]

    if want == "int":
        for i, c in enumerate(low):
            if c == "int":
                return cols[i]
        for i, c in enumerate(low):
            if re.fullmatch(r".*\bint\b.*", c) and "int%" not in c:
                return cols[i]

    if want == "sack":
        for i, c in enumerate(low):
            if "sack" in c:
                return cols[i]

    if want == "tot":
        for i, c in enumerate(low):
            if c == "tot" or "total" in c:
                return cols[i]

    raise RuntimeError(f"Could not find a safe column for '{want}'. Columns={cols}")


def leader_from_name_and_stats(
    name_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    stat_col: str,
    mode: str,
) -> Tuple[str, Number]:
    name_df = flatten_columns(name_df.copy())
    stat_df = flatten_columns(stat_df.copy())

    name_col = name_df.columns[0]
    names = clean_name_series(name_df[name_col])

    raw_names = name_df[name_col].astype(str).map(normalize_spaces)
    bad_mask = raw_names.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False) | (
        raw_names.str.len() == 0
    )

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


def find_named_table_pairs(tables: List[pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
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

    for i, cat in cat_at.items():
        if cat in pairs:
            continue
        j = i - 1
        while j >= 0:
            if j in name_idx:
                nlen = len(tables[j])
                slen = len(tables[i])
                if nlen >= 2 and slen >= 2:
                    pairs[cat] = (tables[j], tables[i])
                break
            j -= 1

    return pairs


# -----------------------------
# ✅ MAIN API your router uses
# -----------------------------
def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Returns dict:
      {
        "Passing Yards": ("1", "Player Name", "", "1234"),
        ...
      }

    Note: "team" field is returned as "" because ESPN team page already implies team.
    Your poster renderer can ignore it or show blank.
    """
    tables = fetch_all_tables(team_url)
    pairs = find_named_table_pairs(tables)

    missing = [k for k in ["pass", "rush", "rec", "def"] if k not in pairs]
    if missing:
        raise RuntimeError(f"Could not detect required ESPN table(s): {missing}")

    name_pass, pass_stats = pairs["pass"]
    name_rush, rush_stats = pairs["rush"]
    name_rec, rec_stats = pairs["rec"]
    name_def, def_stats = pairs["def"]

    pass_stats = flatten_columns(pass_stats)
    rush_stats = flatten_columns(rush_stats)
    rec_stats = flatten_columns(rec_stats)
    def_stats = flatten_columns(def_stats)

    # Strict columns
    col_pass_yds = pick_col_strict(pass_stats, "yds")
    col_pass_td = pick_col_strict(pass_stats, "td")
    col_pass_int = pick_col_strict(pass_stats, "int")

    col_rush_yds = pick_col_strict(rush_stats, "yds")
    col_rush_td = pick_col_strict(rush_stats, "td")

    col_rec_yds = pick_col_strict(rec_stats, "yds")
    col_rec_td = pick_col_strict(rec_stats, "td")

    col_sack = pick_col_strict(def_stats, "sack")
    col_tot = pick_col_strict(def_stats, "tot")
    col_int = pick_col_strict(def_stats, "int")

    # Leaders
    pass_yds_who, pass_yds = leader_from_name_and_stats(name_pass, pass_stats, col_pass_yds, "int")
    pass_td_who, pass_td = leader_from_name_and_stats(name_pass, pass_stats, col_pass_td, "int")
    pass_int_who, pass_int = leader_from_name_and_stats(name_pass, pass_stats, col_pass_int, "int")

    rush_yds_who, rush_yds = leader_from_name_and_stats(name_rush, rush_stats, col_rush_yds, "int")
    rush_td_who, rush_td = leader_from_name_and_stats(name_rush, rush_stats, col_rush_td, "int")

    rec_yds_who, rec_yds = leader_from_name_and_stats(name_rec, rec_stats, col_rec_yds, "int")
    rec_td_who, rec_td = leader_from_name_and_stats(name_rec, rec_stats, col_rec_td, "int")

    sack_who, sack_val = leader_from_name_and_stats(name_def, def_stats, col_sack, "float1")
    tot_who, tot_val = leader_from_name_and_stats(name_def, def_stats, col_tot, "int")
    int_who, int_val = leader_from_name_and_stats(name_def, def_stats, col_int, "int")

    # Build output in your router's expected format: (rank, player, team, value)
    leaders: Dict[str, Tuple[str, str, str, str]] = {
        "Passing Yards": ("1", pass_yds_who, "", str(int(pass_yds))),
        "Passing TDs": ("1", pass_td_who, "", str(int(pass_td))),
        "Interceptions Thrown": ("1", pass_int_who, "", str(int(pass_int))),
        "Rushing Yards": ("1", rush_yds_who, "", str(int(rush_yds))),
        "Rushing TDs": ("1", rush_td_who, "", str(int(rush_td))),
        "Receiving Yards": ("1", rec_yds_who, "", str(int(rec_yds))),
        "Receiving TDs": ("1", rec_td_who, "", str(int(rec_td))),
        "Sacks": ("1", sack_who, "", f"{float(sack_val):.1f}"),
        "Tackles": ("1", tot_who, "", str(int(tot_val))),
        "Interceptions": ("1", int_who, "", str(int(int_val))),
    }
    return leaders


# -----------------------------
# KEEP your existing renderer
# -----------------------------
# If you already have draw_leaders_grid_poster in this file, KEEP IT.
# If not, you can paste your existing implementation here unchanged.
