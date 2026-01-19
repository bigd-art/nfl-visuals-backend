import os
import re
from io import StringIO
from datetime import datetime
from typing import Optional, Tuple, Union, List, Dict

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HEADERS = {"User-Agent": "Mozilla/5.0"}
Number = Union[int, float]

# --- ESPN team slugs (same as your map) ---
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
    "IND": "indianapolis-colts/ind/indianapolis-colts".split("/")[-2] if False else "ind/indianapolis-colts",
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

def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except:
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

def _drop_total_rows(name_df: pd.DataFrame, stat_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    name_df = name_df.copy()
    stat_df = stat_df.copy()
    name_df.iloc[:, 0] = name_df.iloc[:, 0].map(normalize_spaces)

    mask_total = name_df.iloc[:, 0].str.lower().str.contains(r"\btotal\b", na=False)
    if mask_total.any():
        name_df = name_df.loc[~mask_total].reset_index(drop=True)
        stat_df = stat_df.loc[~mask_total].reset_index(drop=True)
    return name_df, stat_df

def leader_from_two_tables(name_df: pd.DataFrame, stat_df: pd.DataFrame, stat_col: str, mode: str) -> Tuple[str, Number]:
    n = min(len(name_df), len(stat_df))
    name_df = name_df.iloc[:n].reset_index(drop=True).copy()
    stat_df = stat_df.iloc[:n].reset_index(drop=True).copy()

    name_df, stat_df = _drop_total_rows(name_df, stat_df)

    if stat_col not in stat_df.columns:
        raise RuntimeError(f"Missing stat col '{stat_col}' in {list(stat_df.columns)}")

    if mode == "float1":
        vals = stat_df[stat_col].map(safe_float)
    else:
        vals = stat_df[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")
    if vals_num.isna().all():
        raise RuntimeError(f"All values NaN for '{stat_col}'")

    best_i = vals_num.idxmax()
    best_val = vals_num.loc[best_i]
    who = normalize_spaces(name_df.iloc[best_i, 0])
    return who, best_val

def _pick_def_col(def_df: pd.DataFrame, wants: List[str]) -> str:
    """
    Robust defense column picker.
    ESPN can label sacks as: 'SACK', 'SACKS', 'SK', 'Sacks SACK', etc.
    """
    cols = [str(c) for c in def_df.columns]
    low_map = {c.lower(): c for c in cols}

    # exact / contains search over synonyms
    for w in wants:
        w = w.lower()
        # exact
        if w in low_map:
            return low_map[w]
        # contains
        for lk, orig in low_map.items():
            if w in lk:
                return orig

    # last resort: if wants is ['sack'] try regex boundaries
    for w in wants:
        pat = re.compile(rf"\b{re.escape(w.lower())}\b")
        for lk, orig in low_map.items():
            if pat.search(lk):
                return orig

    raise RuntimeError(f"Could not find defense col for {wants}. Have: {cols}")

def _find_name_and_stat_tables(tables: List[pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    ESPN team stats page typically returns pairs:
      name table, stat table, name table, stat table, ...
    But ordering can vary. We detect them by presence of key columns.
    """
    flat = [flatten_columns(t.copy()) for t in tables]

    def has_cols(df, any_of: List[str]) -> bool:
        cols = [c.upper() for c in df.columns]
        return any(x.upper() in cols for x in any_of)

    out = {}

    # Passing pair (needs YDS/TD/INT)
    for i in range(len(flat) - 1):
        if has_cols(flat[i+1], ["YDS", "TD", "INT"]) and flat[i].shape[1] >= 1:
            out["pass"] = (flat[i].iloc[:, [0]].rename(columns={flat[i].columns[0]: "Name"}), flat[i+1])
            break

    # Rushing pair (needs YDS/TD, often no INT)
    for i in range(len(flat) - 1):
        if has_cols(flat[i+1], ["YDS", "TD"]) and not has_cols(flat[i+1], ["REC", "TGTS", "TGT"]) and flat[i].shape[1] >= 1:
            # avoid re-using passing
            if "pass" in out and out["pass"][1].equals(flat[i+1]):
                continue
            out["rush"] = (flat[i].iloc[:, [0]].rename(columns={flat[i].columns[0]: "Name"}), flat[i+1])
            break

    # Receiving pair (needs YDS/TD and usually REC)
    for i in range(len(flat) - 1):
        if has_cols(flat[i+1], ["YDS", "TD"]) and has_cols(flat[i+1], ["REC", "RECS", "RECEPTIONS"]) and flat[i].shape[1] >= 1:
            out["rec"] = (flat[i].iloc[:, [0]].rename(columns={flat[i].columns[0]: "Name"}), flat[i+1])
            break

    # Defense pair (needs something like SACK/TOT/INT)
    for i in range(len(flat) - 1):
        if (
            has_cols(flat[i+1], ["INT", "INTERCEPTIONS"])
            and (has_cols(flat[i+1], ["SACK", "SACKS", "SK"]) or has_cols(flat[i+1], ["TOT", "TACK", "TACKLES"]))
            and flat[i].shape[1] >= 1
        ):
            out["def"] = (flat[i].iloc[:, [0]].rename(columns={flat[i].columns[0]: "Name"}), flat[i+1])
            break

    missing = [k for k in ["pass", "rush", "rec", "def"] if k not in out]
    if missing:
        raise RuntimeError(f"Could not detect tables for: {missing}. Table count={len(flat)}")

    return out

def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Returns mapping:
      category -> (player_name, team_abbr, display_value)
    We don't actually need team_abbr for display, but router expects 3 items.
    """
    tables = fetch_tables(team_url)
    pairs = _find_name_and_stat_tables(tables)

    name_pass, pass_stats = pairs["pass"]
    name_rush, rush_stats = pairs["rush"]
    name_rec,  rec_stats  = pairs["rec"]
    name_def,  def_stats  = pairs["def"]

    # OFFENSE
    pass_yds_who, pass_yds = leader_from_two_tables(name_pass, pass_stats, "YDS", "int")
    pass_td_who, pass_td   = leader_from_two_tables(name_pass, pass_stats, "TD", "int")
    pass_int_who, pass_int = leader_from_two_tables(name_pass, pass_stats, "INT", "int")

    rush_yds_who, rush_yds = leader_from_two_tables(name_rush, rush_stats, "YDS", "int")
    rush_td_who, rush_td   = leader_from_two_tables(name_rush, rush_stats, "TD", "int")

    rec_yds_who, rec_yds   = leader_from_two_tables(name_rec, rec_stats, "YDS", "int")
    rec_td_who, rec_td     = leader_from_two_tables(name_rec, rec_stats, "TD", "int")

    # DEFENSE (robust pick)
    col_sack = _pick_def_col(def_stats, ["sack", "sacks", "sk", "sacks sack", "sack sack"])
    col_tot  = _pick_def_col(def_stats, ["tot", "tackles", "tackles tot", "total", "total tackles"])
    col_int  = _pick_def_col(def_stats, ["int", "interceptions", "interceptions int"])

    sack_who, sack_val = leader_from_two_tables(name_def, def_stats, col_sack, "float1")
    tot_who, tot_val   = leader_from_two_tables(name_def, def_stats, col_tot, "int")
    int_who, int_val   = leader_from_two_tables(name_def, def_stats, col_int, "int")

    # Return 3-tuple values because your router builds (cat, leaders[cat][0], leaders[cat][1], leaders[cat][2])
    # We'll use team placeholder "-" (not needed for poster)
    return {
        "Passing Yards": (pass_yds_who, "-", str(int(pass_yds))),
        "Passing TDs": (pass_td_who, "-", str(int(pass_td))),
        "Interceptions Thrown": (pass_int_who, "-", str(int(pass_int))),
        "Rushing Yards": (rush_yds_who, "-", str(int(rush_yds))),
        "Rushing TDs": (rush_td_who, "-", str(int(rush_td))),
        "Receiving Yards": (rec_yds_who, "-", str(int(rec_yds))),
        "Receiving TDs": (rec_td_who, "-", str(int(rec_td))),
        "Sacks": (sack_who, "-", f"{float(sack_val):.1f}"),
        "Tackles": (tot_who, "-", str(int(tot_val))),
        "Interceptions": (int_who, "-", str(int(int_val))),
    }

# Keep your router compatibility:
# Router imports draw_leaders_grid_poster — so we provide it.
def draw_leaders_grid_poster(out_path: str, title: str, subtitle: str, sections, cols=2, rows=4):
    """
    Minimal implementation that uses your existing poster style.
    sections: list of (category, player, team, value)
    """
    # Convert to rows for the same style
    rows_out = []
    for cat, player, _team, value in sections:
        rows_out.append((cat, player, str(value)))

    draw_team_poster(out_path, title, subtitle, rows_out)
