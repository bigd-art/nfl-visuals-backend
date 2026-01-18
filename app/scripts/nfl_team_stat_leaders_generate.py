import os
import re
import argparse
from io import StringIO
from datetime import datetime
from typing import Optional, Tuple, Union, List

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HEADERS = {"User-Agent": "Mozilla/5.0"}
Number = Union[int, float]

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
            " ".join([str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]).strip()
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

def leader_from_two_tables(name_df: pd.DataFrame, stat_df: pd.DataFrame, stat_col: str, mode: str) -> Tuple[str, Number]:
    # align by row index
    n = min(len(name_df), len(stat_df))
    name_df = name_df.iloc[:n].reset_index(drop=True).copy()
    stat_df = stat_df.iloc[:n].reset_index(drop=True).copy()

    # Normalize the name column (ESPN sometimes includes "Total" row)
    name_df.iloc[:, 0] = name_df.iloc[:, 0].map(normalize_spaces)

    # Drop "Total" rows so leaders are actual players
    mask_total = name_df.iloc[:, 0].str.lower().str.contains(r"\btotal\b", na=False)
    if mask_total.any():
        name_df = name_df.loc[~mask_total].reset_index(drop=True)
        stat_df = stat_df.loc[~mask_total].reset_index(drop=True)

    if stat_col not in stat_df.columns:
        raise RuntimeError(f"Missing stat col {stat_col} in {list(stat_df.columns)}")

    # Parse values
    if mode == "float1":
        vals = stat_df[stat_col].map(safe_float)
    else:
        vals = stat_df[stat_col].map(safe_int)

    # Some columns might have blanks; drop Nones
    vals_num = pd.to_numeric(vals, errors="coerce")
    if vals_num.isna().all():
        raise RuntimeError(f"All values NaN for {stat_col}")

    best_i = vals_num.idxmax()
    best_val = vals_num.loc[best_i]

    who = normalize_spaces(name_df.iloc[best_i, 0])
    return who, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="ARI, SEA, DAL, etc.")
    args = ap.parse_args()

    team = args.team.strip().upper()
    if team not in TEAM_TO_ESPN:
        raise SystemExit(f"Unknown team: {team}")

    url = f"https://www.espn.com/nfl/team/stats/_/name/{TEAM_TO_ESPN[team]}"
    print("Fetching:", url)

    tables = fetch_tables(url)

    # Based on your debug:
    # 0 name (pass), 1 passing stats
    # 2 name (rush), 3 rushing stats
    # 4 name (rec), 5 receiving stats
    # 6 name (def), 7 defense stats (MultiIndex)
    name_pass = flatten_columns(tables[0]).rename(columns={tables[0].columns[0]:"Name"})[["Name"]]
    pass_stats = flatten_columns(tables[1])

    name_rush = flatten_columns(tables[2]).rename(columns={tables[2].columns[0]:"Name"})[["Name"]]
    rush_stats = flatten_columns(tables[3])

    name_rec = flatten_columns(tables[4]).rename(columns={tables[4].columns[0]:"Name"})[["Name"]]
    rec_stats = flatten_columns(tables[5])

    name_def = flatten_columns(tables[6]).rename(columns={tables[6].columns[0]:"Name"})[["Name"]]
    def_stats = flatten_columns(tables[7])

    # OFFENSE leaders
    pass_yds_who, pass_yds = leader_from_two_tables(name_pass, pass_stats, "YDS", "int")
    pass_td_who, pass_td   = leader_from_two_tables(name_pass, pass_stats, "TD", "int")
    pass_int_who, pass_int = leader_from_two_tables(name_pass, pass_stats, "INT", "int")

    rush_yds_who, rush_yds = leader_from_two_tables(name_rush, rush_stats, "YDS", "int")
    rush_td_who, rush_td   = leader_from_two_tables(name_rush, rush_stats, "TD", "int")

    rec_yds_who, rec_yds   = leader_from_two_tables(name_rec, rec_stats, "YDS", "int")
    rec_td_who, rec_td     = leader_from_two_tables(name_rec, rec_stats, "TD", "int")

    # DEFENSE leaders (from TABLE 7 after flatten -> "Sacks SACK", "Tackles TOT", "Interceptions INT")
    # Some teams might have slightly different capitalization, so we match robustly:
    cols = {c.lower(): c for c in def_stats.columns}

    def pick_col(want: List[str]) -> str:
        for w in want:
            w = w.lower()
            for k, orig in cols.items():
                if k == w or w in k:
                    return orig
        raise RuntimeError(f"Could not find defense col for {want}. Have: {list(def_stats.columns)}")

    col_sack = pick_col(["sacks sack", "sack"])
    col_tot  = pick_col(["tackles tot", "tot"])
    col_int  = pick_col(["interceptions int", "int"])

    sack_who, sack_val = leader_from_two_tables(name_def, def_stats, col_sack, "float1")
    tot_who, tot_val   = leader_from_two_tables(name_def, def_stats, col_tot, "int")
    int_who, int_val   = leader_from_two_tables(name_def, def_stats, col_int, "int")

    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"{team} • Updated {updated}"

    outdir = os.path.join(os.path.expanduser("~"), "Desktop")
    out_off = os.path.join(outdir, f"team_offense_leaders_{team}.png")
    out_def = os.path.join(outdir, f"team_defense_leaders_{team}.png")

    offense_rows = [
        ("Passing Yards", pass_yds_who, str(int(pass_yds))),
        ("Passing TDs", pass_td_who, str(int(pass_td))),
        ("Interceptions Thrown", pass_int_who, str(int(pass_int))),
        ("Rushing Yards", rush_yds_who, str(int(rush_yds))),
        ("Rushing TDs", rush_td_who, str(int(rush_td))),
        ("Receiving Yards", rec_yds_who, str(int(rec_yds))),
        ("Receiving TDs", rec_td_who, str(int(rec_td))),
    ]

    defense_rows = [
        ("Sacks", sack_who, f"{float(sack_val):.1f}"),
        ("Tackles", tot_who, str(int(tot_val))),
        ("Interceptions", int_who, str(int(int_val))),
    ]

    draw_team_poster(out_off, "Offensive Team Leaders", subtitle, offense_rows)
    draw_team_poster(out_def, "Defensive Team Leaders", subtitle, defense_rows)

    print("\nDONE ✅")
    print(out_off)
    print(out_def)
    print("\nOpen them by double-clicking the PNGs on your Desktop.")

if __name__ == "__main__":
    main()

