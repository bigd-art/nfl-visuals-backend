import os
import re
from io import StringIO
from datetime import datetime
from typing import Optional, Tuple, Union, List, Dict

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from app.services.storage_supabase import upload_file_return_url

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
}

# ---------------- helpers ---------------- #

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def safe_int(x) -> Optional[int]:
    try:
        return int(str(x).replace(",", ""))
    except:
        return None

def fetch_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))

def flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(c) for c in col if "Unnamed" not in str(c)]) for col in df.columns]
    return df

def load_font(size, bold=False):
    try:
        return ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            size
        )
    except:
        return ImageFont.load_default()

def draw_poster(path, title, subtitle, rows):
    img = Image.new("RGB", (1400, 2400), (12, 12, 16))
    d = ImageDraw.Draw(img)

    d.text((60, 60), title, font=load_font(64, True), fill="white")
    d.text((60, 140), subtitle, font=load_font(28), fill="gray")

    y = 260
    for label, name, val in rows:
        d.text((80, y), label, font=load_font(36, True), fill="white")
        d.text((80, y + 50), name, font=load_font(28), fill="gray")
        d.text((1200, y + 25), str(val), font=load_font(36, True), fill="white", anchor="ra")
        y += 260

    img.save(path)

# ---------------- MAIN API FUNCTION ---------------- #

def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, str, int]]:
    tables = fetch_tables(team_url)

    name_pass, pass_df = flatten(tables[0]), flatten(tables[1])
    name_rush, rush_df = flatten(tables[2]), flatten(tables[3])
    name_rec, rec_df   = flatten(tables[4]), flatten(tables[5])
    name_def, def_df   = flatten(tables[6]), flatten(tables[7])

    def top(name_df, stat_df, col):
        idx = stat_df[col].astype(str).str.replace(",", "").astype(int).idxmax()
        return normalize_spaces(name_df.iloc[idx, 0]), int(stat_df.loc[idx, col])

    return {
        "Passing Yards": top(name_pass, pass_df, "YDS"),
        "Passing TDs": top(name_pass, pass_df, "TD"),
        "Interceptions Thrown": top(name_pass, pass_df, "INT"),
        "Rushing Yards": top(name_rush, rush_df, "YDS"),
        "Rushing TDs": top(name_rush, rush_df, "TD"),
        "Receiving Yards": top(name_rec, rec_df, "YDS"),
        "Receiving TDs": top(name_rec, rec_df, "TD"),
        "Sacks": top(name_def, def_df, "SACK"),
        "Tackles": top(name_def, def_df, "TOT"),
        "Interceptions": top(name_def, def_df, "INT"),
    }

def generate_team_stat_leaders(team: str) -> List[str]:
    team = team.upper()
    url = f"https://www.espn.com/nfl/team/stats/_/name/{TEAM_TO_ESPN[team]}"
    leaders = extract_team_leaders(url)

    now = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"{team} • Updated {now}"

    outdir = "/tmp"
    off_path = f"{outdir}/{team}_off.png"
    def_path = f"{outdir}/{team}_def.png"

    draw_poster(
        off_path,
        "Offensive Team Leaders",
        subtitle,
        [
            ("Passing Yards", *leaders["Passing Yards"]),
            ("Passing TDs", *leaders["Passing TDs"]),
            ("INTs Thrown", *leaders["Interceptions Thrown"]),
            ("Rushing Yards", *leaders["Rushing Yards"]),
            ("Rushing TDs", *leaders["Rushing TDs"]),
            ("Receiving Yards", *leaders["Receiving Yards"]),
            ("Receiving TDs", *leaders["Receiving TDs"]),
        ],
    )

    draw_poster(
        def_path,
        "Defensive Team Leaders",
        subtitle,
        [
            ("Sacks", *leaders["Sacks"]),
            ("Tackles", *leaders["Tackles"]),
            ("Interceptions", *leaders["Interceptions"]),
        ],
    )

    off_url = upload_file_return_url(off_path, f"team/{team}/offense.png")
    def_url = upload_file_return_url(def_path, f"team/{team}/defense.png")

    return [off_url, def_url]
