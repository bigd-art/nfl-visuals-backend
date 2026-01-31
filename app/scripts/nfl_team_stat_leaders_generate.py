# app/scripts/nfl_team_stat_leaders_generate.py
# ==========================================================
# ESPN Team Leaders scraper (Render/GitHub Actions safe)
#
# Key fix:
#   DO NOT pair separate [name] + [stats] tables.
#   Instead scrape ESPN "table pages" where PLAYER + stats are in the SAME table:
#     .../table/passing
#     .../table/rushing
#     .../table/receiving
#     .../table/defensive
#
# Supports:
#   season + seasontype (2=regular, 3=postseason)
#   scope: "regular" | "playoffs" | "both"
#
# Public API used by router:
#   extract_team_leaders(team_url, season, seasontype) -> leaders dict
# ==========================================================

import os
import re
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests

Number = Union[int, float]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# ----------------------------------------------------------
# helpers
# ----------------------------------------------------------
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


def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def _strip_position(player_cell: str) -> str:
    """
    ESPN player cells often look like: 'Ernest Jones IV LB' or 'Devon Witherspoon CB'
    We keep it as-is (your posters show name + position which is fine),
    but normalize whitespace.
    """
    return normalize_spaces(player_cell)


def _looks_like_player_col(colname: str) -> bool:
    c = _norm_col(colname)
    return c in {"player", "name"} or "player" in c


def _pick_col(df: pd.DataFrame, wants: List[str]) -> str:
    """
    Pick the best matching stat column from df.
    wants are tokens like ["yds"], ["td"], ["int"], ["sack"], ["tot","tkl","tack"]
    """
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]

    # exact-first
    for w in wants:
        w = _norm_col(w)
        for i, c in enumerate(low):
            if c == w:
                return cols[i]

    # token-based
    for w in wants:
        w = _norm_col(w)
        for i, c in enumerate(low):
            if re.search(rf"\b{re.escape(w)}\b", c):
                return cols[i]

    # contains-based
    for w in wants:
        w = _norm_col(w)
        for i, c in enumerate(low):
            if w in c:
                return cols[i]

    raise RuntimeError(f"Could not find stat col for {wants}. Have columns: {cols}")


# ----------------------------------------------------------
# ESPN URL building
# ----------------------------------------------------------
def _extract_team_name_slug(team_url: str) -> str:
    """
    Accepts URLs like:
      https://www.espn.com/nfl/team/stats/_/name/sea/seattle-seahawks
      https://www.espn.com/nfl/team/stats/_/name/sea/table//sort/interceptions/dir/desc
      https://www.espn.com/nfl/team/stats/_/name/sea/season/2025/seasontype/3
    Returns:
      'sea' (the ESPN team key)
    """
    m = re.search(r"/name/([a-z0-9]+)/", team_url, flags=re.IGNORECASE)
    if not m:
        # sometimes ends right after /name/sea
        m = re.search(r"/name/([a-z0-9]+)(?:$|[/?#])", team_url, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not parse team from team_url: {team_url}")
    return m.group(1).lower()


def build_table_url(team_url: str, season: int, seasontype: int, table: str) -> str:
    """
    Canonical, stable ESPN table page URL:
      https://www.espn.com/nfl/team/stats/_/name/sea/season/2025/seasontype/2/table/defensive
    """
    team_key = _extract_team_name_slug(team_url)
    table = table.strip().lower()
    if table not in {"passing", "rushing", "receiving", "defensive"}:
        raise ValueError(f"Invalid table={table}")
    return (
        f"https://www.espn.com/nfl/team/stats/_/name/{team_key}"
        f"/season/{int(season)}/seasontype/{int(seasontype)}/table/{table}"
    )


# ----------------------------------------------------------
# fetch + read_html (CI-safe retries)
# ----------------------------------------------------------
def fetch_html(url: str, timeout: int = 30, retries: int = 4) -> str:
    last_err = None
    session = requests.Session()

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            html = r.text or ""

            # ESPN sometimes returns soft-block pages; table pages should include <table
            if "<table" not in html.lower():
                snippet = normalize_spaces(html[:400])
                raise RuntimeError(f"ESPN HTML had no <table>. Snippet: {snippet}")

            return html
        except Exception as e:
            last_err = e
            time.sleep(1.25 * attempt)

    raise RuntimeError(f"Failed to fetch ESPN after {retries} tries. Last error: {last_err}")


def read_best_table(url: str) -> pd.DataFrame:
    html = fetch_html(url)
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        raise RuntimeError(f"pd.read_html failed for url={url}. Error: {e}")

    if not tables:
        raise RuntimeError(f"pd.read_html returned 0 tables for url={url}")

    # Pick the biggest table that contains a player/name column
    best = None
    best_score = -1
    for t in tables:
        if t is None or t.empty:
            continue
        tt = flatten_columns(t.copy())
        cols = list(tt.columns)
        has_player = any(_looks_like_player_col(c) for c in cols)
        score = (len(tt) * len(cols)) + (100000 if has_player else 0)
        if score > best_score:
            best = tt
            best_score = score

    if best is None or best.empty:
        raise RuntimeError(f"No usable table found for url={url}")

    return best


def leader_from_single_table(
    df: pd.DataFrame,
    player_col: str,
    stat_col: str,
    mode: str,  # "int" or "float1"
) -> Tuple[str, Number]:
    """
    Compute leader from a single table where player + stats exist together.
    """
    d = df.copy()

    # clean player col
    players = d[player_col].astype(str).map(_strip_position)

    # drop team/total/etc rows just in case
    bad = players.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False) | (players.str.len() == 0)
    d = d.loc[~bad].reset_index(drop=True)
    players = players.loc[~bad].reset_index(drop=True)

    if mode == "float1":
        vals = d[stat_col].map(safe_float)
    else:
        vals = d[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")
    good = vals_num.notna().to_numpy()
    players = players.iloc[good].reset_index(drop=True)
    vals_num = vals_num.iloc[good].reset_index(drop=True)

    if vals_num.empty:
        raise RuntimeError(f"All values empty/NaN for stat_col={stat_col}")

    best_pos = int(vals_num.values.argmax())
    return players.iloc[best_pos], vals_num.iloc[best_pos]


# ----------------------------------------------------------
# PUBLIC API
# ----------------------------------------------------------
def extract_team_leaders(team_url: str, season: int, seasontype: int) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Returns dict:
      leaders["Passing Yards"] = ("1", "Player Name", "", "1234")
    """
    # 1) Pull each table page (stable)
    url_pass = build_table_url(team_url, season, seasontype, "passing")
    url_rush = build_table_url(team_url, season, seasontype, "rushing")
    url_rec  = build_table_url(team_url, season, seasontype, "receiving")
    url_def  = build_table_url(team_url, season, seasontype, "defensive")

    pass_df = read_best_table(url_pass)
    rush_df = read_best_table(url_rush)
    rec_df  = read_best_table(url_rec)
    def_df  = read_best_table(url_def)

    # 2) Identify player columns
    def pick_player_col(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        for c in cols:
            if _looks_like_player_col(c):
                return c
        # fallback: first column
        return cols[0]

    p_player = pick_player_col(pass_df)
    r_player = pick_player_col(rush_df)
    rc_player = pick_player_col(rec_df)
    d_player = pick_player_col(def_df)

    # 3) Pick stat columns robustly
    col_pass_yds = _pick_col(pass_df, ["yds"])
    col_pass_td  = _pick_col(pass_df, ["td"])
    col_pass_int = _pick_col(pass_df, ["int"])

    col_rush_yds = _pick_col(rush_df, ["yds"])
    col_rush_td  = _pick_col(rush_df, ["td"])

    col_rec_yds  = _pick_col(rec_df, ["yds"])
    col_rec_td   = _pick_col(rec_df, ["td"])

    col_sack     = _pick_col(def_df, ["sack", "sacks"])
    col_tackles  = _pick_col(def_df, ["tot", "tkl", "tack", "combined"])
    col_def_int  = _pick_col(def_df, ["int", "ints", "interceptions"])

    # 4) Compute leaders (single-table = correct names)
    pass_yds_who, pass_yds = leader_from_single_table(pass_df, p_player, col_pass_yds, "int")
    pass_td_who, pass_td   = leader_from_single_table(pass_df, p_player, col_pass_td, "int")
    pass_int_who, pass_int = leader_from_single_table(pass_df, p_player, col_pass_int, "int")

    rush_yds_who, rush_yds = leader_from_single_table(rush_df, r_player, col_rush_yds, "int")
    rush_td_who, rush_td   = leader_from_single_table(rush_df, r_player, col_rush_td, "int")

    rec_yds_who, rec_yds   = leader_from_single_table(rec_df, rc_player, col_rec_yds, "int")
    rec_td_who, rec_td     = leader_from_single_table(rec_df, rc_player, col_rec_td, "int")

    sack_who, sack_val     = leader_from_single_table(def_df, d_player, col_sack, "float1")
    tack_who, tack_val     = leader_from_single_table(def_df, d_player, col_tackles, "int")
    int_who, int_val       = leader_from_single_table(def_df, d_player, col_def_int, "int")

    leaders: Dict[str, Tuple[str, str, str, str]] = {
        "Passing Yards": ("1", str(pass_yds_who), "", str(int(pass_yds))),
        "Passing TDs": ("1", str(pass_td_who), "", str(int(pass_td))),
        "Interceptions Thrown": ("1", str(pass_int_who), "", str(int(pass_int))),
        "Rushing Yards": ("1", str(rush_yds_who), "", str(int(rush_yds))),
        "Rushing TDs": ("1", str(rush_td_who), "", str(int(rush_td))),
        "Receiving Yards": ("1", str(rec_yds_who), "", str(int(rec_yds))),
        "Receiving TDs": ("1", str(rec_td_who), "", str(int(rec_td))),
        "Sacks": ("1", str(sack_who), "", f"{float(sack_val):.1f}"),
        "Tackles": ("1", str(tack_who), "", str(int(tack_val))),
        "Interceptions": ("1", str(int_who), "", str(int(int_val))),
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
