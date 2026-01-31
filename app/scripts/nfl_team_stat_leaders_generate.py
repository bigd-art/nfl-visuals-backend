# app/scripts/nfl_team_stat_leaders_generate.py
# ==========================================================
# ESPN Team Stat Leaders (Render-safe, CI-safe)
#
# Fixes:
# - Wrong defensive player names (by using correct table views per stat)
# - "Could not detect pass/rush/rec/def" (we don't rely on guessing all tables)
# - "Could not find yards" (robust column pickers)
#
# Public API used by router:
#   leaders = extract_team_leaders(team="SEA", season=2025, seasontype=2)
#
# Poster renderer stays IDENTICAL below.
# ==========================================================

import os
import re
import time
import argparse
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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

# -----------------------------
# text / number helpers
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
            " ".join([str(x).strip() for x in tup if str(x).strip() and "Unnamed" not in str(x)]).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


# -----------------------------
# robust column picker
# -----------------------------
def pick_col(df: pd.DataFrame, kind: str) -> str:
    """
    kind: "name" | "yds" | "td" | "int" | "sack" | "tackles"
    """
    df = flatten_columns(df.copy())
    cols = list(df.columns)
    low = [_norm_col(c) for c in cols]

    def first_where(pred):
        for i, c in enumerate(low):
            if pred(c):
                return cols[i]
        return None

    if kind == "name":
        # ESPN usually has Name or PLAYER, but sometimes it's "Name  " etc.
        c = first_where(lambda c: c == "name" or c == "player" or c.startswith("name"))
        if c:
            return c
        # fallback: first column
        return cols[0]

    if kind == "yds":
        c = first_where(lambda c: c == "yds")
        if c:
            return c
        c = first_where(lambda c: "yds" in c and "yds/g" not in c and "/g" not in c)
        if c:
            return c
        c = first_where(lambda c: "yds" in c)
        if c:
            return c
        raise RuntimeError(f"Could not find YDS column. Columns={cols}")

    if kind == "td":
        c = first_where(lambda c: c == "td")
        if c:
            return c
        c = first_where(lambda c: re.search(r"\btd\b", c) and "td%" not in c)
        if c:
            return c
        raise RuntimeError(f"Could not find TD column. Columns={cols}")

    if kind == "int":
        c = first_where(lambda c: c == "int")
        if c:
            return c
        c = first_where(lambda c: re.search(r"\bint\b", c) and "int%" not in c)
        if c:
            return c
        c = first_where(lambda c: "interception" in c)
        if c:
            return c
        raise RuntimeError(f"Could not find INT column. Columns={cols}")

    if kind == "sack":
        c = first_where(lambda c: "sack" in c)
        if c:
            return c
        raise RuntimeError(f"Could not find SACK column. Columns={cols}")

    if kind == "tackles":
        # Prefer total tackles
        for target in ["tot", "total", "tkl", "tack", "combined", "comb"]:
            c = first_where(lambda c, t=target: c == t)
            if c:
                return c
        c = first_where(lambda c: any(x in c for x in ["tot", "total", "tkl", "tack", "combined", "comb"]))
        if c:
            return c
        raise RuntimeError(f"Could not find TACKLES column. Columns={cols}")

    raise RuntimeError(f"Unknown kind='{kind}'")


# -----------------------------
# fetching with retries
# -----------------------------
def fetch_html(url: str, timeout: int = 30, retries: int = 4) -> str:
    last_err = None
    session = requests.Session()

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            html = r.text or ""
            if "<table" not in html.lower():
                snippet = normalize_spaces(html[:500])
                raise RuntimeError(f"ESPN HTML had no <table>. Snippet: {snippet}")
            return html
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to fetch ESPN page after {retries} tries: {last_err}")


def read_tables(url: str) -> List[pd.DataFrame]:
    html = fetch_html(url)
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:
        raise RuntimeError(f"pd.read_html failed for URL={url}. Ensure lxml/html5lib installed. Error: {e}")
    if not tables:
        raise RuntimeError(f"pd.read_html returned 0 tables for URL={url}")
    return tables


# -----------------------------
# table selection
# ESPN team pages usually render:
#  - a 1-col Name table
#  - an adjacent stats table
# We find the best pair on the page by looking for a 1-col name table
# followed immediately by a stats table containing a needed stat column.
# -----------------------------
def is_name_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    df = flatten_columns(df.copy())
    if df.shape[1] != 1:
        return False
    sample = df.iloc[:10, 0].astype(str).map(normalize_spaces).tolist()
    alpha = sum(bool(re.search(r"[A-Za-z]", s)) for s in sample)
    return alpha >= max(2, len(sample) // 2)


def find_pair_for_stat(tables: List[pd.DataFrame], required_stat_kind: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find (name_df, stat_df) such that:
      - name_df is 1-col name table
      - stat_df is next table and contains the required stat column kind
    """
    for i in range(len(tables) - 1):
        a = tables[i]
        b = tables[i + 1]
        if not is_name_table(a):
            continue
        b2 = flatten_columns(b.copy())
        try:
            # just validate column exists
            _ = pick_col(b2, required_stat_kind)
            return a, b
        except Exception:
            continue

    # If not found, raise with helpful debugging
    debug = []
    for i, t in enumerate(tables[:10]):
        try:
            tt = flatten_columns(t.copy())
            debug.append((i, list(tt.columns)))
        except Exception:
            debug.append((i, ["<unreadable>"]))
    raise RuntimeError(f"Could not find name+stats pair for kind='{required_stat_kind}'. Table columns sample: {debug}")


def leader_from_pair(
    name_df: pd.DataFrame,
    stat_df: pd.DataFrame,
    stat_kind: str,        # "yds" | "td" | "int" | "sack" | "tackles"
    mode: str,             # "int" | "float1"
) -> Tuple[str, Number]:
    name_df = flatten_columns(name_df.copy()).reset_index(drop=True)
    stat_df = flatten_columns(stat_df.copy()).reset_index(drop=True)

    n0 = min(len(name_df), len(stat_df))
    if n0 <= 0:
        raise RuntimeError("Empty name/stats table after trim.")

    name_df = name_df.iloc[:n0].reset_index(drop=True)
    stat_df = stat_df.iloc[:n0].reset_index(drop=True)

    name_col = name_df.columns[0]
    raw_names = name_df[name_col].astype(str).map(normalize_spaces)

    # Drop Total/TEAM rows positionally
    bad = (
        raw_names.str.lower().str.contains(r"\b(total|team|opp|opponent)\b", na=False)
        | (raw_names.str.len() == 0)
    ).to_numpy()

    keep = ~bad
    names = raw_names.iloc[keep].reset_index(drop=True)
    stats = stat_df.iloc[keep].reset_index(drop=True)

    stat_col = pick_col(stats, stat_kind)

    if mode == "float1":
        vals = stats[stat_col].map(safe_float)
    else:
        vals = stats[stat_col].map(safe_int)

    vals_num = pd.to_numeric(vals, errors="coerce")

    good = vals_num.notna().to_numpy()
    names = names.iloc[good].reset_index(drop=True)
    vals_num = vals_num.iloc[good].reset_index(drop=True)

    if vals_num.empty:
        raise RuntimeError(f"No valid numeric values for stat_kind='{stat_kind}' col='{stat_col}'")

    best_pos = int(vals_num.values.argmax())
    return normalize_spaces(names.iloc[best_pos]), vals_num.iloc[best_pos]


# -----------------------------
# URL builder
# -----------------------------
def team_stats_url(team: str, season: int, seasontype: int, table: str, sort: str) -> str:
    """
    Example:
    https://www.espn.com/nfl/team/stats/_/name/sea/season/2025/seasontype/3/table/defensive%2Cdefense/sort/interceptions/dir/desc
    """
    team_key = team.strip().lower()
    return (
        f"https://www.espn.com/nfl/team/stats/_/name/{team_key}"
        f"/season/{season}/seasontype/{seasontype}"
        f"/table/{table}/sort/{sort}/dir/desc"
    )


# -----------------------------
# MAIN PUBLIC API
# -----------------------------
def extract_team_leaders(team: str, season: int, seasontype: int) -> Dict[str, Tuple[str, str, str, str]]:
    """
    Returns dict like:
      leaders["Passing Yards"] = ("1", "Player Name", "", "1234")
    """
    # OFFENSE: compute multiple stats from the correct table view
    # Passing page (sorted by passing yards so the table is definitely the passing table)
    passing_url = team_stats_url(team, season, seasontype, table="passing", sort="passingYards")
    rushing_url = team_stats_url(team, season, seasontype, table="rushing", sort="rushingYards")
    receiving_url = team_stats_url(team, season, seasontype, table="receiving", sort="receivingYards")

    # DEFENSE: IMPORTANT — use separate defense views so columns match
    sacks_url = team_stats_url(team, season, seasontype, table="defensive%2Cdefense", sort="sacks")
    tackles_url = team_stats_url(team, season, seasontype, table="defensive%2Cdefense", sort="totalTackles")
    ints_url = team_stats_url(team, season, seasontype, table="defensive%2Cdefense", sort="interceptions")

    # Passing leaders
    pass_tables = read_tables(passing_url)
    name_pass, pass_stats = find_pair_for_stat(pass_tables, "yds")
    pass_yds_who, pass_yds = leader_from_pair(name_pass, pass_stats, "yds", "int")
    pass_td_who, pass_td = leader_from_pair(name_pass, pass_stats, "td", "int")
    pass_int_who, pass_int = leader_from_pair(name_pass, pass_stats, "int", "int")

    # Rushing leaders
    rush_tables = read_tables(rushing_url)
    name_rush, rush_stats = find_pair_for_stat(rush_tables, "yds")
    rush_yds_who, rush_yds = leader_from_pair(name_rush, rush_stats, "yds", "int")
    rush_td_who, rush_td = leader_from_pair(name_rush, rush_stats, "td", "int")

    # Receiving leaders
    rec_tables = read_tables(receiving_url)
    name_rec, rec_stats = find_pair_for_stat(rec_tables, "yds")
    rec_yds_who, rec_yds = leader_from_pair(name_rec, rec_stats, "yds", "int")
    rec_td_who, rec_td = leader_from_pair(name_rec, rec_stats, "td", "int")

    # Defense — sacks
    sack_tables = read_tables(sacks_url)
    name_def_s, def_s = find_pair_for_stat(sack_tables, "sack")
    sack_who, sack_val = leader_from_pair(name_def_s, def_s, "sack", "float1")

    # Defense — tackles
    tack_tables = read_tables(tackles_url)
    name_def_t, def_t = find_pair_for_stat(tack_tables, "tackles")
    tackles_who, tackles_val = leader_from_pair(name_def_t, def_t, "tackles", "int")

    # Defense — interceptions
    int_tables = read_tables(ints_url)
    name_def_i, def_i = find_pair_for_stat(int_tables, "int")
    int_who, int_val = leader_from_pair(name_def_i, def_i, "int", "int")

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


# ==========================================================
# LOCAL TEST HARNESS (Desktop)
# ==========================================================
def _build_sections(leaders: Dict[str, Tuple[str, str, str, str]]):
    offense_order = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
    ]
    defense_order = ["Sacks", "Tackles", "Interceptions"]

    offense_sections = [(cat, leaders[cat][0], leaders[cat][1], leaders[cat][3]) for cat in offense_order]
    defense_sections = [(cat, leaders[cat][0], leaders[cat][1], leaders[cat][3]) for cat in defense_order]
    return offense_sections, defense_sections


def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="Team abbreviation like SEA")
    ap.add_argument("--season", type=int, required=True, help="Season year like 2025")
    ap.add_argument("--seasontype", type=int, choices=[2, 3], required=True, help="2=regular, 3=postseason")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    team = args.team.upper().strip()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    leaders = extract_team_leaders(team=team, season=args.season, seasontype=args.seasontype)

    label = "Regular Season" if args.seasontype == 2 else "Postseason"
    subtitle = f"{team} • {label} • Updated {time.strftime('%b %d, %Y • %I:%M %p')}"

    off_sections, def_sections = _build_sections(leaders)

    out_off = os.path.join(outdir, f"{team.lower()}_{args.season}_{args.seasontype}_offense.png")
    out_def = os.path.join(outdir, f"{team.lower()}_{args.season}_{args.seasontype}_defense.png")

    draw_leaders_grid_poster(out_off, "Offensive Statistical Leaders", subtitle, off_sections, cols=2, rows=4)
    draw_leaders_grid_poster(out_def, "Defensive Statistical Leaders", subtitle, def_sections, cols=1, rows=3)

    print("Wrote:")
    print(" -", out_off)
    print(" -", out_def)


if __name__ == "__main__":
    cli_main()
