# app/scripts/nfl_team_stat_leaders_generate.py
import os
import re
import argparse
from io import StringIO
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict

import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )
}

Number = Union[int, float]

# ----------------------------
# Normalization helpers (same vibe as your existing script)
# ----------------------------
def normalize_spaces(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)  # zero-width chars
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip()]).strip()
            for tup in df.columns
        ]
    df.columns = [str(c).strip() for c in df.columns]
    return df


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
    except ValueError:
        return None


def safe_int(x) -> Optional[int]:
    f = safe_float(x)
    if f is None:
        return None
    return int(round(f))


def col_lookup(cols: List[str]) -> dict:
    return {str(c).strip().lower(): c for c in cols}


# ----------------------------
# Fetch + parse ESPN team stats page
# ----------------------------
def fetch_tables(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    return [flatten_columns(t.copy()) for t in tables]


def find_table_with_cols(tables: List[pd.DataFrame], must_have: List[str]) -> Optional[pd.DataFrame]:
    """
    Find a table that contains all required column keywords (case-insensitive, substring ok).
    Example: must_have=["player","yds","td"]
    """
    must = [m.lower().strip() for m in must_have]
    for t in tables:
        cols_l = [str(c).lower().strip() for c in t.columns]
        ok = True
        for m in must:
            if not any((m == c) or (m in c) for c in cols_l):
                ok = False
                break
        if ok:
            return t
    return None


def choose_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Pick best matching column from candidates (exact then contains).
    """
    cols = list(df.columns)
    cols_l = [str(c).lower().strip() for c in cols]
    cand_l = [c.lower().strip() for c in candidates]

    for ck in cand_l:
        for i, colk in enumerate(cols_l):
            if colk == ck:
                return cols[i]
    for ck in cand_l:
        for i, colk in enumerate(cols_l):
            if ck in colk:
                return cols[i]
    raise RuntimeError(f"Could not find column. candidates={candidates} cols={list(df.columns)}")


def pick_player_col(df: pd.DataFrame) -> str:
    """
    ESPN team tables usually have 'PLAYER' or 'Player'.
    """
    cmap = col_lookup(list(df.columns))
    for k in ["player", "name"]:
        if k in cmap:
            return cmap[k]
    # fallback: first column
    return list(df.columns)[0]


def leader_from_table(df: pd.DataFrame, stat_candidates: List[str], mode: str) -> Tuple[str, Number]:
    """
    Return (player_name, value) where value is max for that stat.
    """
    player_col = pick_player_col(df)
    stat_col = choose_col(df, stat_candidates)

    work = df[[player_col, stat_col]].copy()
    if mode == "float1":
        work["__val__"] = work[stat_col].map(safe_float)
    else:
        work["__val__"] = work[stat_col].map(safe_int)

    work[player_col] = work[player_col].map(normalize_spaces)
    work = work.dropna(subset=["__val__"])

    if work.empty:
        raise RuntimeError(f"No values for stat col {stat_col} in table cols={list(df.columns)}")

    best = work.sort_values("__val__", ascending=False).iloc[0]
    return str(best[player_col]), best["__val__"]


# ----------------------------
# Mapping: team page -> the tables we need
# ----------------------------
def extract_team_leaders(team_url: str) -> Dict[str, Tuple[str, Number, str]]:
    """
    Returns dict mapping category -> (player, value, mode)
    Categories:
      Passing Yards, Passing TDs, Interceptions Thrown,
      Rushing Yards, Rushing TDs,
      Receiving Yards, Receiving TDs,
      Sacks, Tackles, Interceptions
    """
    tables = fetch_tables(team_url)
    if not tables:
        raise RuntimeError("No tables found on team stats page.")

    # ESPN team stats pages typically have separate tables for:
    # Passing (C/ATT/YDS/TD/INT), Rushing (ATT/YDS/TD), Receiving (REC/YDS/TD),
    # Defense (TOT/SACK/INT) or similar.
    #
    # We locate each by “must_have” columns.

    passing = find_table_with_cols(tables, ["player", "yds", "td", "int"])
    rushing = find_table_with_cols(tables, ["player", "yds", "td"])
    receiving = find_table_with_cols(tables, ["player", "rec", "yds", "td"])
    defense = find_table_with_cols(tables, ["player", "tot", "sack", "int"]) or find_table_with_cols(
        tables, ["player", "tackles", "sack", "int"]
    )

    if passing is None:
        # Sometimes passing table might not include INT column name exactly; try looser
        passing = find_table_with_cols(tables, ["player", "yds", "td"])
    if passing is None:
        raise RuntimeError("Could not find Passing table on team stats page.")

    if rushing is None:
        raise RuntimeError("Could not find Rushing table on team stats page.")
    if receiving is None:
        raise RuntimeError("Could not find Receiving table on team stats page.")
    if defense is None:
        raise RuntimeError("Could not find Defense table on team stats page.")

    out: Dict[str, Tuple[str, Number, str]] = {}

    # Passing
    py_name, py_val = leader_from_table(passing, ["YDS", "PASS YDS", "Pass YDS"], "int")
    ptd_name, ptd_val = leader_from_table(passing, ["TD", "PASS TD", "Pass TD"], "int")
    pint_name, pint_val = leader_from_table(passing, ["INT", "Interceptions"], "int")
    out["Passing Yards"] = (py_name, py_val, "int")
    out["Passing TDs"] = (ptd_name, ptd_val, "int")
    out["Interceptions Thrown"] = (pint_name, pint_val, "int")

    # Rushing
    ry_name, ry_val = leader_from_table(rushing, ["YDS", "RUSH YDS", "Rush YDS"], "int")
    rtd_name, rtd_val = leader_from_table(rushing, ["TD", "RUSH TD", "Rush TD"], "int")
    out["Rushing Yards"] = (ry_name, ry_val, "int")
    out["Rushing TDs"] = (rtd_name, rtd_val, "int")

    # Receiving
    rey_name, rey_val = leader_from_table(receiving, ["YDS", "REC YDS", "Rec YDS"], "int")
    retd_name, retd_val = leader_from_table(receiving, ["TD", "REC TD", "Rec TD"], "int")
    out["Receiving Yards"] = (rey_name, rey_val, "int")
    out["Receiving TDs"] = (retd_name, retd_val, "int")

    # Defense
    sack_name, sack_val = leader_from_table(defense, ["SACK", "Sacks", "SACKS"], "float1")
    tack_name, tack_val = leader_from_table(defense, ["TOT", "Tackles", "Total", "Total Tackles"], "int")
    dint_name, dint_val = leader_from_table(defense, ["INT", "Interceptions"], "int")
    out["Sacks"] = (sack_name, sack_val, "float1")
    out["Tackles"] = (tack_name, tack_val, "int")
    out["Interceptions"] = (dint_name, dint_val, "int")

    return out


# ----------------------------
# Poster drawing (same style as your offense/defense poster, but phone-big)
# ----------------------------
def load_font(size: int, bold: bool = False):
    candidates = []
    # Linux (Render/GHA)
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    # macOS fallback
    candidates += [
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


def fmt_value(val: Number, mode: str) -> str:
    if mode == "float1":
        return f"{float(val):.1f}"
    return str(int(val))


def draw_leaders_grid_poster(
    out_path: str,
    title: str,
    subtitle: str,
    sections: List[Tuple[str, str, Number, str]],
    cols: int,
    rows: int,
):
    """
    sections: list of (category, leader_name, value, mode)
    """
    W, H = 1440, 2560
    img = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(img)

    title_font = load_font(78, bold=True)
    sub_font = load_font(40, bold=False)
    head_font = load_font(44, bold=True)
    name_font = load_font(44, bold=False)
    val_font = load_font(52, bold=True)

    d.text((90, 120), title, font=title_font, fill=(245, 245, 245))
    d.text((90, 220), subtitle, font=sub_font, fill=(180, 180, 190))

    left = 90
    top = 320
    gap_x = 40
    gap_y = 40
    box_w = (W - left * 2 - gap_x * (cols - 1)) // cols
    box_h = (H - top - 140 - gap_y * (rows - 1)) // rows

    for idx, (cat, leader, val, mode) in enumerate(sections):
        c = idx % cols
        r = idx // cols
        x0 = left + c * (box_w + gap_x)
        y0 = top + r * (box_h + gap_y)
        x1 = x0 + box_w
        y1 = y0 + box_h

        d.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=26,
            fill=(20, 20, 28),
            outline=(45, 45, 60),
            width=3,
        )

        d.text((x0 + 24, y0 + 18), cat, font=head_font, fill=(240, 240, 245))

        # Leader name (wrap-ish by truncation)
        leader = normalize_spaces(leader)
        max_chars = 22 if cols == 2 else 28
        if len(leader) > max_chars:
            leader = leader[: max_chars - 1] + "…"

        d.text((x0 + 24, y0 + 82), leader, font=name_font, fill=(210, 210, 220))

        # Value right-aligned
        val_txt = fmt_value(val, mode)
        tw = d.textlength(val_txt, font=val_font)
        d.text((x1 - 24 - tw, y0 + 78), val_txt, font=val_font, fill=(255, 255, 255))

    img.save(out_path, "PNG", optimize=True)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url",
        type=str,
        default="https://www.espn.com/nfl/team/stats/_/name/ari/arizona-cardinals",
        help="ESPN team stats URL",
    )
    ap.add_argument("--outdir", type=str, default=os.path.join(os.path.expanduser("~"), "Desktop"))
    ap.add_argument("--team", type=str, default="ARI", help="Team abbreviation for filenames (e.g., ARI)")
    args = ap.parse_args()

    team_url = args.url
    outdir = args.outdir
    team = args.team.strip().upper()

    os.makedirs(outdir, exist_ok=True)

    updated = datetime.now().strftime("%b %d, %Y • %I:%M %p")
    subtitle = f"{team} • Updated {updated}"

    leaders = extract_team_leaders(team_url)

    # Offensive poster (7 categories)
    offense_order = [
        "Passing Yards",
        "Passing TDs",
        "Interceptions Thrown",
        "Rushing Yards",
        "Rushing TDs",
        "Receiving Yards",
        "Receiving TDs",
    ]
    offense_sections = []
    for cat in offense_order:
        player, val, mode = leaders[cat]
        offense_sections.append((cat, player, val, mode))

    # Defensive poster (3 categories)
    defense_order = ["Sacks", "Tackles", "Interceptions"]
    defense_sections = []
    for cat in defense_order:
        player, val, mode = leaders[cat]
        defense_sections.append((cat, player, val, mode))

    out_off = os.path.join(outdir, f"{team.lower()}_offense_stat_leaders.png")
    out_def = os.path.join(outdir, f"{team.lower()}_defense_stat_leaders.png")

    # Layout choices:
    # offense: 2 cols x 4 rows (7 boxes filled)
    draw_leaders_grid_poster(
        out_off,
        "Offensive Statistical Leaders",
        subtitle,
        offense_sections,
        cols=2,
        rows=4,
    )

    # defense: 1 col x 3 rows
    draw_leaders_grid_poster(
        out_def,
        "Defensive Statistical Leaders",
        subtitle,
        defense_sections,
        cols=1,
        rows=3,
    )

    print("\nDONE ✅")
    print(out_off)
    print(out_def)
    print("")


if __name__ == "__main__":
    main()
