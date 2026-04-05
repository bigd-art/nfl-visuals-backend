#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import os
import random
import textwrap
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.services.storage_supabase import upload_file_return_url


# ============================================================
# CONFIG
# ============================================================

W = 1080
H = 1350
MARGIN = 54

BG = (8, 12, 20)
TEXT = (245, 247, 250)
MUTED = (170, 178, 190)
LINE = (38, 48, 66)
CARD = (14, 20, 32)
CARD2 = (18, 26, 42)
GREEN = (66, 200, 120)
YELLOW = (255, 196, 66)
RED = (255, 100, 100)
BLUE = (88, 158, 255)

CATEGORY_ORDER = [
    "success_rate_by_down_and_distance",
    "qb_masterclass",
    "play_that_won_the_game",
    "better_than_expected",
    "clutch_gene",
]

CATEGORY_LABELS = {
    "success_rate_by_down_and_distance": "Success Rate by Down and Distance",
    "qb_masterclass": "QB Masterclass",
    "play_that_won_the_game": "Play That Won the Game",
    "better_than_expected": "Better Than Expected",
    "clutch_gene": "Clutch Gene",
}

TEAM_COLORS = {
    "ARI": (151, 35, 63), "ATL": (167, 25, 48), "BAL": (26, 25, 95),
    "BUF": (0, 51, 141), "CAR": (0, 133, 202), "CHI": (11, 22, 42),
    "CIN": (251, 79, 20), "CLE": (49, 29, 0), "DAL": (0, 53, 148),
    "DEN": (0, 34, 68), "DET": (0, 118, 182), "GB": (24, 48, 40),
    "HOU": (3, 32, 47), "IND": (0, 44, 95), "JAX": (0, 103, 120),
    "KC": (227, 24, 55), "LV": (0, 0, 0), "LAC": (0, 128, 198),
    "LAR": (0, 53, 148), "MIA": (0, 142, 151), "MIN": (79, 38, 131),
    "NE": (0, 34, 68), "NO": (211, 188, 141), "NYG": (1, 35, 82),
    "NYJ": (18, 87, 64), "PHI": (0, 76, 84), "PIT": (255, 182, 18),
    "SEA": (0, 34, 68), "SF": (170, 0, 0), "TB": (213, 10, 10),
    "TEN": (12, 35, 64), "WAS": (90, 20, 20), "WSH": (90, 20, 20),
}

VALID_PLAY_TYPES = {"PASS", "RUSH", "SACK", "FIELD_GOAL", "PENALTY"}

ROTATION_ANCHOR_DATE = date(2026, 1, 1)  # day 0 = success_rate_by_down_and_distance
MIN_SEASON = 2018

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ============================================================
# FONT HELPERS
# ============================================================

def find_font_bold() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def find_font_regular() -> str:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


FONT_BOLD_PATH = find_font_bold()
FONT_REG_PATH = find_font_regular()


def font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_REG_PATH
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rounded_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def safe_text(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def normalize_team(team: str) -> str:
    if not team:
        return ""
    team = str(team).upper()
    if team == "WAS":
        return "WSH"
    return team


def clean_desc(s: str, max_len: int = None) -> str:
    s = safe_text(s).replace("\n", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def yardline_bin(series: pd.Series, step: int = 5) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return (np.floor(vals / step) * step + step / 2).astype("float")


def distance_bucket(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype="object")
    out[(vals >= 1) & (vals <= 3)] = "1-3 YDS"
    out[(vals >= 4) & (vals <= 6)] = "4-6 YDS"
    out[(vals >= 7) & (vals <= 10)] = "7-10 YDS"
    out[vals >= 11] = "11+ YDS"
    return out


def format_down_distance(down, ydstogo) -> str:
    try:
        d = int(float(down))
        y = int(round(float(ydstogo)))
        return f"{ordinal(d)} Down & {y}"
    except Exception:
        return "High-leverage snap"


def weekly_context(season: int, week: int) -> str:
    return f"{season} • Week {week} • Regular Season"


def week_limit_for_year(season: int) -> int:
    return 18 if season >= 2021 else 17


def validate_week(season: int, week: int) -> None:
    max_week = week_limit_for_year(season)
    if week < 1 or week > max_week:
        raise ValueError(f"Invalid week {week} for season {season}. Use 1-{max_week}.")


def load_logo(team: str):
    if not team:
        return None
    team = team.lower()
    urls = [
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{team}.png",
        f"https://a.espncdn.com/i/teamlogos/nfl/500-dark/{team}.png",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
            if r.ok and r.content:
                return Image.open(io.BytesIO(r.content)).convert("RGBA")
        except Exception:
            pass
    return None


def fit_multiline_text(draw, text: str, max_width: int, max_height: int,
                       start_size: int, min_size: int = 16, bold: bool = False,
                       line_spacing: int = 6):
    text = safe_text(text).strip()
    if not text:
        return "", font(start_size, bold=bold), line_spacing

    for size in range(start_size, min_size - 1, -1):
        f = font(size, bold=bold)
        approx_chars = max(18, int(max_width / max(size * 0.55, 1)))
        wrapped = wrap_text(text, approx_chars)

        bbox = draw.multiline_textbbox((0, 0), wrapped, font=f, spacing=line_spacing)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if w <= max_width and h <= max_height:
            return wrapped, f, line_spacing

    f = font(min_size, bold=bold)
    approx_chars = max(18, int(max_width / max(min_size * 0.55, 1)))
    wrapped = wrap_text(text, approx_chars)
    return wrapped, f, line_spacing


def now_eastern_date() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


def rotation_index_for_day(day: date) -> int:
    return (day - ROTATION_ANCHOR_DATE).days % len(CATEGORY_ORDER)


def category_for_day(day: date) -> str:
    return CATEGORY_ORDER[rotation_index_for_day(day)]


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


# ============================================================
# DATA LOADING
# ============================================================

def load_csv_gz_url(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=180, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    with gzip.GzipFile(fileobj=bio) as gz:
        return pd.read_csv(gz, low_memory=False)


def load_pbp_one_season(season: int) -> pd.DataFrame:
    urls = [
        f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz",
        f"https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/data/play_by_play_{season}.csv.gz",
    ]
    last_err = None
    for url in urls:
        try:
            print(f"trying {url}")
            return load_csv_gz_url(url)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load play-by-play for {season}. Last error: {last_err}")


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "week", "qtr", "down", "ydstogo", "yardline_100", "ep", "wp",
        "wpa", "epa", "cpoe", "air_yards", "yards_gained", "score_differential",
        "complete_pass", "incomplete_pass", "interception", "pass_touchdown",
        "rushing_yards", "passing_yards", "receiving_yards", "rush_touchdown",
        "receiving_touchdown", "game_seconds_remaining", "pass", "rush"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "play_type_nfl" in df.columns:
        df = df[df["play_type_nfl"].astype(str).isin(VALID_PLAY_TYPES)].copy()

    df = df[df["ep"].notna()].copy()
    df = df[df["yardline_100"].between(1, 99, inclusive="both")].copy()

    if "success" not in df.columns:
        df["success"] = (pd.to_numeric(df["epa"], errors="coerce") > 0).astype(float)
    else:
        df["success"] = pd.to_numeric(df["success"], errors="coerce")

    for col in ["posteam", "defteam", "home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_team)

    if "desc" not in df.columns and "play_description" in df.columns:
        df["desc"] = df["play_description"]

    if "score_differential" in df.columns:
        df["score_margin"] = pd.to_numeric(df["score_differential"], errors="coerce")
    elif "posteam_score" in df.columns and "defteam_score" in df.columns:
        df["score_margin"] = (
            pd.to_numeric(df["posteam_score"], errors="coerce")
            - pd.to_numeric(df["defteam_score"], errors="coerce")
        )
    else:
        df["score_margin"] = np.nan

    if "game_seconds_remaining" in df.columns:
        df["minutes_remaining"] = pd.to_numeric(df["game_seconds_remaining"], errors="coerce") / 60.0
    else:
        df["minutes_remaining"] = np.nan

    return df


def filter_regular_week(df: pd.DataFrame, week: int) -> pd.DataFrame:
    out = df.copy()
    if "game_type" in out.columns:
        out = out[out["game_type"].astype(str) == "REG"].copy()
    out = out[out["week"] == week].copy()
    return out


# ============================================================
# STATLINE BUILDERS
# ============================================================

def qb_week_statline(df_week: pd.DataFrame, player: str) -> str:
    d = df_week[df_week["passer_player_name"].astype(str) == str(player)].copy()
    if d.empty:
        return "Stat line unavailable"

    completions = int(d.get("complete_pass", pd.Series(dtype=float)).fillna(0).sum())
    incompletions = int(d.get("incomplete_pass", pd.Series(dtype=float)).fillna(0).sum())
    interceptions = int(d.get("interception", pd.Series(dtype=float)).fillna(0).sum())
    attempts = completions + incompletions + interceptions

    pass_yards = int(d.get("passing_yards", pd.Series(dtype=float)).fillna(0).sum())
    pass_tds = int(d.get("pass_touchdown", pd.Series(dtype=float)).fillna(0).sum())

    rushes = d[d.get("rush", pd.Series(dtype=float)).fillna(0) == 1]
    rush_yards = int(rushes.get("rushing_yards", pd.Series(dtype=float)).fillna(0).sum()) if not rushes.empty else 0
    rush_tds = int(rushes.get("rush_touchdown", pd.Series(dtype=float)).fillna(0).sum()) if not rushes.empty else 0

    return f"{completions}/{attempts}, {pass_yards} Pass Yds, {pass_tds} Pass TD, {interceptions} INT, {rush_yards} Rush Yds, {rush_tds} Rush TD"


def skill_week_statline(df_week: pd.DataFrame, player: str) -> str:
    recv = df_week[df_week.get("receiver_player_name", pd.Series(dtype=object)).astype(str) == str(player)].copy()
    rush = df_week[df_week.get("rusher_player_name", pd.Series(dtype=object)).astype(str) == str(player)].copy()

    receptions = int(recv.get("complete_pass", pd.Series(dtype=float)).fillna(0).sum()) if not recv.empty else 0
    rec_yards = int(recv.get("receiving_yards", pd.Series(dtype=float)).fillna(0).sum()) if not recv.empty else 0
    rec_tds = int(recv.get("receiving_touchdown", pd.Series(dtype=float)).fillna(0).sum()) if not recv.empty else 0

    carries = len(rush) if not rush.empty else 0
    rush_yards = int(rush.get("rushing_yards", pd.Series(dtype=float)).fillna(0).sum()) if not rush.empty else 0
    rush_tds = int(rush.get("rush_touchdown", pd.Series(dtype=float)).fillna(0).sum()) if not rush.empty else 0

    if receptions > 0 and carries > 0:
        return f"{receptions} Rec, {rec_yards} Rec Yds, {rec_tds} Rec TD • {carries} Car, {rush_yards} Rush Yds, {rush_tds} Rush TD"
    if receptions > 0:
        return f"{receptions} Rec, {rec_yards} Rec Yds, {rec_tds} Rec TD"
    return f"{carries} Car, {rush_yards} Rush Yds, {rush_tds} Rush TD"


def actor_statline(df_week: pd.DataFrame, player: str) -> str:
    if player in set(df_week.get("passer_player_name", pd.Series(dtype=object)).dropna().astype(str)):
        return qb_week_statline(df_week, player)
    return skill_week_statline(df_week, player)


# ============================================================
# CHART POSTER
# ============================================================

def apply_chart_style(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str, xlabel: str, ylabel: str) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.28, linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_alpha(0.45)

    ax.set_title(title, fontsize=28, fontweight="bold", loc="left", pad=14)
    fig.text(0.125, 0.02, subtitle, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold", labelpad=10)
    ax.tick_params(labelsize=12)


def save_chart(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def plot_success_rate_by_down_and_distance(df_week: pd.DataFrame, season: int, week: int, out_path: str) -> None:
    d = df_week.copy()
    d["dist_bucket"] = distance_bucket(d["ydstogo"])
    d = d[d["dist_bucket"].notna()].copy()
    d = d[d["success"].notna()].copy()
    d = d[d["down"].between(1, 4, inclusive="both")].copy()

    g = (
        d.groupby(["down", "dist_bucket"], as_index=False)
        .agg(success_rate=("success", "mean"), plays=("success", "size"))
    )
    g = g[g["plays"] >= 6].copy()

    order = ["1-3 YDS", "4-6 YDS", "7-10 YDS", "11+ YDS"]
    pivot = g.pivot(index="dist_bucket", columns="down", values="success_rate").reindex(order)

    labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}

    fig, ax = plt.subplots(figsize=(10.8, 7.9))
    apply_chart_style(
        fig, ax,
        "Success Rate by Down and Distance",
        weekly_context(season, week),
        "Yards to go",
        "Success rate",
    )

    x = np.arange(len(order))
    width = 0.18

    for i, down in enumerate([1, 2, 3, 4]):
        vals = pivot[down].values if down in pivot.columns else np.full(len(order), np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=labels[down])

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1)
    ax.legend(title="Down", fontsize=12, title_fontsize=16)

    save_chart(fig, out_path)


# ============================================================
# IMAGE POSTERS
# ============================================================

@dataclass
class PosterItem:
    title: str
    subtitle: str
    player: str
    team: str
    big_value: str
    big_label: str
    description: str
    statline: str
    chip1: str
    chip2: str
    chip3: str
    accent_rgb: Tuple[int, int, int]
    visual_kind: str
    visual_values: Dict[str, float]


def add_gradient_background(img: Image.Image, accent_rgb: Tuple[int, int, int]) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    px = overlay.load()
    for y in range(H):
        alpha = int(95 * (1 - (y / H)))
        r = int(accent_rgb[0] * 0.42)
        g = int(accent_rgb[1] * 0.42)
        b = int(accent_rgb[2] * 0.42)
        for x in range(W):
            px[x, y] = (r, g, b, alpha)
    img.alpha_composite(overlay)


def paste_logo(base: Image.Image, team: str) -> None:
    logo = load_logo(team)
    if logo is None:
        return
    logo = ImageOps.contain(logo, (155, 155))
    lx = W - MARGIN - logo.width
    ly = 28
    base.alpha_composite(logo, (lx, ly))


def draw_header(draw, item: PosterItem):
    draw.text((MARGIN, 34), "STAT OF THE DAY", font=font(24, bold=True), fill=MUTED)
    draw.text((MARGIN, 74), item.title, font=font(54, bold=True), fill=TEXT)
    draw.text((MARGIN, 140), item.subtitle, font=font(24), fill=MUTED)


def draw_player_row(draw, item: PosterItem):
    y = 196
    rounded_rect(draw, (MARGIN, y, W - MARGIN, y + 98), 24, fill=CARD, outline=LINE, width=2)
    draw.text((MARGIN + 22, y + 16), item.player, font=font(40, bold=True), fill=TEXT)
    draw.text((MARGIN + 22, y + 58), item.team, font=font(22), fill=MUTED)


def draw_big_stat(draw, item: PosterItem):
    y = 318
    rounded_rect(draw, (MARGIN, y, W - MARGIN, y + 160), 28, fill=CARD2, outline=item.accent_rgb, width=3)
    draw.text((MARGIN + 24, y + 18), item.big_label, font=font(24, bold=True), fill=MUTED)
    draw.text((MARGIN + 24, y + 56), item.big_value, font=font(84, bold=True), fill=TEXT)


def draw_description(draw, item: PosterItem):
    y = 500
    rounded_rect(draw, (MARGIN, y, W - MARGIN, y + 118), 22, fill=CARD, outline=LINE, width=2)

    max_width = (W - 2 * MARGIN) - 36
    max_height = 118 - 32

    wrapped, fitted_font, spacing = fit_multiline_text(
        draw=draw,
        text=item.description,
        max_width=max_width,
        max_height=max_height,
        start_size=22,
        min_size=15,
        bold=False,
        line_spacing=6,
    )
    draw.multiline_text((MARGIN + 18, y + 16), wrapped, font=fitted_font, fill=TEXT, spacing=spacing)


def draw_statline(draw, item: PosterItem):
    y = 640
    rounded_rect(draw, (MARGIN, y, W - MARGIN, y + 98), 22, fill=CARD, outline=LINE, width=2)
    draw.text((MARGIN + 18, y + 14), "STAT LINE", font=font(18, bold=True), fill=MUTED)
    draw.multiline_text((MARGIN + 18, y + 42), wrap_text(item.statline, 70), font=font(20, bold=False), fill=TEXT, spacing=4)


def draw_visual(draw, item: PosterItem):
    y = 760
    rounded_rect(draw, (MARGIN, y, W - MARGIN, y + 250), 28, fill=CARD, outline=LINE, width=2)
    draw.text((MARGIN + 20, y + 16), "VISUAL BREAKDOWN", font=font(20, bold=True), fill=MUTED)

    if item.visual_kind == "before_after":
        left = MARGIN + 30
        right = W - MARGIN - 30
        before = max(0.0, min(1.0, item.visual_values.get("before", 0.0)))
        after = max(0.0, min(1.0, item.visual_values.get("after", 0.0)))

        draw.text((left, y + 56), f"Before: {before * 100:.0f}%", font=font(24, bold=True), fill=TEXT)
        draw.text((left + 470, y + 56), f"After: {after * 100:.0f}%", font=font(24, bold=True), fill=TEXT)

        bar_y1 = y + 104
        bar_y2 = y + 140
        rounded_rect(draw, (left, bar_y1, right, bar_y2), 18, fill=(28, 34, 48))
        rounded_rect(draw, (left, bar_y1, left + int((right - left) * before), bar_y2), 18, fill=YELLOW)

        bar2_y1 = y + 176
        bar2_y2 = y + 212
        rounded_rect(draw, (left, bar2_y1, right, bar2_y2), 18, fill=(28, 34, 48))
        rounded_rect(draw, (left, bar2_y1, left + int((right - left) * after), bar2_y2), 18, fill=GREEN)

    elif item.visual_kind == "percentile":
        pct = max(1, min(99, int(item.visual_values.get("percentile", 50))))
        left = MARGIN + 30
        right = W - MARGIN - 30

        draw.text((left, y + 54), f"{pct}th percentile", font=font(50, bold=True), fill=TEXT)
        draw.text((left, y + 112), "Compared to other performances that week", font=font(20), fill=MUTED)

        bar_y1 = y + 172
        bar_y2 = y + 210
        rounded_rect(draw, (left, bar_y1, right, bar_y2), 18, fill=(28, 34, 48))
        fill_x = left + int((right - left) * (pct / 100.0))
        fill_color = GREEN if pct >= 75 else YELLOW if pct >= 50 else RED
        rounded_rect(draw, (left, bar_y1, fill_x, bar_y2), 18, fill=fill_color)

    elif item.visual_kind == "expected_vs_actual":
        left_val = max(0.0, float(item.visual_values.get("expected", 0.0)))
        right_val = max(0.0, float(item.visual_values.get("actual", 0.0)))
        top = max(left_val, right_val, 1.0)

        chart_left = MARGIN + 150
        chart_right = W - MARGIN - 150
        base_y = y + 208
        max_h = 120
        bar_w = 150

        left_h = int(max_h * (left_val / top))
        right_h = int(max_h * (right_val / top))

        draw.line((chart_left - 60, base_y, chart_right + 60, base_y), fill=LINE, width=3)

        lx1 = chart_left
        lx2 = lx1 + bar_w
        rx1 = chart_right - bar_w
        rx2 = chart_right

        rounded_rect(draw, (lx1, base_y - left_h, lx2, base_y), 18, fill=YELLOW)
        rounded_rect(draw, (rx1, base_y - right_h, rx2, base_y), 18, fill=GREEN)

        draw.text((lx1 + 28, base_y - left_h - 34), f"{left_val:.0f}", font=font(28, bold=True), fill=TEXT)
        draw.text((rx1 + 28, base_y - right_h - 34), f"{right_val:.0f}", font=font(28, bold=True), fill=TEXT)
        draw.text((lx1 + 12, base_y + 12), "Expected", font=font(22, bold=True), fill=TEXT)
        draw.text((rx1 + 28, base_y + 12), "Actual", font=font(22, bold=True), fill=TEXT)


def draw_chips(draw, item: PosterItem):
    y = 1034
    gap = 16
    chip_w = (W - (2 * MARGIN) - (2 * gap)) // 3
    chips = [item.chip1, item.chip2, item.chip3]
    for i, txt in enumerate(chips):
        x1 = MARGIN + i * (chip_w + gap)
        x2 = x1 + chip_w
        rounded_rect(draw, (x1, y, x2, y + 88), 22, fill=CARD2, outline=LINE, width=2)
        draw.multiline_text((x1 + 14, y + 14), wrap_text(txt, 18), font=font(20, bold=True), fill=TEXT, spacing=4)


def draw_footer(draw, season: int, week: int):
    draw.text((MARGIN, H - 36), f"{season} Week {week} • Regular Season", font=font(18), fill=MUTED)


def render_player_poster(item: PosterItem, season: int, week: int, out_path: str) -> None:
    base = Image.new("RGBA", (W, H), BG + (255,))
    add_gradient_background(base, item.accent_rgb)
    draw = ImageDraw.Draw(base)

    draw_header(draw, item)
    draw_player_row(draw, item)
    draw_big_stat(draw, item)
    draw_description(draw, item)
    draw_statline(draw, item)
    draw_visual(draw, item)
    draw_chips(draw, item)
    draw_footer(draw, season, week)
    paste_logo(base, item.team)

    base.convert("RGB").save(out_path, quality=95)
    print(f"saved: {out_path}")


# ============================================================
# CATEGORY BUILDERS
# ============================================================

def choose_actor(row: pd.Series) -> str:
    for key in ["receiver_player_name", "rusher_player_name", "passer_player_name"]:
        v = safe_text(row.get(key))
        if v:
            return v
    return "Impact Player"


def percentile_from_series(series: pd.Series, value: float) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 50
    pct = (s <= value).mean()
    return max(1, min(99, int(round(pct * 100))))


def build_qb_masterclass(df_week: pd.DataFrame, season: int, week: int) -> PosterItem:
    c = df_week.copy()
    c = c[c["passer_player_name"].notna()].copy()
    c = c[c["epa"].notna()].copy()

    if "qb_dropback" in c.columns:
        c = c[(c["pass"] == 1) | (c["qb_dropback"] == 1)].copy()
    else:
        c = c[c["pass"] == 1].copy()

    grp = (
        c.groupby(["posteam", "passer_player_name"], dropna=False)
        .agg(
            attempts=("epa", "size"),
            epa_per_play=("epa", "mean"),
            total_epa=("epa", "sum"),
            cpoe=("cpoe", "mean"),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )
    grp = grp[grp["attempts"] >= 15].copy()
    grp = grp.sort_values(["epa_per_play", "total_epa"], ascending=[False, False])

    if grp.empty:
        raise RuntimeError("No QB Masterclass candidate found for that week.")

    row = grp.iloc[0]
    player = safe_text(row["passer_player_name"])
    team = safe_text(row["posteam"])
    statline = qb_week_statline(df_week, player)
    pct = percentile_from_series(grp["epa_per_play"], float(row["epa_per_play"]))

    return PosterItem(
        title="QB MASTERCLASS",
        subtitle=weekly_context(season, week),
        player=player,
        team=team,
        big_value=f"{float(row['epa_per_play']):+.2f}",
        big_label="IMPACT PER PLAY",
        description=f"{player} created efficient offense all game long, not just empty passing volume.",
        statline=statline,
        chip1=f"{int(row['attempts'])} dropbacks",
        chip2=f"{float(row['cpoe']):+.1f}% accuracy vs expected" if pd.notna(row["cpoe"]) else "Accuracy data unavailable",
        chip3=f"{float(row['success_rate']):.0%} positive plays",
        accent_rgb=TEAM_COLORS.get(team, BLUE),
        visual_kind="percentile",
        visual_values={"percentile": pct},
    )


def build_play_that_won_game(df_week: pd.DataFrame, season: int, week: int) -> PosterItem:
    c = df_week.copy()
    c = c[c["wpa"].notna()].copy()
    c["abs_wpa"] = c["wpa"].abs()
    c = c[c["abs_wpa"] > 0.05].copy()
    c = c[c["desc"].notna()].copy()
    c = c.sort_values(["abs_wpa", "epa"], ascending=[False, False])

    if c.empty:
        raise RuntimeError("No Play That Won the Game candidate found for that week.")

    row = c.iloc[0]
    player = choose_actor(row)
    team = safe_text(row.get("posteam"))
    wp_before = float(row.get("wp")) if pd.notna(row.get("wp")) else 0.0
    wpa = float(row.get("wpa")) if pd.notna(row.get("wpa")) else 0.0
    wp_after = max(0.0, min(1.0, wp_before + wpa))

    teams_line = ""
    away = safe_text(row.get("away_team"))
    home = safe_text(row.get("home_team"))
    if away and home:
        teams_line = f"{away} at {home}"

    statline = actor_statline(df_week, player)

    return PosterItem(
        title="PLAY THAT WON THE GAME",
        subtitle=f"{weekly_context(season, week)} • {teams_line}" if teams_line else weekly_context(season, week),
        player=player,
        team=team,
        big_value=f"{wpa * 100:+.1f}%",
        big_label="WIN CHANCE SWING",
        description=clean_desc(row.get("desc"), max_len=None),
        statline=statline,
        chip1=f"Impact {float(row['epa']):+.2f}" if pd.notna(row.get("epa")) else "Impact unavailable",
        chip2=f"Q{int(row['qtr'])}" if pd.notna(row.get("qtr")) else "Quarter unavailable",
        chip3=format_down_distance(row.get("down"), row.get("ydstogo")),
        accent_rgb=TEAM_COLORS.get(team, BLUE),
        visual_kind="before_after",
        visual_values={"before": wp_before, "after": wp_after},
    )


def build_better_than_expected(df_week: pd.DataFrame, season: int, week: int) -> PosterItem:
    c = df_week.copy()
    c = c[c["yards_gained"].notna()].copy()
    c = c[(c["rush"] == 1) | (c["receiver_player_name"].notna())].copy()

    c["ydstogo_bucket"] = distance_bucket(c["ydstogo"])
    c["yard_bin"] = yardline_bin(c["yardline_100"], step=20)
    c["down_bucket"] = c["down"].fillna(0).astype(int)

    baseline = (
        c.groupby(["down_bucket", "ydstogo_bucket", "yard_bin"], dropna=False)
        .agg(expected_yards=("yards_gained", "mean"), sample=("yards_gained", "size"))
        .reset_index()
    )
    baseline = baseline[baseline["sample"] >= 3].copy()

    c = c.merge(
        baseline[["down_bucket", "ydstogo_bucket", "yard_bin", "expected_yards"]],
        on=["down_bucket", "ydstogo_bucket", "yard_bin"],
        how="left",
    )
    c["expected_yards"] = c["expected_yards"].fillna(c["yards_gained"].median())
    c["yards_over_expected"] = c["yards_gained"] - c["expected_yards"]
    c["player_name"] = c["receiver_player_name"].fillna(c["rusher_player_name"])
    c = c[c["player_name"].notna()].copy()

    grp = (
        c.groupby(["posteam", "player_name"], dropna=False)
        .agg(
            total_yoe=("yards_over_expected", "sum"),
            total_actual=("yards_gained", "sum"),
            total_expected=("expected_yards", "sum"),
            plays=("yards_over_expected", "size"),
        )
        .reset_index()
    )
    grp = grp[grp["plays"] >= 4].copy()
    grp = grp.sort_values(["total_yoe", "total_actual"], ascending=[False, False])

    if grp.empty:
        raise RuntimeError("No Better Than Expected candidate found for that week.")

    row = grp.iloc[0]
    player = safe_text(row["player_name"])
    team = safe_text(row["posteam"])
    statline = skill_week_statline(df_week, player)

    return PosterItem(
        title="BETTER THAN EXPECTED",
        subtitle=weekly_context(season, week),
        player=player,
        team=team,
        big_value=f"{float(row['total_yoe']):+.1f}",
        big_label="YARDS ABOVE EXPECTATION",
        description=f"{player} got much more out of his touches than an average player would in the same situations.",
        statline=statline,
        chip1=f"Actual {float(row['total_actual']):.0f} yds",
        chip2=f"Expected {float(row['total_expected']):.0f} yds",
        chip3=f"{int(row['plays'])} plays",
        accent_rgb=TEAM_COLORS.get(team, BLUE),
        visual_kind="expected_vs_actual",
        visual_values={
            "expected": float(row["total_expected"]),
            "actual": float(row["total_actual"]),
        },
    )


def build_clutch_gene(df_week: pd.DataFrame, season: int, week: int) -> PosterItem:
    c = df_week.copy()
    c = c[c["epa"].notna()].copy()
    c["actor"] = c.apply(choose_actor, axis=1)
    c = c[c["actor"] != "Impact Player"].copy()
    c = c[c["qtr"].fillna(0) >= 4].copy()
    c = c[c["wp"].between(0.20, 0.80, inclusive="both")].copy()

    grp = (
        c.groupby(["posteam", "actor"], dropna=False)
        .agg(
            clutch_epa=("epa", "sum"),
            plays=("epa", "size"),
            avg_epa=("epa", "mean"),
        )
        .reset_index()
    )
    grp = grp[grp["plays"] >= 3].copy()
    grp = grp.sort_values(["clutch_epa", "avg_epa"], ascending=[False, False])

    if grp.empty:
        raise RuntimeError("No Clutch Gene candidate found for that week.")

    row = grp.iloc[0]
    player = safe_text(row["actor"])
    team = safe_text(row["posteam"])
    statline = actor_statline(df_week, player)
    pct = percentile_from_series(grp["clutch_epa"], float(row["clutch_epa"]))

    return PosterItem(
        title="CLUTCH GENE",
        subtitle=weekly_context(season, week),
        player=player,
        team=team,
        big_value=f"{float(row['clutch_epa']):+.2f}",
        big_label="CLUTCH IMPACT",
        description=f"{player} delivered late when the game still felt up for grabs.",
        statline=statline,
        chip1=f"{int(row['plays'])} clutch plays",
        chip2=f"{float(row['avg_epa']):+.2f} per play",
        chip3="4th quarter, live game state",
        accent_rgb=TEAM_COLORS.get(team, BLUE),
        visual_kind="percentile",
        visual_values={"percentile": pct},
    )


# ============================================================
# SEASON/WEEK PICKING
# ============================================================

def load_prepped_regular_week(season: int, week: int) -> pd.DataFrame:
    validate_week(season, week)
    df = load_pbp_one_season(season)
    df = prep_df(df)
    df_week = filter_regular_week(df, week)
    if df_week.empty:
        raise RuntimeError(f"No regular-season plays found for {season} week {week}.")
    return df_week


def category_candidate_works(category_key: str, df_week: pd.DataFrame, season: int, week: int) -> bool:
    try:
        if category_key == "success_rate_by_down_and_distance":
            d = df_week.copy()
            d["dist_bucket"] = distance_bucket(d["ydstogo"])
            d = d[d["dist_bucket"].notna()].copy()
            d = d[d["success"].notna()].copy()
            d = d[d["down"].between(1, 4, inclusive="both")].copy()
            g = (
                d.groupby(["down", "dist_bucket"], as_index=False)
                .agg(success_rate=("success", "mean"), plays=("success", "size"))
            )
            g = g[g["plays"] >= 6].copy()
            return not g.empty

        if category_key == "qb_masterclass":
            build_qb_masterclass(df_week, season, week)
            return True
        if category_key == "play_that_won_the_game":
            build_play_that_won_game(df_week, season, week)
            return True
        if category_key == "better_than_expected":
            build_better_than_expected(df_week, season, week)
            return True
        if category_key == "clutch_gene":
            build_clutch_gene(df_week, season, week)
            return True
        return False
    except Exception:
        return False


def choose_random_valid_season_week(category_key: str, start_season: int = MIN_SEASON, end_season: int = None,
                                    max_tries: int = 60) -> Tuple[int, int, pd.DataFrame]:
    if end_season is None:
        end_season = now_eastern_date().year

    candidates = []
    for season in range(start_season, end_season + 1):
        max_week = week_limit_for_year(season)
        for week in range(1, max_week + 1):
            candidates.append((season, week))

    random.shuffle(candidates)

    last_error = None
    for season, week in candidates[:max_tries]:
        try:
            print(f"trying candidate season={season} week={week} for category={category_key}")
            df_week = load_prepped_regular_week(season, week)
            if category_candidate_works(category_key, df_week, season, week):
                return season, week, df_week
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not find valid season/week for category={category_key}. Last error: {last_error}")


# ============================================================
# GENERATION
# ============================================================

def generate_single_category(category_key: str, season: int, week: int, df_week: pd.DataFrame, out_path: str) -> Dict[str, str]:
    if category_key == "success_rate_by_down_and_distance":
        plot_success_rate_by_down_and_distance(df_week, season, week, out_path)
        return {
            "title": CATEGORY_LABELS[category_key],
            "team": "",
            "player": "",
        }

    if category_key == "qb_masterclass":
        item = build_qb_masterclass(df_week, season, week)
        render_player_poster(item, season, week, out_path)
        return {"title": item.title, "team": item.team, "player": item.player}

    if category_key == "play_that_won_the_game":
        item = build_play_that_won_game(df_week, season, week)
        render_player_poster(item, season, week, out_path)
        return {"title": item.title, "team": item.team, "player": item.player}

    if category_key == "better_than_expected":
        item = build_better_than_expected(df_week, season, week)
        render_player_poster(item, season, week, out_path)
        return {"title": item.title, "team": item.team, "player": item.player}

    if category_key == "clutch_gene":
        item = build_clutch_gene(df_week, season, week)
        render_player_poster(item, season, week, out_path)
        return {"title": item.title, "team": item.team, "player": item.player}

    raise ValueError(f"Unknown category: {category_key}")


def publish_stat_of_the_day(run_day: date = None, keep_versioned: bool = False) -> Dict[str, str]:
    if run_day is None:
        run_day = now_eastern_date()

    category_key = category_for_day(run_day)

    # daily deterministic randomness so reruns that same day stay stable
    random.seed(f"stat-of-day::{run_day.isoformat()}::{category_key}")

    season, week, df_week = choose_random_valid_season_week(category_key=category_key)

    local_png = f"/tmp/stat_of_the_day_{run_day.isoformat()}.png"
    info = generate_single_category(category_key, season, week, df_week, local_png)

    current_storage_key = "stat_of_the_day/current.png"
    current_url = upload_file_return_url(local_png, current_storage_key)

    metadata = {
        "date": run_day.isoformat(),
        "category_key": category_key,
        "category_label": CATEGORY_LABELS[category_key],
        "season": season,
        "week": week,
        "image_url": current_url,
        "title": info.get("title", ""),
        "team": info.get("team", ""),
        "player": info.get("player", ""),
        "storage_key": current_storage_key,
    }

    local_json = f"/tmp/stat_of_the_day_{run_day.isoformat()}.json"
    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    current_meta_key = "stat_of_the_day/current.json"
    metadata_url = upload_file_return_url(local_json, current_meta_key)

    versioned_image_url = ""
    versioned_meta_url = ""

    if keep_versioned:
        versioned_prefix = f"stat_of_the_day/history/{run_day.isoformat()}"
        versioned_image_key = f"{versioned_prefix}/poster.png"
        versioned_meta_key = f"{versioned_prefix}/metadata.json"
        versioned_image_url = upload_file_return_url(local_png, versioned_image_key)
        versioned_meta_url = upload_file_return_url(local_json, versioned_meta_key)

    metadata["metadata_url"] = metadata_url
    if versioned_image_url:
        metadata["versioned_image_url"] = versioned_image_url
    if versioned_meta_url:
        metadata["versioned_metadata_url"] = versioned_meta_url

    return metadata


def get_current_stat_of_the_day_payload() -> Dict[str, str]:
    return {
        "image_url": public_storage_url("stat_of_the_day/current.png"),
        "metadata_url": public_storage_url("stat_of_the_day/current.json"),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Publish the nightly stat of the day poster.")
    p.add_argument("--date", type=str, default="", help="Optional YYYY-MM-DD in America/New_York rotation logic")
    p.add_argument("--keep_versioned", action="store_true", help="Also upload date-stamped history copy")
    return p.parse_args()


def main():
    args = parse_args()
    run_day = date.fromisoformat(args.date) if args.date else None
    result = publish_stat_of_the_day(run_day=run_day, keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
