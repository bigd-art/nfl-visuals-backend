import os
import shutil
import re
import sys
import argparse
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from PIL import Image


# ===================== STYLE =====================

STYLE = {
    "figsize": (8, 14),
    "fig_facecolor": "#050816",

    "top_band_color": "#0b1020",
    "middle_band_color": "#0b1224",
    "bottom_band_color": "#050816",

    "text_primary": "white",
    "text_secondary": "#e5e7eb",
    "text_muted": "#9ca3af",
    "accent": "#fbbf24",

    "font_family": "DejaVu Sans",
}

# ================================================


# ---------------------- ESPN HELPERS ----------------------

def fetch_url(url: str, timeout: int = 25) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_summary(event_id: str) -> Dict:
    api_url = (
        "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary"
        f"?event={event_id}"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(api_url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


def safe_int(value, default: Optional[int] = 0) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_logo_image(url: Optional[str]) -> Optional[Image.Image]:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return None


def scoreboard_url(year: int, week: int, seasontype: int) -> str:
    # seasontype: 1=preseason, 2=regular season, 3=postseason
    return f"https://www.espn.com/nfl/scoreboard/_/week/{week}/year/{year}/seasontype/{seasontype}"


def extract_game_ids_from_scoreboard_html(html: str) -> List[str]:
    ids = re.findall(r"gameId/(\d+)", html)
    seen = set()
    out = []
    for gid in ids:
        if gid not in seen:
            seen.add(gid)
            out.append(gid)
    return out



def _strip_html_to_text(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;|&#160;", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def get_scoring_periods_from_summary(event_id: str) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Returns (period_labels, periods_by_team_abbr)

    - period_labels: ["1Q","2Q","3Q","4Q"] and includes "OT" ONLY if OT exists.
    - periods_by_team_abbr: {"SEA":[q1,q2,q3,q4,(ot?)], "WSH":[...]}
    """
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
    j = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).json()
    comp = j["header"]["competitions"][0]

    # Build team scores per period using ESPN linescores when available
    tmp: Dict[str, List[int]] = {}
    max_periods = 0
    for c in comp["competitors"]:
        abbr = c["team"]["abbreviation"]
        ls = c.get("linescores", []) or []
        vals = [_safe_int(q.get("value", q.get("displayValue", 0))) for q in ls]
        tmp[abbr] = vals
        max_periods = max(max_periods, len(vals))

    # If linescores missing for either team, fall back to scraping boxscore header table
    if max_periods == 0 or any(len(v) == 0 for v in tmp.values()):
        labels, scraped = fetch_scoring_periods_from_boxscore(event_id)
        return labels, scraped

    # Determine if OT occurred based on number of periods
    # NFL regulation has 4 quarters. ESPN may include OT as 5th period.
    has_ot = max_periods > 4

    labels = ["1Q", "2Q", "3Q", "4Q"] + (["OT"] if has_ot else [])
    target_len = 5 if has_ot else 4

    # Normalize all teams to same length (pad with 0s if needed)
    out: Dict[str, List[int]] = {}
    for abbr, vals in tmp.items():
        out[abbr] = (vals + [0] * target_len)[:target_len]

    return labels, out


def fetch_scoring_periods_from_boxscore(event_id: str) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Scrapes the top linescore table from the ESPN boxscore page.
    Returns (period_labels, periods_by_team_abbr) where period_labels includes OT only if present.
    """
    url = f"https://www.espn.com/nfl/boxscore/_/gameId/{event_id}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    text = _strip_html_to_text(r.text)

    # Match header like: "Final 1 2 3 4 T" or "Final 1 2 3 4 OT T"
    m = re.search(r"\bFinal\b\s+((?:(?:\d+|OT)\s+)+)T\b", text)
    if not m:
        return (["1Q", "2Q", "3Q", "4Q"], {})

    raw_labels = m.group(1).strip().split()  # e.g. ["1","2","3","4"] or ["1","2","3","4","OT"]
    labels = [f"{x}Q" if x.isdigit() else "OT" for x in raw_labels]
    n = len(labels)

    # After header: "ABBR q1 q2 ... qn total"
    window = text[m.end(): m.end() + 2200]
    row_pat = re.compile(rf"\b([A-Z]{{2,4}})\b\s+((?:\d+\s+){{{n}}}\d+)\b")
    rows = row_pat.findall(window)[:2]

    out: Dict[str, List[int]] = {}
    for abbr, nums_blob in rows:
        nums = [_safe_int(x) for x in nums_blob.strip().split()]
        # nums = period scores (n) + total (1)
        if len(nums) == n + 1:
            out[abbr] = nums[:n]

    # If somehow only 4 labels but OT exists in data, keep labels as parsed.
    # If labels has OT, we will show it; otherwise we won't.
    return labels, out


# ---------------------- STAT PARSING ----------------------

def parse_stat_group(stat_group: Dict) -> List[Dict]:
    labels = stat_group.get("labels", [])
    athletes = stat_group.get("athletes", [])
    rows: List[Dict] = []
    for a in athletes:
        athlete_info = a.get("athlete", {}) or {}
        stats = a.get("stats", []) or []
        row = dict(zip(labels, stats))
        row["player"] = athlete_info.get("displayName") or athlete_info.get("shortName")
        rows.append(row)
    return rows


def extract_passing_leader(stat_group: Dict) -> Optional[Dict]:
    rows = parse_stat_group(stat_group)
    if not rows:
        return None

    row = rows[0]
    cmp_att = row.get("C/ATT") or row.get("CMP/ATT") or ""
    completions = attempts = None
    if "/" in cmp_att:
        c_str, a_str = cmp_att.split("/", 1)
        completions = safe_int(c_str, default=None)
        attempts = safe_int(a_str, default=None)

    return {
        "name": row.get("player"),
        "completions": completions,
        "attempts": attempts,
        "yards": safe_int(row.get("YDS"), 0),
        "td": safe_int(row.get("TD"), 0),
        "ints": safe_int(row.get("INT"), 0),
    }


def extract_yardage_leader(stat_group: Dict, kind: str) -> Optional[Dict]:
    rows = parse_stat_group(stat_group)
    if not rows:
        return None

    leader = None
    max_yds = -1
    for row in rows:
        yds = safe_int(row.get("YDS"), 0)
        if yds > max_yds:
            max_yds = yds
            leader = row

    if not leader:
        return None

    if kind == "rushing":
        return {
            "name": leader.get("player"),
            "carries": safe_int(leader.get("CAR"), 0),
            "yards": safe_int(leader.get("YDS"), 0),
            "td": safe_int(leader.get("TD"), 0),
        }

    return {
        "name": leader.get("player"),
        "receptions": safe_int(leader.get("REC"), 0),
        "yards": safe_int(leader.get("YDS"), 0),
        "td": safe_int(leader.get("TD"), 0),
    }


def extract_team_leaders_from_players_block(team_block: Dict) -> Dict:
    team_info = team_block.get("team", {}) or {}
    team_name = team_info.get("displayName") or team_info.get("name")

    leaders: Dict[str, Optional[Dict]] = {
        "team": team_name,
        "passing_leader": None,
        "rushing_leader": None,
        "receiving_leader": None,
    }

    for stat_group in team_block.get("statistics", []):
        group_name = (stat_group.get("name") or "").lower()
        if group_name == "passing":
            leaders["passing_leader"] = extract_passing_leader(stat_group)
        elif group_name == "rushing":
            leaders["rushing_leader"] = extract_yardage_leader(stat_group, "rushing")
        elif group_name == "receiving":
            leaders["receiving_leader"] = extract_yardage_leader(stat_group, "receiving")

    return leaders


def extract_all_team_leaders(summary: Dict) -> Dict[str, Dict]:
    boxscore = summary.get("boxscore", {}) or {}
    players_blocks = boxscore.get("players", []) or []
    results: Dict[str, Dict] = {}
    for team_block in players_blocks:
        team_leaders = extract_team_leaders_from_players_block(team_block)
        team_name = team_leaders["team"]
        if team_name:
            results[team_name] = team_leaders
    return results


def extract_interception_leader_from_players_block(team_block: Dict) -> Optional[Dict]:
    stat_groups = team_block.get("statistics", []) or []
    best_player = None
    best_ints = 0

    for stat_group in stat_groups:
        name = (stat_group.get("name") or "").lower()
        display_name = (
            stat_group.get("displayName")
            or stat_group.get("shortDisplayName")
            or ""
        ).lower()

        if "interception" not in name and "interception" not in display_name:
            continue

        rows = parse_stat_group(stat_group)
        for row in rows:
            ints = safe_int(row.get("INT") or row.get("INTS") or row.get("NO.") or 0, 0)
            if ints > best_ints:
                best_ints = ints
                best_player = {"name": row.get("player") or "N/A", "ints": ints}

    if best_player and best_player["ints"] > 0:
        return best_player
    return None


def extract_defensive_leaders_from_players_block(team_block: Dict) -> Dict:
    team_info = team_block.get("team", {}) or {}
    team_name = team_info.get("displayName") or team_info.get("name")

    leaders = {
        "team": team_name,
        "tackles_leader": None,
        "sacks_leader": None,
        "ints_leader": None,
    }

    defensive_group = None
    for stat_group in team_block.get("statistics", []):
        if (stat_group.get("name") or "").lower() == "defensive":
            defensive_group = stat_group
            break

    if defensive_group:
        rows = parse_stat_group(defensive_group)
        max_tackles = 0
        max_sacks = 0.0

        for row in rows:
            tackles = safe_int(row.get("TOT") or row.get("Total") or row.get("TKL") or 0, 0)
            sacks = safe_float(row.get("SACKS") or row.get("SACK") or row.get("SK") or 0, 0.0)
            name = row.get("player") or "N/A"

            if tackles > max_tackles and tackles > 0:
                max_tackles = tackles
                leaders["tackles_leader"] = {"name": name, "tackles": tackles}

            if sacks > max_sacks and sacks > 0:
                max_sacks = sacks
                leaders["sacks_leader"] = {"name": name, "sacks": sacks}

    int_leader = extract_interception_leader_from_players_block(team_block)
    if int_leader:
        leaders["ints_leader"] = int_leader

    return leaders


def extract_all_defensive_leaders(summary: Dict) -> Dict[str, Dict]:
    boxscore = summary.get("boxscore", {}) or {}
    players_blocks = boxscore.get("players", []) or []
    results: Dict[str, Dict] = {}
    for team_block in players_blocks:
        team_leaders = extract_defensive_leaders_from_players_block(team_block)
        team_name = team_leaders["team"]
        if team_name:
            results[team_name] = team_leaders
    return results


def extract_game_meta(summary: Dict, meta_event_id: str) -> Dict:
    header = summary.get("header", {}) or {}
    competitions = header.get("competitions", []) or []
    comp = competitions[0] if competitions else {}
    competitors = comp.get("competitors", []) or []

    status_type = (comp.get("status") or {}).get("type") or {}
    completed = bool(status_type.get("completed", False))

    teams = []
    for c in competitors:
        team = c.get("team", {}) or {}
        name = team.get("displayName")
        abbr = team.get("abbreviation")
        record = (c.get("records") or [{}])[0].get("summary", "")
        score = c.get("score")
        home_away = c.get("homeAway", "home")

        logo = None
        logos = team.get("logos") or []
        if logos:
            logo = logos[0].get("href") or team.get("logo")

        # quarter/OT scores will be filled later via get_scoring_periods_from_summary
        q_scores: List[int] = []

        teams.append(
            {
                "name": name,
                "abbr": abbr,
                "record": record,
                "score": score,
                "home_away": home_away,
                "logo_url": logo,
                "quarter_scores": q_scores,
            }
        )

    # away first, then home
    teams_sorted = sorted(teams, key=lambda t: t["home_away"] != "away")

    # Fill quarter scores (and OT if it exists) for both teams
    period_labels, periods_by_abbr = get_scoring_periods_from_summary(meta_event_id)

    for t in teams_sorted:
        ab = t.get("abbr") or ""
        t["quarter_scores"] = periods_by_abbr.get(ab, [])

    return {"teams": teams_sorted, "completed": completed, "period_labels": period_labels}




def extract_team_yardage(summary: Dict) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    boxscore = summary.get("boxscore", {}) or {}
    team_blocks = boxscore.get("teams", []) or []

    for tb in team_blocks:
        team = tb.get("team", {}) or {}
        name = team.get("displayName") or team.get("name")
        stats = tb.get("statistics", []) or []

        total = rush = passing = 0
        for s in stats:
            label = (s.get("label") or "").lower()
            val_str = s.get("displayValue") or "0"
            val = safe_int(val_str.split(" ")[0], 0)

            if "total yards" in label:
                total = val
            elif "rushing yards" in label:
                rush = val
            elif "passing yards" in label:
                passing = val

        if name:
            results[name] = {"total_yards": total, "rush_yards": rush, "pass_yards": passing}

    return results


# ---------------------- POSTER GENERATION ----------------------

def make_poster_style_image(
    meta: Dict,
    offensive_leaders: Dict[str, Dict],
    defensive_leaders: Dict[str, Dict],
    yardage: Dict[str, Dict],
    output_path: str,
    style: Dict = STYLE,
) -> None:
    teams = meta["teams"]
    if len(teams) < 2:
        print("Not a standard two-team game, skipping.")
        return

    away, home = teams[0], teams[1]
    completed = meta.get("completed", False)

    away_name = away["name"]
    home_name = home["name"]

    away_off = offensive_leaders.get(away_name, {})
    home_off = offensive_leaders.get(home_name, {})

    away_def = defensive_leaders.get(away_name, {})
    home_def = defensive_leaders.get(home_name, {})

    away_yards = yardage.get(away_name, {"total_yards": 0, "rush_yards": 0, "pass_yards": 0})
    home_yards = yardage.get(home_name, {"total_yards": 0, "rush_yards": 0, "pass_yards": 0})

    plt.rcParams["font.family"] = style["font_family"]
    fig = plt.figure(figsize=style["figsize"], facecolor=style["fig_facecolor"])
    gs = gridspec.GridSpec(4, 1, height_ratios=[3.0, 2.0, 1.5, 3.5], hspace=0.28)

    # TOP
    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")
    ax_top.set_facecolor(style["top_band_color"])
    ax_top.add_patch(
        patches.FancyBboxPatch(
            (0.02, 0.08),
            0.96,
            0.84,
            boxstyle="round,pad=0.03",
            linewidth=0,
            facecolor=style["top_band_color"],
        )
    )

    away_logo = fetch_logo_image(away["logo_url"])
    home_logo = fetch_logo_image(home["logo_url"])

    if away_logo is not None:
        logo_ax = fig.add_axes([0.07, 0.78, 0.14, 0.14])
        logo_ax.imshow(away_logo)
        logo_ax.axis("off")

    if home_logo is not None:
        logo_ax = fig.add_axes([0.79, 0.78, 0.14, 0.14])
        logo_ax.imshow(home_logo)
        logo_ax.axis("off")

    ax_top.text(0.18, 0.55, f"{away['abbr'] or ''}", ha="center", va="center",
                fontsize=28, fontweight="bold", color=style["text_secondary"])
    ax_top.text(0.18, 0.32, away["record"] or "", ha="center", va="center",
                fontsize=11, color=style["text_muted"])

    ax_top.text(0.82, 0.55, f"{home['abbr'] or ''}", ha="center", va="center",
                fontsize=28, fontweight="bold", color=style["text_secondary"])
    ax_top.text(0.82, 0.32, home["record"] or "", ha="center", va="center",
                fontsize=11, color=style["text_muted"])

    left_score = away["score"] if completed and away["score"] is not None else "–"
    right_score = home["score"] if completed and home["score"] is not None else "–"

    ax_top.text(0.40, 0.55, str(left_score), ha="center", va="center",
                fontsize=46, fontweight="bold", color=style["text_primary"])
    ax_top.text(0.60, 0.55, str(right_score), ha="center", va="center",
                fontsize=46, fontweight="bold", color=style["text_primary"])
    ax_top.text(0.50, 0.55, "AT", ha="center", va="center",
                fontsize=14, color=style["text_muted"])

    # QUARTERS
    ax_q = fig.add_subplot(gs[1])
    ax_q.axis("off")
    ax_q.set_facecolor(style["middle_band_color"])
    ax_q.text(0.5, 0.86, "SCORING BY QUARTER", ha="center", va="center",
              fontsize=13, fontweight="bold", color=style["accent"])

    period_labels = meta.get("period_labels", ["1Q", "2Q", "3Q", "4Q"])
    labels = ["TEAM"] + period_labels

    # Dynamically space columns depending on whether OT exists
    n_cols = len(labels)  # TEAM + periods
    left, right = 0.14, 0.90
    step = (right - left) / max(1, (n_cols - 1))
    x_positions = [left + i * step for i in range(n_cols)]

    for x, lbl in zip(x_positions, labels):
        ax_q.text(x, 0.68, lbl, ha="center", va="center",
                  fontsize=11, color=style["text_secondary"])

    def row_scores(team: Dict, y_pos: float):
        abbr = team.get("abbr") or ""
        ax_q.text(x_positions[0], y_pos, abbr, ha="center", va="center",
                  fontsize=11, color=style["text_secondary"], fontweight="bold")

        q_scores = team.get("quarter_scores") or []
        q_scores = (q_scores + [0] * len(period_labels))[:len(period_labels)]

        for i in range(len(period_labels)):
            ax_q.text(x_positions[i + 1], y_pos, str(q_scores[i]),
                      ha="center", va="center", fontsize=11,
                      color=style["text_secondary"])

    row_scores(away, 0.50)
    row_scores(home, 0.32)


    # TOTAL YARDS
    ax_ty = fig.add_subplot(gs[2])
    ax_ty.axis("off")
    ax_ty.set_facecolor(style["middle_band_color"])
    ax_ty.text(0.5, 0.75, "TOTAL YARDS", ha="center", va="center",
               fontsize=13, fontweight="bold", color=style["accent"])
    ax_ty.text(
        0.5,
        0.42,
        f"{away['abbr']}: {away_yards['total_yards']}    |    {home['abbr']}: {home_yards['total_yards']}",
        ha="center",
        va="center",
        fontsize=12,
        color=style["text_primary"],
    )

    # TEAM LEADERS
    ax_leaders = fig.add_subplot(gs[3])
    ax_leaders.axis("off")
    ax_leaders.set_facecolor(style["bottom_band_color"])

    ax_leaders.text(
        0.5, 1.08, "TEAM LEADERS",
        ha="center", va="top",
        fontsize=14, fontweight="bold",
        color=style["text_primary"],
        clip_on=False,
    )

    def render_offensive(off_dict: Dict) -> List[str]:
        p = off_dict.get("passing_leader") or {}
        r = off_dict.get("rushing_leader") or {}
        rc = off_dict.get("receiving_leader") or {}
        lines = []
        lines.append(
            f"PASS: {p.get('name','N/A')} – {p.get('yards',0)} YDS, {p.get('td',0)} TD, {p.get('ints',0)} INT"
            if p else "PASS: N/A"
        )
        lines.append(
            f"RUSH: {r.get('name','N/A')} – {r.get('yards',0)} YDS, {r.get('td',0)} TD"
            if r else "RUSH: N/A"
        )
        lines.append(
            f"REC: {rc.get('name','N/A')} – {rc.get('yards',0)} YDS, {rc.get('td',0)} TD"
            if rc else "REC: N/A"
        )
        return lines

    def render_defensive(def_dict: Dict) -> List[str]:
        lines = []
        t = def_dict.get("tackles_leader")
        s = def_dict.get("sacks_leader")
        i = def_dict.get("ints_leader")
        if t: lines.append(f"TACKLES: {t['name']} – {t['tackles']}")
        if s: lines.append(f"SACKS: {s['name']} – {s['sacks']}")
        if i: lines.append(f"INTERCEPTIONS: {i['name']} – {i['ints']}")
        return lines

    def render_team_column(x_center: float, team_label: str, off_dict: Dict, def_dict: Dict) -> None:
        y_team = 0.90
        y_off_label = 0.82
        y_off_lines = [0.74, 0.66, 0.58]
        y_def_label = 0.46
        y_def_lines = [0.38, 0.30, 0.22]

        ax_leaders.text(x_center, y_team, team_label, ha="center", va="center",
                        fontsize=12, color=style["accent"])

        ax_leaders.text(x_center, y_off_label, "OFFENSIVE TEAM LEADERS",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color=style["text_secondary"])
        for y, line in zip(y_off_lines, render_offensive(off_dict)):
            ax_leaders.text(x_center, y, line, ha="center", va="center",
                            fontsize=9.5, color=style["text_secondary"])

        ax_leaders.text(x_center, y_def_label, "DEFENSIVE TEAM LEADERS",
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color=style["text_secondary"])
        dlines = render_defensive(def_dict)
        if not dlines:
            ax_leaders.text(x_center, y_def_lines[0], "N/A", ha="center", va="center",
                            fontsize=9.5, color=style["text_muted"])
        else:
            for y, line in zip(y_def_lines, dlines):
                ax_leaders.text(x_center, y, line, ha="center", va="center",
                                fontsize=9.5, color=style["text_secondary"])

    render_team_column(0.25, away["name"], away_off, away_def)
    render_team_column(0.75, home["name"], home_off, home_def)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_poster_for_game(game_id: str, out_dir: str) -> Tuple[bool, str]:
    try:
        summary = fetch_summary(game_id)
        meta = extract_game_meta(summary, game_id)

        offensive_leaders = extract_all_team_leaders(summary)
        defensive_leaders = extract_all_defensive_leaders(summary)
        yardage = extract_team_yardage(summary)

        out_path = os.path.join(out_dir, f"game_{game_id}_poster.png")
        make_poster_style_image(meta, offensive_leaders, defensive_leaders, yardage, out_path)
        return True, out_path
    except Exception as e:
        return False, f"{game_id}: {e}"


# ---------------------- MAIN ----------------------

def main():
    parser = argparse.ArgumentParser(description="Generate NFL posters for any ESPN week.")
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g., 2025)")
    parser.add_argument("--week", type=int, required=True, help="Week number (e.g., 13)")
    parser.add_argument("--seasontype", type=int, default=2, help="1=preseason, 2=regular, 3=postseason")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of games (0=all)")
    args = parser.parse_args()

    url = scoreboard_url(args.year, args.week, args.seasontype)
    print(f"Fetching scoreboard: {url}")
    html = fetch_url(url)
    game_ids = extract_game_ids_from_scoreboard_html(html)

    if args.limit and args.limit > 0:
        game_ids = game_ids[: args.limit]

    if not game_ids:
        print("No gameIds found. ESPN page format may have changed.")
        sys.exit(1)


def generate_week(year: int, week: int, seasontype: int = 2, limit: int = 0) -> str:
    """
    Wrapper used by FastAPI.
    Generates posters for a given week and returns the output folder path.
    """
    url = scoreboard_url(year, week, seasontype)
    html = fetch_url(url)
    game_ids = extract_game_ids_from_scoreboard_html(html)

    if limit and limit > 0:
        game_ids = game_ids[:limit]

    if not game_ids:
        raise RuntimeError("No gameIds found. ESPN page format may have changed.")

    kind = "regular" if seasontype == 2 else "playoffs" 
    out_dir = os.path.join("game_visuals", str(year), kind, f"week{str(week).zfill(2)}")

    # ALWAYS start clean so we don't upload leftover PNGs from a previous run
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    for gid in game_ids:
        generate_poster_for_game(gid, out_dir)

    return out_dir



if __name__ == "__main__":
    main()


