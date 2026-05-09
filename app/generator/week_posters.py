import os
import shutil
import re
import sys
import argparse
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont


W, H = 1080, 1920

BLUE = (128, 183, 255)
BG = (10, 14, 24)
CARD = (24, 29, 42)
BORDER = (64, 74, 98)
WHITE = (246, 248, 252)
MUTED = (188, 198, 217)
HEADER = (22, 38, 74)


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


def fetch_logo_image(url: Optional[str], size: int = 155) -> Image.Image:
    if not url:
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)

        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
        return canvas
    except Exception:
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))


def scoreboard_url(year: int, week: int, seasontype: int) -> str:
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
    url = f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
    j = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15).json()
    comp = j["header"]["competitions"][0]

    tmp: Dict[str, List[int]] = {}
    max_periods = 0

    for c in comp["competitors"]:
        abbr = c["team"]["abbreviation"]
        ls = c.get("linescores", []) or []
        vals = [_safe_int(q.get("value", q.get("displayValue", 0))) for q in ls]
        tmp[abbr] = vals
        max_periods = max(max_periods, len(vals))

    if max_periods == 0 or any(len(v) == 0 for v in tmp.values()):
        labels, scraped = fetch_scoring_periods_from_boxscore(event_id)
        return labels, scraped

    has_ot = max_periods > 4
    labels = ["1Q", "2Q", "3Q", "4Q"] + (["OT"] if has_ot else [])
    target_len = 5 if has_ot else 4

    out: Dict[str, List[int]] = {}
    for abbr, vals in tmp.items():
        out[abbr] = (vals + [0] * target_len)[:target_len]

    return labels, out


def fetch_scoring_periods_from_boxscore(event_id: str) -> Tuple[List[str], Dict[str, List[int]]]:
    url = f"https://www.espn.com/nfl/boxscore/_/gameId/{event_id}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    text = _strip_html_to_text(r.text)

    m = re.search(r"\bFinal\b\s+((?:(?:\d+|OT)\s+)+)T\b", text)
    if not m:
        return (["1Q", "2Q", "3Q", "4Q"], {})

    raw_labels = m.group(1).strip().split()
    labels = [f"{x}Q" if x.isdigit() else "OT" for x in raw_labels]
    n = len(labels)

    window = text[m.end(): m.end() + 2200]
    row_pat = re.compile(rf"\b([A-Z]{{2,4}})\b\s+((?:\d+\s+){{{n}}}\d+)\b")
    rows = row_pat.findall(window)[:2]

    out: Dict[str, List[int]] = {}
    for abbr, nums_blob in rows:
        nums = [_safe_int(x) for x in nums_blob.strip().split()]
        if len(nums) == n + 1:
            out[abbr] = nums[:n]

    return labels, out


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

        teams.append(
            {
                "name": name,
                "abbr": abbr,
                "record": record,
                "score": score,
                "home_away": home_away,
                "logo_url": logo,
                "quarter_scores": [],
            }
        )

    teams_sorted = sorted(teams, key=lambda t: t["home_away"] != "away")

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
            results[name] = {
                "total_yards": total,
                "rush_yards": rush,
                "pass_yards": passing,
            }

    return results


def load_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


def fit_text(draw, text: str, font, max_width: int) -> str:
    text = str(text)

    if draw.textlength(text, font=font) <= max_width:
        return text

    while len(text) > 3 and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]

    return text.rstrip() + "…"


def wrap_line(draw, text: str, font, max_width: int, max_lines: int = 2) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines = []
    current = ""

    for word in words:
        test = word if not current else current + " " + word

        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit_text(draw, lines[-1], font, max_width)

    return lines


def draw_center(draw, box, text: str, font, fill):
    x1, y1, x2, y2 = box
    text = str(text)

    tw = draw.textlength(text, font=font)
    bb = draw.textbbox((0, 0), text, font=font)
    th = bb[3] - bb[1]

    draw.text(
        (x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2 - 2),
        text,
        font=font,
        fill=fill,
    )


def make_background() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.rectangle((0, 0, W, 190), fill=HEADER)
    d.rectangle((0, 190, W, 199), fill=BLUE)

    for y in range(210, H, 30):
        color = (14, 18, 28) if (y // 30) % 2 == 0 else (12, 16, 26)
        d.rectangle((0, y, W, y + 15), fill=color)

    return img


def leader_offense_lines(off_dict: Dict) -> List[str]:
    p = off_dict.get("passing_leader") or {}
    r = off_dict.get("rushing_leader") or {}
    rc = off_dict.get("receiving_leader") or {}

    lines = []

    if p:
        lines.append(f"PASS: {p.get('name','N/A')} • {p.get('yards',0)} YDS, {p.get('td',0)} TD, {p.get('ints',0)} INT")
    else:
        lines.append("PASS: N/A")

    if r:
        lines.append(f"RUSH: {r.get('name','N/A')} • {r.get('yards',0)} YDS, {r.get('td',0)} TD")
    else:
        lines.append("RUSH: N/A")

    if rc:
        lines.append(f"REC: {rc.get('name','N/A')} • {rc.get('yards',0)} YDS, {rc.get('td',0)} TD")
    else:
        lines.append("REC: N/A")

    return lines


def leader_defense_lines(def_dict: Dict) -> List[str]:
    lines = []

    t = def_dict.get("tackles_leader")
    s = def_dict.get("sacks_leader")
    i = def_dict.get("ints_leader")

    if t:
        lines.append(f"TACKLES: {t['name']} • {t['tackles']}")
    else:
        lines.append("TACKLES: N/A")

    if s:
        lines.append(f"SACKS: {s['name']} • {s['sacks']}")
    else:
        lines.append("SACKS: N/A")

    if i:
        lines.append(f"INT: {i['name']} • {i['ints']}")
    else:
        lines.append("INT: None")

    return lines


def draw_leader_column(draw, x1, y1, x2, title, lines):
    section_font = load_font(23, True)
    line_font = load_font(21, True)

    draw.text((x1, y1), title, font=section_font, fill=WHITE)

    y = y1 + 42
    max_width = x2 - x1

    for line in lines:
        wrapped = wrap_line(draw, line, line_font, max_width, max_lines=2)

        for wrapped_line in wrapped:
            draw.text((x1, y), wrapped_line, font=line_font, fill=WHITE)
            y += 26

        y += 8


def draw_team_leader_card(draw, x1, y1, x2, y2, team_name, off_dict, def_dict):
    team_font = load_font(33, True)

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=26,
        fill=CARD,
        outline=BORDER,
        width=3,
    )

    team_text = fit_text(draw, str(team_name).upper(), team_font, x2 - x1 - 36)
    draw.text((x1 + 18, y1 + 18), team_text, font=team_font, fill=BLUE)

    mid = x1 + (x2 - x1) // 2
    top = y1 + 84
    bottom = y2 - 24

    draw.line((mid, top, mid, bottom), fill=BORDER, width=2)

    draw_leader_column(
        draw,
        x1 + 24,
        top,
        mid - 24,
        "OFFENSE",
        leader_offense_lines(off_dict),
    )

    draw_leader_column(
        draw,
        mid + 24,
        top,
        x2 - 24,
        "DEFENSE",
        leader_defense_lines(def_dict),
    )


def stat_line(draw, x, y, label, value, label_font, value_font):
    draw.text((x, y), label, font=label_font, fill=MUTED)
    draw.text((x, y + 30), value, font=value_font, fill=WHITE)


def make_poster_style_image(
    meta: Dict,
    offensive_leaders: Dict[str, Dict],
    defensive_leaders: Dict[str, Dict],
    yardage: Dict[str, Dict],
    output_path: str,
    style: Dict = None,
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

    img = make_background()
    d = ImageDraw.Draw(img)

    title_font = load_font(52, True)
    small_font = load_font(26, False)
    abbr_font = load_font(52, True)
    record_font = load_font(28, True)
    score_font = load_font(90, True)
    at_font = load_font(24, True)
    section_font = load_font(31, True)
    q_font = load_font(31, True)
    q_header_font = load_font(25, True)
    stat_label_font = load_font(22, True)
    stat_value_font = load_font(38, True)

    title = "NFL GAME RECAP"
    subtitle = "FINAL SCORE • GAME RECAP"

    d.text(((W - d.textlength(title, font=title_font)) / 2, 28), title, font=title_font, fill=WHITE)
    d.text(((W - d.textlength(subtitle, font=small_font)) / 2, 105), subtitle, font=small_font, fill=(208, 218, 238))

    x0, y0, x1, y1 = 42, 230, W - 42, 555

    d.rounded_rectangle((x0, y0, x1, y1), radius=32, fill=CARD, outline=BORDER, width=3)

    away_logo = fetch_logo_image(away.get("logo_url"), 155)
    home_logo = fetch_logo_image(home.get("logo_url"), 155)

    img.paste(away_logo, (x0 + 36, y0 + 36), away_logo)
    img.paste(home_logo, (x1 - 191, y0 + 36), home_logo)

    draw_center(d, (x0 + 28, y0 + 190, x0 + 185, y0 + 250), away.get("abbr") or "", abbr_font, WHITE)
    draw_center(d, (x1 - 185, y0 + 190, x1 - 28, y0 + 250), home.get("abbr") or "", abbr_font, WHITE)

    draw_center(d, (x0 + 28, y0 + 248, x0 + 185, y0 + 294), away.get("record") or "", record_font, MUTED)
    draw_center(d, (x1 - 185, y0 + 248, x1 - 28, y0 + 294), home.get("record") or "", record_font, MUTED)

    left_score = away.get("score") if completed and away.get("score") is not None else "–"
    right_score = home.get("score") if completed and home.get("score") is not None else "–"

    d.text((368, y0 + 92), str(left_score), font=score_font, fill=WHITE)
    draw_center(d, (498, y0 + 127, 582, y0 + 175), "AT", at_font, MUTED)
    d.text((600, y0 + 92), str(right_score), font=score_font, fill=WHITE)

    qx0, qy0, qx1, qy1 = 42, 590, W - 42, 850

    d.rounded_rectangle((qx0, qy0, qx1, qy1), radius=28, fill=CARD, outline=BORDER, width=3)
    draw_center(d, (qx0, qy0 + 18, qx1, qy0 + 70), "SCORING BY QUARTER", section_font, BLUE)

    period_labels = meta.get("period_labels", ["1Q", "2Q", "3Q", "4Q"])
    labels = ["TEAM"] + period_labels

    start_x = 145
    end_x = 890
    step = (end_x - start_x) / max(1, len(labels) - 1)
    col_x = [int(start_x + i * step) for i in range(len(labels))]

    for label, x in zip(labels, col_x):
        draw_center(d, (x - 55, qy0 + 92, x + 55, qy0 + 130), label, q_header_font, WHITE)

    rows = [
        (away.get("abbr") or "", away.get("quarter_scores") or []),
        (home.get("abbr") or "", home.get("quarter_scores") or []),
    ]

    for r_i, (abbr, scores) in enumerate(rows):
        y = qy0 + 148 + r_i * 62
        scores = (scores + [0] * len(period_labels))[:len(period_labels)]

        draw_center(d, (col_x[0] - 55, y, col_x[0] + 55, y + 42), abbr, q_font, BLUE)

        for score, x in zip(scores, col_x[1:]):
            draw_center(d, (x - 55, y, x + 55, y + 42), str(score), q_font, WHITE)

    sx0, sy0, sx1, sy1 = 42, 890, W - 42, 1090

    d.rounded_rectangle((sx0, sy0, sx1, sy1), radius=28, fill=CARD, outline=BORDER, width=3)
    draw_center(d, (sx0, sy0 + 20, sx1, sy0 + 70), "TEAM YARDAGE", section_font, BLUE)

    away_total = safe_int(away_yards.get("total_yards"), 0)
    home_total = safe_int(home_yards.get("total_yards"), 0)
    diff = home_total - away_total
    diff_text = f"+{diff}" if diff >= 0 else str(diff)

    stat_line(d, sx0 + 80, sy0 + 95, f"{away.get('abbr')} TOTAL", f"{away_total}", stat_label_font, stat_value_font)
    stat_line(d, sx0 + 380, sy0 + 95, f"{home.get('abbr')} TOTAL", f"{home_total}", stat_label_font, stat_value_font)
    stat_line(d, sx0 + 670, sy0 + 95, "DIFFERENCE", diff_text, stat_label_font, stat_value_font)

    draw_center(d, (42, 1120, W - 42, 1170), "TEAM LEADERS", section_font, WHITE)

    draw_team_leader_card(d, 42, 1190, W - 42, 1518, away_name, away_off, away_def)
    draw_team_leader_card(d, 42, 1545, W - 42, 1878, home_name, home_off, home_def)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")


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


def no_poster_message(year: int, week: int) -> str:
    return f"No poster available yet for {year} week {week}. Please try another week or season type"


def generate_week(year: int, week: int, seasontype: int = 2, limit: int = 0) -> str:
    url = scoreboard_url(year, week, seasontype)
    html = fetch_url(url)
    game_ids = extract_game_ids_from_scoreboard_html(html)

    if limit and limit > 0:
        game_ids = game_ids[:limit]

    if not game_ids:
        raise RuntimeError(no_poster_message(year, week))

    summaries: Dict[str, Dict] = {}

    for gid in game_ids:
        summary = fetch_summary(gid)
        summaries[gid] = summary
        meta = extract_game_meta(summary, gid)

        if not meta.get("completed", False):
            raise RuntimeError(no_poster_message(year, week))

    kind = "regular" if seasontype == 2 else "playoffs"
    out_dir = os.path.join("game_visuals", str(year), kind, f"week{str(week).zfill(2)}")

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    for gid in game_ids:
        summary = summaries[gid]

        offensive_leaders = extract_all_team_leaders(summary)
        defensive_leaders = extract_all_defensive_leaders(summary)
        yardage = extract_team_yardage(summary)
        meta = extract_game_meta(summary, gid)

        out_path = os.path.join(out_dir, f"game_{gid}_poster.png")
        make_poster_style_image(meta, offensive_leaders, defensive_leaders, yardage, out_path)

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate NFL posters for any ESPN week.")
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2025")
    parser.add_argument("--week", type=int, required=True, help="Week number, e.g. 13")
    parser.add_argument("--seasontype", type=int, default=2, help="1=preseason, 2=regular, 3=postseason")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of games, 0=all")
    args = parser.parse_args()

    try:
        out_dir = generate_week(
            year=args.year,
            week=args.week,
            seasontype=args.seasontype,
            limit=args.limit,
        )
        print(f"Posters generated in: {out_dir}")
    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
