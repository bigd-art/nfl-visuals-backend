#!/usr/bin/env python3
import os
import re
import sys
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

SOURCE_URL = "https://www.tankathon.com/nfl/mock_draft"
OUTPUT_DIR = "tankathon_mock_posters"

POSTER_WIDTH = 1600
POSTER_HEIGHT = 2000
ROWS_PER_POSTER = 8
TOTAL_POSTERS = 4

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

TEAM_LOGO_SLUGS = {
    "ARI": "ari",
    "ATL": "atl",
    "BAL": "bal",
    "BUF": "buf",
    "CAR": "car",
    "CHI": "chi",
    "CIN": "cin",
    "CLE": "cle",
    "DAL": "dal",
    "DEN": "den",
    "DET": "det",
    "GB": "gb",
    "HOU": "hou",
    "IND": "ind",
    "JAX": "jax",
    "KC": "kc",
    "LV": "lv",
    "LAC": "lac",
    "LAR": "lar",
    "MIA": "mia",
    "MIN": "min",
    "NE": "ne",
    "NO": "no",
    "NYG": "nyg",
    "NYJ": "nyj",
    "PHI": "phi",
    "PIT": "pit",
    "SEA": "sea",
    "SF": "sf",
    "TB": "tb",
    "TEN": "ten",
    "WSH": "wsh",
}

TEAM_FIXES = {
    "JAC": "JAX",
    "WAS": "WSH",
    "ARZ": "ARI",
}

POSITION_TOKENS = [
    "LB/EDGE",
    "IOL",
    "EDGE",
    "QB", "RB", "WR", "TE", "OT", "OG", "C",
    "DE", "DT", "DL", "LB", "OLB", "ILB",
    "CB", "FS", "SS", "S", "DB",
]

POSITION_PATTERN = "|".join(sorted([re.escape(x) for x in POSITION_TOKENS], key=len, reverse=True))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass

    return ImageFont.load_default()


FONT_TITLE = load_font(74, bold=True)
FONT_SUBTITLE = load_font(30, bold=False)
FONT_PICK = load_font(44, bold=True)
FONT_TEAM = load_font(32, bold=True)
FONT_PLAYER = load_font(42, bold=True)
FONT_META = load_font(28, bold=False)


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def normalize_team_abbr(team: str) -> str:
    team = (team or "").strip().upper()
    return TEAM_FIXES.get(team, team)


def get_logo_url(team_abbr: str) -> str:
    team_abbr = normalize_team_abbr(team_abbr)
    slug = TEAM_LOGO_SLUGS.get(team_abbr)
    if not slug:
        return ""
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"


def fetch_logo(team_abbr: str, size: int = 116) -> Image.Image:
    url = get_logo_url(team_abbr)
    if url:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
            img.thumbnail((size, size), Image.LANCZOS)
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            x = (size - img.width) // 2
            y = (size - img.height) // 2
            canvas.paste(img, (x, y), img)
            return canvas
        except Exception:
            pass

    canvas = Image.new("RGBA", (size, size), (30, 34, 48, 255))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((2, 2, size - 2, size - 2), radius=20, outline=(120, 130, 160), width=3)
    f = load_font(26, bold=True)
    text = team_abbr if team_abbr else "NFL"
    tw = d.textlength(text, font=f)
    d.text(((size - tw) / 2, size / 2 - 12), text, fill=(235, 240, 250), font=f)
    return canvas


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def soup_to_text_with_img_alts(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for img in soup.find_all("img"):
        alt = clean_text(img.get("alt", ""))
        if alt:
            img.replace_with(f"\nImage: {alt}\n")
        else:
            img.replace_with("\n")

    text = soup.get_text("\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_round1_text(full_text: str) -> str:
    m = re.search(r"\bRound 1\b(.*?)(?:\bRound 2\b|$)", full_text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError("Could not find Round 1 section.")
    return m.group(1)


def parse_pick_block(block_text: str):
    team_matches = re.findall(r"Image:\s*([A-Z]{2,3})\b", block_text)
    team = ""
    for t in team_matches:
        t = normalize_team_abbr(t)
        if t in TEAM_LOGO_SLUGS:
            team = t
            break

    player_match = re.search(
        rf"\b([A-Z][A-Za-z\.'\-]+(?:\s+[A-Z][A-Za-z\.'\-]+)*)\s+({POSITION_PATTERN})\s*\|\s*([^\n]+)",
        block_text
    )

    if not player_match:
        return None

    player = clean_text(player_match.group(1))
    position = clean_text(player_match.group(2))
    college = clean_text(player_match.group(3))

    college = re.sub(r"\s+$", "", college)
    college = re.sub(r"\s{2,}.*$", "", college)

    return {
        "team": team,
        "player": player,
        "position": position,
        "college": college,
    }


def parse_round1_picks(round1_text: str):
    pattern = re.compile(r"(?m)^\s*(\d{1,2})\s*$")
    matches = list(pattern.finditer(round1_text))

    picks = []

    for idx, match in enumerate(matches):
        pick_num = int(match.group(1))
        if not (1 <= pick_num <= 32):
            continue

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(round1_text)
        block = round1_text[start:end]

        parsed = parse_pick_block(block)
        if parsed:
            parsed["pick"] = pick_num
            picks.append(parsed)

    deduped = {}
    for p in picks:
        deduped[p["pick"]] = p

    out = [deduped[k] for k in sorted(deduped.keys()) if 1 <= k <= 32]

    if len(out) != 32:
        debug_path = os.path.join(OUTPUT_DIR, "round1_debug.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(round1_text)
        print(f"DEBUG WRITTEN: {debug_path}")
        for p in out:
            print(f"#{p['pick']} {p['team']} - {p['player']} ({p['position']}, {p['college']})")
        raise RuntimeError(f"Expected 32 picks, found {len(out)}.")

    return out


def rounded_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(draw, text, font, max_width, max_lines=2):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        test = current + " " + word
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_width:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"

    return lines


def make_background():
    img = Image.new("RGB", (POSTER_WIDTH, POSTER_HEIGHT), (10, 14, 24))
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, POSTER_WIDTH, 165), fill=(22, 38, 74))
    draw.rectangle((0, 165, POSTER_WIDTH, 172), fill=(128, 183, 255))

    for y in range(180, POSTER_HEIGHT, 30):
        color = (14, 18, 28) if (y // 30) % 2 == 0 else (12, 16, 26)
        draw.rectangle((0, y, POSTER_WIDTH, y + 15), fill=color)

    return img


def draw_card(base_img, draw, item, box):
    x1, y1, x2, y2 = box
    rounded_rect(draw, box, radius=28, fill=(24, 29, 42), outline=(64, 74, 98), width=3)

    pill = (x1 + 22, y1 + 20, x1 + 160, y1 + 78)
    rounded_rect(draw, pill, radius=18, fill=(128, 183, 255))
    pick_text = f"#{item['pick']}"
    tw = draw.textlength(pick_text, font=FONT_PICK)
    draw.text((pill[0] + (pill[2] - pill[0] - tw) / 2, pill[1] + 7), pick_text, font=FONT_PICK, fill=(15, 20, 28))

    team_text = item["team"] if item["team"] else "NFL"
    draw.text((x1 + 186, y1 + 28), team_text, font=FONT_TEAM, fill=(223, 231, 244))

    logo = fetch_logo(item["team"], size=116)
    logo_x = x2 - 150
    logo_y = y1 + (y2 - y1 - 116) // 2
    base_img.paste(logo, (logo_x, logo_y), logo)

    text_left = x1 + 28
    text_right = logo_x - 28
    max_width = text_right - text_left

    player_lines = wrap_text(draw, item["player"], FONT_PLAYER, max_width, max_lines=2)
    py = y1 + 92
    for line in player_lines:
        draw.text((text_left, py), line, font=FONT_PLAYER, fill=(246, 248, 252))
        py += 46

    meta = f"{item['position']}  •  {item['college']}"
    meta_lines = wrap_text(draw, meta, FONT_META, max_width, max_lines=2)
    for line in meta_lines:
        draw.text((text_left, py + 2), line, font=FONT_META, fill=(188, 198, 217))
        py += 32


def render_poster(chunk):
    img = make_background()
    draw = ImageDraw.Draw(img)

    title = "NFL MOCK DRAFT"
    subtitle = f"PICKS {chunk[0]['pick']}-{chunk[-1]['pick']}"

    tw = draw.textlength(title, font=FONT_TITLE)
    sw = draw.textlength(subtitle, font=FONT_SUBTITLE)

    draw.text(((POSTER_WIDTH - tw) / 2, 36), title, font=FONT_TITLE, fill=(245, 247, 252))
    draw.text(((POSTER_WIDTH - sw) / 2, 114), subtitle, font=FONT_SUBTITLE, fill=(208, 218, 238))

    left = 70
    right = POSTER_WIDTH - 70
    top = 215
    bottom = POSTER_HEIGHT - 40
    gap = 18

    usable_h = bottom - top
    row_h = int((usable_h - gap * (ROWS_PER_POSTER - 1)) / ROWS_PER_POSTER)

    for idx, item in enumerate(chunk):
        y1 = top + idx * (row_h + gap)
        y2 = y1 + row_h
        draw_card(img, draw, item, (left, y1, right, y2))

    return img


def main():
    ensure_dir(OUTPUT_DIR)

    print("Fetching Tankathon mock draft...")
    html = fetch_html(SOURCE_URL)

    print("Extracting text...")
    full_text = soup_to_text_with_img_alts(html)

    print("Extracting Round 1 section...")
    round1_text = extract_round1_text(full_text)

    print("Parsing picks 1-32...")
    picks = parse_round1_picks(round1_text)

    print(f"Found {len(picks)} picks.")
    for p in picks:
        print(f"#{p['pick']} {p['team']} - {p['player']} ({p['position']}, {p['college']})")

    chunks = [picks[i:i + ROWS_PER_POSTER] for i in range(0, 32, ROWS_PER_POSTER)]
    if len(chunks) != 4:
        raise RuntimeError(f"Expected 4 posters, got {len(chunks)}")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"Rendering poster {idx}/4...")
        poster = render_poster(chunk)
        out_path = os.path.join(OUTPUT_DIR, f"tankathon_mock_poster_{idx}.png")
        poster.save(out_path)
        print(f"Saved: {out_path}")

    print("\nDone.")
    print(f"Open folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
