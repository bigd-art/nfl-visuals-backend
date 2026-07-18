#!/usr/bin/env python3
import argparse
import io
import os
import re
import shutil
from typing import Dict, List, Tuple, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

TEAM_NEEDS = {
    "ARI": ["QB", "RB", "G", "T"],
    "ATL": ["WR", "TE", "DI", "CB"],
    "BAL": ["WR", "G", "DL"],
    "BUF": ["WR", "ED", "LB", "DB"],
    "CAR": ["WR", "DL", "LB"],
    "CHI": ["DL", "LB", "S"],
    "CIN": ["C", "G", "DI", "S"],
    "CLE": ["QB", "WR", "G", "T"],
    "DAL": ["RB", "ED", "LB", "S"],
    "DEN": ["RB", "TE", "C", "LB"],
    "DET": ["C", "T", "ED", "CB"],
    "GB": ["T", "DI", "CB"],
    "HOU": ["RB", "C", "G", "DI"],
    "IND": ["QB", "DI", "LB", "S"],
    "JAX": ["C", "G", "DI", "S"],
    "KC": ["RB", "WR", "G", "T", "DL"],
    "LV": ["QB", "WR", "G", "T", "DI", "LB", "CB"],
    "LAC": ["G", "C", "DL"],
    "LAR": ["QB", "T", "CB"],
    "MIA": ["QB", "WR", "G", "CB"],
    "MIN": ["RB", "C", "DB"],
    "NE": ["G", "T", "ED", "LB"],
    "NO": ["WR", "G", "DL"],
    "NYG": ["WR", "G", "T", "CB"],
    "NYJ": ["QB", "WR", "DI", "LB", "CB"],
    "PHI": ["TE", "G", "ED", "CB"],
    "PIT": ["QB", "WR", "T", "DB"],
    "SEA": ["RB", "C", "G", "LB", "CB"],
    "SF": ["WR", "G", "ED", "S"],
    "TB": ["TE", "G", "ED", "LB", "CB"],
    "TEN": ["RB", "WR", "C", "G", "ED", "CB"],
    "WSH": ["TE", "G", "ED", "LB", "DB"],
}

TEAM_META = {
    "ARI": ("22", "ari", "Arizona Cardinals"),
    "ATL": ("1", "atl", "Atlanta Falcons"),
    "BAL": ("33", "bal", "Baltimore Ravens"),
    "BUF": ("2", "buf", "Buffalo Bills"),
    "CAR": ("29", "car", "Carolina Panthers"),
    "CHI": ("3", "chi", "Chicago Bears"),
    "CIN": ("4", "cin", "Cincinnati Bengals"),
    "CLE": ("5", "cle", "Cleveland Browns"),
    "DAL": ("6", "dal", "Dallas Cowboys"),
    "DEN": ("7", "den", "Denver Broncos"),
    "DET": ("8", "det", "Detroit Lions"),
    "GB": ("9", "gb", "Green Bay Packers"),
    "HOU": ("34", "hou", "Houston Texans"),
    "IND": ("11", "ind", "Indianapolis Colts"),
    "JAX": ("30", "jax", "Jacksonville Jaguars"),
    "KC": ("12", "kc", "Kansas City Chiefs"),
    "LV": ("13", "lv", "Las Vegas Raiders"),
    "LAC": ("24", "lac", "Los Angeles Chargers"),
    "LAR": ("14", "lar", "Los Angeles Rams"),
    "MIA": ("15", "mia", "Miami Dolphins"),
    "MIN": ("16", "min", "Minnesota Vikings"),
    "NE": ("17", "ne", "New England Patriots"),
    "NO": ("18", "no", "New Orleans Saints"),
    "NYG": ("19", "nyg", "New York Giants"),
    "NYJ": ("20", "nyj", "New York Jets"),
    "PHI": ("21", "phi", "Philadelphia Eagles"),
    "PIT": ("23", "pit", "Pittsburgh Steelers"),
    "SEA": ("26", "sea", "Seattle Seahawks"),
    "SF": ("25", "sf", "San Francisco 49ers"),
    "TB": ("27", "tb", "Tampa Bay Buccaneers"),
    "TEN": ("10", "ten", "Tennessee Titans"),
    "WSH": ("28", "wsh", "Washington Commanders"),
}

POSITION_MAP = {
    "QB": ["QB"],
    "RB": ["RB", "HB", "FB"],
    "WR": ["WR"],
    "TE": ["TE"],
    "C": ["C"],
    "G": ["G", "OG", "LG", "RG"],
    "T": ["T", "OT", "LT", "RT"],
    "ED": ["DE", "EDGE", "OLB"],
    "DI": ["DT", "NT", "DL"],
    "DL": ["DE", "DT", "NT", "DL", "EDGE"],
    "LB": ["LB", "ILB", "OLB", "MLB"],
    "CB": ["CB"],
    "S": ["S", "FS", "SS"],
    "DB": ["CB", "S", "FS", "SS", "DB"],
}

TEAM_COLORS = {
    "ARI": ("#97233F", "#000000"),
    "ATL": ("#A71930", "#000000"),
    "BAL": ("#241773", "#000000"),
    "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#101820"),
    "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"),
    "CLE": ("#311D00", "#FF3C00"),
    "DAL": ("#041E42", "#869397"),
    "DEN": ("#FB4F14", "#002244"),
    "DET": ("#0076B6", "#B0B7BC"),
    "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"),
    "IND": ("#002C5F", "#A2AAAD"),
    "JAX": ("#006778", "#101820"),
    "KC": ("#E31837", "#FFB81C"),
    "LV": ("#000000", "#A5ACAF"),
    "LAC": ("#0080C6", "#FFC20E"),
    "LAR": ("#003594", "#FFA300"),
    "MIA": ("#008E97", "#FC4C02"),
    "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"),
    "NO": ("#101820", "#D3BC8D"),
    "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#000000"),
    "PHI": ("#004C54", "#A5ACAF"),
    "PIT": ("#101820", "#FFB612"),
    "SEA": ("#002244", "#69BE28"),
    "SF": ("#AA0000", "#B3995D"),
    "TB": ("#D50A0A", "#34302B"),
    "TEN": ("#0C2340", "#4B92DB"),
    "WSH": ("#5A1414", "#FFB612"),
}

ALIASES = {
    "WAS": "WSH",
}


def load_font(size: int, bold: bool = False):
    paths = [
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/Library/Fonts/Arial Bold.ttf"
            if bold
            else "/Library/Fonts/Arial.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]

    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    return ImageFont.load_default()


def clean_text(value) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).replace("\xa0", " ").strip(),
    )


def normalize_name(name: str) -> str:
    name = clean_text(name).lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def fetch_json(url: str) -> Dict:
    last_error = None

    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=30,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Bad status code: {resp.status_code}"
                )

            return resp.json()

        except Exception as e:
            last_error = e
            print(f"Retry {attempt + 1}/3 failed for {url}")

    raise RuntimeError(
        f"Failed API request: {url}\n{last_error}"
    )


def fetch_image(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()

        return Image.open(
            io.BytesIO(resp.content)
        ).convert("RGBA")

    except Exception:
        return None


def get_logo(team: str) -> Optional[Image.Image]:
    _, slug, _ = TEAM_META[team]

    return fetch_image(
        f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
    )


def normalize_position(pos: str) -> str:
    p = clean_text(pos).upper()

    if p == "QB":
        return "QB"

    if p in {"RB", "HB", "FB"}:
        return "RB"

    if p == "WR":
        return "WR"

    if p == "TE":
        return "TE"

    if p in {"G", "OG", "LG", "RG"}:
        return "G"

    if p in {"T", "OT", "LT", "RT"}:
        return "T"

    if p == "C":
        return "C"

    if p in {"DE", "EDGE", "LDE", "RDE"}:
        return "DE"

    if p in {"DT", "NT", "DL", "LDT", "RDT"}:
        return "DT"

    if p in {
        "LB",
        "ILB",
        "OLB",
        "MLB",
        "WLB",
        "SLB",
    }:
        return "LB"

    if p in {"CB", "LCB", "RCB", "NB", "DB"}:
        return "CB"

    if p in {"S", "FS", "SS"}:
        return "S"

    return p


def fetch_roster_json(team: str) -> Dict:
    team_id, _, _ = TEAM_META[team]

    return fetch_json(
        "https://site.web.api.espn.com/apis/common/v3/"
        f"sports/football/nfl/teams/{team_id}/roster"
    )


def fetch_depthchart_json(team: str) -> Dict:
    team_id, _, _ = TEAM_META[team]

    return fetch_json(
        "https://site.api.espn.com/apis/site/v2/"
        f"sports/football/nfl/teams/{team_id}/depthcharts"
    )


def parse_player(raw: Dict) -> Optional[Dict]:
    name = clean_text(
        raw.get("displayName")
        or raw.get("fullName")
        or raw.get("shortName")
        or raw.get("name")
        or ""
    )

    pos_obj = raw.get("position") or {}

    if isinstance(pos_obj, dict):
        pos = clean_text(
            pos_obj.get("abbreviation")
            or pos_obj.get("name")
            or ""
        )
    else:
        pos = clean_text(pos_obj)

    pos = normalize_position(pos)

    if not name or not pos:
        return None

    return {
        "name": name,
        "name_key": normalize_name(name),
        "position": pos,
    }


def depth_position_from_position_data(
    position_data: Dict,
) -> str:
    position_obj = position_data.get("position", {}) or {}
    parent_obj = position_obj.get("parent", {}) or {}

    own_abbr = clean_text(
        position_obj.get("abbreviation", "")
    )

    parent_abbr = clean_text(
        parent_obj.get("abbreviation", "")
    )

    if own_abbr:
        normalized_own = normalize_position(own_abbr)

        if normalized_own not in {"OFF", "DEF", "ST"}:
            return normalized_own

    if parent_abbr:
        normalized_parent = normalize_position(parent_abbr)

        if normalized_parent not in {"OFF", "DEF", "ST"}:
            return normalized_parent

    return normalize_position(
        own_abbr or parent_abbr
    )


def parse_roster_players(team: str) -> List[Dict]:
    data = fetch_roster_json(team)
    players: List[Dict] = []
    seen = set()

    for group in data.get("positionGroups", []):
        for raw in group.get("athletes", []) or []:
            parsed = parse_player(raw)

            if not parsed:
                continue

            key = parsed["name_key"]

            if key in seen:
                continue

            seen.add(key)
            players.append(parsed)

    if not players:
        raise RuntimeError(
            f"No roster players parsed for {team}."
        )

    return players


def parse_depthchart_order(team: str) -> List[Dict]:
    data = fetch_depthchart_json(team)
    depthchart = data.get("depthchart", []) or []

    ordered_players: List[Dict] = []
    seen = set()

    for group in depthchart:
        positions = group.get("positions", {})

        if not isinstance(positions, dict):
            continue

        for position_key, position_data in positions.items():
            if not isinstance(position_data, dict):
                continue

            display_pos = depth_position_from_position_data(
                position_data
            )

            athletes = position_data.get("athletes", [])

            if not isinstance(athletes, list):
                continue

            for depth_rank, athlete in enumerate(
                athletes,
                start=1,
            ):
                if not isinstance(athlete, dict):
                    continue

                name = clean_text(
                    athlete.get("displayName")
                    or athlete.get("fullName")
                    or athlete.get("shortName")
                    or athlete.get("name")
                    or ""
                )

                if not name:
                    continue

                name_key = normalize_name(name)

                if name_key in seen:
                    continue

                seen.add(name_key)

                ordered_players.append(
                    {
                        "name": name,
                        "name_key": name_key,
                        "position": display_pos,
                        "depth_position_key": clean_text(
                            position_key
                        ).upper(),
                        "depth_rank": depth_rank,
                    }
                )

    return ordered_players


def order_roster_by_depthchart(
    roster_players: List[Dict],
    depth_players: List[Dict],
) -> List[Dict]:
    roster_by_name = {
        player["name_key"]: player
        for player in roster_players
    }

    ordered: List[Dict] = []
    used = set()

    for depth_player in depth_players:
        name_key = depth_player["name_key"]

        if name_key not in roster_by_name:
            continue

        player = roster_by_name[name_key].copy()

        depth_position = depth_player.get("position")

        if depth_position:
            player["position"] = depth_position

        player["depth_position_key"] = (
            depth_player.get("depth_position_key", "")
        )

        player["depth_rank"] = depth_player.get(
            "depth_rank",
            "",
        )

        ordered.append(player)
        used.add(name_key)

    for player in roster_players:
        if player["name_key"] not in used:
            ordered.append(player)

    return ordered


def get_players(team: str) -> List[Dict]:
    roster_players = parse_roster_players(team)

    try:
        depth_players = parse_depthchart_order(team)

        if depth_players:
            ordered_players = order_roster_by_depthchart(
                roster_players,
                depth_players,
            )

            print(
                f"{team}: Using ESPN depth chart ordering "
                f"for {len(depth_players)} players."
            )

            return ordered_players

        print(
            f"WARNING: No depth chart rows found for {team}. "
            "Using roster API order."
        )

    except Exception as e:
        print(
            f"WARNING: Depth chart request failed for {team}: {e}"
        )
        print("Using roster API order.")

    return roster_players


def players_for_position(
    pos: str,
    roster: List[Dict],
    max_players: int = 8,
) -> List[str]:
    allowed_positions = {
        normalize_position(mapped_pos)
        for mapped_pos in POSITION_MAP.get(pos, [pos])
    }

    result: List[str] = []

    for player in roster:
        player_position = normalize_position(
            player.get("position", "")
        )

        if player_position not in allowed_positions:
            continue

        name = player.get("name", "")

        if name and name not in result:
            result.append(name)

        if len(result) >= max_players:
            break

    return result


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
) -> Tuple[int, int]:
    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    )


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> str:
    text = str(text)

    if draw.textlength(text, font=font) <= max_width:
        return text

    while (
        len(text) > 3
        and draw.textlength(
            text + "…",
            font=font,
        ) > max_width
    ):
        text = text[:-1]

    return text.rstrip() + "…"


def draw_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font,
    fill,
    canvas_width: int,
) -> int:
    width, height = text_size(
        draw,
        text,
        font,
    )

    x = (canvas_width - width) // 2

    draw.text(
        (x, y),
        text,
        fill=fill,
        font=font,
    )

    return y + height


def hex_to_rgb(
    hex_color: str,
) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")

    return tuple(
        int(hex_color[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def make_gradient(
    width: int,
    height: int,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int],
) -> Image.Image:
    img = Image.new(
        "RGB",
        (width, height),
        top_color,
    )

    pixels = img.load()

    for y in range(height):
        ratio = y / max(1, height - 1)

        red = int(
            top_color[0] * (1 - ratio)
            + bottom_color[0] * ratio
        )

        green = int(
            top_color[1] * (1 - ratio)
            + bottom_color[1] * ratio
        )

        blue = int(
            top_color[2] * (1 - ratio)
            + bottom_color[2] * ratio
        )

        for x in range(width):
            pixels[x, y] = (
                red,
                green,
                blue,
            )

    return img


def draw_need_block(
    draw: ImageDraw.ImageDraw,
    pos: str,
    players: List[str],
    x: int,
    y: int,
    width: int,
    accent: str,
    pos_font,
    player_font,
    small_font,
) -> int:
    draw.text(
        (x, y),
        pos,
        fill=accent,
        font=pos_font,
    )

    label = "POSITION GROUP NEED"

    label_width, _ = text_size(
        draw,
        label,
        small_font,
    )

    draw.text(
        (
            x + width - label_width,
            y + 8,
        ),
        label,
        fill="#555555",
        font=small_font,
    )

    y += 52

    if not players:
        draw.text(
            (x + 18, y),
            "1 - No players found",
            fill="black",
            font=player_font,
        )

        return y + 42

    for index, player in enumerate(players, 1):
        line = f"{index} - {player}"

        line = fit_text(
            draw,
            line,
            player_font,
            width - 24,
        )

        draw.text(
            (x + 18, y),
            line,
            fill="black",
            font=player_font,
        )

        y += 38

    return y


def poster(
    team: str,
    out_file: Optional[str] = None,
):
    team = ALIASES.get(team, team)

    if team not in TEAM_META:
        raise ValueError(f"Invalid team: {team}")

    roster = get_players(team)

    _, _, name = TEAM_META[team]
    primary_hex, accent_hex = TEAM_COLORS[team]

    width = 1600
    height = 2000

    img = make_gradient(
        width,
        height,
        hex_to_rgb(primary_hex),
        (10, 10, 10),
    ).convert("RGBA")

    draw = ImageDraw.Draw(img)

    title_font = load_font(82, True)
    team_font = load_font(56, True)
    pos_font = load_font(38, True)
    player_font = load_font(27, False)
    small_font = load_font(19, True)
    footer_font = load_font(22, False)

    y = 65

    logo = get_logo(team)

    if logo:
        logo.thumbnail(
            (215, 215),
            Image.LANCZOS,
        )

        logo_x = (width - logo.width) // 2

        img.alpha_composite(
            logo,
            (logo_x, y),
        )

        y += logo.height + 28

    y = draw_centered(
        draw,
        name.upper(),
        y,
        team_font,
        "white",
        width,
    ) + 18

    y = draw_centered(
        draw,
        "TEAM NEEDS BOARD",
        y,
        title_font,
        "white",
        width,
    ) + 40

    panel_x1 = 105
    panel_y1 = y
    panel_x2 = width - 105
    panel_y2 = height - 120

    draw.rounded_rectangle(
        (
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y2,
        ),
        radius=34,
        fill=(245, 245, 245, 238),
        outline=(255, 255, 255, 85),
        width=3,
    )

    y = panel_y1 + 38
    left = panel_x1 + 45
    right = panel_x2 - 45
    block_width = right - left

    needs = TEAM_NEEDS[team]

    for index, pos in enumerate(needs, 1):
        players = players_for_position(
            pos,
            roster,
            max_players=8,
        )

        y = draw_need_block(
            draw=draw,
            pos=f"{index}. {pos}",
            players=players,
            x=left,
            y=y,
            width=block_width,
            accent=accent_hex,
            pos_font=pos_font,
            player_font=player_font,
            small_font=small_font,
        )

        y += 18

        if index != len(needs):
            draw.line(
                (left, y, right, y),
                fill=(190, 190, 190),
                width=2,
            )

            y += 26

        if y > panel_y2 - 120:
            break

    draw_centered(
        draw,
        f"{team} TEAM NEEDS",
        height - 60,
        footer_font,
        "#DADADA",
        width,
    )

    out_file = (
        out_file
        or f"{team.lower()}_team_needs.png"
    )

    os.makedirs(
        os.path.dirname(out_file) or ".",
        exist_ok=True,
    )

    img.convert("RGB").save(
        out_file,
        quality=95,
    )

    return out_file


def generate_all_team_needs_posters(
    outdir: str,
):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    outputs = {}
    failures = {}

    for team in TEAM_META:
        try:
            out_file = os.path.join(
                outdir,
                f"{team.lower()}_team_needs.png",
            )

            poster(
                team,
                out_file=out_file,
            )

            outputs[team] = out_file

        except Exception as e:
            failures[team] = str(e)

    return outputs, failures


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "team",
        nargs="?",
        help="Example: BUF, DAL, WSH",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all 32 teams",
    )

    parser.add_argument(
        "--outdir",
        default="team_needs_numbered_posters",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.all:
        outputs, failures = (
            generate_all_team_needs_posters(
                args.outdir
            )
        )

        print(
            f"Generated {len(outputs)} posters "
            f"in {args.outdir}"
        )

        if failures:
            print("Failures:")

            for team, error in failures.items():
                print(f"{team}: {error}")

        return

    if not args.team:
        raise SystemExit(
            "Provide a team abbreviation like BUF, "
            "or use --all"
        )

    team = ALIASES.get(
        args.team.upper(),
        args.team.upper(),
    )

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    out_file = os.path.join(
        args.outdir,
        f"{team.lower()}_team_needs.png",
    )

    saved = poster(
        team,
        out_file=out_file,
    )

    print(f"Saved {saved}")


if __name__ == "__main__":
    main()
