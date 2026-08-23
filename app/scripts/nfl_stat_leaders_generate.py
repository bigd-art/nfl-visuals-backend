import os
import re
import argparse
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict, Any

import requests
from PIL import Image, ImageDraw, ImageFont


TOP_N = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

Number = Union[int, float]


# ============================================================
# TEAM ABBREVIATIONS
# ============================================================

TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS", "WSH",
}


# ============================================================
# TEAM COLORS
# ============================================================

TEAM_COLORS = {
    "ARI": ("#97233F", "#E6A13B"),
    "ATL": ("#A71930", "#111111"),
    "BAL": ("#4F2A86", "#D5A824"),
    "BUF": ("#1769AA", "#C8102E"),
    "CAR": ("#0085CA", "#101820"),
    "CHI": ("#C95C12", "#5A3216"),
    "CIN": ("#FB4F14", "#111111"),
    "CLE": ("#E85D04", "#5B2C12"),
    "DAL": ("#17365D", "#B7C1CC"),
    "DEN": ("#F36C21", "#17365D"),
    "DET": ("#1673B1", "#A7B1BA"),
    "GB": ("#1F5132", "#E3B23C"),
    "HOU": ("#17365D", "#C51F35"),
    "IND": ("#194F90", "#FFFFFF"),
    "JAX": ("#D5A52A", "#008C95"),
    "KC": ("#C62828", "#EFB32B"),
    "LV": ("#252525", "#B8B8B8"),
    "LAC": ("#168DD0", "#F5C542"),
    "LAR": ("#E0A520", "#1C4C8C"),
    "MIA": ("#1499A5", "#ED6A29"),
    "MIN": ("#5B328A", "#E5B735"),
    "NE": ("#17365D", "#C6283E"),
    "NO": ("#C9A13B", "#171717"),
    "NYG": ("#1F5594", "#B92739"),
    "NYJ": ("#1F573D", "#FFFFFF"),
    "PHI": ("#126A70", "#D5D8DA"),
    "PIT": ("#E3B52B", "#CB273A"),
    "SF": ("#C5412D", "#D9A640"),
    "SEA": ("#17365D", "#58A64A"),
    "TB": ("#B82729", "#E36D2E"),
    "TEN": ("#4D85BD", "#17365D"),
    "WSH": ("#8F2433", "#E6B640"),
}


# ============================================================
# STAT CONFIG
# ============================================================

STAT_CONFIG = [
    (
        "passing_yards",
        "Passing Yards",
        "Passing Yards",
        [
            "passingYards",
            "passing yards",
        ],
    ),
    (
        "passing_tds",
        "Passing TDs",
        "Passing TDs",
        [
            "passingTouchdowns",
            "passing touchdowns",
            "passing tds",
        ],
    ),
    (
        "rushing_yards",
        "Rushing Yards",
        "Rushing Yards",
        [
            "rushingYards",
            "rushing yards",
        ],
    ),
    (
        "rushing_tds",
        "Rushing TDs",
        "Rushing TDs",
        [
            "rushingTouchdowns",
            "rushing touchdowns",
            "rushing tds",
        ],
    ),
    (
        "receiving_yards",
        "Receiving Yards",
        "Receiving Yards",
        [
            "receivingYards",
            "receiving yards",
        ],
    ),
    (
        "receiving_tds",
        "Receiving TDs",
        "Receiving TDs",
        [
            "receivingTouchdowns",
            "receiving touchdowns",
            "receiving tds",
        ],
    ),
    (
        "sacks",
        "Sacks",
        "Sacks",
        [
            "sacks",
        ],
    ),
    (
        "tackles",
        "Tackles",
        "Tackles",
        [
            "totalTackles",
            "total tackles",
            "tackles",
        ],
    ),
    (
        "interceptions_defense",
        "Interceptions (Defense)",
        "Interceptions",
        [
            "defensiveInterceptions",
            "interceptions",
        ],
    ),
]


# ============================================================
# ESPN CORE API
#
# Lowercase league path is required internally by ESPN.
# It is never displayed on the poster.
# ============================================================

CORE_LEADERS_URL = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl/"
    "seasons/{season}/"
    "types/{seasontype}/"
    "leaders?lang=en&region=us"
)


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_spaces(
    s: str,
) -> str:

    s = str(
        s or ""
    )

    s = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        s,
    )

    s = (
        s
        .replace(
            "\u00a0",
            " ",
        )
        .replace(
            "\u2009",
            " ",
        )
        .replace(
            "\u202f",
            " ",
        )
        .replace(
            "\u00ad",
            "",
        )
    )

    return re.sub(
        r"\s+",
        " ",
        s,
    ).strip()


def safe_float(
    x,
) -> Optional[float]:

    if x is None:
        return None

    s = str(
        x
    ).replace(
        ",",
        "",
    ).strip()

    s = re.sub(
        r"[^\d\.\-]",
        "",
        s,
    )

    if s in {
        "",
        "-",
        ".",
    }:
        return None

    try:
        return float(
            s
        )

    except Exception:
        return None


# ============================================================
# HTTP
# ============================================================

def fetch_json(
    url: str,
) -> Dict[
    str,
    Any,
]:

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def resolve_ref(
    obj: Any,
) -> Dict[
    str,
    Any,
]:

    if (
        isinstance(
            obj,
            dict,
        )
        and "$ref" in obj
    ):

        try:
            return fetch_json(
                obj["$ref"]
            )

        except Exception:
            return obj

    if isinstance(
        obj,
        dict,
    ):
        return obj

    return {}


# ============================================================
# CATEGORY MATCHING
# ============================================================

def text_key(
    value: str,
) -> str:

    return (
        normalize_spaces(
            value
        )
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )


def category_matches(
    category: Dict[
        str,
        Any,
    ],
    aliases: List[str],
    slug: str,
) -> bool:

    raw_fields = [
        category.get(
            "name"
        ),
        category.get(
            "displayName"
        ),
        category.get(
            "shortDisplayName"
        ),
        category.get(
            "description"
        ),
        category.get(
            "abbreviation"
        ),
    ]

    combined = " ".join(
        text_key(
            value
        )
        for value in raw_fields
        if value
    )

    for alias in aliases:

        alias_key = text_key(
            alias
        )

        if (
            alias_key
            and alias_key in combined
        ):
            return True

    if (
        slug
        == "interceptions_thrown"
    ):

        return (
            "interception" in combined
            and "defensive" not in combined
        )

    if (
        slug
        == "interceptions_defense"
    ):

        return (
            "interception" in combined
            and (
                "defensive" in combined
                or "defense" in combined
                or "def" in combined
            )
        )

    return False


def find_leader_categories(
    data: Any,
) -> List[
    Dict[
        str,
        Any,
    ]
]:

    found = []

    if isinstance(
        data,
        dict,
    ):

        if (
            "leaders" in data
            and isinstance(
                data["leaders"],
                list,
            )
        ):
            found.append(
                data
            )

        for value in data.values():
            found.extend(
                find_leader_categories(
                    value
                )
            )

    elif isinstance(
        data,
        list,
    ):

        for item in data:
            found.extend(
                find_leader_categories(
                    item
                )
            )

    return found


# ============================================================
# LEADER EXTRACTION
# ============================================================

def extract_athlete_name(
    leader: Dict[
        str,
        Any,
    ],
) -> str:

    athlete = resolve_ref(
        leader.get(
            "athlete"
        )
        or leader.get(
            "player"
        )
        or leader.get(
            "person"
        )
        or {}
    )

    return normalize_spaces(
        athlete.get(
            "displayName"
        )
        or athlete.get(
            "fullName"
        )
        or athlete.get(
            "shortName"
        )
        or leader.get(
            "displayName"
        )
        or leader.get(
            "name"
        )
        or "Unknown Player"
    )


def extract_team_abbr(
    leader: Dict[
        str,
        Any,
    ],
) -> str:

    team_obj = (
        leader.get(
            "team"
        )
        or leader.get(
            "teamAthlete"
        )
        or {}
    )

    if (
        isinstance(
            team_obj,
            dict,
        )
        and "$ref" in team_obj
    ):

        team_obj = resolve_ref(
            team_obj
        )

    abbr = normalize_spaces(
        team_obj.get(
            "abbreviation"
        )
        or team_obj.get(
            "shortDisplayName"
        )
        or team_obj.get(
            "name"
        )
        or ""
    ).upper()

    if abbr == "WAS":
        abbr = "WSH"

    if abbr in TEAM_ABBRS:
        return abbr

    return ""


def extract_leader_value(
    leader: Dict[
        str,
        Any,
    ],
) -> Optional[float]:

    for key in [
        "value",
        "displayValue",
        "stat",
        "score",
    ]:

        if key in leader:

            val = safe_float(
                leader.get(
                    key
                )
            )

            if val is not None:
                return val

    statistics = (
        leader.get(
            "statistics"
        )
        or leader.get(
            "stats"
        )
        or []
    )

    if isinstance(
        statistics,
        list,
    ):

        for stat in statistics:

            if isinstance(
                stat,
                dict,
            ):

                val = safe_float(
                    stat.get(
                        "value"
                    )
                    or stat.get(
                        "displayValue"
                    )
                )

                if val is not None:
                    return val

    return None


# ============================================================
# FETCH TOP LEADERS
# ============================================================

def fetch_top_from_leaders_api(
    season: int,
    seasontype: int,
    slug: str,
    aliases: List[str],
    mode: str,
) -> List[
    Tuple[
        int,
        str,
        Number,
    ]
]:

    url = CORE_LEADERS_URL.format(
        season=season,
        seasontype=seasontype,
    )

    data = fetch_json(
        url
    )

    categories = (
        find_leader_categories(
            data
        )
    )

    matched = None

    for category in categories:

        if category_matches(
            category,
            aliases,
            slug,
        ):

            matched = category
            break

    if not matched:

        available = []

        for category in categories[
            :40
        ]:

            available.append(
                {
                    "name": category.get(
                        "name"
                    ),
                    "displayName": category.get(
                        "displayName"
                    ),
                    "shortDisplayName": category.get(
                        "shortDisplayName"
                    ),
                }
            )

        raise RuntimeError(
            f"No matching category "
            f"for {slug}. "
            f"Available sample: "
            f"{available}"
        )

    rows = []

    for leader in matched.get(
        "leaders",
        [],
    ):

        if not isinstance(
            leader,
            dict,
        ):
            continue

        name = extract_athlete_name(
            leader
        )

        team = extract_team_abbr(
            leader
        )

        value = extract_leader_value(
            leader
        )

        if value is None:
            continue

        display_name = (
            f"{name} {team}"
            .strip()
        )

        rows.append(
            (
                display_name,
                value,
            )
        )

    rows = sorted(
        rows,
        key=lambda item: item[1],
        reverse=True,
    )[:TOP_N]

    output = []

    for (
        index,
        (
            name,
            value,
        ),
    ) in enumerate(
        rows,
        start=1,
    ):

        if mode == "float1":

            output.append(
                (
                    index,
                    name,
                    float(
                        value
                    ),
                )
            )

        else:

            output.append(
                (
                    index,
                    name,
                    int(
                        round(
                            value
                        )
                    ),
                )
            )

    if not output:

        raise RuntimeError(
            f"No usable leaders "
            f"for {slug}"
        )

    return output


# ============================================================
# FONT HELPERS
# ============================================================

def load_font(
    size: int,
    bold: bool = False,
):

    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else
            "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        (
            "/Library/Fonts/Arial Bold.ttf"
            if bold
            else
            "/Library/Fonts/Arial.ttf"
        ),
        (
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"
            if bold
            else
            "/System/Library/Fonts/Supplemental/Helvetica.ttf"
        ),
    ]

    for path in candidates:

        try:

            return ImageFont.truetype(
                path,
                size=size,
            )

        except Exception:
            pass

    return ImageFont.load_default()


def fmt_value(
    val: Number,
    mode: str,
) -> str:

    if mode == "float1":
        return (
            f"{float(val):.1f}"
        )

    return (
        f"{int(val):,}"
    )


def fit_text(
    draw,
    text: str,
    font,
    max_width: int,
) -> str:

    text = str(
        text
    )

    if (
        draw.textlength(
            text,
            font=font,
        )
        <= max_width
    ):
        return text

    while (
        len(text) > 3
        and draw.textlength(
            text + "…",
            font=font,
        )
        > max_width
    ):

        text = text[:-1]

    return (
        text.rstrip()
        + "…"
    )


def split_name_team(
    display_name: str,
) -> Tuple[
    str,
    str,
]:

    display_name = normalize_spaces(
        display_name
    )

    parts = (
        display_name.split()
    )

    if (
        parts
        and parts[-1].upper()
        in TEAM_ABBRS
    ):

        team = (
            parts[-1]
            .upper()
        )

        if team == "WAS":
            team = "WSH"

        return (
            " ".join(
                parts[:-1]
            ),
            team,
        )

    return (
        display_name,
        "",
    )


# ============================================================
# COLOR HELPER
# ============================================================

def hex_to_rgb(
    value: str,
) -> Tuple[
    int,
    int,
    int,
]:

    value = (
        value
        .replace(
            "#",
            "",
        )
        .strip()
    )

    return tuple(
        int(
            value[
                i:i + 2
            ],
            16,
        )
        for i in (
            0,
            2,
            4,
        )
    )


# ============================================================
# ORIGINAL PIXEL ICON HELPERS
# ============================================================

def draw_star(
    draw,
    cx,
    cy,
    radius,
    fill,
):

    points = [
        (cx, cy - radius),
        (
            cx + radius // 4,
            cy - radius // 4,
        ),
        (
            cx + radius,
            cy - radius // 4,
        ),
        (
            cx + radius // 3,
            cy + radius // 5,
        ),
        (
            cx + radius // 2,
            cy + radius,
        ),
        (
            cx,
            cy + radius // 2,
        ),
        (
            cx - radius // 2,
            cy + radius,
        ),
        (
            cx - radius // 3,
            cy + radius // 5,
        ),
        (
            cx - radius,
            cy - radius // 4,
        ),
        (
            cx - radius // 4,
            cy - radius // 4,
        ),
    ]

    draw.polygon(
        points,
        fill=fill,
    )


def draw_paw(
    draw,
    cx,
    cy,
    fill,
):

    draw.ellipse(
        (
            cx - 14,
            cy - 2,
            cx + 14,
            cy + 22,
        ),
        fill=fill,
    )

    for (
        dx,
        dy,
    ) in [
        (-20, -18),
        (-7, -25),
        (7, -25),
        (20, -18),
    ]:

        draw.ellipse(
            (
                cx + dx - 5,
                cy + dy - 7,
                cx + dx + 5,
                cy + dy + 7,
            ),
            fill=fill,
        )


def draw_football(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.ellipse(
        (
            cx - 34,
            cy - 17,
            cx + 34,
            cy + 17,
        ),
        fill=fill,
    )

    draw.line(
        (
            cx - 13,
            cy,
            cx + 13,
            cy,
        ),
        fill=accent,
        width=3,
    )

    for offset in (
        -8,
        0,
        8,
    ):

        draw.line(
            (
                cx + offset,
                cy - 5,
                cx + offset,
                cy + 5,
            ),
            fill=accent,
            width=2,
        )


def draw_feather(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 27,
                cy + 26,
            ),
            (
                cx - 14,
                cy - 18,
            ),
            (
                cx + 27,
                cy - 33,
            ),
            (
                cx + 18,
                cy + 8,
            ),
            (
                cx - 9,
                cy + 27,
            ),
        ],
        fill=fill,
    )

    draw.line(
        (
            cx - 24,
            cy + 30,
            cx + 18,
            cy - 24,
        ),
        fill=accent,
        width=3,
    )


def draw_claws(
    draw,
    cx,
    cy,
    fill,
):

    for offset in (
        -19,
        0,
        19,
    ):

        draw.polygon(
            [
                (
                    cx
                    + offset
                    - 5,
                    cy + 27,
                ),
                (
                    cx
                    + offset
                    + 3,
                    cy - 30,
                ),
                (
                    cx
                    + offset
                    + 10,
                    cy - 34,
                ),
                (
                    cx
                    + offset
                    + 3,
                    cy + 27,
                ),
            ],
            fill=fill,
        )


def draw_stripes(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 31,
            cy - 30,
            cx + 31,
            cy + 30,
        ),
        fill=fill,
    )

    for offset in (
        -34,
        -9,
        16,
    ):

        draw.polygon(
            [
                (
                    cx + offset,
                    cy - 30,
                ),
                (
                    cx + offset + 10,
                    cy - 30,
                ),
                (
                    cx + offset + 35,
                    cy + 30,
                ),
                (
                    cx + offset + 25,
                    cy + 30,
                ),
            ],
            fill=accent,
        )


def draw_mountain(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 36,
                cy + 27,
            ),
            (
                cx,
                cy - 31,
            ),
            (
                cx + 36,
                cy + 27,
            ),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx - 9,
                cy - 17,
            ),
            (
                cx,
                cy - 31,
            ),
            (
                cx + 10,
                cy - 15,
            ),
            (
                cx + 2,
                cy - 20,
            ),
        ],
        fill=accent,
    )


def draw_texas(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 28,
                cy - 30,
            ),
            (
                cx + 7,
                cy - 30,
            ),
            (
                cx + 8,
                cy - 13,
            ),
            (
                cx + 30,
                cy - 12,
            ),
            (
                cx + 23,
                cy + 8,
            ),
            (
                cx + 7,
                cy + 15,
            ),
            (
                cx - 2,
                cy + 33,
            ),
            (
                cx - 16,
                cy + 16,
            ),
            (
                cx - 30,
                cy + 4,
            ),
        ],
        fill=fill,
    )

    draw_star(
        draw,
        cx,
        cy,
        8,
        accent,
    )


def draw_arc(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.arc(
        (
            cx - 31,
            cy - 31,
            cx + 31,
            cy + 31,
        ),
        205,
        335,
        fill=fill,
        width=9,
    )

    draw.arc(
        (
            cx - 20,
            cy - 20,
            cx + 20,
            cy + 20,
        ),
        205,
        335,
        fill=accent,
        width=4,
    )


def draw_pennant(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 30,
            cy - 32,
            cx - 26,
            cy + 30,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (
                cx - 26,
                cy - 27,
            ),
            (
                cx + 33,
                cy - 4,
            ),
            (
                cx - 26,
                cy + 17,
            ),
        ],
        fill=fill,
    )


def draw_lightning(
    draw,
    cx,
    cy,
    fill,
):

    draw.polygon(
        [
            (
                cx + 5,
                cy - 34,
            ),
            (
                cx - 22,
                cy + 1,
            ),
            (
                cx - 5,
                cy + 1,
            ),
            (
                cx - 14,
                cy + 34,
            ),
            (
                cx + 25,
                cy - 8,
            ),
            (
                cx + 8,
                cy - 8,
            ),
        ],
        fill=fill,
    )


def draw_wave(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 34,
                cy + 18,
            ),
            (
                cx - 24,
                cy - 5,
            ),
            (
                cx - 10,
                cy - 20,
            ),
            (
                cx + 4,
                cy - 23,
            ),
            (
                cx + 21,
                cy - 14,
            ),
            (
                cx + 34,
                cy + 4,
            ),
            (
                cx + 13,
                cy - 1,
            ),
            (
                cx,
                cy + 8,
            ),
            (
                cx - 7,
                cy + 20,
            ),
        ],
        fill=fill,
    )

    draw.line(
        (
            cx - 29,
            cy + 22,
            cx + 30,
            cy + 22,
        ),
        fill=accent,
        width=4,
    )


def draw_spiral(
    draw,
    cx,
    cy,
    fill,
):

    draw.arc(
        (
            cx - 28,
            cy - 28,
            cx + 28,
            cy + 28,
        ),
        20,
        340,
        fill=fill,
        width=8,
    )

    draw.arc(
        (
            cx - 15,
            cy - 15,
            cx + 15,
            cy + 15,
        ),
        20,
        300,
        fill=fill,
        width=6,
    )


def draw_ship(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 34,
                cy + 17,
            ),
            (
                cx + 34,
                cy + 17,
            ),
            (
                cx + 22,
                cy + 29,
            ),
            (
                cx - 22,
                cy + 29,
            ),
        ],
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 2,
            cy - 31,
            cx + 2,
            cy + 15,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (
                cx + 2,
                cy - 27,
            ),
            (
                cx + 23,
                cy - 5,
            ),
            (
                cx + 2,
                cy - 5,
            ),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx - 3,
                cy - 24,
            ),
            (
                cx - 22,
                cy - 4,
            ),
            (
                cx - 3,
                cy - 4,
            ),
        ],
        fill=fill,
    )


def draw_hat(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 33,
                cy + 16,
            ),
            (
                cx - 19,
                cy - 13,
            ),
            (
                cx,
                cy - 25,
            ),
            (
                cx + 19,
                cy - 13,
            ),
            (
                cx + 33,
                cy + 16,
            ),
            (
                cx + 10,
                cy + 9,
            ),
            (
                cx,
                cy + 21,
            ),
            (
                cx - 10,
                cy + 9,
            ),
        ],
        fill=fill,
    )

    draw_star(
        draw,
        cx,
        cy,
        8,
        accent,
    )


def draw_trumpet(
    draw,
    cx,
    cy,
    fill,
):

    draw.rectangle(
        (
            cx - 23,
            cy - 5,
            cx + 10,
            cy + 5,
        ),
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx + 10,
                cy - 14,
            ),
            (
                cx + 32,
                cy - 22,
            ),
            (
                cx + 32,
                cy + 22,
            ),
            (
                cx + 10,
                cy + 14,
            ),
        ],
        fill=fill,
    )

    draw.arc(
        (
            cx - 29,
            cy - 1,
            cx - 11,
            cy + 24,
        ),
        10,
        190,
        fill=fill,
        width=4,
    )


def draw_skyline(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    buildings = [
        (
            -31,
            11,
            -22,
            29,
        ),
        (
            -20,
            -3,
            -8,
            29,
        ),
        (
            -6,
            -29,
            6,
            29,
        ),
        (
            8,
            -10,
            21,
            29,
        ),
        (
            23,
            3,
            32,
            29,
        ),
    ]

    for (
        x1,
        y1,
        x2,
        y2,
    ) in buildings:

        draw.rectangle(
            (
                cx + x1,
                cy + y1,
                cx + x2,
                cy + y2,
            ),
            fill=fill,
        )

    draw.line(
        (
            cx - 35,
            cy + 30,
            cx + 35,
            cy + 30,
        ),
        fill=accent,
        width=4,
    )


def draw_jet(
    draw,
    cx,
    cy,
    fill,
):

    draw.polygon(
        [
            (
                cx - 35,
                cy + 6,
            ),
            (
                cx - 7,
                cy - 4,
            ),
            (
                cx + 18,
                cy - 28,
            ),
            (
                cx + 25,
                cy - 23,
            ),
            (
                cx + 12,
                cy - 3,
            ),
            (
                cx + 34,
                cy + 7,
            ),
            (
                cx + 10,
                cy + 10,
            ),
            (
                cx + 3,
                cy + 27,
            ),
            (
                cx - 5,
                cy + 27,
            ),
            (
                cx - 6,
                cy + 11,
            ),
        ],
        fill=fill,
    )


def draw_diamonds(
    draw,
    cx,
    cy,
):

    data = [
        (
            cx,
            cy - 18,
            "#F2C230",
        ),
        (
            cx - 19,
            cy + 13,
            "#1E5FA8",
        ),
        (
            cx + 19,
            cy + 13,
            "#D32E3E",
        ),
    ]

    for (
        x,
        y,
        color,
    ) in data:

        draw.polygon(
            [
                (
                    x,
                    y - 9,
                ),
                (
                    x + 9,
                    y,
                ),
                (
                    x,
                    y + 9,
                ),
                (
                    x - 9,
                    y,
                ),
            ],
            fill=color,
        )


def draw_bridge(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 26,
            cy - 27,
            cx - 21,
            cy + 27,
        ),
        fill=fill,
    )

    draw.rectangle(
        (
            cx + 21,
            cy - 27,
            cx + 26,
            cy + 27,
        ),
        fill=fill,
    )

    draw.line(
        (
            cx - 34,
            cy + 18,
            cx + 34,
            cy + 18,
        ),
        fill=fill,
        width=5,
    )

    draw.arc(
        (
            cx - 24,
            cy - 22,
            cx + 24,
            cy + 31,
        ),
        180,
        360,
        fill=accent,
        width=3,
    )


def draw_flag(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 28,
            cy - 32,
            cx - 24,
            cy + 30,
        ),
        fill=accent,
    )

    draw.polygon(
        [
            (
                cx - 24,
                cy - 26,
            ),
            (
                cx + 29,
                cy - 21,
            ),
            (
                cx + 20,
                cy + 3,
            ),
            (
                cx - 24,
                cy + 8,
            ),
        ],
        fill=fill,
    )


def draw_sword(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.polygon(
        [
            (
                cx - 4,
                cy - 33,
            ),
            (
                cx + 5,
                cy - 33,
            ),
            (
                cx + 6,
                cy + 13,
            ),
            (
                cx - 6,
                cy + 13,
            ),
        ],
        fill=fill,
    )

    draw.polygon(
        [
            (
                cx - 4,
                cy - 33,
            ),
            (
                cx,
                cy - 42,
            ),
            (
                cx + 5,
                cy - 33,
            ),
        ],
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 18,
            cy + 11,
            cx + 18,
            cy + 16,
        ),
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 4,
            cy + 16,
            cx + 5,
            cy + 31,
        ),
        fill=accent,
    )


def draw_column(
    draw,
    cx,
    cy,
    fill,
    accent,
):

    draw.rectangle(
        (
            cx - 27,
            cy - 28,
            cx + 27,
            cy - 22,
        ),
        fill=accent,
    )

    draw.rectangle(
        (
            cx - 31,
            cy + 23,
            cx + 31,
            cy + 29,
        ),
        fill=accent,
    )

    for x in (
        -19,
        -6,
        6,
        19,
    ):

        draw.rectangle(
            (
                cx + x - 3,
                cy - 19,
                cx + x + 3,
                cy + 21,
            ),
            fill=fill,
        )


# ============================================================
# TEAM SYMBOL
# ============================================================

def draw_team_symbol(
    draw,
    team: str,
    cx: int,
    cy: int,
    primary,
    secondary,
):

    if team == "ARI":

        draw.rectangle(
            (
                cx - 5,
                cy - 29,
                cx + 5,
                cy + 27,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx - 21,
                cy - 6,
                cx - 4,
                cy + 3,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx - 21,
                cy - 18,
                cx - 13,
                cy + 3,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx + 4,
                cy - 10,
                cx + 20,
                cy - 2,
            ),
            fill=primary,
        )

        draw.rectangle(
            (
                cx + 13,
                cy - 21,
                cx + 20,
                cy - 2,
            ),
            fill=primary,
        )

    elif team in {
        "ATL",
        "BAL",
        "PHI",
    }:

        draw_feather(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "BUF":

        draw.ellipse(
            (
                cx - 26,
                cy - 26,
                cx - 4,
                cy + 5,
            ),
            fill=primary,
        )

        draw.ellipse(
            (
                cx + 4,
                cy - 26,
                cx + 26,
                cy + 5,
            ),
            fill=primary,
        )

        draw.ellipse(
            (
                cx - 9,
                cy + 9,
                cx + 9,
                cy + 26,
            ),
            fill=secondary,
        )

    elif team == "CAR":

        draw_claws(
            draw,
            cx,
            cy,
            primary,
        )

    elif team in {
        "CHI",
        "DET",
        "JAX",
    }:

        draw_paw(
            draw,
            cx,
            cy,
            primary,
        )

    elif team == "CIN":

        draw_stripes(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team in {
        "CLE",
        "GB",
    }:

        draw_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "DAL":

        draw_star(
            draw,
            cx,
            cy,
            30,
            primary,
        )

    elif team == "DEN":

        draw_mountain(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "HOU":

        draw_texas(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "IND":

        draw_arc(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "KC":

        draw_pennant(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "LV":

        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "LAC":

        draw_lightning(
            draw,
            cx,
            cy,
            secondary,
        )

    elif team == "LAR":

        draw_spiral(
            draw,
            cx,
            cy,
            primary,
        )

    elif team == "MIA":

        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "MIN":

        draw_ship(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "NE":

        draw_hat(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "NO":

        draw_trumpet(
            draw,
            cx,
            cy,
            primary,
        )

    elif team == "NYG":

        draw_skyline(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "NYJ":

        draw_jet(
            draw,
            cx,
            cy,
            primary,
        )

    elif team == "PIT":

        draw_diamonds(
            draw,
            cx,
            cy,
        )

    elif team == "SF":

        draw_bridge(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "SEA":

        draw_wave(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "TB":

        draw_flag(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "TEN":

        draw_sword(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )

    elif team == "WSH":

        draw_column(
            draw,
            cx,
            cy,
            secondary,
            primary,
        )

    else:

        draw_football(
            draw,
            cx,
            cy,
            primary,
            secondary,
        )


# ============================================================
# CREATE TEAM ICON
#
# Each icon includes abbreviation below the symbol.
# ============================================================

def create_team_icon(
    team: str,
    size: int = 72,
) -> Optional[
    Image.Image
]:

    if not team:
        return None

    team = (
        team
        .upper()
        .strip()
    )

    if team == "WAS":
        team = "WSH"

    if team not in TEAM_COLORS:
        return None

    primary_hex, secondary_hex = (
        TEAM_COLORS[
            team
        ]
    )

    primary = hex_to_rgb(
        primary_hex
    )

    secondary = hex_to_rgb(
        secondary_hex
    )

    base_width = 104
    base_height = 118

    icon = Image.new(
        "RGBA",
        (
            base_width,
            base_height,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    draw = ImageDraw.Draw(
        icon
    )

    draw_team_symbol(
        draw,
        team,
        base_width // 2,
        45,
        primary,
        secondary,
    )

    abbreviation_font = (
        load_font(
            18,
            bold=True,
        )
    )

    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        team,
        font=abbreviation_font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    draw.rounded_rectangle(
        (
            13,
            88,
            base_width - 13,
            115,
        ),
        radius=4,
        fill=(
            8,
            12,
            18,
            235,
        ),
    )

    text_fill = primary

    if sum(
        primary
    ) < 140:

        text_fill = (
            230,
            230,
            230,
        )

    draw.text(
        (
            (
                base_width
                - text_width
            )
            // 2,
            91,
        ),
        team,
        font=abbreviation_font,
        fill=text_fill,
    )

    ratio = (
        size
        / base_height
    )

    output_width = max(
        1,
        int(
            base_width
            * ratio
        ),
    )

    return icon.resize(
        (
            output_width,
            size,
        ),
        Image.Resampling.NEAREST,
    )


# ============================================================
# POSTER
# ============================================================

def draw_single_stat_poster(
    out_path: str,
    poster_title: str,
    stat_title: str,
    subtitle: str,
    items: List[
        Tuple[
            int,
            str,
            Number,
        ]
    ],
    mode: str,
):

    width = 1080
    height = 1920

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        (
            10,
            14,
            24,
        ),
    )

    draw = ImageDraw.Draw(
        image
    )

    title_font = load_font(
        48,
        bold=True,
    )

    stat_font = load_font(
        68,
        bold=True,
    )

    sub_font = load_font(
        22,
        bold=False,
    )

    rank_font = load_font(
        31,
        bold=True,
    )

    name_font = load_font(
        39,
        bold=True,
    )

    value_font = load_font(
        45,
        bold=True,
    )

    white = (
        246,
        248,
        252,
    )

    muted = (
        208,
        218,
        238,
    )

    blue = (
        128,
        183,
        255,
    )

    dark = (
        24,
        29,
        42,
    )

    border = (
        64,
        74,
        98,
    )

    logo_cache: Dict[
        str,
        Optional[
            Image.Image
        ],
    ] = {}

    draw.rectangle(
        (
            0,
            0,
            width,
            178,
        ),
        fill=(
            22,
            38,
            74,
        ),
    )

    draw.rectangle(
        (
            0,
            178,
            width,
            187,
        ),
        fill=blue,
    )

    for y in range(
        198,
        height,
        30,
    ):

        color = (
            (
                14,
                18,
                28,
            )
            if (
                y // 30
            )
            % 2
            == 0
            else (
                12,
                16,
                26,
            )
        )

        draw.rectangle(
            (
                0,
                y,
                width,
                y + 15,
            ),
            fill=color,
        )

    title = fit_text(
        draw,
        poster_title.upper(),
        title_font,
        width - 90,
    )

    stat = fit_text(
        draw,
        stat_title.upper(),
        stat_font,
        width - 90,
    )

    sub = fit_text(
        draw,
        subtitle,
        sub_font,
        width - 90,
    )

    draw.text(
        (
            (
                width
                - draw.textlength(
                    title,
                    font=title_font,
                )
            )
            / 2,
            24,
        ),
        title,
        font=title_font,
        fill=white,
    )

    draw.text(
        (
            (
                width
                - draw.textlength(
                    stat,
                    font=stat_font,
                )
            )
            / 2,
            78,
        ),
        stat,
        font=stat_font,
        fill=white,
    )

    draw.text(
        (
            (
                width
                - draw.textlength(
                    sub,
                    font=sub_font,
                )
            )
            / 2,
            143,
        ),
        sub,
        font=sub_font,
        fill=muted,
    )

    x0 = 42
    x1 = width - 42

    top = 220
    bottom = height - 42

    gap = 14

    row_h = int(
        (
            bottom
            - top
            - gap
            * (
                TOP_N - 1
            )
        )
        / TOP_N
    )

    for (
        rank,
        display_name,
        val,
    ) in items:

        y0 = (
            top
            + (
                rank - 1
            )
            * (
                row_h
                + gap
            )
        )

        y1 = (
            y0
            + row_h
        )

        draw.rounded_rectangle(
            (
                x0,
                y0,
                x1,
                y1,
            ),
            radius=26,
            fill=dark,
            outline=border,
            width=3,
        )

        pill = (
            x0 + 18,
            y0 + 22,
            x0 + 92,
            y0 + 84,
        )

        draw.rounded_rectangle(
            pill,
            radius=18,
            fill=blue,
        )

        rank_text = str(
            rank
        )

        rank_width = (
            draw.textlength(
                rank_text,
                font=rank_font,
            )
        )

        draw.text(
            (
                pill[0]
                + (
                    pill[2]
                    - pill[0]
                    - rank_width
                )
                / 2,
                pill[1]
                + 12,
            ),
            rank_text,
            font=rank_font,
            fill=(
                15,
                20,
                28,
            ),
        )

        player_name, team = (
            split_name_team(
                display_name
            )
        )

        value_text = fmt_value(
            val,
            mode,
        )

        value_width = (
            draw.textlength(
                value_text,
                font=value_font,
            )
        )

        draw.text(
            (
                x1
                - 30
                - value_width,
                y0 + 31,
            ),
            value_text,
            font=value_font,
            fill=white,
        )

        # ----------------------------------------------------
        # ORIGINAL TEAM ICON
        # ----------------------------------------------------

        logo_x = (
            x0 + 108
        )

        logo_y = (
            y0 + 17
        )

        logo_box = 82

        if team:

            if team not in logo_cache:

                logo_cache[
                    team
                ] = create_team_icon(
                    team,
                    size=78,
                )

            logo = (
                logo_cache.get(
                    team
                )
            )

            if logo:

                lx = (
                    logo_x
                    + (
                        logo_box
                        - logo.width
                    )
                    // 2
                )

                ly = (
                    logo_y
                    + (
                        logo_box
                        - logo.height
                    )
                    // 2
                )

                image.paste(
                    logo,
                    (
                        lx,
                        ly,
                    ),
                    logo,
                )

        # ----------------------------------------------------
        # PLAYER NAME
        # ----------------------------------------------------

        name_x = (
            x0 + 205
        )

        max_name_width = (
            x1
            - name_x
            - value_width
            - 50
        )

        name_text = fit_text(
            draw,
            player_name.upper(),
            name_font,
            max_name_width,
        )

        draw.text(
            (
                name_x,
                y0 + 43,
            ),
            name_text,
            font=name_font,
            fill=white,
        )

        draw.rectangle(
            (
                x0 + 18,
                y1 - 12,
                x1 - 18,
                y1 - 8,
            ),
            fill=blue,
        )

    os.makedirs(
        os.path.dirname(
            out_path
        )
        or ".",
        exist_ok=True,
    )

    image.save(
        out_path,
        "PNG",
    )


# ============================================================
# STAT MODE
# ============================================================

def stat_mode_for_slug(
    slug: str,
) -> str:

    if slug == "sacks":
        return "float1"

    return "int"


# ============================================================
# BUILD SECTIONS
# ============================================================

def build_stat_sections(
    season: int,
    seasontype: int,
) -> Dict[
    str,
    Tuple[
        str,
        List[
            Tuple[
                int,
                str,
                Number,
            ]
        ],
        str,
    ],
]:

    output = {}

    for (
        slug,
        _source_title,
        short_title,
        aliases,
    ) in STAT_CONFIG:

        mode = (
            stat_mode_for_slug(
                slug
            )
        )

        try:

            items = (
                fetch_top_from_leaders_api(
                    season=season,
                    seasontype=seasontype,
                    slug=slug,
                    aliases=aliases,
                    mode=mode,
                )
            )

            output[
                slug
            ] = (
                short_title,
                items,
                mode,
            )

            print(
                f"Generated data "
                f"for {slug}: "
                f"{len(items)} rows"
            )

        except Exception as exc:

            print(
                f"WARNING: Failed "
                f"to fetch {slug}: "
                f"{exc}"
            )

    return output


# ============================================================
# GENERATE ALL POSTERS
# ============================================================

def generate_all_stat_leader_posters(
    season: int,
    seasontype: int,
    outdir: str,
) -> Dict[
    str,
    str,
]:

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    phase = (
        "Regular Season"
        if seasontype == 2
        else "Postseason"
    )

    updated = (
        datetime.utcnow()
        .strftime(
            "%b %d, %Y • %I:%M %p UTC"
        )
    )

    subtitle = (
        f"Season {season} • "
        f"{phase} • "
        f"Updated {updated}"
    )

    sections = (
        build_stat_sections(
            season,
            seasontype,
        )
    )

    outputs = {}

    for (
        slug,
        _source_title,
        short_title,
        _aliases,
    ) in STAT_CONFIG:

        if slug not in sections:

            print(
                f"WARNING: Skipping poster "
                f"for {slug}; "
                f"no data available"
            )

            continue

        (
            stat_title,
            items,
            mode,
        ) = (
            sections[
                slug
            ]
        )

        out_path = os.path.join(
            outdir,
            (
                f"{slug}_"
                f"s{season}_"
                f"t{seasontype}.png"
            ),
        )

        draw_single_stat_poster(
            out_path=out_path,
            poster_title=(
                "STATISTICAL LEADERS"
            ),
            stat_title=stat_title,
            subtitle=subtitle,
            items=items,
            mode=mode,
        )

        outputs[
            slug
        ] = out_path

    if not outputs:

        raise RuntimeError(
            f"No stat leader posters "
            f"generated for "
            f"season={season}, "
            f"seasontype={seasontype}"
        )

    return outputs


# ============================================================
# CLI
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--seasontype",
        type=int,
        default=2,
        choices=[
            2,
            3,
        ],
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(
            os.path.expanduser(
                "~"
            ),
            "Desktop",
        ),
    )

    args = (
        parser.parse_args()
    )

    outputs = (
        generate_all_stat_leader_posters(
            season=args.season,
            seasontype=args.seasontype,
            outdir=args.outdir,
        )
    )

    print(
        "\nDONE"
    )

    for (
        slug,
        path,
    ) in outputs.items():

        print(
            slug,
            "->",
            path,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
