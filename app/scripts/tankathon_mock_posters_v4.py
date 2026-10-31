#!/usr/bin/env python3
import os
import re
import sys

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

SOURCE_URL = "https://www.tankathon.com/nfl/mock-draft"
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

TEAM_ABBRS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WSH",
}

TEAM_FIXES = {
    "JAC": "JAX",
    "WAS": "WSH",
    "ARZ": "ARI",
}


# Tankathon currently exposes the team name in the same text chunk
# that the old parser may capture as part of the player name.
# Keep the original parser intact, then clean that prefix afterward.
TEAM_NAME_PREFIXES = {
    "ARIZONA CARDINALS": "ARI",
    "ARIZONA": "ARI",
    "ATLANTA FALCONS": "ATL",
    "ATLANTA": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BALTIMORE": "BAL",
    "BUFFALO BILLS": "BUF",
    "BUFFALO": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CAROLINA": "CAR",
    "CHICAGO BEARS": "CHI",
    "CHICAGO": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CINCINNATI": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "CLEVELAND": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DALLAS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DENVER": "DEN",
    "DETROIT LIONS": "DET",
    "DETROIT": "DET",
    "GREEN BAY PACKERS": "GB",
    "GREEN BAY": "GB",
    "HOUSTON TEXANS": "HOU",
    "HOUSTON": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "FROM INDIANAPOLIS": "IND",
    "INDIANAPOLIS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "JACKSONVILLE": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "KANSAS CITY": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LAS VEGAS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "LA CHARGERS": "LAC",
    "CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "LA RAMS": "LAR",
    "RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "MIAMI": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "MINNESOTA": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
