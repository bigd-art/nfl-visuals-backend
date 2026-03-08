# app/scripts/nightly_publish_posters.py
import os
import json
import argparse
from datetime import datetime, timezone

from app.services.storage_supabase import upload_file_return_url
from app.scripts.nfl_stat_leaders_generate import generate_all_stat_leader_posters
from app.scripts.nfl_standings_conference_generate import generate_standings_conference_png


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ensure_tmp():
    os.makedirs("/tmp", exist_ok=True)


def generate_standings_png(season: int) -> str:
    ensure_tmp()
    out_path = f"/tmp/standings_conference_{season}.png"
    generate_standings_conference_png(season, out_path)
    if not os.path.exists(out_path):
        raise RuntimeError(f"Missing generated file: {out_path}")
    return out_path


def generate_stat_leaders_pngs(season: int, seasontype: int) -> dict[str, str]:
    ensure_tmp()
    outputs = generate_all_stat_leader_posters(
        season=season,
        seasontype=seasontype,
        outdir="/tmp",
    )
    if not outputs:
        raise RuntimeError("No stat leader posters were generated.")
    return outputs


def write_manifest(manifest: dict) -> str:
    ensure_tmp()
    path = "/tmp/latest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_catalog_for_year(
    season: int,
    season_types: list[int],
    keep_versioned: bool,
) -> dict:
    """
    Returns:
    {
      "2": {
        "standings_url": "...",
        "stat_leaders": {...}
      },
      "3": {
        "stat_leaders": {...}
      }
    }
    """
    day = utc_day()
    season_catalog: dict[str, dict] = {}

    for seasontype in season_types:
        type_key = str(seasontype)
        base = (
            f"posters/{day}/{season}/{seasontype}"
            if keep_versioned
            else f"posters/latest/{season}/{seasontype}"
        )

        entry: dict = {}

        # Standings only for regular season
        if seasontype == 2:
            standings_local = generate_standings_png(season)
            standings_key = f"{base}/standings_conference_s{season}.png"
            entry["standings_url"] = upload_file_return_url(standings_local, standings_key)

        stat_outputs = generate_stat_leaders_pngs(season, seasontype)
        stat_urls: dict[str, str] = {}
        for slug, local_path in stat_outputs.items():
            storage_key = f"{base}/{slug}_s{season}_t{seasontype}.png"
            stat_urls[slug] = upload_file_return_url(local_path, storage_key)

        entry["stat_leaders"] = stat_urls
        season_catalog[type_key] = entry

    return season_catalog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="Single season, e.g. 2025")
    ap.add_argument(
        "--seasontypes",
        type=str,
        default="2,3",
        help='CSV list, e.g. "2,3"',
    )
    ap.add_argument(
        "--keep_versioned",
        action="store_true",
        help="Store posters in dated paths.",
    )
    args = ap.parse_args()

    season = args.season
    if season < 2025:
        raise ValueError("This setup only supports seasons 2025 and onward.")

    season_types = [int(x.strip()) for x in args.seasontypes.split(",") if x.strip()]

    catalog = {
        str(season): build_catalog_for_year(
            season=season,
            season_types=season_types,
            keep_versioned=args.keep_versioned,
        )
    }

    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "season_min": 2025,
        "season_max_generated": season,
        "seasontypes": season_types,
        "versioned": bool(args.keep_versioned),
        "catalog": catalog,
    }

    manifest_local = write_manifest(manifest)
    manifest_url = upload_file_return_url(manifest_local, "posters/latest.json")

    print("\n✅ Uploaded manifest")
    print("manifest_url:", manifest_url)
    print(json.dumps(manifest, indent=2))
    print("")


if __name__ == "__main__":
    main()
