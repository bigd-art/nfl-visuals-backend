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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--seasontype", type=int, default=2, choices=[2, 3])
    ap.add_argument("--keep_versioned", action="store_true", help="Store posters in dated paths (new URL daily).")
    args = ap.parse_args()

    season = args.season
    seasontype = args.seasontype
    day = utc_day()
    base = f"posters/{day}" if args.keep_versioned else "posters/latest"

    # Generate locally
    standings_local = generate_standings_png(season)
    stat_outputs = generate_stat_leaders_pngs(season, seasontype)

    # Upload standings
    standings_key = f"{base}/standings_conference_s{season}.png"
    standings_url = upload_file_return_url(standings_local, standings_key)

    # Upload all stat posters
    stat_urls: dict[str, str] = {}
    for slug, local_path in stat_outputs.items():
        storage_key = f"{base}/{slug}_s{season}_t{seasontype}.png"
        stat_urls[slug] = upload_file_return_url(local_path, storage_key)

    # Manifest
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "seasontype": seasontype,
        "versioned": bool(args.keep_versioned),
        "base_path": base,
        "standings_url": standings_url,
        "stat_leaders": stat_urls,
    }

    manifest_local = write_manifest(manifest)
    manifest_url = upload_file_return_url(manifest_local, "posters/latest.json")

    print("\n✅ Uploaded URLs")
    print("standings_url:", standings_url)
    for slug, url in stat_urls.items():
        print(f"{slug}_url:", url)
    print("manifest_url:", manifest_url)
    print("")


if __name__ == "__main__":
    main()
