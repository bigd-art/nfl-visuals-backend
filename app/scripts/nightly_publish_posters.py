# app/scripts/nightly_publish_posters.py
import os
import json
import argparse
from datetime import datetime, timezone

from app.services.storage_supabase import upload_file_return_url
from app.scripts.nfl_stat_leaders_generate import main as stat_leaders_main
from app.scripts.nfl_standings_conference_generate import generate_standings_conference_png


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ensure_tmp():
    os.makedirs("/tmp", exist_ok=True)


def generate_stat_leaders_pngs(season: int, seasontype: int) -> tuple[str, str]:
    """
    Runs your existing CLI generator, but forces output to /tmp (GitHub Actions safe).
    Returns (offense_path, defense_path).
    """
    ensure_tmp()
    tag = "reg" if seasontype == 2 else "post"

    out_off = f"/tmp/offense_stat_leaders_{season}_{tag}.png"
    out_def = f"/tmp/defense_stat_leaders_{season}_{tag}.png"

    # Call your script's main() by temporarily patching argv
    import sys
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "nfl_stat_leaders_generate.py",
            "--season", str(season),
            "--seasontype", str(seasontype),
            "--outdir", "/tmp",
        ]
        stat_leaders_main()
    finally:
        sys.argv = old_argv

    if not os.path.exists(out_off):
        raise RuntimeError(f"Missing generated file: {out_off}")
    if not os.path.exists(out_def):
        raise RuntimeError(f"Missing generated file: {out_def}")

    return out_off, out_def


def generate_standings_png(season: int) -> str:
    ensure_tmp()
    out_path = f"/tmp/standings_conference_{season}.png"
    generate_standings_conference_png(season, out_path)
    if not os.path.exists(out_path):
        raise RuntimeError(f"Missing generated file: {out_path}")
    return out_path


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

    # Storage keys
    # If keep_versioned: posters/2026-02-17/...
    # Else: posters/latest/...
    base = f"posters/{day}" if args.keep_versioned else "posters/latest"

    # Generate locally
    standings_local = generate_standings_png(season)
    off_local, def_local = generate_stat_leaders_pngs(season, seasontype)

    # Upload to Supabase (your uploader returns public URL)
    standings_key = f"{base}/standings_conference_s{season}.png"
    offense_key = f"{base}/leaders_offense_s{season}_t{seasontype}.png"
    defense_key = f"{base}/leaders_defense_s{season}_t{seasontype}.png"

    standings_url = upload_file_return_url(standings_local, standings_key)
    offense_url = upload_file_return_url(off_local, offense_key)
    defense_url = upload_file_return_url(def_local, defense_key)

    # Manifest ALWAYS at a stable path
    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "seasontype": seasontype,
        "standings_url": standings_url,
        "leaders_offense_url": offense_url,
        "leaders_defense_url": defense_url,
        "versioned": bool(args.keep_versioned),
        "base_path": base,
    }

    manifest_local = write_manifest(manifest)
    manifest_url = upload_file_return_url(manifest_local, "posters/latest.json")

    print("\n✅ Uploaded URLs")
    print("standings_url:", standings_url)
    print("leaders_offense_url:", offense_url)
    print("leaders_defense_url:", defense_url)
    print("manifest_url:", manifest_url)
    print("")


if __name__ == "__main__":
    main()
