# app/jobs/nightly_stat_leaders.py
import os
import tempfile
from datetime import datetime
from pathlib import Path

from supabase import create_client

# Import your generator script (already in repo)
from app.scripts.nfl_stat_leaders_generate import main as generate_main


def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def public_url(supabase_url: str, bucket: str, path: str) -> str:
    # Works if bucket is PUBLIC
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{path}"


def upload_file(supabase, bucket: str, local_path: str, dest_path: str):
    with open(local_path, "rb") as f:
        data = f.read()

    # upsert=True overwrites existing (perfect for nightly refresh)
    res = supabase.storage.from_(bucket).upload(
        path=dest_path,
        file=data,
        file_options={"content-type": "image/png", "upsert": True},
    )
    return res


def run_one(season: int, seasontype: int, bucket: str, folder: str, supabase_url: str, supabase):
    # Generate into a temp folder
    with tempfile.TemporaryDirectory() as td:
        outdir = Path(td)

        # Call generator main() with args (simulate CLI)
        # It expects argparse args; easiest is to run by setting sys.argv.
        import sys
        sys_argv_backup = sys.argv[:]
        try:
            sys.argv = [
                "nfl_stat_leaders_generate.py",
                "--season", str(season),
                "--seasontype", str(seasontype),
                "--outdir", str(outdir),
            ]
            generate_main()
        finally:
            sys.argv = sys_argv_backup

        tag = "reg" if seasontype == 2 else "post"

        off_local = outdir / f"offense_stat_leaders_{season}_{tag}.png"
        def_local = outdir / f"defense_stat_leaders_{season}_{tag}.png"

        if not off_local.exists() or not def_local.exists():
            raise RuntimeError(f"Missing generated files for {season} {tag}")

        off_dest = f"{folder}/{season}/{tag}/offense.png"
        def_dest = f"{folder}/{season}/{tag}/defense.png"

        upload_file(supabase, bucket, str(off_local), off_dest)
        upload_file(supabase, bucket, str(def_local), def_dest)

        return {
            "offense_url": public_url(supabase_url, bucket, off_dest),
            "defense_url": public_url(supabase_url, bucket, def_dest),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }


def main():
    # REQUIRED ENV VARS on Render
    supabase_url = _env("SUPABASE_URL")
    supabase_key = _env("SUPABASE_SERVICE_ROLE_KEY")  # service role key
    bucket = _env("SUPABASE_BUCKET")                  # your bucket name (PUBLIC)
    season = int(os.getenv("STAT_LEADERS_SEASON", "2025"))
    folder = os.getenv("STAT_LEADERS_FOLDER", "stat-leaders")

    supabase = create_client(supabase_url, supabase_key)

    reg = run_one(season, 2, bucket, folder, supabase_url, supabase)
    post = run_one(season, 3, bucket, folder, supabase_url, supabase)

    print("DONE ✅")
    print("REG:", reg)
    print("POST:", post)


if __name__ == "__main__":
    main()

