#!/usr/bin/env python3
import argparse
import json
import os
import tempfile

from app.services.storage_supabase import upload_file_return_url
import app.scripts.tankathon_mock_posters_v4 as tankathon


def public_storage_url(storage_key: str) -> str:
    base = os.environ["SUPABASE_URL"].rstrip("/")
    bucket = os.environ.get("SUPABASE_BUCKET", "nfl-posters")
    return f"{base}/storage/v1/object/public/{bucket}/{storage_key}"


def publish_tankathon_mock(keep_versioned: bool = False):
    with tempfile.TemporaryDirectory() as tmpdir:
        tankathon.OUTPUT_DIR = tmpdir
        tankathon.ensure_dir(tmpdir)

        html = tankathon.fetch_html(tankathon.SOURCE_URL)
        full_text = tankathon.soup_to_text_with_img_alts(html)
        round1_text = tankathon.extract_round1_text(full_text)
        picks = tankathon.parse_round1_picks(round1_text)
        chunks = [picks[i:i + tankathon.ROWS_PER_POSTER] for i in range(0, 32, tankathon.ROWS_PER_POSTER)]

        poster_urls = []
        for idx, chunk in enumerate(chunks, start=1):
            img = tankathon.render_poster(chunk)
            filename = f"tankathon_mock_poster_{idx}.png"
            local_path = os.path.join(tmpdir, filename)
            img.save(local_path)
            poster_urls.append(
                upload_file_return_url(local_path, f"tankathon_mock/current/{filename}")
            )

        payload = {
            "count": len(poster_urls),
            "poster_urls": poster_urls,
        }

        local_json = os.path.join(tmpdir, "current.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        payload["metadata_url"] = upload_file_return_url(local_json, "tankathon_mock/current.json")

        if keep_versioned:
            versioned_urls = []
            for idx in range(1, len(poster_urls) + 1):
                filename = f"tankathon_mock_poster_{idx}.png"
                local_path = os.path.join(tmpdir, filename)
                versioned_urls.append(
                    upload_file_return_url(local_path, f"tankathon_mock/history/{filename}")
                )
            payload["versioned_poster_urls"] = versioned_urls
            payload["versioned_metadata_url"] = upload_file_return_url(
                local_json,
                "tankathon_mock/history/metadata.json"
            )

        return payload


def get_current_tankathon_mock_payload():
    return {
        "metadata_url": public_storage_url("tankathon_mock/current.json"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keep_versioned", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    result = publish_tankathon_mock(keep_versioned=args.keep_versioned)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
