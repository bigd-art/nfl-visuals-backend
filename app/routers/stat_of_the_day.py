from fastapi import APIRouter, HTTPException
import requests

from app.scripts.stat_of_the_day_publish import get_current_stat_of_the_day_payload

router = APIRouter(prefix="/stat-of-the-day", tags=["stat-of-the-day"])


@router.get("/current")
def stat_of_the_day_current():
    try:
        payload = get_current_stat_of_the_day_payload()

        meta_resp = requests.get(payload["metadata_url"], timeout=20)
        if meta_resp.ok:
            meta = meta_resp.json()
            return {
                "image_url": payload["image_url"],
                "metadata_url": payload["metadata_url"],
                "category_key": meta.get("category_key"),
                "category_label": meta.get("category_label"),
                "season": meta.get("season"),
                "week": meta.get("week"),
                "title": meta.get("title"),
                "team": meta.get("team"),
                "player": meta.get("player"),
                "date": meta.get("date"),
            }

        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
