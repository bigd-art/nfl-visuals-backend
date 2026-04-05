from fastapi import APIRouter, HTTPException, Query
import requests

from app.scripts.nightly_publish_pff_big_board import get_current_pff_big_board_payload

router = APIRouter(prefix="/pff-big-board", tags=["pff-big-board"])


@router.get("/current")
def pff_big_board_current(position: str = Query(default="")):
    try:
        payload = get_current_pff_big_board_payload()
        resp = requests.get(payload["metadata_url"], timeout=20)
        if not resp.ok:
            return payload

        meta = resp.json()

        if position:
            pos = position.strip().upper()
            return {
                "season": meta.get("season"),
                "position": pos,
                "image_url": meta.get("posters", {}).get(pos),
                "metadata_url": meta.get("metadata_url"),
            }

        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
