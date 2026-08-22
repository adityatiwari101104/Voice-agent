import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from livekit import api

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
    raise RuntimeError(
        "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must all be "
        "set in .env -- these are needed to mint room tokens for the "
        "test frontend."
    )

app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    """Serve the test page. Same-origin as /token, so no CORS needed."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/token")
def get_token(identity: str = "teacher", room: str = "question-paper-room"):
    """Mint a short-lived LiveKit room token for the browser to connect
    with. The API secret never reaches the browser -- only this signed
    JWT does."""

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )

    return {
        "token": token,
        "url": LIVEKIT_URL,
        "room": room,
        "identity": identity,
    }


# NOTE: this serves the single fixed output/question_paper.pdf path that
# pdf_builder.py always writes to -- fine for testing one conversation at
# a time, but it gets overwritten on every new paper. Once multiple
# teachers/sessions run concurrently, this needs to become session- or
# room-specific (e.g. name the output file after the room/job ID and
# look it up here by room name) rather than always serving "the latest
# paper anyone generated".
@app.get("/paper")
def get_latest_paper():
    path = Path("output") / "question_paper.pdf"

    if not path.exists():
        return JSONResponse(
            {"error": "No paper has been generated yet."},
            status_code=404,
        )

    return FileResponse(
        path,
        media_type="application/pdf",
        filename="question_paper.pdf",
        headers={
            # Without this, browsers can silently serve a cached
            # response for a repeat GET /paper -- meaning a second
            # download after generating a *new* paper can return the
            # *previous* paper's bytes instead of re-fetching.
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )