#!/usr/bin/env python3
"""main.py — FastAPI inference server for AntID.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /health                      -> status + species_count + geo_index_loaded
    GET  /species                     -> all known species
    POST /identify?lat=&lon=          -> top-3 results (lat/lon optional)
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from inference import AntIdentifier

app = FastAPI(title="AntID", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts once. If this raises, the server won't start — intended.
identifier = AntIdentifier()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "species_count": identifier.species_count,
        "geo_index_loaded": identifier.geo_index_loaded,
    }


@app.get("/species")
def species() -> dict:
    return {"species": identifier.species_list()}


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    lat: float | None = Query(None, ge=-90.0, le=90.0),
    lon: float | None = Query(None, ge=-180.0, le=180.0),
) -> dict:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Upload must be an image.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty file.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="Could not decode image.")
    # If only one of lat/lon arrives, ignore location entirely.
    if (lat is None) != (lon is None):
        lat = lon = None
    return identifier.identify(img, lat=lat, lon=lon)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
