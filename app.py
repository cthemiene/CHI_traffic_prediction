"""
REST API for parking lot monitoring.

This module exposes a basic API using FastAPI that allows clients (e.g., a
Flutter mobile application) to retrieve current occupancy status and
short‑term forecasts.  The API endpoints are documented using OpenAPI
automatically by FastAPI.  To run the API server, install the ``fastapi``
and ``uvicorn`` packages and execute this module.

Example
-------
```
pip install fastapi uvicorn
uvicorn parking_lot_monitor.app:app --reload
```

Endpoints
---------
* ``GET /current``: Return the latest occupancy state and annotated frame.
* ``GET /forecast``: Return a forecast for the next 12 hours and the best
  time to visit during operating hours.

This file is optional for the user and is provided for completeness.  It
demonstrates how to integrate the core detection and forecasting logic
within a web service.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.requests import Request
import cv2  # type: ignore

from .detection import ParkingLotMonitor, ParkingSpot
from .forecasting import Observation, OccupancyForecaster


# Initialise FastAPI app
app = FastAPI(title="Parking Lot Monitoring API", version="1.0.0")


# Define Pydantic models for responses
class OccupancyResponse(BaseModel):
    timestamp: float
    occupancy: List[bool]
    image_base64: str


class ForecastResponse(BaseModel):
    forecast: List[dict]
    best_time: str
    predicted_ratio: float


# Example configuration: two parking spots and model path
PARKING_SPOTS = [
    ParkingSpot("A1", (0, 0), (100, 100)),
    ParkingSpot("A2", (120, 0), (220, 100)),
]

MODEL_PATH = "yolov8n.pt"  # Replace with real path to YOLO weights

try:
    monitor = ParkingLotMonitor(model_weights=MODEL_PATH, parking_spots=PARKING_SPOTS, device="cpu")
except RuntimeError as exc:
    # Delay error until first request
    monitor = None  # type: ignore

forecaster = OccupancyForecaster(timezone="America/Chicago", operating_hours=(7, 19), resample_minutes=15)
observations: List[Observation] = []


def frame_to_base64(frame) -> str:
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.on_event("startup")
async def startup_event() -> None:
    # Placeholder for any startup initialization.  You could load model weights
    # here or read existing logs into `observations`.
    pass


@app.get("/current", response_model=OccupancyResponse)
async def current_occupancy() -> OccupancyResponse:
    """Return the latest occupancy status and annotated image.

    This endpoint captures a single frame from the default webcam, runs the
    detector and returns the occupancy state along with a JPEG of the
    annotated image encoded in base64.
    """
    if monitor is None:
        raise HTTPException(status_code=500, detail="Object detection model is not initialized.")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=500, detail="Unable to capture frame from webcam.")
    occupancy = monitor.analyze_frame(frame)
    annotated = monitor.draw_overlay(frame, occupancy)
    # Append observation with current time and occupancy ratio
    now = dt.datetime.now(tz=forecaster.tz)
    ratio = sum(occupancy) / len(occupancy)
    observations.append(Observation(timestamp=now, ratio=ratio))
    return OccupancyResponse(
        timestamp=now.timestamp(),
        occupancy=occupancy,
        image_base64=frame_to_base64(annotated),
    )


@app.get("/forecast", response_model=ForecastResponse)
async def forecast() -> ForecastResponse:
    """Generate a forecast for the next 12 hours and suggest a best visit time."""
    if not observations:
        raise HTTPException(status_code=400, detail="Not enough data for forecasting.")
    df = forecaster.load_observations(observations)
    if not forecaster._fitted:
        forecaster.fit(df)
    fc = forecaster.forecast(horizon_hours=12)
    best_time, pred_ratio = forecaster.best_time_to_visit(fc)
    # Serialize forecast
    serialized = [
        {"timestamp": row.timestamp.isoformat(), "ratio_pred": float(row.ratio_pred)}
        for row in fc.itertuples(index=False)
    ]
    return ForecastResponse(
        forecast=serialized,
        best_time=best_time.isoformat(),
        predicted_ratio=pred_ratio,
    )