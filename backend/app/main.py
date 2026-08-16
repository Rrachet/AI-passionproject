from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .tracker import Point, TemporalTracker, classify, stable_gesture

app = FastAPI(title="AirCanvas Python Vision Engine", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

Gesture = Literal["DRAW", "PAUSE", "CLEAR", "UNKNOWN"]


class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0


class HandFrame(BaseModel):
    landmarks: list[Landmark] = Field(min_length=21, max_length=21)
    timestamp: float = 0.0
    confidence: float = 1.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aircanvas-python-vision", "engine": "temporal-v0.2"}


def points(frame: HandFrame) -> list[Point]:
    return [Point(p.x, p.y) for p in frame.landmarks]


def analyse(frame: HandFrame, tracker: TemporalTracker) -> dict:
    pts = points(frame)
    gesture: Gesture = stable_gesture(tracker, classify(pts))
    tip = tracker.update(pts[8], frame.timestamp or 0.0, frame.confidence)
    return {"gesture": gesture, "tip": tip, "tracking": tip["confidence"] >= 0.28}


@app.post("/gesture")
def gesture(frame: HandFrame) -> dict:
    tracker = TemporalTracker()
    return analyse(frame, tracker)


@app.websocket("/ws/vision")
async def vision_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    tracker = TemporalTracker()
    try:
        while True:
            payload = await websocket.receive_json()
            frame = HandFrame.model_validate(payload)
            result = analyse(frame, tracker)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        tracker.reset()
        return
