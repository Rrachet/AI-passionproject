from __future__ import annotations

from math import hypot
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AirCanvas Vision Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0


class HandFrame(BaseModel):
    landmarks: list[Landmark] = Field(min_length=21, max_length=21)


Gesture = Literal["DRAW", "PAUSE", "UNKNOWN"]


def classify_gesture(points: list[Landmark]) -> Gesture:
    """Classify the first MVP gesture from normalized hand landmarks.

    MediaPipe landmark indices used here:
    6/8 index PIP/tip, 10/12 middle, 14/16 ring, 18/20 pinky.
    The backend owns the gesture decision so the browser remains a thin client.
    """
    index_extended = points[8].y < points[6].y
    middle_folded = points[12].y > points[10].y
    ring_folded = points[16].y > points[14].y
    pinky_folded = points[20].y > points[18].y

    if index_extended and middle_folded and ring_folded and pinky_folded:
        return "DRAW"
    if not index_extended:
        return "PAUSE"
    return "UNKNOWN"


def movement_score(previous: Landmark | None, current: Landmark) -> float:
    if previous is None:
        return 0.0
    return hypot(current.x - previous.x, current.y - previous.y)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aircanvas-python-vision"}


@app.post("/gesture")
def gesture(frame: HandFrame) -> dict[str, object]:
    result = classify_gesture(frame.landmarks)
    return {"gesture": result, "confidence": 1.0 if result != "UNKNOWN" else 0.0}


@app.websocket("/ws/vision")
async def vision_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    previous: Landmark | None = None

    try:
        while True:
            payload = await websocket.receive_json()
            frame = HandFrame.model_validate(payload)
            tip = frame.landmarks[8]
            gesture_name = classify_gesture(frame.landmarks)
            movement = movement_score(previous, tip)
            previous = tip

            await websocket.send_json(
                {
                    "gesture": gesture_name,
                    "movement": round(movement, 6),
                    "tip": {"x": tip.x, "y": tip.y},
                }
            )
    except WebSocketDisconnect:
        return
