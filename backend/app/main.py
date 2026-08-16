from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .tracker import Point, TemporalTracker, classify, stable_gesture
from .polygon import align_shape

app = FastAPI(title='AirCanvas Python Vision Engine', version='0.6.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=False, allow_methods=['*'], allow_headers=['*'])
Gesture = Literal['DRAW', 'PAUSE', 'CLEAR', 'UNKNOWN']

class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0

class HandFrame(BaseModel):
    landmarks: list[Landmark] = Field(min_length=21, max_length=21)
    timestamp: float = 0.0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class Stroke(BaseModel):
    points: list[Landmark] = Field(min_length=4, max_length=5000)

@app.get('/')
def root() -> dict[str, str]:
    return {'service': 'aircanvas-python-vision', 'version': '0.6.0', 'websocket': '/ws/vision'}

@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'aircanvas-python-vision', 'engine': 'temporal-v0.6'}

def analyse(frame: HandFrame, tracker: TemporalTracker) -> dict:
    pts = [Point(p.x, p.y) for p in frame.landmarks]
    gesture: Gesture = stable_gesture(tracker, classify(pts))
    timestamp = frame.timestamp if frame.timestamp > 0 else (tracker.last_timestamp or 0.0) + 1 / 30
    tip = tracker.update(pts[8], timestamp, frame.confidence)
    return {'gesture': gesture, 'tip': tip, 'tracking': tip['confidence'] >= 0.28}

@app.post('/gesture')
def gesture(frame: HandFrame) -> dict:
    return analyse(frame, TemporalTracker())

@app.post('/shape')
def shape(frame: Stroke) -> dict:
    return align_shape([Point(p.x, p.y) for p in frame.points])

@app.websocket('/ws/vision')
async def vision_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    tracker = TemporalTracker()
    try:
        while True:
            frame = HandFrame.model_validate(await websocket.receive_json())
            await websocket.send_json(analyse(frame, tracker))
    except WebSocketDisconnect:
        tracker.reset()
    except Exception:
        tracker.reset()
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
