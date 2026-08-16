from __future__ import annotations

from math import cos, hypot, pi, sin
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .tracker import Point, TemporalTracker, classify, stable_gesture

app = FastAPI(title='AirCanvas Python Vision Engine', version='0.4.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

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
    return {'service': 'aircanvas-python-vision', 'version': '0.4.0', 'websocket': '/ws/vision'}


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'aircanvas-python-vision', 'engine': 'temporal-v0.4'}


def analyse(frame: HandFrame, tracker: TemporalTracker) -> dict:
    pts = [Point(p.x, p.y) for p in frame.landmarks]
    gesture: Gesture = stable_gesture(tracker, classify(pts))
    timestamp = frame.timestamp if frame.timestamp > 0 else (tracker.last_timestamp or 0.0) + 1 / 30
    tip = tracker.update(pts[8], timestamp, frame.confidence)
    return {
        'gesture': gesture,
        'tip': tip,
        'tracking': tip['confidence'] >= 0.28,
    }


@app.post('/gesture')
def gesture(frame: HandFrame) -> dict:
    # REST requests are intentionally stateless. Use /ws/vision for continuous tracking.
    return analyse(frame, TemporalTracker())


def shape_alignment(raw: list[Point]) -> dict:
    if len(raw) < 4:
        return {'shape': 'freehand', 'points': []}
    xs = [p.x for p in raw]
    ys = [p.y for p in raw]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    w, h = max_x - min_x, max_y - min_y
    closed = hypot(raw[0].x - raw[-1].x, raw[0].y - raw[-1].y) <= max(w, h) * 0.22
    if not closed:
        return {'shape': 'line', 'points': [{'x': raw[0].x, 'y': raw[0].y}, {'x': raw[-1].x, 'y': raw[-1].y}]}
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    if w == 0 or h == 0:
        return {'shape': 'line', 'points': [{'x': raw[0].x, 'y': raw[0].y}, {'x': raw[-1].x, 'y': raw[-1].y}]}
    aspect = w / h
    if 0.78 <= aspect <= 1.28:
        r = (w + h) / 4
        points = [
            {'x': cx + r * cos(i * 2 * pi / 64), 'y': cy + r * sin(i * 2 * pi / 64)}
            for i in range(65)
        ]
        return {'shape': 'circle', 'points': points}
    return {
        'shape': 'rectangle',
        'points': [
            {'x': min_x, 'y': min_y},
            {'x': max_x, 'y': min_y},
            {'x': max_x, 'y': max_y},
            {'x': min_x, 'y': max_y},
            {'x': min_x, 'y': min_y},
        ],
    }


@app.post('/shape')
def shape(frame: Stroke) -> dict:
    return shape_alignment([Point(p.x, p.y) for p in frame.points])


@app.websocket('/ws/vision')
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
    except Exception:
        tracker.reset()
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
