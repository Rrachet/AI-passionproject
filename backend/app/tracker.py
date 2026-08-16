from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, degrees, hypot
from typing import Literal

Gesture = Literal['DRAW', 'PAUSE', 'CLEAR', 'UNKNOWN']


@dataclass
class Point:
    x: float
    y: float


@dataclass
class OneEuro:
    """Adaptive low-pass filter for noisy fingertip landmarks."""
    min_cutoff: float = 1.4
    beta: float = 0.08
    d_cutoff: float = 1.0
    x_prev: float | None = None
    dx_prev: float = 0.0

    @staticmethod
    def alpha(cutoff: float, dt: float) -> float:
        r = 2.0 * 3.141592653589793 * cutoff * dt
        return r / (r + 1.0)

    def filter(self, value: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = value
            self.dx_prev = 0.0
            return value
        safe_dt = max(dt, 1e-3)
        raw_dx = (value - self.x_prev) / safe_dt
        a_d = self.alpha(self.d_cutoff, safe_dt)
        self.dx_prev += a_d * (raw_dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self.alpha(cutoff, safe_dt)
        self.x_prev += a * (value - self.x_prev)
        return self.x_prev

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0


@dataclass
class TemporalTracker:
    """Stateful fingertip tracker used by the WebSocket vision engine."""
    x_filter: OneEuro = field(default_factory=OneEuro)
    y_filter: OneEuro = field(default_factory=OneEuro)
    previous: Point | None = None
    velocity: Point = field(default_factory=lambda: Point(0.0, 0.0))
    confidence: float = 0.0
    last_timestamp: float | None = None
    missing_frames: int = 0
    gesture_memory: list[Gesture] = field(default_factory=list)

    def update(self, raw: Point, timestamp: float, detection_confidence: float = 1.0) -> dict:
        dt = 1 / 30 if self.last_timestamp is None else max(1 / 120, min(0.2, timestamp - self.last_timestamp))
        self.last_timestamp = timestamp
        x = self.x_filter.filter(raw.x, dt)
        y = self.y_filter.filter(raw.y, dt)
        current = Point(x, y)
        if self.previous:
            self.velocity = Point((x - self.previous.x) / dt, (y - self.previous.y) / dt)
        self.previous = current
        self.missing_frames = 0
        self.confidence = min(1.0, max(self.confidence * 0.65, max(0.0, min(1.0, detection_confidence))))
        return {
            'x': round(x, 6),
            'y': round(y, 6),
            'vx': round(self.velocity.x, 6),
            'vy': round(self.velocity.y, 6),
            'speed': round(hypot(self.velocity.x, self.velocity.y), 6),
            'confidence': round(self.confidence, 4),
        }

    def miss(self) -> dict:
        self.missing_frames += 1
        self.confidence *= 0.72
        predicted = False
        if self.previous and self.missing_frames <= 3:
            self.previous = Point(
                self.previous.x + self.velocity.x / 30,
                self.previous.y + self.velocity.y / 30,
            )
            predicted = True
        return {
            'x': round(self.previous.x, 6) if self.previous else None,
            'y': round(self.previous.y, 6) if self.previous else None,
            'confidence': round(self.confidence, 4),
            'predicted': predicted,
        }

    def reset(self) -> None:
        self.x_filter.reset()
        self.y_filter.reset()
        self.previous = None
        self.velocity = Point(0.0, 0.0)
        self.confidence = 0.0
        self.last_timestamp = None
        self.missing_frames = 0
        self.gesture_memory.clear()


def angle(a: Point, b: Point, c: Point) -> float:
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    denom = (hypot(*ab) * hypot(*cb)) or 1e-9
    cosine = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / denom))
    return degrees(acos(cosine))


def extended(points: list[Point], mcp: int, pip: int, tip: int) -> bool:
    return angle(points[mcp], points[pip], points[tip]) > 138 and hypot(
        points[tip].x - points[mcp].x,
        points[tip].y - points[mcp].y,
    ) > hypot(
        points[pip].x - points[mcp].x,
        points[pip].y - points[mcp].y,
    ) * 1.03


def classify(points: list[Point]) -> Gesture:
    if len(points) != 21:
        return 'UNKNOWN'
    index = extended(points, 5, 6, 8)
    middle = extended(points, 9, 10, 12)
    ring = extended(points, 13, 14, 16)
    pinky = extended(points, 17, 18, 20)
    if index and not middle and not ring and not pinky:
        return 'DRAW'
    if index and middle and ring and pinky:
        return 'CLEAR'
    if not index and not middle and not ring and not pinky:
        return 'PAUSE'
    return 'UNKNOWN'


def stable_gesture(tracker: TemporalTracker, gesture: Gesture, window: int = 5) -> Gesture:
    tracker.gesture_memory.append(gesture)
    del tracker.gesture_memory[:-window]
    counts = {g: tracker.gesture_memory.count(g) for g in ('DRAW', 'PAUSE', 'CLEAR', 'UNKNOWN')}
    winner = max(counts, key=counts.get)
    return winner if counts[winner] >= 3 else 'UNKNOWN'
