from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atan2, cos, degrees, hypot, pi, sin
from typing import Literal

Gesture = Literal['DRAW', 'PAUSE', 'CLEAR', 'UNKNOWN']

@dataclass
class Point:
    x: float
    y: float

@dataclass
class OneEuro:
    min_cutoff: float = 1.4
    beta: float = 0.08
    d_cutoff: float = 1.0
    x_prev: float | None = None
    dx_prev: float = 0.0
    @staticmethod
    def alpha(cutoff: float, dt: float) -> float:
        r = 2.0 * pi * cutoff * dt
        return r / (r + 1.0)
    def filter(self, value: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = value; self.dx_prev = 0.0; return value
        safe_dt = max(dt, 1e-3)
        raw_dx = (value - self.x_prev) / safe_dt
        a_d = self.alpha(self.d_cutoff, safe_dt)
        self.dx_prev += a_d * (raw_dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self.alpha(cutoff, safe_dt)
        self.x_prev += a * (value - self.x_prev)
        return self.x_prev
    def reset(self) -> None:
        self.x_prev = None; self.dx_prev = 0.0

@dataclass
class TemporalTracker:
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
        x = self.x_filter.filter(raw.x, dt); y = self.y_filter.filter(raw.y, dt)
        current = Point(x, y)
        if self.previous: self.velocity = Point((x - self.previous.x) / dt, (y - self.previous.y) / dt)
        self.previous = current; self.missing_frames = 0
        self.confidence = min(1.0, max(self.confidence * 0.65, max(0.0, min(1.0, detection_confidence))))
        return {'x': round(x, 6), 'y': round(y, 6), 'vx': round(self.velocity.x, 6), 'vy': round(self.velocity.y, 6), 'speed': round(hypot(self.velocity.x, self.velocity.y), 6), 'confidence': round(self.confidence, 4)}
    def miss(self) -> dict:
        self.missing_frames += 1; self.confidence *= 0.72; predicted = False
        if self.previous and self.missing_frames <= 3:
            self.previous = Point(self.previous.x + self.velocity.x / 30, self.previous.y + self.velocity.y / 30); predicted = True
        return {'x': round(self.previous.x, 6) if self.previous else None, 'y': round(self.previous.y, 6) if self.previous else None, 'confidence': round(self.confidence, 4), 'predicted': predicted}
    def reset(self) -> None:
        self.x_filter.reset(); self.y_filter.reset(); self.previous = None; self.velocity = Point(0.0, 0.0); self.confidence = 0.0; self.last_timestamp = None; self.missing_frames = 0; self.gesture_memory.clear()

def angle(a: Point, b: Point, c: Point) -> float:
    ab = (a.x - b.x, a.y - b.y); cb = (c.x - b.x, c.y - b.y); denom = (hypot(*ab) * hypot(*cb)) or 1e-9
    return degrees(acos(max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / denom))))

def extended(points: list[Point], mcp: int, pip: int, tip: int) -> bool:
    return angle(points[mcp], points[pip], points[tip]) > 138 and hypot(points[tip].x - points[mcp].x, points[tip].y - points[mcp].y) > hypot(points[pip].x - points[mcp].x, points[pip].y - points[mcp].y) * 1.03

def classify(points: list[Point]) -> Gesture:
    if len(points) != 21: return 'UNKNOWN'
    index = extended(points, 5, 6, 8); middle = extended(points, 9, 10, 12); ring = extended(points, 13, 14, 16); pinky = extended(points, 17, 18, 20)
    if index and not middle and not ring and not pinky: return 'DRAW'
    if index and middle and ring and pinky: return 'CLEAR'
    if not index and not middle and not ring and not pinky: return 'PAUSE'
    return 'UNKNOWN'

def stable_gesture(tracker: TemporalTracker, gesture: Gesture, window: int = 5) -> Gesture:
    tracker.gesture_memory.append(gesture); del tracker.gesture_memory[:-window]
    counts = {g: tracker.gesture_memory.count(g) for g in ('DRAW', 'PAUSE', 'CLEAR', 'UNKNOWN')}; winner = max(counts, key=counts.get)
    return winner if counts[winner] >= 3 else 'UNKNOWN'

def _distance_to_line(p: Point, a: Point, b: Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    if dx == 0 and dy == 0: return hypot(p.x - a.x, p.y - a.y)
    return abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / hypot(dx, dy)

def simplify(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) <= 2: return points[:]
    i = max(range(1, len(points) - 1), key=lambda j: _distance_to_line(points[j], points[0], points[-1]))
    if _distance_to_line(points[i], points[0], points[-1]) > epsilon:
        return simplify(points[:i + 1], epsilon)[:-1] + simplify(points[i:], epsilon)
    return [points[0], points[-1]]

def regular_polygon(points: list[Point]) -> dict:
    if len(points) < 4: return {'shape': 'freehand', 'sides': 0, 'points': []}
    min_x, max_x = min(p.x for p in points), max(p.x for p in points); min_y, max_y = min(p.y for p in points), max(p.y for p in points)
    scale = max(max_x - min_x, max_y - min_y, 1.0)
    closed = hypot(points[0].x - points[-1].x, points[0].y - points[-1].y) <= scale * 0.28
    if not closed: return {'shape': 'line', 'sides': 2, 'points': [{'x': points[0].x, 'y': points[0].y}, {'x': points[-1].x, 'y': points[-1].y}]}
    ring = points[:-1] if points[0].x == points[-1].x and points[0].y == points[-1].y else points
    cx = sum(p.x for p in ring) / len(ring); cy = sum(p.y for p in ring) / len(ring)
    radius = sum(hypot(p.x - cx, p.y - cy) for p in ring) / len(ring)
    if radius < 4: return {'shape': 'freehand', 'sides': 0, 'points': []}
    radial = [hypot(p.x - cx, p.y - cy) for p in ring]; mean = sum(radial) / len(radial)
    radial_error = (sum((r - mean) ** 2 for r in radial) / len(radial)) ** 0.5 / max(mean, 1e-6)
    corners = simplify(ring + [ring[0]], max(4.0, scale * 0.075))
    if len(corners) > 1 and hypot(corners[0].x - corners[-1].x, corners[0].y - corners[-1].y) < scale * 0.2: corners = corners[:-1]
    n = max(3, min(12, len(corners)))
    if radial_error < 0.10 or n >= 9: n = 64
    if n == 64:
        return {'shape': 'circle', 'sides': 0, 'points': [{'x': cx + radius * cos(2*pi*i/64), 'y': cy + radius * sin(2*pi*i/64)} for i in range(65)]}
    start = atan2(corners[0].y - cy, corners[0].x - cx)
    regular = [{'x': cx + radius * cos(start + 2*pi*i/n), 'y': cy + radius * sin(start + 2*pi*i/n)} for i in range(n)]
    regular.append(regular[0].copy())
    names = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon'}
    return {'shape': names.get(n, f'{n}-gon'), 'sides': n, 'points': regular}
