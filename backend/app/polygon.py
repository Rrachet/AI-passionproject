from __future__ import annotations

from math import atan2, cos, hypot, pi, sin
from .tracker import Point


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def convex_hull(points: list[Point]) -> list[Point]:
    unique = sorted({(round(p.x, 5), round(p.y, 5)) for p in points})
    if len(unique) <= 2:
        return [Point(x, y) for x, y in unique]
    pts = [Point(x, y) for x, y in unique]
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _distance_to_segment(p: Point, a: Point, b: Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return hypot(p.x - a.x, p.y - a.y)
    t = max(0.0, min(1.0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length2))
    qx, qy = a.x + t * dx, a.y + t * dy
    return hypot(p.x - qx, p.y - qy)


def simplify(points: list[Point], epsilon: float) -> list[Point]:
    if len(points) <= 2:
        return points[:]
    best_i = 1
    best = 0.0
    for i in range(1, len(points) - 1):
        d = _distance_to_segment(points[i], points[0], points[-1])
        if d > best:
            best, best_i = d, i
    if best > epsilon:
        left = simplify(points[:best_i + 1], epsilon)
        right = simplify(points[best_i:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def _regular(cx: float, cy: float, radius: float, sides: int, start: float) -> list[dict[str, float]]:
    out = [
        {'x': cx + radius * cos(start + 2 * pi * i / sides),
         'y': cy + radius * sin(start + 2 * pi * i / sides)}
        for i in range(sides)
    ]
    out.append(out[0].copy())
    return out


def align_shape(points: list[Point]) -> dict:
    if len(points) < 8:
        return {'shape': 'freehand', 'sides': 0, 'confidence': 0.0, 'points': []}

    min_x, max_x = min(p.x for p in points), max(p.x for p in points)
    min_y, max_y = min(p.y for p in points), max(p.y for p in points)
    width, height = max_x - min_x, max_y - min_y
    scale = max(width, height, 1.0)
    close_limit = max(14.0, scale * 0.22)
    closed = hypot(points[0].x - points[-1].x, points[0].y - points[-1].y) <= close_limit

    if not closed:
        return {
            'shape': 'line', 'sides': 2, 'confidence': 1.0,
            'points': [{'x': points[0].x, 'y': points[0].y}, {'x': points[-1].x, 'y': points[-1].y}],
        }

    ring = points[:-1] if hypot(points[0].x - points[-1].x, points[0].y - points[-1].y) <= close_limit else points[:]
    cx = sum(p.x for p in ring) / len(ring)
    cy = sum(p.y for p in ring) / len(ring)
    radii = [hypot(p.x - cx, p.y - cy) for p in ring]
    radius = sum(radii) / len(radii)
    if radius < 8:
        return {'shape': 'freehand', 'sides': 0, 'confidence': 0.0, 'points': []}

    radial_error = (sum((r - radius) ** 2 for r in radii) / len(radii)) ** 0.5 / radius
    hull = convex_hull(ring)
    if len(hull) < 3:
        return {'shape': 'freehand', 'sides': 0, 'confidence': 0.0, 'points': []}

    # Round strokes have a dense convex hull and low radial error.
    simplified = simplify(hull + [hull[0]], max(3.0, scale * 0.06))
    if len(simplified) > 1 and hypot(simplified[0].x - simplified[-1].x, simplified[0].y - simplified[-1].y) < scale * 0.16:
        simplified = simplified[:-1]

    if radial_error < 0.10 or len(simplified) >= 9:
        return {
            'shape': 'circle', 'sides': 0,
            'confidence': round(max(0.0, min(1.0, 1.0 - radial_error)), 3),
            'points': _regular(cx, cy, radius, 72, 0.0),
        }

    sides = max(3, min(8, len(simplified)))
    names = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon'}

    # Choose orientation from the strongest corner, then regularize every side.
    corner = max(simplified, key=lambda p: hypot(p.x - cx, p.y - cy))
    start = atan2(corner.y - cy, corner.x - cx)
    target_radius = min(radius, scale * 0.70)
    confidence = max(0.0, min(1.0, 1.0 - radial_error - abs(len(simplified) - sides) * 0.04))
    return {
        'shape': names[sides], 'sides': sides,
        'confidence': round(confidence, 3),
        'points': _regular(cx, cy, target_radius, sides, start),
    }
