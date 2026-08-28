"""
hole_shapes.py
--------------
Generatory konturu otworu jako lista punktow (u, v) [mm] w LOKALNYM ukladzie
otworu:
    u -> wzdluz osi rury
    v -> obwodowo (dlugosc luku), zanim zostanie zamieniona na kat

Uzywane w trybie "manualnym" (otwory podane parametrycznie, bez modelu 3D)
oraz w testach. Kontur jest zawsze ZAMKNIETY (pierwszy punkt == ostatni)
i skierowany przeciwnie do wskazowek zegara (CCW) patrzac "z zewnatrz rury
do wewnatrz" -- to zalozenie wykorzystuje pozniej geometry_core przy
liczeniu przesuniecia (offsetu) o promien freza.

Krzywe (okrag, zaokraglenia naroznikow) sa dyskretyzowane adaptacyjnie:
liczba segmentow dobierana tak, by blad cieciwy < tolerance_mm.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Point2 = Tuple[float, float]


def _arc_segments(radius_mm: float, angle_span_rad: float, tolerance_mm: float) -> int:
    """Liczba segmentow potrzebna, by blad cieciwy luku byl < tolerance_mm."""
    radius_mm = max(radius_mm, 1e-6)
    tolerance_mm = max(tolerance_mm, 1e-4)
    # blad cieciwy e = r*(1-cos(dtheta/2))  =>  dtheta = 2*acos(1 - e/r)
    ratio = max(0.0, 1.0 - tolerance_mm / radius_mm)
    ratio = min(1.0, ratio)
    max_dtheta = 2.0 * math.acos(ratio) if ratio < 1.0 else math.pi / 4
    max_dtheta = max(max_dtheta, math.radians(1.0))
    n = max(4, math.ceil(abs(angle_span_rad) / max_dtheta))
    return n


def circle(diameter_mm: float, tolerance_mm: float = 0.03) -> List[Point2]:
    r = diameter_mm / 2.0
    n = _arc_segments(r, 2 * math.pi, tolerance_mm)
    pts = []
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def rectangle(width_mm: float, height_mm: float) -> List[Point2]:
    """width -> wzdluz u (os X rury), height -> wzdluz v (obwod)."""
    hw, hh = width_mm / 2.0, height_mm / 2.0
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh), (-hw, -hh)]
    return pts


def rounded_rectangle(width_mm: float, height_mm: float, corner_radius_mm: float,
                       tolerance_mm: float = 0.03) -> List[Point2]:
    hw, hh = width_mm / 2.0, height_mm / 2.0
    r = max(0.0, min(corner_radius_mm, min(hw, hh)))
    if r < 1e-6:
        return rectangle(width_mm, height_mm)

    n_arc = _arc_segments(r, math.pi / 2, tolerance_mm)
    pts: List[Point2] = []

    def arc(cx, cy, a_start, a_end):
        for i in range(n_arc + 1):
            a = a_start + (a_end - a_start) * i / n_arc
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    # start: srodek prawej krawedzi dolnej, idziemy CCW
    pts.append((hw, -hh + r))
    arc(hw - r, -hh + r, 0.0, -math.pi / 2)          # prawy-dolny naroznik
    pts.append((-hw + r, -hh))
    arc(-hw + r, -hh + r, -math.pi / 2, -math.pi)     # lewy-dolny
    pts.append((-hw, hh - r))
    arc(-hw + r, hh - r, math.pi, math.pi / 2)        # lewy-gorny
    pts.append((hw - r, hh))
    arc(hw - r, hh - r, math.pi / 2, 0.0)             # prawy-gorny
    pts.append((hw, -hh + r))
    return _dedupe_consecutive(pts)


def polygon(points_mm: List[List[float]]) -> List[Point2]:
    pts = [(float(p[0]), float(p[1])) for p in points_mm]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _dedupe_consecutive(pts: List[Point2], eps: float = 1e-9) -> List[Point2]:
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    return out


def signed_area(pts: List[Point2]) -> float:
    """Dodatnie pole -> CCW. Uzywane do wymuszenia spojnej orientacji konturow."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def ensure_ccw(pts: List[Point2]) -> List[Point2]:
    return pts if signed_area(pts) > 0 else list(reversed(pts))
