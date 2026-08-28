"""
geometry_core.py
-----------------
Serce calego programu. Implementuje dokladnie metode zaproponowana w briefie:

    "lokalizowac punkty na krawedziach otworow, laczyc je z osia obrotu
    i tak wyliczac katy obrotu, uwzgledniajac grubosc frezu/wiertla"

Krok po kroku:

1) `axis_project(points_xyz)`:
   Dla kazdego punktu P=(x,y,z) w KANONICZNYM ukladzie modelu (os rury = os X)
   liczymy rzut na os obrotu: najblizszy punkt na osi to (x,0,0). Wektor
   od osi do P to (0,y,z) - to jest doslownie "polaczenie punktu z osia".
   Jego dlugosc to promien r(P) = sqrt(y^2+z^2), a kat to theta = atan2(z,y).
   x -> bezposrednio pozycja na osi maszyny X.
   theta -> po odwinieciu (unwrap) i konwersji na stopnie -> os maszyny A.

   Dlaczego to jest fizycznie poprawne dla maszyny XZA (bez Y):
   Frez moze podchodzic WYLACZNIE promieniowo (Z), a to, KTORY punkt
   powierzchni rury znajduje sie aktualnie "pod" wrzecionem, zalezy
   WYLACZNIE od kata obrotu A. Zatem obrobka kazdego punktu konturu siega
   promieniowo dokladnie do osi obrotu z jego wlasnego kata -- innej drogi
   fizycznie nie ma przy trzech osiach X/Z/A. Patrz README p.2.

2) `unroll(points, ref_radius)`:
   Zamiana (x, theta) -> (x, s) gdzie s = ref_radius * theta_unwrapped
   (dlugosc luku na promieniu odniesienia, domyslnie promien zewnetrzny
   rury). To pozwala policzyc offset narzedzia w zwyklej plaskiej
   geometrii euklidesowej (dobre przyblizenie o ile grubosc sciany
   << promien rury -- typowe dla tulei rakietowych; patrz README,
   sekcja "Zalozenia i ograniczenia").

3) `offset_polygon_inward(points_xy, offset_mm)`:
   Przesuniecie konturu do wnetrza otworu o promien narzedzia, realizowane
   przez pyclipper (Clipper2-Offset) - biblioteka odporna na przypadki
   brzegowe (ostre/wklesle naroza, samo-przeciecia przy zbyt duzym offsecie
   wzgledem malych detali konturu).

4) `roll_back(points_xs, ref_radius)`:
   Powrot z (x, s) do (x, theta[deg]) juz PO offsetowaniu -- to jest
   scizka faktycznie wysylana do G-code.

5) Funkcje pomocnicze: unwrap kata (ciaglosc bez skokow 0/360),
   walidacja (czy narzedzie miesci sie w otworze / promieniu naroznika).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    import pyclipper
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Brakuje pakietu 'pyclipper' (pip install pyclipter). Jest on "
        "wymagany do poprawnego (odpornego na przypadki brzegowe) "
        "przesuniecia konturu o promien narzedzia."
    ) from e


CLIPPER_SCALE = 10_000.0  # pyclipper dziala na liczbach calkowitych - skalujemy mm


# --------------------------------------------------------------------------- #
#  Krok 1: rzut punktu na os obrotu
# --------------------------------------------------------------------------- #

@dataclass
class AxisPoint:
    x_mm: float          # pozycja wzdluz osi rury (kanoniczna, PRZED zmiana referencji)
    radius_mm: float      # odleglosc punktu od osi (promien w tym miejscu)
    theta_rad: float      # kat wokol osi, NIEODWINIETY (atan2 w zakresie -pi..pi)


def axis_project(points_xyz: np.ndarray) -> List[AxisPoint]:
    """points_xyz: (N,3) w ukladzie kanonicznym (os rury = os X globalna)."""
    out = []
    for x, y, z in points_xyz:
        r = math.hypot(y, z)
        theta = math.atan2(z, y)
        out.append(AxisPoint(x_mm=float(x), radius_mm=float(r), theta_rad=float(theta)))
    return out


def unwrap_theta(axis_points: Sequence[AxisPoint]) -> np.ndarray:
    """Zwraca ciagly (bez skokow +/-2pi) przebieg katow w radianach."""
    thetas = np.array([p.theta_rad for p in axis_points], dtype=float)
    return np.unwrap(thetas)


# --------------------------------------------------------------------------- #
#  Krok 2 / 4: rozwiniecie walca na plaszczyzne i z powrotem
# --------------------------------------------------------------------------- #

def unroll(x_mm: np.ndarray, theta_unwrapped_rad: np.ndarray,
           ref_radius_mm: float) -> np.ndarray:
    """(x, theta) -> (x, s) gdzie s = r_ref * theta.  Zwraca (N,2)."""
    s = ref_radius_mm * theta_unwrapped_rad
    return np.column_stack([x_mm, s])


def roll_back(xy_unrolled: np.ndarray, ref_radius_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """(x, s) -> (x, theta_unwrapped_rad)."""
    x = xy_unrolled[:, 0]
    theta = xy_unrolled[:, 1] / ref_radius_mm
    return x, theta


def theta_to_deg_continuous(theta_unwrapped_rad: np.ndarray, sign: int = 1,
                             zero_offset_deg: float = 0.0) -> np.ndarray:
    """
    Zamiana na stopnie dla osi A. `sign` pozwala odwrocic kierunek obrotu,
    jesli dodatni kat matematyczny (CCW patrzac od strony +X) odpowiada
    ujemnemu kierunkowi obrotu fizycznej osi A danej maszyny (do
    skalibrowania jednorazowo na maszynie -- patrz README p. "Kalibracja").
    """
    return sign * np.degrees(theta_unwrapped_rad) + zero_offset_deg


# --------------------------------------------------------------------------- #
#  Krok 3: offset konturu (promien narzedzia) w plaszczyznie rozwinietej
# --------------------------------------------------------------------------- #

def offset_polygon_inward(xy: np.ndarray, offset_mm: float,
                           arc_tolerance_mm: float = 0.02) -> List[np.ndarray]:
    """
    Przesuwa zamkniety kontur (N,2) o `offset_mm` DO WEWNATRZ (w strone
    usuwanego materialu = wnetrza otworu). Zwraca liste petli (moze byc >1,
    jesli offset "rozdzieli" ksztalt, lub 0 jesli offset > polowa najwezszego
    miejsca otworu -- wtedy narzedzie fizycznie sie nie miesci).

    Kontur WEJSCIOWY musi byc CCW (patrz hole_shapes.ensure_ccw) - przy
    CCW dodatni ClipperOffset z ujemnym argumentem daje offset do wnetrza.

    `arc_tolerance_mm` ogranicza blad cieciwy, z jakim pyclipper aproksymuje
    zaokraglone (JT_ROUND) naroza wielokatem -- BEZ jawnego ustawienia
    pyclipper uzywa dosc zgrubnej wartosci domyslnej, co przy malych
    promieniach (rzedu promienia narzedzia) daje zauwazalnie "kanciasty"
    luk i psuje pozniejsze oszacowanie lokalnej krzywizny konturu
    (geometry_core.find_tight_corners) fałszywymi ostrzezeniami nawet na
    gladkim okragu.
    """
    assert offset_mm >= 0
    path = (xy[:-1] if np.allclose(xy[0], xy[-1]) else xy)  # bez powielonego punktu koncowego
    scaled = pyclipper.scale_to_clipper(path.tolist(), CLIPPER_SCALE)

    pco = pyclipper.PyclipperOffset()
    pco.ArcTolerance = arc_tolerance_mm * CLIPPER_SCALE
    # JT_ROUND: frez jest okragly, wiec zaokraglone naroza to fizycznie
    # poprawny ksztalt sladu narzedzia w naroznikach wypuklych "od zewnatrz"
    pco.AddPath(scaled, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(-offset_mm * CLIPPER_SCALE)  # ujemny = do wewnatrz

    loops = []
    for loop in solution:
        pts = np.array(pyclipper.scale_from_clipper(loop, CLIPPER_SCALE), dtype=float)
        if len(pts) >= 3:
            pts = np.vstack([pts, pts[0]])  # zamknij petle
            loops.append(pts)
    return loops


def max_inscribed_gap(xy: np.ndarray) -> float:
    """
    Przyblizona najmniejsza 'szerokosc' otworu -- offset wiekszy od polowy
    tej wartosci prawie na pewno usunie caly kontur (narzedzie za duze).
    Uzywane tylko do wczesnego ostrzezenia w UI, NIE jako twarda walidacja
    (dokladna odpowiedz daje offset_polygon_inward zwracajac pusta liste).
    """
    xs = xy[:, 0]
    ys = xy[:, 1]
    return float(min(xs.max() - xs.min(), ys.max() - ys.min()))


# --------------------------------------------------------------------------- #
#  Adaptacyjna resampling (dogeszczanie) konturu przed rzutowaniem 3D->2D,
#  tak by po stronie "walcowej" (gdzie krzywizna wplywa na dlugosc luku)
#  nie gubic detalu miedzy zbyt rzadkimi punktami wejsciowymi z BRep/mesh.
# --------------------------------------------------------------------------- #

def resample_polyline(xy: np.ndarray, max_seg_mm: float) -> np.ndarray:
    """Dodaje punkty posrednie tak, by zaden segment nie byl dluzszy niz max_seg_mm."""
    out = [xy[0]]
    for p0, p1 in zip(xy[:-1], xy[1:]):
        seg_len = float(np.linalg.norm(p1 - p0))
        n = max(1, math.ceil(seg_len / max_seg_mm))
        for i in range(1, n + 1):
            out.append(p0 + (p1 - p0) * (i / n))
    return np.array(out)


# --------------------------------------------------------------------------- #
#  Walidacja naroznikow (promien zaokraglenia vs promien narzedzia)
# --------------------------------------------------------------------------- #

def find_tight_corners(xy: np.ndarray, tool_radius_mm: float,
                        angle_thresh_deg: float = 5.0) -> List[Tuple[int, float]]:
    """
    Bardzo prosty detektor: szuka lokalnych wierzcholkow o promieniu krzywizny
    mniejszym niz promien narzedzia (przyblizenie przez 3 kolejne punkty).
    Zwraca liste (indeks, promien_lokalny) -- do wypisania ostrzezenia w UI:
    "w tym miejscu narzedzie nie wejdzie w pelni w naroznik".
    """
    warnings = []
    n = len(xy) - 1  # kontur zamkniety
    for i in range(n):
        p_prev = xy[i - 1]
        p_cur = xy[i]
        p_next = xy[(i + 1) % n]
        r = _circumradius(p_prev, p_cur, p_next)
        if r is not None and r < tool_radius_mm:
            warnings.append((i, r))
    return warnings


def _circumradius(a, b, c):
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    s = (ab + bc + ca) / 2.0
    area = max(s * (s - ab) * (s - bc) * (s - ca), 0.0) ** 0.5
    if area < 1e-9:
        return None
    return (ab * bc * ca) / (4.0 * area)
