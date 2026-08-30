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

6) Lączniki / mostki (tabs), patrz README p. "Łączniki w dużych otworach":
   przy duzych otworach wyciety fragment sciany rury moze odpasc do srodka
   rury zanim frez skonczy obrys - `cumulative_arc_length`, `tab_windows`
   i `point_in_tab_mask` licza, KTORE punkty rozwinietego konturu wypadaja
   w oknach lacznikow (rownomiernie rozlozonych po obwodzie otworu). Sama
   decyzja "czy dodac lączniki" i docelowa glebokosc w oknie lącznika
   (zeby zostawic kawalek nieprzewierconej sciany) siedzi w toolpath.py
   (tam, gdzie znane sa juz przejscia/glebokosci) - tutaj tylko geometria.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
                           arc_tolerance_mm: Optional[float] = None) -> List[np.ndarray]:
    """
    Przesuwa zamkniety kontur (N,2) o `offset_mm` DO WEWNATRZ (w strone
    usuwanego materialu = wnetrza otworu). Zwraca liste petli (moze byc >1,
    jesli offset "rozdzieli" ksztalt, lub 0 jesli offset > polowa najwezszego
    miejsca otworu -- wtedy narzedzie fizycznie sie nie miesci).

    Kontur WEJSCIOWY musi byc CCW (patrz hole_shapes.ensure_ccw) - przy
    CCW dodatni ClipperOffset z ujemnym argumentem daje offset do wnetrza.

    `arc_tolerance_mm`: tolerancja cieciwy dla tesselacji JT_ROUND naroznikow
    wypuklych po offsecie (pyclipper.PyclipperOffset.ArcTolerance, w mm --
    funkcja sama przelicza na skalowane jednostki Clippera). Domyslna wartosc
    biblioteki (0.25 w jednostkach WEWNETRZNYCH, czyli ~0.000025mm przy naszym
    CLIPPER_SCALE) jest absurdalnie drobna i generuje niepotrzebnie ogromna
    liczbe punktow na kazdym zaokraglonym naroznikuu -- przekazanie tu tej
    samej tolerancji cieciwy co reszta programu (typowo JobConfig.tolerance_mm)
    daje spojna gestosc punktow bez utraty dokladnosci potrzebnej do obrobki.
    Gdy None, uzywana jest wartosc domyslna pyclipper.
    """
    assert offset_mm >= 0
    path = (xy[:-1] if np.allclose(xy[0], xy[-1]) else xy)  # bez powielonego punktu koncowego
    scaled = pyclipper.scale_to_clipper(path.tolist(), CLIPPER_SCALE)

    pco = pyclipper.PyclipperOffset()
    if arc_tolerance_mm is not None and arc_tolerance_mm > 0:
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


def bbox_extent(xy: np.ndarray) -> Tuple[float, float]:
    """
    Zwraca (szerokosc, wysokosc) prostokata opisanego na konturze (w
    plaszczyznie rozwinietej x/s). Uzywane m.in. przez max_inscribed_gap
    oraz przez toolpath.py do decyzji o automatycznym dodaniu lacznikow
    (tabs) w duzych otworach -- patrz JobConfig.tab_min_size_factor.
    """
    return float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))


def max_inscribed_gap(xy: np.ndarray) -> float:
    """
    Przyblizona najmniejsza 'szerokosc' otworu -- offset wiekszy od polowy
    tej wartosci prawie na pewno usunie caly kontur (narzedzie za duze).
    Uzywane tylko do wczesnego ostrzezenia w UI, NIE jako twarda walidacja
    (dokladna odpowiedz daje offset_polygon_inward zwracajac pusta liste).
    """
    w, h = bbox_extent(xy)
    return float(min(w, h))


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


# --------------------------------------------------------------------------- #
#  Lączniki / mostki (tabs) -- geometria rozmieszczenia okien na konturze.
#  Decyzje "czy dodac" i "jak glebokie okno" naleza do toolpath.py; tutaj
#  tylko czysta geometria dlugosci luku, dzialajaca na PLASKIM (juz
#  rozwinietym x/s) konturze -- dlatego zwykla euklidesowa dlugosc odcinka
#  jest tu poprawnym przyblizeniem dlugosci luku (patrz zalozenia w naglowku
#  pliku: grubosc sciany << promien rury).
# --------------------------------------------------------------------------- #

def cumulative_arc_length(xy: np.ndarray) -> np.ndarray:
    """
    Skumulowana dlugosc luku wzdluz polilinii, zaczynajac od 0 w pierwszym
    punkcie. Zwraca tablice (N,) tej samej dlugosci co `xy`. Dla petli
    zamknietej (xy[0] == xy[-1]) ostatni element to pelny obwod konturu.
    """
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def tab_windows(perimeter_mm: float, tab_count: int, tab_width_mm: float,
                 phase_offset_mm: float = 0.0) -> List[Tuple[float, float]]:
    """
    Zwraca liste (start, koniec) okien lacznikow rownomiernie rozlozonych
    na obwodzie `perimeter_mm` (te same jednostki dlugosci luku 's' co
    `cumulative_arc_length`). Okna moga wykraczac poza [0, perimeter_mm]
    (np. ujemny start) -- `point_in_tab_mask` obsluguje zawijanie modulo
    obwod, wiec nie trzeba tu nic przycinac.
    """
    if tab_count <= 0 or tab_width_mm <= 0 or perimeter_mm <= 0:
        return []
    half = tab_width_mm / 2.0
    windows = []
    for i in range(tab_count):
        center = phase_offset_mm + perimeter_mm * i / tab_count
        windows.append((center - half, center + half))
    return windows


def point_in_tab_mask(cum_s: np.ndarray, perimeter_mm: float,
                       windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    Maska bool (N,): czy dany punkt (wg dlugosci luku `cum_s`) wypada w
    ktoryms z okien lacznikow, z poprawnym zawijaniem przez zszycie 0/obwod
    (istotne, bo pierwszy punkt konturu prawie nigdy nie wypada dokladnie
    na srodku miedzy dwoma lacznikami).
    """
    n = len(cum_s)
    if not windows or perimeter_mm <= 0:
        return np.zeros(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for start, end in windows:
        s0 = start % perimeter_mm
        s1 = end % perimeter_mm
        if s0 <= s1:
            mask |= (cum_s >= s0) & (cum_s <= s1)
        else:
            # okno przechodzi przez szew 0/perimeter (np. lacznik "na starcie" konturu)
            mask |= (cum_s >= s0) | (cum_s <= s1)
    return mask
