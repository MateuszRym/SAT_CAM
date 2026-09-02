"""
toolpath.py
-----------
Zamienia surowe otwory (RawHole, punkty 3D z model_io) na konkretna
sciezke obrobki: liste operacji (jedna na otwor), kazda zlozona z
przejsc (pass) na kolejnych glebokosciach Z (promieniowo, przez
grubosc sciany rury), gotowa do wygenerowania G-code.

DWA TRYBY OBROBKI OTWORU:

  * CONTOUR (frezowanie konturowe) - domyslny, dziala dla dowolnego
    ksztaltu. Kontur otworu jest przesuwany DO WEWNATRZ o promien
    narzedzia (geometry_core.offset_polygon_inward), a nastepnie
    frez okraza ta (mniejsza) petle na kolejnych glebokosciach Z, az
    do pelnego przebicia sciany.

  * DRILL (wiercenie) - uzywany, gdy kontur nie mozna zoffsetowac
    (narzedzie nie miesci sie w otworze wzgledem jego promienia
    wewnatrz konturu), co w praktyce oznacza: otwor jest (w przyblizeniu)
    KOLOWY, a jego srednica jest bliska lub rowna srednicy narzedzia
    -- klasyczny przypadek "wywiercenia otworu 3mm frezem 3mm martwym
    srodkiem" (endmill uzyty jako wiertlo). W tym trybie NIE liczymy
    offsetu -- frez jedzie prosto w srodek otworu i zaglebia sie
    promieniowo (Z) w cyklu peckingowym (kolejne, coraz glebsze
    naklucia z pelnym odsunieciem miedzy nimi na oczyszczenie wiora),
    z WLASNYM, WOLNIEJSZYM posuwem (ToolConfig.feed_drill_mm_min)
    niz normalny posuw zaglebienia konturu (feed_plunge_mm_min) --
    dokladnie tak, jak przy prawdziwym wierceniu.

    Automatyczne przelaczenie w DRILL nastepuje gdy:
      a) geometry_core.offset_polygon_inward() dla danego otworu zwraca
         pusta liste (narzedzie fizycznie nie miesci sie w konturze), ORAZ
      b) kontur po "rozwinieciu" jest w przyblizeniu kolem (dopasowanie
         okregu metoda najmniejszych kwadratow, niski blad resztkowy),
         ORAZ
      c) srednica narzedzia <= zmierzona srednica otworu + tolerancja
         (inaczej narzedzie fizycznie nie zmiesci sie NAWET jako wiertlo
         -- wtedy zglaszamy blad/ostrzezenie, nie zgadujemy).

    Dla otworow NIE-kolowych, gdzie offset takze sie nie miesci (za
    waskie gniazdo/szczelina wzgledem srednicy narzedzia), nie ma
    bezpiecznego automatycznego trybu -- otwor jest pomijany z jasnym
    ostrzezeniem (uzyj mniejszego narzedzia albo wieksz tolerancje
    projektowa otworu).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import geometry_core as gc
from .config import TubeConfig, ToolConfig, JobConfig
from .model_io import RawHole, TubeAxis, canonicalize


# --------------------------------------------------------------------------- #
#  Struktury wynikowe
# --------------------------------------------------------------------------- #

@dataclass
class ContourPass:
    z_mm: float             # docelowa glebokosc promieniowa tego przejscia (<=0, 0=powierzchnia)
    path_x_mm: np.ndarray    # (N,) pozycje X maszyny
    path_a_deg: np.ndarray   # (N,) pozycje A maszyny [stopnie, CIAGLE - bez zawijania]


@dataclass
class DrillCycle:
    x_mm: float
    a_deg: float
    peck_targets_mm: List[float]   # kolejne coraz glebsze cele Z (<=0), ostatni = pelna glebokosc
    full_retract: bool


@dataclass
class HoleOperation:
    name: str
    mode: str                       # "contour" | "drill" | "skipped"
    contour_passes: List[ContourPass] = field(default_factory=list)
    drill: Optional[DrillCycle] = None
    warnings: List[str] = field(default_factory=list)
    tight_corners: List[Tuple[int, float]] = field(default_factory=list)
    ref_radius_mm: float = 0.0      # promien "rozwiniecia" tego otworu - potrzebny
                                     # gcode_writer do przeliczenia F na deg/min
                                     # dla blokow czysto obrotowych (patrz gcode_writer)


# --------------------------------------------------------------------------- #
#  Dopasowanie okregu (do decyzji o trybie DRILL) - metoda Kasy (liniowa,
#  szybka, wystarczajaco dokladna do klasyfikacji ksztaltu)
# --------------------------------------------------------------------------- #

def _fit_circle_2d(xy: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Zwraca (srodek[2], promien, rms_residual_mm)."""
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx ** 2 + cy ** 2, 1e-12))
    residuals = np.hypot(x - cx, y - cy) - r
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return np.array([cx, cy]), float(r), rms


def _is_circular_enough(xy: np.ndarray, center: np.ndarray, r_fit: float, rms: float,
                         tolerance_mm: float) -> bool:
    """
    Dopasowanie okregu metoda Kasy potrafi dac MALY residuum wzgledem
    duzego promienia takze dla ksztaltow WYDLUZONYCH (np. waska, dluga
    szczelina) - samo RMS wzgledem promienia NIE WYSTARCZY. Najpierw
    twardy test proporcji bounding-boxa (kolo/kwadrat ~1:1), dopiero
    potem test residuum dopasowania.
    """
    bbox_w = float(np.ptp(xy[:, 0]))
    bbox_h = float(np.ptp(xy[:, 1]))
    if min(bbox_w, bbox_h) < 1e-9:
        return False
    aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
    if aspect > 1.25:
        return False
    return rms < max(0.05 * r_fit, tolerance_mm * 3)


def _pass_depths(total_depth_mm: float, pass_depth_mm: float) -> List[float]:
    """Lista celow Z (dodatnich = glebokosc od powierzchni), rowne kroki,
    ostatni krok = dokladnie total_depth_mm."""
    total_depth_mm = max(total_depth_mm, 1e-6)
    pass_depth_mm = max(pass_depth_mm, 1e-3)
    n = max(1, math.ceil(total_depth_mm / pass_depth_mm))
    return [total_depth_mm * (i + 1) / n for i in range(n)]


# --------------------------------------------------------------------------- #
#  Referencja X maszyny (ktory koniec modelu = X0) i kalibracja A
# --------------------------------------------------------------------------- #

def canonical_x_to_machine(x_canonical_mm: np.ndarray, tube: TubeConfig,
                            x_extent: Tuple[float, float]) -> np.ndarray:
    x_lo, x_hi = min(x_extent), max(x_extent)
    if tube.x_zero_at == "min":
        out = x_canonical_mm - x_lo
    elif tube.x_zero_at == "max":
        out = x_hi - x_canonical_mm  # X rosnie w przeciwna strone modelu
    else:
        raise ValueError(f"TubeConfig.x_zero_at musi byc 'min' lub 'max', jest: {tube.x_zero_at!r}")
    return out + tube.x_offset_mm


# --------------------------------------------------------------------------- #
#  Budowa operacji dla jednego otworu
# --------------------------------------------------------------------------- #

def _build_contour_operation(name: str, dense_xy: np.ndarray, ref_radius_mm: float,
                              tube: TubeConfig, tool: ToolConfig, job: JobConfig,
                              x_extent: Tuple[float, float], a_sign: int,
                              a_zero_offset_deg: float) -> Optional[HoleOperation]:
    op = HoleOperation(name=name, mode="contour", ref_radius_mm=ref_radius_mm)

    loops = gc.offset_polygon_inward(dense_xy, tool.radius_mm, arc_tolerance_mm=job.tolerance_mm)

    if not loops:
        return None  # sygnal dla wywolujacego: sprobuj DRILL albo pomin

    # pyclipper (JT_ROUND) zwraca punkty NIEROWNOMIERNIE rozlozone (gesto na
    # zaokraglonych naroznikach, rzadko na prostych) - to psuje 3-punktowe
    # oszacowanie lokalnej krzywizny (find_tight_corners) i dawaloby fałszywe
    # ostrzezenia takze na gladkim okregu. Dogeszczamy rownomiernie PRZED
    # dalsza analiza i PRZED zbudowaniem sciezki G-code.
    loops = [gc.resample_polyline(loop, max_seg_mm=max(tool.radius_mm / 4.0, 0.1))
             for loop in loops]

    total_depth = tube.wall_thickness_mm + job.breakthrough_margin_mm
    depths = _pass_depths(total_depth, job.pass_depth_mm)

    # ostrzezenie o naroznikach ktorych narzedzie nie wyrobi w pelni
    # (find_tight_corners oczekuje ZAMKNIETEJ petli, tj. xy[0]==xy[-1] -
    # to jest dokladnie to, co zwraca offset_polygon_inward/resample_polyline,
    # NIE trzeba i NIE WOLNO tu przycinac [:-1], bo funkcja sama liczy
    # n=len(xy)-1 - podwojne przyciecie psuje indeksowanie modulo)
    for loop in loops:
        op.tight_corners.extend(gc.find_tight_corners(loop, tool.radius_mm))

    for z in depths:
        for loop in loops:
            x_mm, theta_rad = gc.roll_back(loop, ref_radius_mm)
            x_machine = canonical_x_to_machine(x_mm, tube, x_extent)
            a_deg = gc.theta_to_deg_continuous(theta_rad, sign=a_sign,
                                                zero_offset_deg=a_zero_offset_deg)
            op.contour_passes.append(ContourPass(z_mm=-z, path_x_mm=x_machine, path_a_deg=a_deg))

    return op


def _build_drill_operation(name: str, center_xy_unrolled: np.ndarray, ref_radius_mm: float,
                            tube: TubeConfig, job: JobConfig,
                            x_extent: Tuple[float, float], a_sign: int,
                            a_zero_offset_deg: float) -> HoleOperation:
    op = HoleOperation(name=name, mode="drill", ref_radius_mm=ref_radius_mm)

    x_mm, theta_rad = gc.roll_back(center_xy_unrolled.reshape(1, 2), ref_radius_mm)
    x_machine = float(canonical_x_to_machine(x_mm, tube, x_extent)[0])
    a_deg = float(gc.theta_to_deg_continuous(theta_rad, sign=a_sign,
                                              zero_offset_deg=a_zero_offset_deg)[0])

    total_depth = tube.wall_thickness_mm + job.breakthrough_margin_mm
    pecks = _pass_depths(total_depth, job.drill_peck_mm)
    op.drill = DrillCycle(x_mm=x_machine, a_deg=a_deg,
                           peck_targets_mm=[-p for p in pecks],
                           full_retract=job.drill_full_retract)
    return op


def build_operations(raw_holes: List[RawHole], axis: TubeAxis,
                      tube: TubeConfig, tool: ToolConfig, job: JobConfig,
                      x_extent: Optional[Tuple[float, float]] = None,
                      a_sign: int = 1, a_zero_offset_deg: float = 0.0) -> List[HoleOperation]:
    """Glowna funkcja: RawHole (punkty 3D, uklad oryginalny modelu) -> HoleOperation.

    x_extent: rzeczywisty zasieg rury wzdluz osi w ukladzie KANONICZNYM
      (zwrocony przez model_io.load_step_holes/load_mesh_holes). Jesli
      None, uzywany jest TubeConfig.length_mm zaczynajac od X=0 (tryb manualny).
    a_sign / a_zero_offset_deg: kalibracja kierunku/zera osi A wzgledem
      matematycznego theta=atan2(z,y) (patrz geometry_core.theta_to_deg_continuous
      i README p. "Kalibracja osi A").
    """
    if x_extent is None:
        x_extent = (0.0, tube.length_mm)

    ops: List[HoleOperation] = []
    for rh in raw_holes:
        can = canonicalize(rh.points_xyz, axis)
        axis_pts = gc.axis_project(can)
        theta_unwrapped = gc.unwrap_theta(axis_pts)
        x_mm = np.array([p.x_mm for p in axis_pts])
        measured_radius = float(np.mean([p.radius_mm for p in axis_pts]))

        if job.offset_reference == "outer":
            ref_radius = tube.outer_radius_mm
            if abs(ref_radius - measured_radius) > 0.02 * max(ref_radius, 1e-6):
                pct = 100.0 * abs(ref_radius - measured_radius) / max(ref_radius, 1e-6)
                _warn_mismatch(rh.name, ref_radius, measured_radius, pct)
        else:
            ref_radius = measured_radius

        unrolled = gc.unroll(x_mm, theta_unwrapped, ref_radius)
        # zagęszczenie PRZED offsetem i PRZED testem okragosci - kontur o
        # zaledwie kilku punktach (np. goly prostokat: 4 rogi) fituje sie
        # w okrag z zerowym bledem (dowolne 4 punkty maja okrag opisany),
        # wiec test okragosci na surowych punktach jest zawodny.
        dense = gc.resample_polyline(unrolled, max_seg_mm=max(tool.radius_mm / 3.0, 0.15))

        op = _build_contour_operation(rh.name, dense, ref_radius, tube, tool, job,
                                       x_extent, a_sign, a_zero_offset_deg)

        if op is None:
            # kontur sie nie miesci -- sprawdz czy to (w przyblizeniu) kolo,
            # ktore mozna po prostu wywiercic tym narzedziem
            center, r_fit, rms = _fit_circle_2d(dense)
            is_round_enough = _is_circular_enough(dense, center, r_fit, rms, job.tolerance_mm)
            fits_as_drill = tool.diameter_mm <= 2 * r_fit + job.drill_diameter_tolerance_mm

            if is_round_enough and fits_as_drill:
                op = _build_drill_operation(rh.name, center, ref_radius, tube, job,
                                             x_extent, a_sign, a_zero_offset_deg)
                op.warnings.append(
                    f"Frez fi{tool.diameter_mm:.2f}mm nie miesci sie w konturze "
                    f"(otwor ~fi{2*r_fit:.2f}mm) - przelaczono automatycznie na "
                    f"WIERCENIE (peck, posuw={tool.feed_drill_mm_min:.0f} mm/min)."
                )
            else:
                op = HoleOperation(name=rh.name, mode="skipped", ref_radius_mm=ref_radius)
                if is_round_enough:
                    op.warnings.append(
                        f"POMINIETO: otwor kolowy ~fi{2*r_fit:.2f}mm jest MNIEJSZY "
                        f"niz narzedzie fi{tool.diameter_mm:.2f}mm - fizycznie sie nie zmiesci."
                    )
                else:
                    op.warnings.append(
                        f"POMINIETO: kontur otworu jest wezszy niz srednica narzedzia "
                        f"(fi{tool.diameter_mm:.2f}mm) i nie jest kolowy, wiec nie mozna "
                        f"automatycznie przelaczyc na wiercenie. Uzyj mniejszego narzedzia."
                    )
        ops.append(op)
    return ops


def _warn_mismatch(name: str, ref_radius: float, measured: float, pct: float):
    import warnings as _w
    _w.warn(
        f"[{name}] Promien zmierzony z modelu ({measured:.3f}mm) rozni sie o {pct:.1f}% "
        f"od TubeConfig.outer_radius_mm ({ref_radius:.3f}mm) - sprawdz srednice "
        f"zewnetrzna rury w konfiguracji (rozwiniecie liczone jest wzgledem wartosci "
        f"z konfiguracji, wiec bledna wartosc przesunie caly ksztalt otworu)."
    )
