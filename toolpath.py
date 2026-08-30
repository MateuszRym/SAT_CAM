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

WIELE PRZEJSC DO PELNEJ GLEBOKOSCI:
  Zaden pojedynczy przejazd (CONTOUR) ani pojedynczy peck (DRILL) nie
  zaglebia sie od razu na cala grubosc sciany rury -- `_pass_depths()`
  dzieli calkowita glebokosc (`JobConfig.total_cut_depth_mm`, czyli grubosc
  sciany + naddatek na przebicie) na tyle rownych krokow, zeby zaden
  pojedynczy dosuw nie przekroczyl `JobConfig.pass_depth_mm` (CONTOUR) /
  `JobConfig.drill_peck_mm` (DRILL). Kazdy krok to osobne, pelne przejscie
  ruchu (caly obrys konturu, albo jeden peck) na coraz wiekszej glebokosci
  -- to jest wlasnie "kilka powtorzen ruchu do pelnej glebokosci". Liczbe
  przejsc mozna podejrzec w UI PRZED wygenerowaniem sciezki przez
  `JobConfig.estimated_pass_count()` / `estimated_peck_count()`, a po
  wygenerowaniu operacji - w `HoleOperation.pass_count`.

LĄCZNIKI / MOSTKI (tabs) w duzych otworach:
  Wylacznie w trybie CONTOUR. Gdy otwor jest duzy wzgledem narzedzia
  (patrz `JobConfig.tab_min_size_factor` / `HoleDef.tabs_override`),
  program NIE dojezdza w kilku rownomiernie rozlozonych "oknach" na
  obwodzie do pelnej glebokosci, tylko zatrzymuje sie na
  `JobConfig.tab_cap_depth_mm()` -- zostawiajac tam cienki, nieprzewiercony
  mostek materialu, ktory fizycznie podtrzymuje wyciety kawalek sciany az
  do recznego wylamania po zdjeciu obrobionej rury z maszyny. Geometria
  okien (ktore punkty sciezki wypadaja "w lączniku") liczona jest w
  geometry_core (cumulative_arc_length / tab_windows / point_in_tab_mask)
  na JUZ zoffsetowanym konturze (czyli na rzeczywistej sciezce narzedzia).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import geometry_core as gc
from config import TubeConfig, ToolConfig, JobConfig
from model_io import RawHole, TubeAxis, canonicalize


# --------------------------------------------------------------------------- #
#  Struktury wynikowe
# --------------------------------------------------------------------------- #

@dataclass
class ContourPass:
    z_mm: float             # NOMINALNA docelowa glebokosc promieniowa tego przejscia
                             # (<=0, 0=powierzchnia) - gdy path_z_mm is None, CALY obrys
                             # jedzie na tej stalej glebokosci; z_mm sluzy tez do raportow/
                             # podgladu w UI nawet gdy path_z_mm jest ustawione.
    path_x_mm: np.ndarray    # (N,) pozycje X maszyny
    path_a_deg: np.ndarray   # (N,) pozycje A maszyny [stopnie, CIAGLE - bez zawijania]
    path_z_mm: Optional[np.ndarray] = None  # (N,) glebokosc PER PUNKT - ustawiane tylko
                             # gdy ten przejazd wchodzi w okna lacznikow (tabs): wiekszosc
                             # punktow = z_mm, punkty w oknie lacznika = plytsza wartosc
                             # (JobConfig.tab_cap_depth_mm) tak, by zostawic tam mostek
                             # materialu. None -> caly obrys na stalym z_mm (typowy przypadek).


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
    pass_count: int = 0              # liczba przejsc/peckow do pelnej glebokosci (dla UI -
                                      # niezalezna od liczby petli/wysp na ktore rozpadl sie
                                      # kontur po offsecie; patrz JobConfig.estimated_pass_count)
    has_tabs: bool = False           # czy w tej operacji dodano lączniki (tabs)
    tab_count: int = 0               # faktyczna liczba lacznikow na petle (0 gdy has_tabs=False)


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
#  Lączniki / mostki (tabs) - decyzja "czy dodac" dla danego otworu
# --------------------------------------------------------------------------- #

def _hole_triggers_tabs(dense_xy: np.ndarray, tabs_override: Optional[bool],
                         tool: ToolConfig, job: JobConfig) -> bool:
    """
    Decyduje, czy DANY otwor powinien dostac lączniki:
      1. jesli otwor ma jawny `tabs_override` (z HoleDef, przekazany przez
         RawHole) - wygrywa on ZAWSZE, niezaleznie od globalnego przelacznika
         czy rozmiaru (UI: checkbox "wymus/wylacz lączniki dla tego otworu"),
      2. w przeciwnym razie: `JobConfig.use_tabs` I dluzszy z bokow bbox
         konturu (PRZED offsetem o promien narzedzia, czyli rzeczywisty
         rozmiar otworu) >= `JobConfig.tab_threshold_mm(tool.diameter_mm)`.
    `dense_xy` to kontur w plaszczyznie rozwinietej (x, s) - dokladnie to,
    co build_operations juz i tak liczy przed wywolaniem offsetu.
    """
    if tabs_override is not None:
        return bool(tabs_override)
    if not job.use_tabs:
        return False
    w_mm, h_mm = gc.bbox_extent(dense_xy)
    return max(w_mm, h_mm) >= job.tab_threshold_mm(tool.diameter_mm)


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
                              a_zero_offset_deg: float,
                              tabs_enabled: bool = False) -> Optional[HoleOperation]:
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

    total_depth = job.total_cut_depth_mm(tube)
    depths = _pass_depths(total_depth, job.pass_depth_mm)
    op.pass_count = len(depths)

    # ostrzezenie o naroznikach ktorych narzedzie nie wyrobi w pelni
    # (find_tight_corners oczekuje ZAMKNIETEJ petli, tj. xy[0]==xy[-1] -
    # to jest dokladnie to, co zwraca offset_polygon_inward/resample_polyline,
    # NIE trzeba i NIE WOLNO tu przycinac [:-1], bo funkcja sama liczy
    # n=len(xy)-1 - podwojne przyciecie psuje indeksowanie modulo)
    for loop in loops:
        op.tight_corners.extend(gc.find_tight_corners(loop, tool.radius_mm))

    # -------------------------------------------------------------------- #
    #  Lączniki / mostki (tabs): licz OKNA RAZ na kazda petle (geometria nie
    #  zmienia sie miedzy przejsciami na roznych glebokosciach), a nastepnie
    #  przy kazdym przejsciu, ktore siegnie glebiej niz `tab_cap_depth_mm`,
    #  przytnij glebokosc W OKNACH do tej wartosci (poza oknami - bez zmian).
    #  Wczesniejsze, plytsze przejscia i tak nie docieraja do tab_cap_depth_mm,
    #  wiec zostaja bez modyfikacji (path_z_mm=None => szybsza sciezka/gcode).
    # -------------------------------------------------------------------- #
    tabs_enabled = tabs_enabled and job.tab_count > 0 and job.tab_width_mm > 0 \
        and job.tab_remaining_thickness_mm > 0
    tab_cap_depth_mm = None
    loop_tab_masks: List[Optional[np.ndarray]] = [None] * len(loops)
    if tabs_enabled:
        tab_cap_depth_mm = job.tab_cap_depth_mm(tube)
        for li, loop in enumerate(loops):
            cum_s = gc.cumulative_arc_length(loop)
            perimeter = float(cum_s[-1])
            windows = gc.tab_windows(perimeter, job.tab_count, job.tab_width_mm)
            loop_tab_masks[li] = gc.point_in_tab_mask(cum_s, perimeter, windows)
            tab_arc_fraction = (job.tab_count * job.tab_width_mm) / max(perimeter, 1e-6)
            if tab_arc_fraction > 0.4:
                op.warnings.append(
                    f"Lączniki zajmuja ~{100*tab_arc_fraction:.0f}% obwodu otworu "
                    f"(tab_count={job.tab_count} x tab_width_mm={job.tab_width_mm:.1f}mm) "
                    f"- rozwaz mniej/wezsze lączniki, bo znaczna czesc konturu nie zostanie "
                    f"przewiercona."
                )
        op.has_tabs = True
        op.tab_count = job.tab_count
        op.warnings.append(
            f"Dodano {job.tab_count} lącznik(ow) (szer. {job.tab_width_mm:.1f}mm, "
            f"pozostawiony material {job.tab_remaining_thickness_mm:.2f}mm) - otwor "
            f"jest wiekszy niz {job.tab_min_size_factor:.1f}x fi narzedzia (albo wymuszono "
            f"lączniki dla tego otworu recznie)."
        )

    for z in depths:
        clip_for_tabs = tabs_enabled and z > tab_cap_depth_mm
        for li, loop in enumerate(loops):
            x_mm, theta_rad = gc.roll_back(loop, ref_radius_mm)
            x_machine = canonical_x_to_machine(x_mm, tube, x_extent)
            a_deg = gc.theta_to_deg_continuous(theta_rad, sign=a_sign,
                                                zero_offset_deg=a_zero_offset_deg)
            if clip_for_tabs:
                mask = loop_tab_masks[li]
                z_per_point = np.where(mask, -tab_cap_depth_mm, -z)
                op.contour_passes.append(ContourPass(z_mm=-z, path_x_mm=x_machine,
                                                       path_a_deg=a_deg, path_z_mm=z_per_point))
            else:
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

    total_depth = job.total_cut_depth_mm(tube)
    pecks = _pass_depths(total_depth, job.drill_peck_mm)
    op.pass_count = len(pecks)
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

        tabs_enabled = _hole_triggers_tabs(dense, rh.tabs_override, tool, job)

        op = _build_contour_operation(rh.name, dense, ref_radius, tube, tool, job,
                                       x_extent, a_sign, a_zero_offset_deg,
                                       tabs_enabled=tabs_enabled)

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
