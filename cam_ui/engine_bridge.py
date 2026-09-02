"""
engine_bridge.py
-----------------
Warstwa posrednia miedzy UI (Tkinter) a silnikiem CAM w cam_core/. Nie ma
tu ZADNEJ logiki geometrycznej -- to woła wylacznie gotowe funkcje z
cam_core, ale:

  * lapie brakujace opcjonalne zaleznosci (cadquery/OCP dla STEP, trimesh
    dla mesh) i zamienia je na czytelny, PO POLSKU komunikat z podpowiedzia
    `pip install ...` zamiast surowego ImportError,
  * dolicza korekte kata dla trybu manualnego (patrz MANUAL_MODE_AXIS_
    CORRECTION_DEG w app_state.py i uzasadnienie ponizej),
  * przygotowuje dane pod wizualizacje (kontur PRZED offsetem + operacje
    PO offsecie, w ukladzie "rozwinietym" x/s ORAZ jako punkty 3D na
    walcu do podgladu),
  * zbiera ostrzezenia ze wszystkich etapow w jedna liste do wyswietlenia
    w konsoli UI.

WYJASNIENIE KOREKTY KATA W TRYBIE MANUALNYM:
cam_core.model_io.holes_from_manual_defs() generuje punkty juz w ukladzie
KANONICZNYM (dokladnie takim, jakiego oczekuje geometry_core.axis_project:
os rury = globalny X, kat = atan2(z,y)). Natomiast cam_core.toolpath.
build_operations() BEZWARUNKOWO woła model_io.canonicalize(punkty, axis)
na wejsciu -- a ta funkcja, nawet dla "tozsamosciowej" osi (kierunek=+X,
punkt=0,0,0), NIE jest identycznoscia: wewnetrzny wybor wektora pomocniczego
(tmp) w canonicalize() wprowadza STALY obrot ukladu o -90 stopni (zweryfi-
kowane numerycznie w testach). Dla otworow pochodzacych z prawdziwego
pliku STEP nie ma to znaczenia (i tak kalibrujemy a_zero_offset_deg na
maszynie realnym "touch-off"), ale w trybie manualnym uzytkownik wpisuje
`center_angle_deg` oczekujac WPROST tej wartosci na osi A (przed jego
wlasna kalibracja maszyny) -- dlatego doliczamy tu kompensacje.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from cam_core import geometry_core as gc
from cam_core import hole_shapes as hs
from cam_core import model_io
from cam_core import toolpath
from cam_core import gcode_writer
from cam_core.config import TubeConfig, ToolConfig, JobConfig, HoleDef
from cam_core.model_io import RawHole, TubeAxis, TubeGeometryInfo
from cam_core.toolpath import HoleOperation

from .app_state import AppProject, MANUAL_MODE_AXIS_CORRECTION_DEG


class EngineError(RuntimeError):
    """Czytelny blad domenowy do pokazania w UI (zamiast surowego tracebacku)."""
    pass


# --------------------------------------------------------------------------- #
#  Wczytywanie otworow (3 sciezki: manual / step / mesh)
# --------------------------------------------------------------------------- #

@dataclass
class LoadedHoles:
    raw_holes: List[RawHole]
    axis: TubeAxis
    x_extent: Tuple[float, float]
    source: str                      # "manual" | "step" | "mesh"
    info_lines: List[str] = field(default_factory=list)
    tube_geometry: Optional[TubeGeometryInfo] = None  # None dla trybu manualnego -
                                                        # nie ma z czego "odczytac" geometrii


def load_manual(project: AppProject) -> LoadedHoles:
    enabled = [h for h in project.holes if h.enabled]
    if not enabled:
        raise EngineError(
            "Brak wlaczonych otworow w trybie recznym. Dodaj otwor w zakladce "
            "'Otwory' lub zaznacz co najmniej jeden jako aktywny."
        )
    raw = model_io.holes_from_manual_defs(
        enabled, project.tube.outer_radius_mm, project.job.tolerance_mm
    )
    axis = TubeAxis(
        point_on_axis=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        radius_mm=project.tube.outer_radius_mm,
    )
    x_extent = (0.0, project.tube.length_mm)
    return LoadedHoles(
        raw_holes=raw, axis=axis, x_extent=x_extent, source="manual",
        info_lines=[f"Tryb reczny: {len(raw)} otwor(ow) z listy parametrycznej."],
    )


def _resolve_x_extent(project: AppProject, geom: TubeGeometryInfo, raw: List[RawHole],
                       axis: TubeAxis, source_label: str) -> Tuple[Tuple[float, float], List[str]]:
    """
    Ustala ostateczny zasieg X (x_extent) uzywany jako referencja X=0 maszyny
    (patrz cam_core.toolpath.canonical_x_to_machine). Zwraca (x_extent, dodatkowe_info_lines).

    [PATCH/FEATURE] Rozwiazuje zgloszony blad "X=0 wychodzi na otworze, a nie
    na poczatku rury" (patrz README.md, "Historia zmian"). Trzy warstwy, od
    najbardziej do najmniej precyzyjnej:

      1. project.machine.x_extent_source == "manual_length": uzytkownik
         JAWNIE wymusza (0, TubeConfig.length_mm) - pelna reczna kontrola,
         geometria z pliku jest w tym celu calkowicie ignorowana.
      2. geom.x_extent (parametr V sciany walcowej dla STEP / pelny zasieg
         siatki dla mesh) - najdokladniejsze zrodlo, gdy dostepne.
      3. Zasieg samych punktow otworow - ostatecznosc, gdy (2) niedostepne.
         UWAGA: to WLASNIE ta ostatecznosc byla wczesniej JEDYNA sciezka i
         powodowala zglaszany blad - otwory zwykle nie siegaja do samych
         koncow rury, wiec X=0 wypadalo na najblizszym otworze.

    Niezaleznie od wybranej warstwy: jesli finalny zasieg NIE obejmuje w
    pelni wszystkich wykrytych otworow (co sygnalizowaloby blad detekcji,
    np. sciana walcowa podzielona w BRep na kilka niezaleznych fragmentow
    o rozlacznych zakresach V), zasieg jest automatycznie POSZERZANY tak,
    by na pewno je obejmowal, a do UI trafia wyrazne ostrzezenie - nigdy
    nie przycinamy/przesuwamy polozenia otworow po cichu.
    """
    hole_x = model_io.canonicalize(np.vstack([h.points_xyz for h in raw]), axis)[:, 0]
    hole_lo, hole_hi = float(hole_x.min()), float(hole_x.max())
    lines: List[str] = []
    margin = 0.05

    if project.machine.x_extent_source == "manual_length":
        # Reczny, JAWNY wybor uzytkownika - stosujemy BEZWARUNKOWO (bez auto-
        # poszerzania ponizej), bo to dokladnie po to istnieje: przewidywalne,
        # w pelni kontrolowane X=0 niezaleznie od tego, co "myslalaby" detekcja.
        # Nadal OSTRZEGAMY (ale nie zmieniamy zakresu), gdyby otwory wypadaly
        # poza podanym zakresem - to zwykle znaczy: albo 'Dlugosc rury' w
        # zakladce 'Rura' jest nieprawidlowa, albo trzeba przelaczyc 'X=0 przy
        # koncu' (min/max) na drugi koniec - kierunek osi wykryty z pliku bywa
        # dowolny (OCC/mesh nie gwarantuje ktora strona to "+"), wiec to
        # normalne, ze czasem trzeba to raz skalibrowac recznie.
        x_extent = (0.0, project.tube.length_mm)
        lines.append(f"{source_label}: zasieg X wymuszony recznie na (0, "
                      f"{project.tube.length_mm:.2f})mm wg 'Dlugosc rury' z zakladki "
                      f"'Rura' (opcja 'Zakres X' w zakladce 'Kalibracja').")
        lo, hi = x_extent
        if hole_lo < lo - margin or hole_hi > hi + margin:
            lines.append(
                f"{source_label} UWAGA: przy tym recznym zasiegu niektore otwory "
                f"wypadaja POZA nim (wykryto otwory przy {hole_lo:.2f}..{hole_hi:.2f}mm "
                f"wzgledem naturalnego zera osi z modelu, poza (0..{hi:.2f}mm)). "
                f"Sprawdz 'Dlugosc rury' w zakladce 'Rura' oraz czy 'X=0 przy koncu' "
                f"(min/max) nie powinno wskazywac drugiego konca."
            )
        return x_extent, lines

    if geom.x_extent is not None:
        x_extent = geom.x_extent
    else:
        x_extent = (hole_lo, hole_hi)
        lines.append(f"{source_label}: nie udalo sie wyznaczyc pelnego zasiegu rury z "
                      f"geometrii - uzyto zasiegu samych otworow (mniej dokladne; X=0 "
                      f"moze NIE pokrywac sie z fizycznym koncem rury). Rozwaz recznie "
                      f"ustawiona opcje 'Zakres X: reczna dlugosc' w zakladce 'Kalibracja'.")

    lo, hi = min(x_extent), max(x_extent)
    if hole_lo < lo - margin or hole_hi > hi + margin:
        new_lo, new_hi = min(lo, hole_lo), max(hi, hole_hi)
        lines.append(
            f"{source_label} UWAGA: wykryty zasieg dlugosci rury "
            f"({lo:.2f}..{hi:.2f}mm) NIE obejmowal wszystkich otworow "
            f"({hole_lo:.2f}..{hole_hi:.2f}mm) - poszerzono automatycznie do "
            f"({new_lo:.2f}..{new_hi:.2f}mm), zeby X=0 na pewno nie wypadlo na "
            f"otworze. Sprawdz 'Dlugosc rury' w zakladce 'Rura' i w razie potrzeby "
            f"popraw recznie (lub przelacz na reczny 'Zakres X' w 'Kalibracji')."
        )
        x_extent = (new_lo, new_hi)

    return x_extent, lines


def load_step(path: str, project: AppProject) -> LoadedHoles:
    try:
        import cadquery  # noqa: F401
    except ImportError as e:
        raise EngineError(
            "Brak pakietu 'cadquery' (i/lub 'OCP') potrzebnego do odczytu plikow "
            "STEP. Zainstaluj: pip install cadquery\n"
            f"(szczegoly: {e})"
        ) from e

    # [PATCH/FEATURE] Domyslnie (checkbox "auto" w zakladce Kalibracja zaznaczony,
    # czyli machine.outer_radius_hint_mm is None) NIE podajemy JUZ zadnego hinta -
    # pelna automatyczna detekcja bierze najwieksza powierzchnie walcowa w modelu.
    # Wczesniej w tym miejscu podstawialismy tu TubeConfig.outer_radius_mm jako
    # "domyslny hint", co w praktyce wymagalo od uzytkownika wpisania POPRAWNEJ
    # (w granicach radius_match_tol_mm) srednicy PRZED wczytaniem pliku - a jesli
    # sie nie zgadzala, load_step_holes() rzucal ValueError -> EngineError ->
    # blokujacy messagebox.showerror(), mimo ze plik byl calkowicie poprawny.
    # To DOKLADNIE odwracalo pozadany kierunek: srednica ma byc odczytana Z
    # modelu, nie zgadywana z gory pod rygorem bledu. Reczny hint (pole w
    # Kalibracji, gdy checkbox odznaczony) nadal dziala normalnie - przydaje
    # sie, gdy model ma inna, wieksza powierzchnie walcowa (np. mocowanie),
    # ktora bez ograniczenia zostalaby blednie wzieta za rure.
    hint = project.machine.outer_radius_hint_mm
    tol = project.machine.step_tolerance_override_mm or project.job.tolerance_mm

    try:
        raw, axis, geom = model_io.load_step_holes(
            path,
            tolerance_mm=tol,
            outer_radius_hint_mm=hint,
            radius_match_tol_mm=project.machine.radius_match_tol_mm,
        )
    except ValueError as e:
        raise EngineError(str(e)) from e
    except Exception as e:  # pragma: no cover - nieprzewidziane bledy OCC
        raise EngineError(f"Blad podczas czytania pliku STEP: {e}") from e

    x_extent, extent_notes = _resolve_x_extent(project, geom, raw, axis, "STEP")

    info = [f"STEP: wykryto {len(raw)} otwor(ow) na powierzchni walcowej "
            f"o promieniu {axis.radius_mm:.3f}mm."]
    if geom.length_mm is not None:
        info.append(f"STEP: wykryta dlugosc rury (z geometrii modelu): {geom.length_mm:.2f}mm.")
    if geom.inner_radius_mm is not None:
        info.append(
            f"STEP: wykryta wewnetrzna powierzchnia walcowa - promien "
            f"{geom.inner_radius_mm:.3f}mm (grubosc sciany: "
            f"{geom.outer_radius_mm - geom.inner_radius_mm:.3f}mm)."
        )
    else:
        info.append("STEP: nie znaleziono osobnej wewnetrznej powierzchni walcowej - "
                     "grubosc sciany pozostaje wartoscia reczna z zakladki 'Rura'.")
    info.extend(extent_notes)

    return LoadedHoles(
        raw_holes=raw, axis=axis, x_extent=x_extent, source="step",
        info_lines=info, tube_geometry=geom,
    )


def load_mesh(path: str, project: AppProject) -> LoadedHoles:
    try:
        import trimesh  # noqa: F401
    except ImportError as e:
        raise EngineError(
            "Brak pakietu 'trimesh' potrzebnego do odczytu plikow 3MF/STL/OBJ. "
            "Zainstaluj: pip install trimesh\n"
            f"(szczegoly: {e})"
        ) from e

    try:
        raw, axis, geom = model_io.load_mesh_holes(path)
    except ValueError as e:
        raise EngineError(str(e)) from e
    except Exception as e:  # pragma: no cover
        raise EngineError(f"Blad podczas czytania pliku mesh: {e}") from e

    x_extent, extent_notes = _resolve_x_extent(project, geom, raw, axis, "MESH")

    info = [f"MESH: wykryto {len(raw)} petli brzegowych (przyblizona metoda, "
            f"patrz README dot. siatek watertight)."]
    if geom.length_mm is not None:
        info.append(f"MESH: wykryta dlugosc rury (z zasiegu siatki): {geom.length_mm:.2f}mm.")
    info.append("MESH: grubosc sciany nie jest wykrywana automatycznie dla tego formatu - "
                 "pozostaje wartoscia reczna z zakladki 'Rura'.")
    info.extend(extent_notes)

    return LoadedHoles(
        raw_holes=raw, axis=axis, x_extent=x_extent, source="mesh",
        info_lines=info, tube_geometry=geom,
    )


def load_by_mode(project: AppProject, path: Optional[str] = None) -> LoadedHoles:
    if project.source_mode == "manual":
        return load_manual(project)
    elif project.source_mode == "step":
        if not path:
            raise EngineError("Wskaz plik .step / .stp do wczytania.")
        return load_step(path, project)
    elif project.source_mode == "mesh":
        if not path:
            raise EngineError("Wskaz plik .3mf / .stl / .obj do wczytania.")
        return load_mesh(path, project)
    raise EngineError(f"Nieznany tryb zrodla modelu: {project.source_mode!r}")


def apply_detected_tube_geometry(project: AppProject, geom: Optional[TubeGeometryInfo]) -> List[str]:
    """
    [FEATURE] Nadpisuje TubeConfig w `project` wartosciami odczytanymi z
    geometrii modelu (patrz TubeGeometryInfo) - tak, by w trybie STEP/mesh
    uzytkownik NIE musial znac/wpisywac srednicy, grubosci sciany ani
    dlugosci rury PRZED wczytaniem pliku.

    CZYSTA operacja na danych (NIE dotyka zadnego widgetu UI) - bezpieczna
    do wywolania z watku w tle. KRYTYCZNE MIEJSCE WYWOLANIA: musi nastapic
    PRZED eb.build_toolpath(), bo toolpath.build_operations() uzywa
    TubeConfig.outer_radius_mm jako promienia odniesienia przy "rozwijaniu"
    konturow (gdy JobConfig.offset_reference == "outer", czyli domyslnie) -
    zastosowanie wykrytej geometrii DOPIERO PO zbudowaniu sciezki policzyloby
    cala geometrie (offset, rozwiniecie) wzgledem starej, potencjalnie
    calkowicie blednej wartosci sprzed wczytania modelu.

    Grubosc sciany nadpisywana jest TYLKO gdy geom.inner_radius_mm nie jest
    None (model mial osobno zamodelowana powierzchnie wewnetrzna) - w
    przeciwnym razie zostaje dotychczasowa, reczna wartosc uzytkownika.

    Zwraca liste linii tekstowych do zalogowania w UI (samo logowanie -
    jak kazda operacja na widgetach Tk - musi nastapic w GLOWNYM watku,
    stad zwracamy tu tylko tekst zamiast wolac cokolwiek bezposrednio).
    """
    if geom is None:  # tryb reczny - nie ma z czego "odczytac" geometrii
        return []

    lines: List[str] = []
    tube = project.tube

    new_od = geom.outer_radius_mm * 2.0
    if abs(new_od - tube.outer_diameter_mm) > 0.01:
        lines.append(f"Srednica zewnetrzna zaktualizowana z modelu: "
                      f"{tube.outer_diameter_mm:.3f}mm -> {new_od:.3f}mm.")
    tube.outer_diameter_mm = new_od

    if geom.inner_radius_mm is not None:
        new_wall = geom.outer_radius_mm - geom.inner_radius_mm
        if abs(new_wall - tube.wall_thickness_mm) > 0.01:
            lines.append(f"Grubosc sciany zaktualizowana z modelu: "
                          f"{tube.wall_thickness_mm:.3f}mm -> {new_wall:.3f}mm.")
        tube.wall_thickness_mm = new_wall

    if geom.length_mm is not None:
        if abs(geom.length_mm - tube.length_mm) > 0.01:
            lines.append(f"Dlugosc rury zaktualizowana z modelu: "
                          f"{tube.length_mm:.3f}mm -> {geom.length_mm:.3f}mm.")
        tube.length_mm = geom.length_mm

    return lines


# --------------------------------------------------------------------------- #
#  Referencyjny kontur PRZED offsetem (do podgladu) -- powiela pierwsze kroki
#  toolpath.build_operations, tylko bez wywolywania offsetu narzedzia.
# --------------------------------------------------------------------------- #

@dataclass
class HolePreview:
    name: str
    ref_radius_mm: float
    raw_dense_xa: np.ndarray         # (N,2) x_mm, a_deg -- kontur PRZED offsetem, JUZ w
                                      # skalibrowanym ukladzie osi A (ten sam sign/zero_offset
                                      # co operation.contour_passes), wiec porownywalny 1:1
    operation: HoleOperation
    measured_radius_mm: float
    x_mm_range: Tuple[float, float]


def _reference_dense_contour(rh: RawHole, axis: TubeAxis, project: AppProject,
                              a_sign: int, a_zero_offset_deg: float
                              ) -> Tuple[np.ndarray, float, float]:
    """Kontur PRZED offsetem narzedzia, ale PO przeliczeniu na (x, A[deg]) -- dokladnie
    tymi samymi krokami co toolpath.py (unroll -> resample -> roll_back ->
    theta_to_deg_continuous), tylko bez wywolywania offset_polygon_inward. Dzieki
    identycznemu sign/zero_offset kontur ten pokrywa sie 1:1 z operation.contour_passes
    (ktore po offsecie sa PO PROSTU tym samym ksztaltem, przesunietym do wewnatrz)."""
    can = model_io.canonicalize(rh.points_xyz, axis)
    axis_pts = gc.axis_project(can)
    theta_unwrapped = gc.unwrap_theta(axis_pts)
    x_mm = np.array([p.x_mm for p in axis_pts])
    measured_radius = float(np.mean([p.radius_mm for p in axis_pts]))

    if project.job.offset_reference == "outer":
        ref_radius = project.tube.outer_radius_mm
    else:
        ref_radius = measured_radius

    unrolled = gc.unroll(x_mm, theta_unwrapped, ref_radius)
    dense = gc.resample_polyline(unrolled, max_seg_mm=max(project.tool.radius_mm / 3.0, 0.15))

    x_out, theta_out = gc.roll_back(dense, ref_radius)
    a_deg = gc.theta_to_deg_continuous(theta_out, sign=a_sign, zero_offset_deg=a_zero_offset_deg)
    return np.column_stack([x_out, a_deg]), ref_radius, measured_radius


def _effective_a_zero_offset(project: AppProject, source: str) -> float:
    base = project.machine.a_zero_offset_deg
    if source == "manual":
        return base + project.machine.a_sign * MANUAL_MODE_AXIS_CORRECTION_DEG
    return base


def build_toolpath(loaded: LoadedHoles, project: AppProject
                    ) -> Tuple[List[HolePreview], List[str]]:
    """Zwraca (lista_HolePreview, lista_ostrzezen_tekstowych)."""
    warnings_out: List[str] = []

    a_zero = _effective_a_zero_offset(project, loaded.source)

    try:
        operations = toolpath.build_operations(
            loaded.raw_holes, loaded.axis, project.tube, project.tool, project.job,
            x_extent=loaded.x_extent,
            a_sign=project.machine.a_sign,
            a_zero_offset_deg=a_zero,
        )
    except Exception as e:
        raise EngineError(f"Blad podczas budowy sciezki narzedzia: {e}") from e

    previews: List[HolePreview] = []
    for rh, op in zip(loaded.raw_holes, operations):
        try:
            dense_xa, ref_radius, measured_radius = _reference_dense_contour(
                rh, loaded.axis, project, project.machine.a_sign, a_zero)
        except Exception:
            dense_xa, ref_radius, measured_radius = np.zeros((0, 2)), project.tube.outer_radius_mm, 0.0

        x_range = (float(dense_xa[:, 0].min()), float(dense_xa[:, 0].max())) if len(dense_xa) else (0.0, 0.0)
        previews.append(HolePreview(
            name=op.name, ref_radius_mm=ref_radius, raw_dense_xa=dense_xa,
            operation=op, measured_radius_mm=measured_radius, x_mm_range=x_range,
        ))

        for msg in op.warnings:
            warnings_out.append(f"[{op.name}] {msg}")
        for idx, r in op.tight_corners:
            warnings_out.append(
                f"[{op.name}] Naroznik w pkt {idx}: promien lokalny ~{r:.3f}mm < "
                f"promien narzedzia {project.tool.radius_mm:.3f}mm - nie zostanie "
                f"w pelni wyrobiony."
            )
        if op.mode == "skipped":
            warnings_out.append(f"[{op.name}] POMINIETO - brak ruchu skrawajacego.")

    return previews, warnings_out


def generate_gcode(previews: List[HolePreview], project: AppProject
                    ) -> Tuple[str, List[str]]:
    ops = [p.operation for p in previews]
    try:
        text, warns = gcode_writer.generate_gcode(ops, project.tube, project.tool, project.job)
    except Exception as e:
        raise EngineError(f"Blad podczas generowania G-code: {e}") from e
    return text, warns


# --------------------------------------------------------------------------- #
#  Pomoc dla podgladu 3D: kontur (x, s) -> punkty (x, y, z) na walcu
# --------------------------------------------------------------------------- #

def unrolled_to_xyz(x_mm: np.ndarray, s_mm: np.ndarray, ref_radius_mm: float) -> np.ndarray:
    theta = s_mm / max(ref_radius_mm, 1e-6)
    y = ref_radius_mm * np.cos(theta)
    z = ref_radius_mm * np.sin(theta)
    return np.column_stack([x_mm, y, z])


def contour_pass_to_xyz(x_mm: np.ndarray, a_deg: np.ndarray, radius_mm: float) -> np.ndarray:
    theta = np.radians(a_deg)
    y = radius_mm * np.cos(theta)
    z = radius_mm * np.sin(theta)
    return np.column_stack([x_mm, y, z])
