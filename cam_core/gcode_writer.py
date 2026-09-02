"""
gcode_writer.py
----------------
Zamienia liste toolpath.HoleOperation na tekst G-code dla sterownika
grblHAL (OpenBuilds BlackBox X32), maszyna 4-osiowa uzywajaca WYLACZNIE
X (wzdluz osi rury), Z (promieniowy docisk narzedzia), A (obrot wokol X).

RUCH: caly ruch (rowniez "szybki" przejazd miedzy otworami) realizowany
jest jako G1 z jawnie policzonym F, NIGDY G0. Powod: G0 z osia obrotowa
w grblHAL zwykle NIE jest interpolowany liniowo (kazda os jedzie do celu
z wlasna maks. predkoscia, niezsynchronizowana z pozostalymi) - dla ukladu
"frez nad obracana rura" oznacza to nieprzewidywalny, potencjalnie bardzo
szybki i niekontrolowany obrot A. G1 z F wymusza koordynowany, przewidywalny
ruch - kosztem nieco wolniejszych przejazdow nieskrawajacych.

POSUW A vs F (KRYTYCZNE, patrz README p. "Bezpieczenstwo posuwu"):
grblHAL liczy F (mm/min) na podstawie DYSTANSU OSI LINIOWYCH w bloku (X, Z);
os obrotowa A jest zsynchronizowana tak, by dojechac w tym samym czasie,
NIEZALEZNIE jak szybkie put to oznacza dla A. Dla bloku BEZ ruchu liniowego
(czysty obrot A) F jest interpretowany WPROST jako stopnie/min.

To oznacza realne ryzyko: segment o malym dX ale duzym dA (typowe przy
obrysowywaniu krawedzi biegnacej "obwodowo") moze przy zwyklym F dac
NIEBEZPIECZNIE szybki obrot. `audit_feed()` w tym module liczy, jaka
predkosc katowa (deg/min) wynikneloby z zadanego F w NAJGORSZYM (zgodnym
z powyzszym opisem) przypadku, i JESLI trzeba - zmniejsza F tak, by nie
przekroczyc ToolConfig.max_rotary_speed_deg_min. To dziala bezpiecznie
NIEZALEZNIE od dokladnego zachowania konkretnej wersji firmware (jesli
firmware faktycznie blenduje A do dystansu, rzeczywisty ruch bedzie tylko
WOLNIEJSZY niz nasze obliczenie, nigdy szybszy).

MIMO TO: przed pierwszym uruchomieniem na materiale ZAWSZE zrob "przejazd
na sucho" (bez narzedzia w materiale, obnizony override posuwu) i sprawdz
faktyczna predkosc obrotu A - patrz README, sekcja "Kalibracja i pierwsze
uruchomienie".
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import TubeConfig, ToolConfig, JobConfig
from .toolpath import HoleOperation


# --------------------------------------------------------------------------- #
#  Audyt bezpieczenstwa posuwu (patrz docstring modulu)
# --------------------------------------------------------------------------- #

def audit_feed(dx_linear_mm: float, da_deg: float, desired_feed_mm_min: float,
                max_rotary_deg_min: float, ref_radius_mm: float) -> Tuple[float, bool]:
    """
    Zwraca (F_do_wypisania, czy_blok_czysto_obrotowy).

    dx_linear_mm: dystans osi liniowych w tym bloku (zwykle |dX|, bo Z jest
                  stale podczas konturowania; przy nakluciu odwrotnie).
    """
    if abs(da_deg) < 1e-9:
        return desired_feed_mm_min, False  # brak ruchu A - zwykly mm/min

    if dx_linear_mm < 1e-6:
        # blok czysto obrotowy -> F w deg/min, docelowa "predkosc obwodowa"
        # desired_feed_mm_min przeliczona na deg/min przy ref_radius_mm
        want_deg_min = math.degrees(desired_feed_mm_min / max(ref_radius_mm, 1e-6))
        return min(want_deg_min, max_rotary_deg_min), True

    # blok mieszany: w najgorszym razie F rzadzi WYLACZNIE dystansem liniowym,
    # a A ma zdazyc w tym samym czasie - policz jaka predkosc katowa by to dalo
    time_min = dx_linear_mm / max(desired_feed_mm_min, 1e-9)
    implied_deg_min = abs(da_deg) / max(time_min, 1e-12)
    if implied_deg_min <= max_rotary_deg_min:
        return desired_feed_mm_min, False
    # przytnij F tak, by implied_deg_min == max_rotary_deg_min
    capped_time_min = abs(da_deg) / max_rotary_deg_min
    capped_feed = dx_linear_mm / max(capped_time_min, 1e-12)
    return max(capped_feed, 1.0), False  # nie schodzimy ponizej 1 mm/min (0 by zawiesilo GRBL)


# --------------------------------------------------------------------------- #
#  Pisanie linii G-code
# --------------------------------------------------------------------------- #

@dataclass
class _Writer:
    lines: List[str] = field(default_factory=list)
    cur_x: float = 0.0
    cur_z: float = 0.0
    cur_a: float = 0.0
    total_time_min: float = 0.0
    total_rapid_time_min: float = 0.0

    def comment(self, text: str):
        self.lines.append(f"; {text}")

    def raw(self, line: str):
        self.lines.append(line)

    def move(self, x=None, z=None, a=None, feed_mm_min: float = None,
              max_rotary_deg_min: float = 3000.0, ref_radius_mm: float = 20.0,
              is_rapid_reposition: bool = False):
        """Emituje pojedynczy G1 (patrz naglowek modulu: NIGDY G0)."""
        nx = self.cur_x if x is None else x
        nz = self.cur_z if z is None else z
        na = self.cur_a if a is None else a

        dx_lin = math.hypot(nx - self.cur_x, nz - self.cur_z)
        da = na - self.cur_a

        f_used, is_rot = audit_feed(dx_lin, da, feed_mm_min, max_rotary_deg_min, ref_radius_mm)

        parts = ["G1"]
        if x is not None:
            parts.append(f"X{nx:.4f}")
        if z is not None:
            parts.append(f"Z{nz:.4f}")
        if a is not None:
            parts.append(f"A{na:.4f}")
        parts.append(f"F{f_used:.1f}")
        self.lines.append(" ".join(parts))

        # szacowany czas (minuty) - do raportu, nie wplywa na G-code
        if is_rot:
            dt = abs(da) / max(f_used, 1e-6)
        else:
            dist = max(dx_lin, abs(da) * math.radians(1) * ref_radius_mm)
            dt = dist / max(f_used, 1e-6) if dx_lin > 1e-9 else abs(da) / max(f_used, 1e-6)
        if is_rapid_reposition:
            self.total_rapid_time_min += dt
        else:
            self.total_time_min += dt

        self.cur_x, self.cur_z, self.cur_a = nx, nz, na


# --------------------------------------------------------------------------- #
#  Glowna funkcja
# --------------------------------------------------------------------------- #

def generate_gcode(operations: List[HoleOperation], tube: TubeConfig,
                    tool: ToolConfig, job: JobConfig) -> Tuple[str, List[str]]:
    """Zwraca (tekst_gcode, lista_ostrzezen_do_pokazania_w_UI)."""
    w = _Writer()
    warnings_out: List[str] = []

    _write_header(w, tube, tool, job)

    any_cut = False
    for op in operations:
        w.comment(f"--- otwor: {op.name} ({op.mode}) ---")
        for msg in op.warnings:
            w.comment(f"UWAGA: {msg}")
            warnings_out.append(f"[{op.name}] {msg}")
        for idx, r in op.tight_corners:
            msg = (f"Promien lokalny konturu w pkt {idx} (~{r:.3f}mm) MNIEJSZY niz "
                   f"promien narzedzia ({tool.radius_mm:.3f}mm) - naroznik nie zostanie "
                   f"w pelni wyrobiony.")
            w.comment(f"UWAGA: {msg}")
            warnings_out.append(f"[{op.name}] {msg}")

        if op.mode == "contour":
            _write_contour(w, op, tool, job)
            any_cut = True
        elif op.mode == "drill":
            _write_drill(w, op, tool, job)
            any_cut = True
        elif op.mode == "skipped":
            w.comment("POMINIETO - brak ruchu skrawajacego dla tego otworu.")
        else:
            raise ValueError(f"Nieznany tryb operacji: {op.mode!r}")

    if not any_cut:
        warnings_out.append(
            "Zaden otwor nie zostal faktycznie zaprogramowany do obrobki "
            "(wszystkie pominiete) - wygenerowany G-code NIC NIE WYTNIE."
        )

    _write_footer(w, tube, job)

    header_stats = (
        f"; Szacowany czas SKRAWANIA: {w.total_time_min:.1f} min, "
        f"przejazdy nieskrawajace: {w.total_rapid_time_min:.1f} min, "
        f"razem ~{w.total_time_min + w.total_rapid_time_min:.1f} min "
        f"(orientacyjnie - patrz README, nie uwzglednia przyspieszen/opoznien maszyny)"
    )
    text = "\n".join(w.lines) + "\n"
    text = text.replace("__STATS__", header_stats)
    return text, warnings_out


def _write_header(w: _Writer, tube: TubeConfig, tool: ToolConfig, job: JobConfig):
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    w.comment(f"{job.program_name} - wygenerowano {now}")
    w.comment("Maszyna: grblHAL / OpenBuilds BlackBox X32, osie X/Z/A (bez Y)")
    w.comment(f"Rura: OD={tube.outer_diameter_mm:.2f}mm, "
              f"sciana={tube.wall_thickness_mm:.2f}mm, dlugosc={tube.length_mm:.2f}mm")
    w.comment(f"Narzedzie: fi{tool.diameter_mm:.2f}mm, {tool.flutes} ostrza, "
              f"S={tool.spindle_rpm:.0f} RPM")
    w.comment("__STATS__")
    w.comment("UWAGA: caly ruch to G1 (celowo, nie G0) - patrz naglowek gcode_writer.py")
    w.comment("PRZED CIECIEM MATERIALU: wykonaj przejazd na sucho z obnizonym override,"
              " sprawdz faktyczna predkosc obrotu A - patrz README.")
    w.raw("G21")           # milimetry
    w.raw("G90")           # pozycjonowanie bezwzgledne
    w.raw("G94")           # posuw w jednostkach/min
    w.raw("G54")
    spindle_cmd = "M3" if tool.spindle_cw else "M4"
    w.raw(f"{spindle_cmd} S{tool.spindle_rpm:.0f}")
    if job.coolant == "mist":
        w.raw("M7")
    elif job.coolant == "flood":
        w.raw("M8")
    if job.spindle_warmup_s > 0:
        w.raw(f"G4 P{job.spindle_warmup_s:.2f}")
    w.move(x=0.0, z=job.safe_z_mm, a=0.0, feed_mm_min=tool.feed_rapid_mm_min,
           max_rotary_deg_min=tool.max_rotary_speed_deg_min, ref_radius_mm=tube.outer_radius_mm,
           is_rapid_reposition=True)


def _write_footer(w: _Writer, tube: TubeConfig, job: JobConfig):
    w.move(z=job.safe_z_mm, feed_mm_min=800.0, max_rotary_deg_min=1e9,
           ref_radius_mm=tube.outer_radius_mm, is_rapid_reposition=True)
    w.raw("M5")
    if job.coolant:
        w.raw("M9")
    w.raw("M30")


def _write_contour(w: _Writer, op: HoleOperation, tool: ToolConfig, job: JobConfig):
    max_rot = tool.max_rotary_speed_deg_min
    ref_r = op.ref_radius_mm

    first_pass = True
    for p in op.contour_passes:
        x0, a0 = float(p.path_x_mm[0]), float(p.path_a_deg[0])
        # dojazd (na safe_z) do punktu startowego tej petli
        w.move(x=x0, a=a0, feed_mm_min=tool.feed_rapid_mm_min,
               max_rotary_deg_min=max_rot, ref_radius_mm=ref_r, is_rapid_reposition=True)
        # naklucie promieniowe (Z) w miejscu startowym - ruch czysto liniowy
        w.move(z=p.z_mm, feed_mm_min=tool.feed_plunge_mm_min,
               max_rotary_deg_min=max_rot, ref_radius_mm=ref_r)
        # obrys konturu
        for x, a in zip(p.path_x_mm[1:], p.path_a_deg[1:]):
            w.move(x=float(x), a=float(a), feed_mm_min=tool.feed_cut_mm_min,
                   max_rotary_deg_min=max_rot, ref_radius_mm=ref_r)
        first_pass = False

    # odsuniecie po ostatnim przejsciu
    w.move(z=job.safe_z_mm, feed_mm_min=tool.feed_rapid_mm_min,
           max_rotary_deg_min=max_rot, ref_radius_mm=ref_r, is_rapid_reposition=True)


def _write_drill(w: _Writer, op: HoleOperation, tool: ToolConfig, job: JobConfig):
    d = op.drill
    max_rot = tool.max_rotary_speed_deg_min
    ref_r = op.ref_radius_mm

    w.move(x=d.x_mm, a=d.a_deg, feed_mm_min=tool.feed_rapid_mm_min,
           max_rotary_deg_min=max_rot, ref_radius_mm=ref_r, is_rapid_reposition=True)

    for target_z in d.peck_targets_mm:
        # WOLNY posuw wiercenia (osobny od plunge konturu) - patrz ToolConfig.feed_drill_mm_min
        w.move(z=target_z, feed_mm_min=tool.feed_drill_mm_min,
               max_rotary_deg_min=max_rot, ref_radius_mm=ref_r)
        if d.full_retract:
            w.move(z=job.safe_z_mm, feed_mm_min=tool.feed_rapid_mm_min,
                   max_rotary_deg_min=max_rot, ref_radius_mm=ref_r, is_rapid_reposition=True)
        else:
            partial = min(target_z + job.drill_retract_mm, 0.0)
            w.move(z=partial, feed_mm_min=tool.feed_rapid_mm_min,
                   max_rotary_deg_min=max_rot, ref_radius_mm=ref_r, is_rapid_reposition=True)

    if w.cur_z != job.safe_z_mm:
        w.move(z=job.safe_z_mm, feed_mm_min=tool.feed_rapid_mm_min,
               max_rotary_deg_min=max_rot, ref_radius_mm=ref_r, is_rapid_reposition=True)
