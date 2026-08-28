"""
config.py
---------
Wszystkie parametry maszyny, narzedzia, rury i zadania w jednym miejscu.
Trzymane jako dataclasses, latwe do (de)serializacji JSON (UI zapisuje/wczytuje
profile maszyny i zadania jako pliki .json).

WAZNE KONWENCJE UKLADU WSPOLRZEDNYCH (patrz README.md, rozdzial "Konwencje"):

  Os maszyny X  -> wzdluz osi obrotu rury (pokrywa sie z osia A).
  Os maszyny A  -> obrot wokol X, w stopniach, ciagly (bez zawijania 0-360).
  Os maszyny Z  -> promieniowy docisk narzedzia:
                     Z = 0      -> narzedzie styka sie z ZEWNETRZNA powierzchnia rury
                                    (bazowanie recznym dotykiem/touch-off)
                     Z < 0      -> narzedzie zaglebione w materiale (w strone osi rury)
                     Z = -grubosc_sciany - naddatek  -> pelne przebicie sciany

  Ukryty w kodzie "przemysl" ukladu kanonicznego modelu (nie jest to os maszyny!):
     Xc  -> wzdluz osi rury (= os maszyny X, po ustawieniu referencji)
     Yc, Zc -> plaszczyzna przekroju poprzecznego rury
     kat theta = atan2(Zc, Yc)  [rad]  -> po przeliczeniu na stopnie to os A

Rozroznienie "Zc" (wspolrzedna w modelu) od "Z maszyny" (os posuwu promieniowego)
jest krytyczne i konsekwentnie zachowane w calym kodzie: model uzywa Yc/Zc,
G-code i toolpath uzywaja pol `.z` ktore ZAWSZE oznacza os maszyny (promieniowy
docisk), nigdy wspolrzedna przekroju.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


# --------------------------------------------------------------------------- #
#  Parametry tulei / rury
# --------------------------------------------------------------------------- #

@dataclass
class TubeConfig:
    outer_diameter_mm: float = 41.0     # typowa rura BT-50 itp. -- do nadpisania
    wall_thickness_mm: float = 1.6
    length_mm: float = 300.0
    # Ktory koniec modelu (po znalezieniu/kanonizacji osi) odpowiada X=0 maszyny.
    # "min" = poczatek modelu wzdluz osi, "max" = koniec modelu wzdluz osi.
    x_zero_at: str = "min"
    # Dodatkowe przesuniecie (np. rura wystaje z uchwytu o X mm przed
    # punktem, ktory dotykamy jako X=0) -- dodawane do X po zmianie referencji.
    x_offset_mm: float = 0.0

    @property
    def outer_radius_mm(self) -> float:
        return self.outer_diameter_mm / 2.0

    @property
    def inner_radius_mm(self) -> float:
        return self.outer_radius_mm - self.wall_thickness_mm


# --------------------------------------------------------------------------- #
#  Parametry narzedzia
# --------------------------------------------------------------------------- #

class ToolKind(str, Enum):
    ENDMILL = "endmill"          # kontury + kieszenie, dowolny ksztalt
    DRILL = "drill"               # tylko otwory kolowe o srednicy = tool_diameter


@dataclass
class ToolConfig:
    kind: ToolKind = ToolKind.ENDMILL
    diameter_mm: float = 3.0
    flutes: int = 2
    spindle_rpm: float = 12000.0
    spindle_cw: bool = True        # True -> M3, False -> M4

    # Posuwy
    feed_cut_mm_min: float = 400.0     # posuw konturowania (XZ, liniowy ekwiwalent)
    feed_plunge_mm_min: float = 80.0   # posuw wgladu promieniowego (Z) przy KONTUROWANIU
    feed_rapid_mm_min: float = 1500.0  # "rapid" realizowany jako G1 z duzym F
                                        # (patrz README: dlaczego nie G0 na A)

    # Osobny, WOLNIEJSZY posuw dla trybu WIERCENIA (gdy narzedzie uzywane
    # jest jak wiertlo - patrz toolpath.py, otwory o srednicy ~= srednicy
    # narzedzia, gdzie kontur nie miesci sie do zoffsetowania). Wiercenie
    # martwym srodkiem freza wymaga wolniejszego Z niz zwykly plunge przy
    # konturowaniu, bo caly przekroj narzedzia pracuje naraz (nie tylko obwod).
    feed_drill_mm_min: float = 40.0

    # Bezpieczenstwo obrotu -- patrz gcode_writer.audit_and_limit_feed()
    max_rotary_speed_deg_min: float = 3000.0

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0


# --------------------------------------------------------------------------- #
#  Parametry obrobki (wspolne dla zadania)
# --------------------------------------------------------------------------- #

@dataclass
class JobConfig:
    safe_z_mm: float = 5.0             # odsuniecie od powierzchni przed przejazdami
    pass_depth_mm: float = 0.8         # promieniowy dosuw na przejazd
    breakthrough_margin_mm: float = 0.4  # naddatek za grubosc sciany (pewne przebicie)
    tolerance_mm: float = 0.03         # max. odchylka cieciwy przy dyskretyzacji lukow
    offset_reference: str = "outer"    # promien odniesienia do "rozwiniecia" (patrz README)
    lead_in: bool = True               # najazd stycznie zamiast wprost w naroze
    spindle_warmup_s: float = 2.0
    coolant: Optional[str] = None      # None / "mist" / "flood" -> M7/M8 (opcjonalnie)
    program_name: str = "PIPE_HOLES"

    # --- tryb WIERCENIA (patrz toolpath.py: automatyczne przelaczenie gdy
    #     kontur nie miesci sie do zoffsetowania, a otwor jest kolowy) ---
    drill_peck_mm: float = 1.0            # glebokosc pojedynczego "pecka"
    drill_full_retract: bool = True       # True: pelny odwrot do safe_z miedzy peckami
                                           # (bezpieczniejsze, wolniejsze - polecane przy
                                           # braku odciagu/dmuchawy wiorow)
    drill_retract_mm: float = 0.5         # uzywane tylko gdy drill_full_retract=False:
                                           # czesciowy odwrot miedzy peckami
    drill_diameter_tolerance_mm: float = 0.15  # o ile narzedzie MOZE byc wieksze niz
                                           # zmierzona srednica otworu, a mimo to uznajemy
                                           # ze "pasuje jako wiertlo" (naddatek na
                                           # tolerancje pomiaru/tesselacji z modelu)


# --------------------------------------------------------------------------- #
#  Definicja pojedynczego otworu (uzywana w trybie "manualnym" / z JSON,
#  a takze jako wynik posredni ekstrakcji z modelu STEP/3MF)
# --------------------------------------------------------------------------- #

class HoleShape(str, Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    POLYGON = "polygon"        # dowolny ksztalt z listy punktow (u,v) w mm


@dataclass
class HoleDef:
    """
    Otwor definiowany PARAMETRYCZNIE (tryb manualny) w lokalnym ukladzie (u, v):
      u -> wzdluz osi rury [mm], liczone od srodka otworu
      v -> obwodowo [mm luku], liczone od srodka otworu
    Srodek otworu umieszczany jest w (center_x_mm, center_angle_deg).
    """
    name: str
    shape: HoleShape
    center_x_mm: float
    center_angle_deg: float
    # parametry ksztaltu:
    diameter_mm: float = 0.0          # CIRCLE
    width_mm: float = 0.0             # RECTANGLE / ROUNDED_RECTANGLE (wzdluz u / X)
    height_mm: float = 0.0            # RECTANGLE / ROUNDED_RECTANGLE (wzdluz v / obwod)
    corner_radius_mm: float = 0.0     # ROUNDED_RECTANGLE
    polygon_points_mm: List[List[float]] = field(default_factory=list)  # POLYGON [[u,v],...]
    enabled: bool = True


# --------------------------------------------------------------------------- #
#  Cale zadanie (do zapisu/odczytu JSON w UI)
# --------------------------------------------------------------------------- #

@dataclass
class Project:
    tube: TubeConfig = field(default_factory=TubeConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    job: JobConfig = field(default_factory=JobConfig)
    holes: List[HoleDef] = field(default_factory=list)
    source_model_path: Optional[str] = None

    def to_json(self) -> str:
        def _default(o):
            if isinstance(o, Enum):
                return o.value
            raise TypeError(o)
        return json.dumps(asdict(self), indent=2, default=_default, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "Project":
        raw = json.loads(text)
        tube = TubeConfig(**raw.get("tube", {}))
        tool_raw = dict(raw.get("tool", {}))
        if "kind" in tool_raw:
            tool_raw["kind"] = ToolKind(tool_raw["kind"])
        tool = ToolConfig(**tool_raw)
        job = JobConfig(**raw.get("job", {}))
        holes = []
        for h in raw.get("holes", []):
            h = dict(h)
            h["shape"] = HoleShape(h["shape"])
            holes.append(HoleDef(**h))
        return Project(tube=tube, tool=tool, job=job, holes=holes,
                        source_model_path=raw.get("source_model_path"))
