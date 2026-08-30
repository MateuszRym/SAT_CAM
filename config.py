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

WIELOPRZEJSCIOWE FREZOWANIE (kilka nawrotow do pelnej glebokosci):
  Pojedynczy przejazd konturu nigdy nie zaglebia sie od razu na cala grubosc
  sciany -- `JobConfig.pass_depth_mm` to MAKSYMALNY promieniowy dosuw na
  JEDNO przejscie; jesli grubosc sciany + naddatek (patrz
  `JobConfig.total_cut_depth_mm`) jest wieksza, `toolpath._pass_depths()`
  dzieli obrobke na tyle rownych przejsc, zeby zaden pojedynczy dosuw nie
  przekroczyl `pass_depth_mm` (patrz tez `JobConfig.estimated_pass_count`
  do podgladu w UI PRZED wygenerowaniem pelnej sciezki). Analogicznie dla
  wiercenia: `JobConfig.drill_peck_mm` dzieli cykl peckingowy na kroki.

LĄCZNIKI / MOSTKI (tabs) w duzych otworach:
  Przy duzym otworze wyciety fragment sciany rury moze odpasc do jej srodka,
  zanim frez skonczy caly obrys (nic go wtedy juz nie podtrzymuje). Gdy
  najdluzszy bok otworu przekracza `JobConfig.tab_min_size_factor` (typowo
  5-15) razy srednice narzedzia, program automatycznie zostawia
  `JobConfig.tab_count` niewielkich, nieprzewierconych mostkow materialu
  (szerokosc `tab_width_mm`, pozostawiona grubosc `tab_remaining_thickness_mm`)
  rozlozonych rownomiernie po obwodzie -- wycieta czesc zostaje "na zawiasach"
  do wylamania recznie po zdjeciu z maszyny. Mozna to wymusic/wylaczyc per
  otwor przez `HoleDef.tabs_override`, niezaleznie od automatycznej heurystyki
  rozmiaru (patrz tez `JobConfig.tab_threshold_mm` do podgladu progu w UI).
  Geometria okien lacznikow liczona jest w geometry_core (cumulative_arc_length /
  tab_windows / point_in_tab_mask), a sama sciezka -- w toolpath.py.
"""

from __future__ import annotations

import json
import math
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
    feed_plunge_mm_min: float = 80.0   # posuw wgladu promieniowego (Z) - naklucie konturu
    feed_drill_mm_min: float = 60.0    # posuw WLASNY cyklu wiercenia (peck) - wolniejszy
                                        # niz feed_plunge_mm_min, bo pelny przekroj narzedzia
                                        # pracuje jednoczesnie (jak przy prawdziwym wierceniu),
                                        # a nie tylko jego promieniowa krawedz jak w konturze
    feed_rapid_mm_min: float = 1500.0  # "rapid" realizowany jako G1 z duzym F
                                        # (patrz README: dlaczego nie G0 na A)

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
    pass_depth_mm: float = 0.8         # MAKSYMALNY promieniowy dosuw na JEDNO przejscie
                                        # konturu -- jesli grubosc sciany + naddatek jest
                                        # wieksza, obrobka dzieli sie automatycznie na kilka
                                        # kolejnych przejsc (patrz toolpath._pass_depths i
                                        # estimated_pass_count() nizej do podgladu w UI)
    breakthrough_margin_mm: float = 0.4  # naddatek za grubosc sciany (pewne przebicie)
    tolerance_mm: float = 0.03         # max. odchylka cieciwy przy dyskretyzacji lukow
    offset_reference: str = "outer"    # promien odniesienia do "rozwiniecia" (patrz README)
    lead_in: bool = True               # najazd stycznie zamiast wprost w naroze
    spindle_warmup_s: float = 2.0
    coolant: Optional[str] = None      # None / "mist" / "flood" -> M7/M8 (opcjonalnie)
    program_name: str = "PIPE_HOLES"

    # --- Cykl wiercenia (tryb DRILL, patrz toolpath.py naglowek pliku) --- #
    drill_peck_mm: float = 0.5         # MAKSYMALNY promieniowy dosuw na JEDEN peck
                                        # (analogicznie do pass_depth_mm, ale dla wiercenia)
    drill_diameter_tolerance_mm: float = 0.05  # margines przy decyzji "czy narzedzie
                                        # zmiesci sie jako wiertlo" (srednica_narzedzia <=
                                        # srednica_otworu + ta tolerancja)
    drill_full_retract: bool = True    # True -> pelny odwrot do safe_z_mm miedzy peckami
                                        # (bezpieczniejsze, wolniejsze); False -> czesciowy
                                        # odwrot o drill_retract_mm (szybsze, typowe dla
                                        # plytkich/miekkich materialow)
    drill_retract_mm: float = 1.0      # odwrot CZESCIOWY miedzy peckami, uzywany tylko gdy
                                        # drill_full_retract=False

    # --- Lączniki / mostki (tabs) w duzych otworach, patrz naglowek pliku --- #
    use_tabs: bool = True              # globalny wylacznik automatycznego dodawania
                                        # lacznikow wg heurystyki rozmiaru (per-otworowy
                                        # override: HoleDef.tabs_override)
    tab_min_size_factor: float = 8.0   # prog rozmiaru otworu wyzwalajacy automatyczne
                                        # lączniki, w wielokrotnosciach srednicy narzedzia
                                        # (dluzszy z bokow bbox otworu >= tab_min_size_factor
                                        # * tool.diameter_mm) -- typowy sensowny zakres 5-15
    tab_count: int = 4                 # liczba lacznikow rozlozonych rownomiernie po obwodzie
    tab_width_mm: float = 2.0          # szerokosc (dlugosc luku) pojedynczego lacznika
    tab_remaining_thickness_mm: float = 0.3  # grubosc materialu POZOSTAWIONA nieprzewiercona
                                        # w oknie lacznika, liczona od grubosci sciany rury
                                        # (NIE od total_cut_depth_mm z naddatkiem przebicia)

    # --- Metody pomocnicze do podgladu w UI (bez generowania pelnej sciezki) --- #

    def total_cut_depth_mm(self, tube: "TubeConfig") -> float:
        """Calkowita promieniowa glebokosc obrobki (grubosc sciany + naddatek na
        pewne przebicie). Jedno miejsce liczenia tej wartosci - uzywane przez
        toolpath.py oraz przez UI do podgladu razem z estimated_pass_count()."""
        return tube.wall_thickness_mm + self.breakthrough_margin_mm

    def estimated_pass_count(self, total_depth_mm: float) -> int:
        """Szacowana liczba przejsc konturu potrzebna do osiagniecia
        `total_depth_mm` przy aktualnym pass_depth_mm - ta sama formula co
        toolpath._pass_depths(), do podgladu w UI PRZED wygenerowaniem
        pelnej sciezki (np. 'ten otwor: 3 przejscia po 0.67mm')."""
        total_depth_mm = max(total_depth_mm, 1e-6)
        pass_depth = max(self.pass_depth_mm, 1e-3)
        return max(1, math.ceil(total_depth_mm / pass_depth))

    def estimated_peck_count(self, total_depth_mm: float) -> int:
        """Jak estimated_pass_count(), ale dla cyklu wiercenia (drill_peck_mm)."""
        total_depth_mm = max(total_depth_mm, 1e-6)
        peck = max(self.drill_peck_mm, 1e-3)
        return max(1, math.ceil(total_depth_mm / peck))

    def tab_threshold_mm(self, tool_diameter_mm: float) -> float:
        """Prog wielkosci otworu [mm] (dluzszy z wymiarow bbox), powyzej ktorego
        automatycznie wlaczane sa lączniki - do wyswietlenia w UI jako
        podpowiedz przy ustawianiu tab_min_size_factor / srednicy narzedzia."""
        return self.tab_min_size_factor * tool_diameter_mm

    def tab_cap_depth_mm(self, tube: "TubeConfig") -> float:
        """Promieniowa glebokosc, do ktorej WOLNO dojechac w oknach lacznikow
        (reszta otworu dochodzi normalnie do total_cut_depth_mm). Liczona od
        samej grubosci sciany rury (nie od total_cut_depth_mm z naddatkiem),
        zeby mostek byl realnym, nieprzewierconym kawalkiem sciany. Dolny limit
        0.05mm - zawsze lekki nacisk narzedzia, dla widocznego sladu polozenia
        lacznika nawet przy zbyt duzej tab_remaining_thickness_mm."""
        return max(tube.wall_thickness_mm - self.tab_remaining_thickness_mm, 0.05)


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
    # None -> decyduj automatycznie (JobConfig.use_tabs + heurystyka rozmiaru,
    #         patrz JobConfig.tab_min_size_factor); True/False -> wymus dla TEGO
    #         otworu niezaleznie od heurystyki (checkbox w UI per-otworowy)
    tabs_override: Optional[bool] = None


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
