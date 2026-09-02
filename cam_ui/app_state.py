"""
app_state.py
------------
Jeden obiekt `AppProject` trzymajacy KOMPLETNY stan aplikacji: wszystkie
parametry z cam_core.config (TubeConfig/ToolConfig/JobConfig/HoleDef) plus
dodatkowe parametry potrzebne UI, ktorych nie ma w cam_core.config:

  - tryb zrodla otworow: "manual" (lista HoleDef) / "step" (plik .step) /
    "mesh" (plik .3mf/.stl/.obj)
  - sciezka do ostatnio wczytanego pliku modelu
  - kalibracja osi A maszyny (a_sign, a_zero_offset_deg) -- patrz
    cam_core/toolpath.py::build_operations i cam_core/geometry_core.py
    ::theta_to_deg_continuous
  - podpowiedz promienia zewnetrznego rury dla STEP (outer_radius_hint_mm)
    i tolerancja dopasowania (radius_match_tol_mm) -- patrz
    cam_core/model_io.py::load_step_holes

Zapis/odczyt projektu jako JSON jest wlasna implementacja (NIE uzywamy
cam_core.config.Project.to_json/from_json), bo:
  1) Project nie zna pol kalibracji/zrodla modelu, ktore trzyma ten UI,
  2) ToolConfig/JobConfig w tej wersji cam_core maja pola (feed_drill_mm_min,
     drill_peck_mm, ...) ktore Project.from_json by odrzucilo gdyby zapisac
     je inna sciezka i wczytac z powrotem (patrz PATCH w cam_core/config.py).
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from cam_core.config import (
    TubeConfig, ToolConfig, JobConfig, HoleDef, HoleShape, ToolKind,
)

# Domyslny numer wersji formatu pliku projektu (do ewentualnej migracji w przyszlosci)
PROJECT_FILE_VERSION = 1

# Patrz engine_bridge.py -- stala korekta kata dla trybu manualnego, wynika
# z tego, ze cam_core.model_io.canonicalize() z osia tozsamosciowa
# (kierunek=+X, srodek=0) wprowadza STALY obrot ukladu o -90 stopni
# (zweryfikowane numerycznie, patrz komentarz w engine_bridge.py).
MANUAL_MODE_AXIS_CORRECTION_DEG = 90.0


def _default_holes() -> List[HoleDef]:
    return []


@dataclass
class MachineCalibration:
    """Parametry kalibracji maszyny / dopasowania modelu, NIEobecne w cam_core.config."""
    a_sign: int = 1                        # +1 lub -1, patrz geometry_core.theta_to_deg_continuous
    a_zero_offset_deg: float = 0.0         # offset zera osi A wzgledem theta=0 modelu
    outer_radius_hint_mm: Optional[float] = None  # None = pelna automatyczna detekcja (najwiekszy walec)
    radius_match_tol_mm: float = 0.5       # tolerancja dopasowania promienia walca w STEP (uzywana
                                            # tylko gdy outer_radius_hint_mm nie jest None)
    step_tolerance_override_mm: Optional[float] = None  # None = uzyj JobConfig.tolerance_mm

    # [PATCH/FEATURE] Zrodlo zasiegu osi X (gdzie fizycznie wypada X=0 maszyny):
    #   "auto"          - z geometrii modelu (parametr V sciany walcowej dla STEP,
    #                      pelny zasieg siatki dla mesh) - domyslne, najdokladniejsze
    #   "manual_length" - wymuszone (0, TubeConfig.length_mm), ignorujac geometrie
    #                      pliku - reczna "siatka bezpieczenstwa", gdyby auto-detekcja
    #                      dla konkretnego pliku okazala sie zawodna (patrz README,
    #                      "Historia zmian": zgloszony blad "X=0 wychodzil na otworze")
    x_extent_source: str = "auto"


@dataclass
class AppProject:
    tube: TubeConfig = field(default_factory=TubeConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    job: JobConfig = field(default_factory=JobConfig)
    holes: List[HoleDef] = field(default_factory=_default_holes)
    machine: MachineCalibration = field(default_factory=MachineCalibration)

    source_mode: str = "manual"            # "manual" | "step" | "mesh"
    source_model_path: Optional[str] = None

    project_path: Optional[str] = None     # sciezka ostatniego zapisu/odczytu .json (nie serializowana)

    # ----------------------------------------------------------------- #

    def clone(self) -> "AppProject":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        def enum_default(o):
            from enum import Enum
            if isinstance(o, Enum):
                return o.value
            raise TypeError(o)

        return {
            "version": PROJECT_FILE_VERSION,
            "tube": asdict(self.tube),
            "tool": _tool_to_dict(self.tool),
            "job": _job_to_dict(self.job),
            "holes": [_hole_to_dict(h) for h in self.holes],
            "machine": asdict(self.machine),
            "source_mode": self.source_mode,
            "source_model_path": self.source_model_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @staticmethod
    def from_dict(raw: dict) -> "AppProject":
        tube = TubeConfig(**raw.get("tube", {}))
        tool = _tool_from_dict(raw.get("tool", {}))
        job = _job_from_dict(raw.get("job", {}))
        holes = [_hole_from_dict(h) for h in raw.get("holes", [])]
        machine = MachineCalibration(**raw.get("machine", {}))
        return AppProject(
            tube=tube, tool=tool, job=job, holes=holes, machine=machine,
            source_mode=raw.get("source_mode", "manual"),
            source_model_path=raw.get("source_model_path"),
        )

    @staticmethod
    def from_json(text: str) -> "AppProject":
        return AppProject.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        self.project_path = path

    @staticmethod
    def load(path: str) -> "AppProject":
        with open(path, "r", encoding="utf-8") as f:
            proj = AppProject.from_json(f.read())
        proj.project_path = path
        return proj


# --------------------------------------------------------------------------- #
#  Konwersje enum <-> str dla ToolConfig / JobConfig / HoleDef
# --------------------------------------------------------------------------- #

def _tool_to_dict(tool: ToolConfig) -> dict:
    d = asdict(tool)
    d["kind"] = tool.kind.value if isinstance(tool.kind, ToolKind) else tool.kind
    return d


def _tool_from_dict(raw: dict) -> ToolConfig:
    raw = dict(raw)
    if "kind" in raw:
        raw["kind"] = ToolKind(raw["kind"])
    return ToolConfig(**raw)


def _job_to_dict(job: JobConfig) -> dict:
    return asdict(job)


def _job_from_dict(raw: dict) -> JobConfig:
    return JobConfig(**raw)


def _hole_to_dict(h: HoleDef) -> dict:
    d = asdict(h)
    d["shape"] = h.shape.value if isinstance(h.shape, HoleShape) else h.shape
    return d


def _hole_from_dict(raw: dict) -> HoleDef:
    raw = dict(raw)
    raw["shape"] = HoleShape(raw["shape"])
    return HoleDef(**raw)


def new_hole_default(existing_names: List[str]) -> HoleDef:
    """Nowy otwor z rozsadnymi wartosciami domyslnymi i unikalna nazwa."""
    n = 1
    while f"otwor_{n}" in existing_names:
        n += 1
    return HoleDef(
        name=f"otwor_{n}",
        shape=HoleShape.CIRCLE,
        center_x_mm=50.0,
        center_angle_deg=0.0,
        diameter_mm=8.0,
    )
