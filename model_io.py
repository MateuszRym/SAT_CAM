"""
model_io.py
-----------
Wczytywanie modelu rury z otworami i wyciaganie konturow otworow jako
chmury punktow 3D (w oryginalnym ukladzie modelu -- kanonizacja do
"os rury = os X" jest w `canonicalize()`).

Dwie sciezki:

  * STEP (.step/.stp)  -> `load_step_holes()`  [cadquery / OCP, BRep - PRECYZYJNE]
        Szukamy walcowej powierzchni zewnetrznej rury (dopasowanie promienia),
        a nastepnie jej WEWNETRZNYCH petli (wires) - to sa dokladnie krawedzie
        otworow "wycietych" w tej powierzchni. Kazda krawedz jest
        dyskretyzowana adaptacyjnie (GCPnts_QuasiUniformDeflection) do
        zadanej tolerancji cieciwy.

  * 3MF / STL / OBJ (.3mf/.stl/.obj) -> `load_mesh_holes()` [trimesh - PRZYBLIZONE]
        Dziala WYLACZNIE gdy siatka jest OTWARTA w miejscach otworow
        (tzn. model to "skorupa" powierzchni zewnetrznej rury z dziurami,
        a nie pelna bryla z modelowanymi scianami otworu). W takim
        przypadku krawedzie otworow to krawedzie brzegowe siatki (nalezace
        do dokladnie 1 trojkata) - grupujemy je w petle.
        Jesli 3MF eksportowany jest jako zamknieta (watertight) bryla
        (najczesciej!), automatyczna detekcja NIE zadziala niezawodnie -
        patrz README, sekcja "Ograniczenia formatu 3MF/mesh": w takim
        wypadku nalezy uzyc trybu manualnego (HoleDef w config.py) albo
        wyeksportowac model jako STEP.

Obie sciezki zwracaja liste `RawHole` (punkty 3D w ukladzie modelu +
metadane), oraz funkcje do automatycznego wykrycia osi rury (dopasowanie
walca metoda najmniejszych kwadratow), potrzebnej do kanonizacji ukladu
wspolrzedny przed przekazaniem do geometry_core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RawHole:
    name: str
    points_xyz: np.ndarray   # (N,3) w ukladzie ORYGINALNYM modelu, petla zamknieta
    source: str = ""         # np. "step_face_wire_2"
    # Przekazane z config.HoleDef.tabs_override dla otworow z trybu manualnego
    # (None dla otworow wykrytych z STEP/mesh - tam decyduje wylacznie
    # automatyczna heurystyka rozmiaru w toolpath.build_operations).
    tabs_override: Optional[bool] = None


@dataclass
class TubeAxis:
    point_on_axis: np.ndarray   # (3,)
    direction: np.ndarray       # (3,) znormalizowany
    radius_mm: float


# --------------------------------------------------------------------------- #
#  Dopasowanie osi walca (uzywane dla obu sciezek, gdy model nie jest juz
#  zorientowany osia rury = global X)
# --------------------------------------------------------------------------- #

def fit_cylinder_axis(points_xyz: np.ndarray) -> TubeAxis:
    """
    Dopasowanie osi walca metoda najmniejszych kwadratow (algorytm
    kierunku o minimalnej wariancji rzutow promieniowych -- prosta,
    solidna heurystyka: kierunek osi to kierunek najwiekszej wariancji
    chmury punktow powloki walca, tj. pierwszy wektor wlasny PCA).
    Dziala dobrze dla rury znacznie dluzszej niz szerszej (typowy rocket
    body tube). Nastepnie promien = mediana odleglosci punktow od osi.
    """
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / np.linalg.norm(direction)

    # promien: odleglosc punktow od prostej (centroid, direction)
    t = centered @ direction
    proj = np.outer(t, direction)
    radial = centered - proj
    radii = np.linalg.norm(radial, axis=1)
    radius = float(np.median(radii))

    return TubeAxis(point_on_axis=centroid, direction=direction, radius_mm=radius)


def canonicalize(points_xyz: np.ndarray, axis: TubeAxis) -> np.ndarray:
    """
    Transformuje punkty tak, by os rury pokrywala sie z globalna osia X,
    a x=0 wypadalo w `axis.point_on_axis` (referencje X ustawia sie pozniej
    w toolpath.py wedlug TubeConfig.x_zero_at).
    """
    d = axis.direction
    # budujemy ortonormalna baze (d, u, v)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    R = np.column_stack([d, u, v])  # world = R @ local  =>  local = R.T @ world
    local = (points_xyz - axis.point_on_axis) @ R
    return local  # kolumny: [x_wzdluz_osi, y_lokalny, z_lokalny]


# --------------------------------------------------------------------------- #
#  STEP (cadquery / OCP)
# --------------------------------------------------------------------------- #

def load_step_holes(path: str, tolerance_mm: float = 0.03,
                     outer_radius_hint_mm: Optional[float] = None,
                     radius_match_tol_mm: float = 0.5) -> Tuple[List[RawHole], TubeAxis]:
    """
    Zwraca (lista_otworow, os_rury) w oryginalnym ukladzie modelu.

    Strategia (patrz README, sekcja "Jak dziala ekstrakcja z STEP" po
    szczegolowe uzasadnienie -- w skrocie: NIE polegamy na strukturze
    'wire' zwracanej przez BRep, bo OCC czesto reprezentuje sciane
    walcowa z otworami jako JEDEN wire polaczony "mostkami" wzdluz szwu
    parametryzacji (u=0/2pi), zwlaszcza gdy otwor lezy blisko szwu.
    Zamiast tego triangulujemy KAZDA sciane walcowa z osobna (siatka
    tylko tej jednej sciany), sklejamy wierzcholki-duplikaty na szwie
    (te same wspolrzedne 3D, rozne indeksy przez periodycznosc), a
    nastepnie szukamy petli krawedzi brzegowych triangulacji (naleza-
    cych do dokladnie 1 trojkata) - to zawsze daje CZYSTO ROZDZIELONE
    petle: obrys(y) konca rury + jedna petla na kazdy otwor, niezaleznie
    od tego, jak skomplikowana byla oryginalna topologia BRep.

    1. Zaladuj shape, znajdz wszystkie sciany walcowe (GeomType == CYLINDER).
    2. Wybierz te o promieniu ~ outer_radius_hint_mm (lub najwiekszym, jesli
       nie podano hinta) - to zewnetrzna powloka rury.
    3. Zsiatkuj kazda taka sciane (BRepMesh_IncrementalMesh, deflection =
       tolerance_mm) i wyciagnij jej wlasna triangulacje.
    4. Znajdz petle brzegowe, sklasyfikuj: petla obejmujaca ~caly obwod
       (360 st.) I majaca prawie zerowy zasieg wzdluz osi = obrys konca
       rury (odrzucamy); kazda inna petla = otwor.
    """
    import cadquery as cq
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    result = cq.importers.importStep(path)
    shape = result.val().wrapped

    # --- znajdz sciany walcowe ---
    cyl_faces = []  # (face, radius, gp_Cyl)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        surf = BRepAdaptor_Surface(face, True)
        if surf.GetType() == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            cyl_faces.append((face, cyl.Radius(), cyl))
        exp.Next()

    if not cyl_faces:
        raise ValueError("Nie znaleziono zadnej powierzchni walcowej w modelu STEP - "
                          "sprawdz, czy model to rura (obiekt walcowy).")

    if outer_radius_hint_mm is not None:
        cyl_faces = [f for f in cyl_faces
                     if abs(f[1] - outer_radius_hint_mm) < radius_match_tol_mm]
        if not cyl_faces:
            raise ValueError(
                f"Zaden fragment walcowy nie pasuje do podanej srednicy zewnetrznej "
                f"({outer_radius_hint_mm*2:.2f} mm +/- {radius_match_tol_mm*2:.2f} mm). "
                f"Sprawdz TubeConfig.outer_diameter_mm."
            )
    outer_radius = max(f[1] for f in cyl_faces)
    outer = [f for f in cyl_faces if abs(f[1] - outer_radius) < 1e-3]

    # os walca -- ANALITYCZNA, wprost z geometrii BRep (nie z fitu PCA,
    # ktory bylby wrazliwy na nierownomierny rozklad punktow siatki)
    ax1 = outer[0][2].Axis()
    loc = ax1.Location()
    d = ax1.Direction()
    axis = TubeAxis(point_on_axis=np.array([loc.X(), loc.Y(), loc.Z()]),
                     direction=np.array([d.X(), d.Y(), d.Z()]),
                     radius_mm=outer_radius)

    holes: List[RawHole] = []
    hole_idx = 0
    for face, radius, _cyl in outer:
        loops_xyz = _mesh_face_boundary_loops(face, tolerance_mm)
        for pts in loops_xyz:
            can = canonicalize(pts, axis)
            x_range = float(np.ptp(can[:, 0]))
            theta = np.unwrap(np.arctan2(can[:, 2], can[:, 1]))
            theta_span_deg = float(np.degrees(theta.max() - theta.min()))
            is_rim = (theta_span_deg > 300.0) and (x_range < 2.0)
            if is_rim:
                continue
            hole_idx += 1
            holes.append(RawHole(name=f"hole_{hole_idx}", points_xyz=pts,
                                  source=f"step_cyl_r{outer_radius:.2f}"))

    if not holes:
        raise ValueError(
            "Znaleziono powierzchnie walcowa zewnetrzna, ale zaden otwor. "
            "Sprawdz model (czy otwory naprawde przebijaja sciane rury) "
            "oraz TubeConfig.outer_diameter_mm (musi pasowac do promienia "
            "zewnetrznego w pliku STEP)."
        )
    return holes, axis


def _mesh_face_boundary_loops(face, tolerance_mm: float) -> List[np.ndarray]:
    """Siatkuje POJEDYNCZA sciane i zwraca jej petle brzegowe jako punkty 3D (world)."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(face, max(tolerance_mm, 0.001), False, 0.3, False)
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbNodes() == 0:
        return []
    trsf = loc.Transformation()

    nodes = np.empty((tri.NbNodes(), 3), dtype=float)
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i).Transformed(trsf)
        nodes[i - 1] = (p.X(), p.Y(), p.Z())

    merged_nodes, remap = _merge_duplicate_points(nodes, tol=1e-4)

    edge_count = {}
    for i in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(i).Get()
        a, b, c = int(remap[a - 1]), int(remap[b - 1]), int(remap[c - 1])
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [k for k, v in edge_count.items() if v == 1]

    loops_idx = _chain_edges_into_loops(boundary_edges)
    return [merged_nodes[loop] for loop in loops_idx if len(loop) >= 4]


def _merge_duplicate_points(points: np.ndarray, tol: float = 1e-4):
    """
    Sklejenie wierzcholkow o (prawie) identycznych wspolrzednych 3D w jeden
    indeks. Niezbedne dla scian OKRESOWYCH (walec ma szew u=0/2pi): triangu-
    lacja OCC duplikuje wierzcholki na szwie pod dwoma roznymi indeksami
    mimo tych samych wspolrzednych, co bez sklejenia sztucznie tworzy
    falszywe krawedzie "brzegowe" wzdluz calego szwu.
    """
    rounded = np.round(points / tol).astype(np.int64)
    seen = {}
    remap = np.empty(len(points), dtype=np.int64)
    merged = []
    for i, key in enumerate(map(tuple, rounded)):
        j = seen.get(key)
        if j is None:
            j = len(merged)
            seen[key] = j
            merged.append(points[i])
        remap[i] = j
    return np.array(merged), remap


# --------------------------------------------------------------------------- #
#  Mesh: 3MF / STL / OBJ (trimesh) - metoda przyblizona (krawedzie brzegowe)
# --------------------------------------------------------------------------- #

def load_mesh_holes(path: str, min_loop_points: int = 6) -> Tuple[List[RawHole], TubeAxis]:
    """
    UWAGA (patrz naglowek pliku): dziala tylko dla siatek "otwartych" w
    miejscu otworow (skorupa z dziurami), NIE dla pelnych/zamknietych brył.
    Dla zamknietych brył 3MF nalezy uzyc trybu manualnego albo eksportu STEP.
    """
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if mesh.is_watertight:
        raise ValueError(
            "Siatka jest zamknieta (watertight) - automatyczna detekcja otworow "
            "z mesha dziala tylko dla otwartych skorup z dziurami. Uzyj trybu "
            "manualnego (lista otworow z parametrow) albo wyeksportuj model jako STEP."
        )

    # krawedzie brzegowe = nalezace do dokladnie 1 trojkata
    edges = mesh.edges_sorted
    from collections import Counter
    counts = Counter(map(tuple, edges))
    boundary_edges = [e for e, c in counts.items() if c == 1]

    if not boundary_edges:
        raise ValueError("Nie znaleziono zadnych krawedzi brzegowych (otworow) w siatce.")

    loops = _chain_edges_into_loops(boundary_edges)
    axis = fit_cylinder_axis(mesh.vertices)

    holes = []
    hole_idx = 0
    # petla o najwiekszym obwodzie zwykle to koncowka rury (jesli otwarta na
    # koncach) a nie otwor - odsiewamy najwieksza, jesli jest > 3x wieksza
    # od mediany (heurystyka; w praktyce lepiej zweryfikowac w UI).
    loop_arrays = []
    for loop in loops:
        if len(loop) < min_loop_points:
            continue
        pts = mesh.vertices[loop]
        loop_arrays.append(pts)

    if not loop_arrays:
        raise ValueError("Krawedzie brzegowe znalezione, ale zaden zamknieta petla "
                          "nie ma wystarczajacej liczby punktow.")

    perims = [_polyline_length(p) for p in loop_arrays]
    median_perim = float(np.median(perims))
    for pts, perim in zip(loop_arrays, perims):
        if perim > 3.0 * median_perim and len(loop_arrays) > 1:
            continue  # prawdopodobnie koniec rury, nie otwor
        hole_idx += 1
        holes.append(RawHole(name=f"hole_{hole_idx}", points_xyz=pts,
                              source="mesh_boundary_loop"))

    return holes, axis


def _chain_edges_into_loops(edges: List[Tuple[int, int]]) -> List[List[int]]:
    from collections import defaultdict
    adjacency = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited_edges = set()
    loops = []
    for a, b in edges:
        key = tuple(sorted((a, b)))
        if key in visited_edges:
            continue
        # przejdz petle zaczynajac od a->b
        loop = [a, b]
        visited_edges.add(key)
        current = b
        prev = a
        while True:
            neighbors = [n for n in adjacency[current] if n != prev
                         or tuple(sorted((current, n))) not in visited_edges]
            nxt = None
            for n in adjacency[current]:
                ek = tuple(sorted((current, n)))
                if ek in visited_edges:
                    continue
                nxt = n
                break
            if nxt is None:
                break
            visited_edges.add(tuple(sorted((current, nxt))))
            loop.append(nxt)
            prev, current = current, nxt
            if current == loop[0]:
                break
        if loop[0] == loop[-1] and len(loop) > 3:
            loops.append(loop)
    return loops


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# --------------------------------------------------------------------------- #
#  Otwory z definicji manualnej (config.HoleDef) -> RawHole w ukladzie
#  kanonicznym (od razu, bo nie ma tu zadnego "oryginalnego" ukladu modelu)
# --------------------------------------------------------------------------- #

def holes_from_manual_defs(hole_defs, tube_outer_radius_mm: float, tolerance_mm: float = 0.03):
    """Zwraca liste RawHole juz w ukladzie KANONICZNYM (x wzdluz osi X, promien=outer)."""
    import hole_shapes as hs

    holes = []
    for hd in hole_defs:
        if not hd.enabled:
            continue
        if hd.shape.value == "circle":
            local = hs.circle(hd.diameter_mm, tolerance_mm)
        elif hd.shape.value == "rectangle":
            local = hs.rectangle(hd.width_mm, hd.height_mm)
        elif hd.shape.value == "rounded_rectangle":
            local = hs.rounded_rectangle(hd.width_mm, hd.height_mm, hd.corner_radius_mm,
                                          tolerance_mm)
        elif hd.shape.value == "polygon":
            local = hs.polygon(hd.polygon_points_mm)
        else:
            raise ValueError(f"Nieznany ksztalt otworu: {hd.shape}")

        local = hs.ensure_ccw(local)
        pts3d = []
        for u, v in local:
            x = hd.center_x_mm + u
            theta = math.radians(hd.center_angle_deg) + v / tube_outer_radius_mm
            y = tube_outer_radius_mm * math.cos(theta)
            z = tube_outer_radius_mm * math.sin(theta)
            pts3d.append([x, y, z])
        holes.append(RawHole(name=hd.name, points_xyz=np.array(pts3d), source="manual",
                              tabs_override=hd.tabs_override))
    return holes
