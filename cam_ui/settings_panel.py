"""
settings_panel.py
------------------
Lewy panel z zakladkami: Rura / Narzedzie / Zadanie / Kalibracja / Model /
Otwory. Kazde pole jest na biezaco (Tk variable + trace) zapisywane wprost
do przekazanego obiektu AppProject -- nie ma osobnego przycisku "Zastosuj".

Panel NIE wie nic o cam_core poza typami z cam_core.config (do wypelnienia
combobox-ow) -- cala logika inzynierska zostaje w cam_core/ i cam_ui/
engine_bridge.py.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional

from cam_core.config import ToolKind, HoleShape
from .app_state import AppProject, new_hole_default
from .hole_dialog import HoleDialog, SHAPE_LABELS


class ScrollableFrame(ttk.Frame):
    """Ramka z pionowym scrollem (kolko myszy dziala tylko gdy kursor jest nad nia).

    UWAGA (poprawka bledu "znikajacego" panelu ustawien): tk.Canvas bez
    jawnie podanego width/height przyjmuje domyslny rozmiar Tk (200x150px).
    W polaczeniu z tym, ze ten panel siedzi w lewej, WAGA=0 szufladzie
    ttk.PanedWindow (patrz main_window.py::_build_body), na niektorych
    platformach/rozdzielczosciach potrafilo to doprowadzic do tego, ze
    caly lewy panel byl liczony jako praktycznie zerowej szerokosci i w
    efekcie NIEWIDOCZNY (w tym niedostepna zakladka "Model" z przyciskiem
    wczytywania pliku STEP). Jawny `width=` ponizej usuwa ta niejednoznacznosc
    u zrodla, niezaleznie od zachowania PanedWindow/sashpos.
    """

    def __init__(self, parent, min_width: int = 440):
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, width=min_width)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._win = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._win, width=e.width))
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind(_e):
            canvas.bind_all("<MouseWheel>", _wheel)

        def _unbind(_e):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)


class SettingsPanel(ttk.Frame):
    def __init__(self, parent, project: AppProject,
                 on_change: Callable[[], None] = None,
                 on_browse_model: Callable[[str], None] = None):
        super().__init__(parent, width=470)
        self.pack_propagate(False)  # patrz ScrollableFrame - trzyma stala, widoczna szerokosc
        self.project = project
        self._on_change = on_change or (lambda: None)
        self._on_browse_model = on_browse_model or (lambda mode: None)
        self._loading = False

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self._build_tube_tab()
        self._build_tool_tab()
        self._build_job_tab()
        self._build_calib_tab()
        self._build_model_tab()
        self._build_holes_tab()

        self.refresh_from_project()

    # ------------------------------------------------------------------ #
    #  Helpers do budowy formularzy
    # ------------------------------------------------------------------ #

    def _add_tab(self, title: str) -> ttk.Frame:
        wrapper = ScrollableFrame(self.nb)
        self.nb.add(wrapper, text=title)
        return wrapper.inner

    @staticmethod
    def _section(parent, row, text) -> int:
        ttk.Label(parent, text=text, font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(12, 2))
        ttk.Separator(parent, orient="horizontal").grid(
            row=row + 1, column=0, columnspan=2, sticky="ew", padx=8)
        return row + 2

    def _entry(self, parent, row, label, var, width=12, tip: str = None) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=8, pady=3)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=1, sticky="w", padx=8, pady=3)
        if tip:
            self._tooltip(e, tip)
        return row + 1

    def _check(self, parent, row, label, var, tip: str = None) -> int:
        c = ttk.Checkbutton(parent, text=label, variable=var)
        c.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        if tip:
            self._tooltip(c, tip)
        return row + 1

    def _combo(self, parent, row, label, var, values, width=18) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=8, pady=3)
        cb = ttk.Combobox(parent, textvariable=var, state="readonly", values=values, width=width)
        cb.grid(row=row, column=1, sticky="w", padx=8, pady=3)
        return row + 1

    @staticmethod
    def _tooltip(widget, text):
        tip = {"win": None}

        def show(_e):
            if tip["win"] is not None:
                return
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 6
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")
            ttk.Label(win, text=text, background="#ffffe0", relief="solid",
                      borderwidth=1, padding=4, wraplength=280, justify="left").pack()
            tip["win"] = win

        def hide(_e):
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _bind_float(self, var: tk.DoubleVar, setter: Callable[[float], None]):
        def _cb(*_a):
            if self._loading:
                return
            try:
                setter(float(var.get()))
                self._on_change()
            except (tk.TclError, ValueError):
                pass  # uzytkownik jeszcze pisze (np. samo "-" albo puste pole)
        var.trace_add("write", _cb)

    def _bind_str(self, var: tk.StringVar, setter: Callable[[str], None]):
        def _cb(*_a):
            if self._loading:
                return
            setter(var.get())
            self._on_change()
        var.trace_add("write", _cb)

    def _bind_bool(self, var: tk.BooleanVar, setter: Callable[[bool], None]):
        def _cb(*_a):
            if self._loading:
                return
            try:
                setter(bool(var.get()))
                self._on_change()
            except tk.TclError:
                pass
        var.trace_add("write", _cb)

    def _bind_int(self, var: tk.IntVar, setter: Callable[[int], None]):
        def _cb(*_a):
            if self._loading:
                return
            try:
                setter(int(var.get()))
                self._on_change()
            except (tk.TclError, ValueError):
                pass
        var.trace_add("write", _cb)

    # ------------------------------------------------------------------ #
    #  Zakladka: Rura
    # ------------------------------------------------------------------ #

    def _build_tube_tab(self):
        p = self._add_tab("Rura")
        r = 0
        r = self._section(p, r, "Geometria rury")

        note = ("W trybie STEP/mesh (zakladka 'Model') te trzy pola sa automatycznie "
                 "nadpisywane wartosciami odczytanymi z modelu po kazdym 'Zbuduj "
                 "sciezke narzedzia' - patrz zakladka 'Kalibracja'. W trybie recznym "
                 "pozostaja Twoimi wartosciami.")
        ttk.Label(p, text=note, wraplength=300, justify="left", foreground="#555555").grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        r += 1

        self.v_od = tk.DoubleVar()
        r = self._entry(p, r, "Srednica zewnetrzna [mm]:", self.v_od,
                         tip="TubeConfig.outer_diameter_mm")
        self._bind_float(self.v_od, lambda v: setattr(self.project.tube, "outer_diameter_mm", v))

        self.v_wall = tk.DoubleVar()
        r = self._entry(p, r, "Grubosc sciany [mm]:", self.v_wall,
                         tip="TubeConfig.wall_thickness_mm - decyduje o calkowitej "
                             "glebokosci przebicia (+ naddatek z zakladki Zadanie)")
        self._bind_float(self.v_wall, lambda v: setattr(self.project.tube, "wall_thickness_mm", v))

        self.v_len = tk.DoubleVar()
        r = self._entry(p, r, "Dlugosc rury [mm]:", self.v_len,
                         tip="Uzywana jako zasieg X w trybie recznym (bez modelu 3D)")
        self._bind_float(self.v_len, lambda v: setattr(self.project.tube, "length_mm", v))

        r = self._section(p, r, "Referencja osi X maszyny")

        self.v_xzero = tk.StringVar()
        r = self._combo(p, r, "X=0 przy koncu:", self.v_xzero, ["min", "max"])
        self._bind_str(self.v_xzero, lambda v: setattr(self.project.tube, "x_zero_at", v))

        self.v_xoff = tk.DoubleVar()
        r = self._entry(p, r, "Dodatkowy offset X [mm]:", self.v_xoff,
                         tip="TubeConfig.x_offset_mm - np. rura wystaje z uchwytu")
        self._bind_float(self.v_xoff, lambda v: setattr(self.project.tube, "x_offset_mm", v))

    # ------------------------------------------------------------------ #
    #  Zakladka: Narzedzie
    # ------------------------------------------------------------------ #

    def _build_tool_tab(self):
        p = self._add_tab("Narzedzie")
        r = 0
        r = self._section(p, r, "Narzedzie")

        self.v_tool_kind = tk.StringVar()
        r = self._combo(p, r, "Rodzaj:", self.v_tool_kind, ["endmill", "drill"])
        self._bind_str(self.v_tool_kind, lambda v: setattr(self.project.tool, "kind", ToolKind(v)))

        self.v_diam = tk.DoubleVar()
        r = self._entry(p, r, "Srednica [mm]:", self.v_diam)
        self._bind_float(self.v_diam, lambda v: setattr(self.project.tool, "diameter_mm", v))

        self.v_flutes = tk.IntVar()
        r = self._entry(p, r, "Liczba ostrzy:", self.v_flutes)
        self._bind_int(self.v_flutes, lambda v: setattr(self.project.tool, "flutes", v))

        self.v_rpm = tk.DoubleVar()
        r = self._entry(p, r, "Obroty wrzeciona [RPM]:", self.v_rpm)
        self._bind_float(self.v_rpm, lambda v: setattr(self.project.tool, "spindle_rpm", v))

        self.v_cw = tk.BooleanVar()
        r = self._check(p, r, "Obrot zgodny z ruchem wskazowek (M3, w przeciwnym razie M4)",
                         self.v_cw)
        self._bind_bool(self.v_cw, lambda v: setattr(self.project.tool, "spindle_cw", v))

        r = self._section(p, r, "Posuwy [mm/min]")

        self.v_feed_cut = tk.DoubleVar()
        r = self._entry(p, r, "Konturowanie:", self.v_feed_cut)
        self._bind_float(self.v_feed_cut, lambda v: setattr(self.project.tool, "feed_cut_mm_min", v))

        self.v_feed_plunge = tk.DoubleVar()
        r = self._entry(p, r, "Naklucie promieniowe:", self.v_feed_plunge)
        self._bind_float(self.v_feed_plunge, lambda v: setattr(self.project.tool, "feed_plunge_mm_min", v))

        self.v_feed_rapid = tk.DoubleVar()
        r = self._entry(p, r, "Przejazd nieskrawajacy:", self.v_feed_rapid,
                         tip="Realizowany jako G1 (nie G0) - patrz naglowek gcode_writer.py")
        self._bind_float(self.v_feed_rapid, lambda v: setattr(self.project.tool, "feed_rapid_mm_min", v))

        self.v_feed_drill = tk.DoubleVar()
        r = self._entry(p, r, "Wiercenie (peck):", self.v_feed_drill)
        self._bind_float(self.v_feed_drill, lambda v: setattr(self.project.tool, "feed_drill_mm_min", v))

        r = self._section(p, r, "Bezpieczenstwo")
        self.v_max_rot = tk.DoubleVar()
        r = self._entry(p, r, "Max. predkosc osi A [deg/min]:", self.v_max_rot,
                         tip="gcode_writer.audit_feed() przytnie F, by tego nie przekroczyc")
        self._bind_float(self.v_max_rot, lambda v: setattr(self.project.tool, "max_rotary_speed_deg_min", v))

    # ------------------------------------------------------------------ #
    #  Zakladka: Zadanie
    # ------------------------------------------------------------------ #

    def _build_job_tab(self):
        p = self._add_tab("Zadanie")
        r = 0
        r = self._section(p, r, "Przejscia / dokladnosc")

        self.v_safe_z = tk.DoubleVar()
        r = self._entry(p, r, "Bezpieczna wysokosc Z [mm]:", self.v_safe_z)
        self._bind_float(self.v_safe_z, lambda v: setattr(self.project.job, "safe_z_mm", v))

        self.v_pass_depth = tk.DoubleVar()
        r = self._entry(p, r, "Dosuw na przejscie [mm]:", self.v_pass_depth)
        self._bind_float(self.v_pass_depth, lambda v: setattr(self.project.job, "pass_depth_mm", v))

        self.v_margin = tk.DoubleVar()
        r = self._entry(p, r, "Naddatek przebicia [mm]:", self.v_margin,
                         tip="Dodawany do grubosci sciany dla pewnego przebicia")
        self._bind_float(self.v_margin, lambda v: setattr(self.project.job, "breakthrough_margin_mm", v))

        self.v_tol = tk.DoubleVar()
        r = self._entry(p, r, "Tolerancja cieciwy [mm]:", self.v_tol,
                         tip="Uzywana przy dyskretyzacji lukow oraz jako ArcTolerance offsetu")
        self._bind_float(self.v_tol, lambda v: setattr(self.project.job, "tolerance_mm", v))

        r = self._section(p, r, "Offset / naddatek")

        self.v_offref = tk.StringVar()
        r = self._combo(p, r, "Promien odniesienia rozwiniecia:", self.v_offref,
                         ["outer", "measured"], width=14)
        self._bind_str(self.v_offref, lambda v: setattr(self.project.job, "offset_reference", v))

        self.v_leadin = tk.BooleanVar()
        r = self._check(p, r, "Najazd stycznie (lead-in)", self.v_leadin,
                         tip="Flaga JobConfig.lead_in - w obecnej wersji silnika CAM (cam_core/"
                             "toolpath.py) NIE jest jeszcze zaimplementowana geometrycznie; "
                             "zapisywana do configu na przyszlosc.")
        self._bind_bool(self.v_leadin, lambda v: setattr(self.project.job, "lead_in", v))

        r = self._section(p, r, "Wrzeciono / chlodziwo")

        self.v_warmup = tk.DoubleVar()
        r = self._entry(p, r, "Rozgrzanie wrzeciona [s]:", self.v_warmup)
        self._bind_float(self.v_warmup, lambda v: setattr(self.project.job, "spindle_warmup_s", v))

        self.v_coolant = tk.StringVar()
        r = self._combo(p, r, "Chlodziwo:", self.v_coolant, ["brak", "mist", "flood"], width=14)
        self._bind_str(self.v_coolant, self._set_coolant)

        self.v_progname = tk.StringVar()
        r = self._entry(p, r, "Nazwa programu:", self.v_progname, width=22)
        self._bind_str(self.v_progname, lambda v: setattr(self.project.job, "program_name", v))

        r = self._section(p, r, "Wiercenie (tryb DRILL)")

        self.v_peck = tk.DoubleVar()
        r = self._entry(p, r, "Dosuw peck [mm]:", self.v_peck)
        self._bind_float(self.v_peck, lambda v: setattr(self.project.job, "drill_peck_mm", v))

        self.v_drill_tol = tk.DoubleVar()
        r = self._entry(p, r, "Tolerancja srednicy [mm]:", self.v_drill_tol,
                         tip="Margines decyzji 'czy narzedzie zmiesci sie jako wiertlo'")
        self._bind_float(self.v_drill_tol, lambda v: setattr(self.project.job, "drill_diameter_tolerance_mm", v))

        self.v_full_retract = tk.BooleanVar()
        r = self._check(p, r, "Pelny odwrot po kazdym pecku", self.v_full_retract,
                         tip="Gdy wylaczone: czesciowy odwrot o wartosc ponizej")
        self._bind_bool(self.v_full_retract, self._set_full_retract)

        self.v_retract = tk.DoubleVar()
        r = self._entry(p, r, "Czesciowy odwrot [mm]:", self.v_retract)
        self._bind_float(self.v_retract, lambda v: setattr(self.project.job, "drill_retract_mm", v))

    def _set_coolant(self, v: str):
        self.project.job.coolant = None if v == "brak" else v

    def _set_full_retract(self, v: bool):
        self.project.job.drill_full_retract = v
        self.ent_retract_state()

    def ent_retract_state(self):
        pass  # miejsce na ewentualne wlacz/wylacz pola - pozostawione proste celowo

    # ------------------------------------------------------------------ #
    #  Zakladka: Kalibracja maszyny
    # ------------------------------------------------------------------ #

    def _build_calib_tab(self):
        p = self._add_tab("Kalibracja")
        r = 0
        r = self._section(p, r, "Os A - kierunek i zero")

        self.v_asign = tk.StringVar()
        r = self._combo(p, r, "Kierunek obrotu (a_sign):", self.v_asign, ["+1", "-1"], width=8)
        self._bind_str(self.v_asign, lambda v: setattr(self.project.machine, "a_sign", int(v)))

        self.v_azero = tk.DoubleVar()
        r = self._entry(p, r, "Offset zera A [deg]:", self.v_azero,
                         tip="Kalibrowane 'na sucho' na rzeczywistej maszynie - patrz "
                             "README projektu, sekcja Kalibracja")
        self._bind_float(self.v_azero, lambda v: setattr(self.project.machine, "a_zero_offset_deg", v))

        r = self._section(p, r, "Dopasowanie modelu STEP")

        info_txt = ("Domyslnie (checkbox zaznaczony) srednica, dlugosc, a jesli model ma "
                     "osobna sciane wewnetrzna - takze grubosc sciany, sa ODCZYTYWANE "
                     "AUTOMATYCZNIE z pliku STEP/mesh po kazdym 'Zbuduj sciezke narzedzia' "
                     "i nadpisuja pola w zakladce 'Rura'. Recznie podany promien ponizej "
                     "przydaje sie tylko, gdy model ma inna, wieksza powierzchnie walcowa "
                     "(np. mocowanie), ktora bez ograniczenia zostalaby blednie wzieta za rure.")
        ttk.Label(p, text=info_txt, wraplength=300, justify="left",
                  foreground="#555555").grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        r += 1

        self.v_use_auto_radius = tk.BooleanVar()
        r = self._check(p, r, "Automatyczne wykrycie (najwiekszy walec w modelu)", self.v_use_auto_radius)
        self._bind_bool(self.v_use_auto_radius, self._set_use_auto_radius)

        self.v_radius_hint = tk.DoubleVar()
        r = self._entry(p, r, "Reczna podpowiedz promienia [mm]:", self.v_radius_hint,
                         tip="model_io.load_step_holes(outer_radius_hint_mm=...) - "
                             "uzywane tylko gdy powyzszy checkbox jest odznaczony")
        self._bind_float(self.v_radius_hint, self._set_radius_hint)

        self.v_radius_tol = tk.DoubleVar()
        r = self._entry(p, r, "Tolerancja dopasowania promienia [mm]:", self.v_radius_tol,
                         tip="Uzywana tylko przy recznej podpowiedzi promienia powyzej")
        self._bind_float(self.v_radius_tol, lambda v: setattr(self.project.machine, "radius_match_tol_mm", v))

        r = self._section(p, r, "Zakres osi X (gdzie wypada X=0 maszyny)")

        extent_info = ("Domyslnie X=0 wyznaczany jest z geometrii calej rury w pliku "
                        "(nie z otworow) - patrz README. Jesli dla konkretnego pliku "
                        "wyjdzie to blednie (np. X=0 wychodzi na otworze zamiast na "
                        "koncu rury), przelacz na opcje reczna ponizej: wtedy X=0 "
                        "zawsze wypada dokladnie na poczatku rury, a jej dlugosc to "
                        "wprost 'Dlugosc rury' z zakladki 'Rura'.")
        ttk.Label(p, text=extent_info, wraplength=300, justify="left",
                  foreground="#555555").grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        r += 1

        self.v_extent_source = tk.StringVar()
        for val, label in [("auto", "Automatyczny (z geometrii pliku)"),
                            ("manual_length", "Reczny (0 = poczatek rury, dlugosc z zakladki 'Rura')")]:
            ttk.Radiobutton(p, text=label, value=val, variable=self.v_extent_source,
                             command=self._on_extent_source_change).grid(
                row=r, column=0, columnspan=2, sticky="w", padx=8, pady=2)
            r += 1

    def _on_extent_source_change(self):
        if self._loading:
            return
        self.project.machine.x_extent_source = self.v_extent_source.get()
        self._on_change()

    def _set_use_auto_radius(self, v: bool):
        if v:
            self.project.machine.outer_radius_hint_mm = None
        else:
            self.project.machine.outer_radius_hint_mm = float(self.v_radius_hint.get() or 0.0)

    def _set_radius_hint(self, v: float):
        if not self.v_use_auto_radius.get():
            self.project.machine.outer_radius_hint_mm = v

    # ------------------------------------------------------------------ #
    #  Zakladka: Model
    # ------------------------------------------------------------------ #

    def _build_model_tab(self):
        p = self._add_tab("Model")
        r = 0
        r = self._section(p, r, "Zrodlo otworow")

        self.v_source = tk.StringVar()
        modes = [("manual", "Tryb reczny - lista otworow parametrycznych"),
                 ("step", "Plik STEP (.step / .stp) - precyzyjny, wymaga cadquery"),
                 ("mesh", "Plik mesh (.3mf / .stl / .obj) - przyblizony, wymaga trimesh")]
        for val, label in modes:
            rb = ttk.Radiobutton(p, text=label, value=val, variable=self.v_source,
                                  command=self._on_source_change)
            rb.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=3)
            r += 1

        r += 1
        self.btn_browse = ttk.Button(p, text="Wczytaj plik modelu...", command=self._browse_model)
        self.btn_browse.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 3))
        r += 1

        self.lbl_model_path = ttk.Label(p, text="(brak wczytanego pliku)", foreground="#666666",
                                         wraplength=280, justify="left")
        self.lbl_model_path.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        r += 1

        r = self._section(p, r, "Informacja")
        info = ("W trybie recznym otwory definiuje sie parametrycznie w zakladce "
                "'Otwory' (bez pliku 3D). W trybach STEP/mesh otwory sa wykrywane "
                "automatycznie z geometrii pliku - patrz naglowki cam_core/model_io.py.")
        ttk.Label(p, text=info, wraplength=300, justify="left", foreground="#555555").grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8, pady=4)

    def _on_source_change(self):
        if self._loading:
            return
        self.project.source_mode = self.v_source.get()
        self.btn_browse.config(state=("disabled" if self.v_source.get() == "manual" else "normal"))
        self._on_change()

    def _browse_model(self):
        mode = self.v_source.get()
        if mode == "step":
            path = filedialog.askopenfilename(
                title="Wybierz plik STEP", filetypes=[("STEP", "*.step *.stp"), ("Wszystkie", "*.*")])
        elif mode == "mesh":
            path = filedialog.askopenfilename(
                title="Wybierz plik mesh", filetypes=[("Mesh", "*.3mf *.stl *.obj"), ("Wszystkie", "*.*")])
        else:
            return
        if not path:
            return
        self.project.source_model_path = path
        self.lbl_model_path.config(text=path)
        self._on_browse_model(path)

    def set_model_status(self, text: str):
        self.lbl_model_path.config(text=text)

    # ------------------------------------------------------------------ #
    #  Zakladka: Otwory (tryb reczny)
    # ------------------------------------------------------------------ #

    def _build_holes_tab(self):
        p = self._add_tab("Otwory")
        p.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(p)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        ttk.Button(toolbar, text="Dodaj", command=self._hole_add).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Edytuj", command=self._hole_edit).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Duplikuj", command=self._hole_duplicate).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Usun", command=self._hole_remove).pack(side="left", padx=2)

        hint = ttk.Label(p, text="Dwuklik na kolumnie 'Akt.' wlacza/wylacza otwor.",
                          foreground="#777777", font=("", 8))
        hint.grid(row=1, column=0, sticky="w", padx=8)

        cols = ("enabled", "name", "shape", "x", "angle", "dims")
        self.tree_holes = ttk.Treeview(p, columns=cols, show="headings", height=14, selectmode="browse")
        headers = {"enabled": "Akt.", "name": "Nazwa", "shape": "Ksztalt", "x": "X [mm]",
                   "angle": "Kat [deg]", "dims": "Wymiary"}
        widths = {"enabled": 36, "name": 85, "shape": 85, "x": 62, "angle": 62, "dims": 145}
        for c in cols:
            self.tree_holes.heading(c, text=headers[c])
            self.tree_holes.column(c, width=widths[c], anchor="center" if c != "dims" else "w")
        self.tree_holes.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        p.rowconfigure(2, weight=1)

        vsb = ttk.Scrollbar(p, orient="vertical", command=self.tree_holes.yview)
        self.tree_holes.configure(yscrollcommand=vsb.set)
        vsb.grid(row=2, column=1, sticky="ns")

        self.tree_holes.bind("<Double-1>", self._on_holes_tree_double_click)
        self.tree_holes.bind("<Delete>", lambda e: self._hole_remove())

    def _dims_text(self, h) -> str:
        if h.shape == HoleShape.CIRCLE:
            return f"fi{h.diameter_mm:g}"
        if h.shape == HoleShape.RECTANGLE:
            return f"{h.width_mm:g} x {h.height_mm:g}"
        if h.shape == HoleShape.ROUNDED_RECTANGLE:
            return f"{h.width_mm:g} x {h.height_mm:g}, r{h.corner_radius_mm:g}"
        if h.shape == HoleShape.POLYGON:
            return f"{len(h.polygon_points_mm)} pkt"
        return ""

    def refresh_holes_tree(self):
        self.tree_holes.delete(*self.tree_holes.get_children())
        for i, h in enumerate(self.project.holes):
            self.tree_holes.insert("", "end", iid=str(i), values=(
                "\u2713" if h.enabled else "\u2014",
                h.name, SHAPE_LABELS[h.shape], f"{h.center_x_mm:g}",
                f"{h.center_angle_deg:g}", self._dims_text(h),
            ))

    def _selected_hole_index(self) -> Optional[int]:
        sel = self.tree_holes.selection()
        if not sel:
            return None
        return int(sel[0])

    def _on_holes_tree_double_click(self, event):
        region = self.tree_holes.identify_region(event.x, event.y)
        if region != "cell":
            return
        col = self.tree_holes.identify_column(event.x)
        row = self.tree_holes.identify_row(event.y)
        if not row:
            return
        idx = int(row)
        if col == "#1":  # kolumna "Akt."
            self.project.holes[idx].enabled = not self.project.holes[idx].enabled
            self.refresh_holes_tree()
            self._on_change()
        else:
            self._hole_edit()

    def _hole_add(self):
        existing = [h.name for h in self.project.holes]
        default_hole = new_hole_default(existing)

        def _save(new_hole):
            self.project.holes.append(new_hole)
            self.refresh_holes_tree()
            self._on_change()

        dlg = HoleDialog(self, None, existing, _save)
        dlg.var_name.set(default_hole.name)
        dlg.var_center_x.set(default_hole.center_x_mm)
        dlg.var_diameter.set(default_hole.diameter_mm)

    def _hole_edit(self):
        idx = self._selected_hole_index()
        if idx is None:
            messagebox.showinfo("Edycja otworu", "Zaznacz najpierw otwor na liscie.", parent=self)
            return
        hole = self.project.holes[idx]
        existing = [h.name for j, h in enumerate(self.project.holes) if j != idx]

        def _save(updated):
            self.project.holes[idx] = updated
            self.refresh_holes_tree()
            self._on_change()

        HoleDialog(self, hole, existing, _save)

    def _hole_duplicate(self):
        idx = self._selected_hole_index()
        if idx is None:
            return
        import copy
        src = self.project.holes[idx]
        clone = copy.deepcopy(src)
        existing = {h.name for h in self.project.holes}
        n = 2
        base = src.name
        while f"{base}_kopia{n if n > 2 else ''}" in existing:
            n += 1
        clone.name = f"{base}_kopia" if f"{base}_kopia" not in existing else f"{base}_kopia{n}"
        self.project.holes.insert(idx + 1, clone)
        self.refresh_holes_tree()
        self._on_change()

    def _hole_remove(self):
        idx = self._selected_hole_index()
        if idx is None:
            return
        name = self.project.holes[idx].name
        if messagebox.askyesno("Usun otwor", f"Usunac otwor '{name}'?", parent=self):
            del self.project.holes[idx]
            self.refresh_holes_tree()
            self._on_change()

    # ------------------------------------------------------------------ #
    #  Zaladowanie danych z (nowego) projektu do wszystkich pol
    # ------------------------------------------------------------------ #

    def refresh_from_project(self):
        self._loading = True
        try:
            t = self.project.tube
            self.v_od.set(t.outer_diameter_mm)
            self.v_wall.set(t.wall_thickness_mm)
            self.v_len.set(t.length_mm)
            self.v_xzero.set(t.x_zero_at)
            self.v_xoff.set(t.x_offset_mm)

            tool = self.project.tool
            self.v_tool_kind.set(tool.kind.value)
            self.v_diam.set(tool.diameter_mm)
            self.v_flutes.set(tool.flutes)
            self.v_rpm.set(tool.spindle_rpm)
            self.v_cw.set(tool.spindle_cw)
            self.v_feed_cut.set(tool.feed_cut_mm_min)
            self.v_feed_plunge.set(tool.feed_plunge_mm_min)
            self.v_feed_rapid.set(tool.feed_rapid_mm_min)
            self.v_feed_drill.set(tool.feed_drill_mm_min)
            self.v_max_rot.set(tool.max_rotary_speed_deg_min)

            job = self.project.job
            self.v_safe_z.set(job.safe_z_mm)
            self.v_pass_depth.set(job.pass_depth_mm)
            self.v_margin.set(job.breakthrough_margin_mm)
            self.v_tol.set(job.tolerance_mm)
            self.v_offref.set(job.offset_reference)
            self.v_leadin.set(job.lead_in)
            self.v_warmup.set(job.spindle_warmup_s)
            self.v_coolant.set("brak" if job.coolant is None else job.coolant)
            self.v_progname.set(job.program_name)
            self.v_peck.set(job.drill_peck_mm)
            self.v_drill_tol.set(job.drill_diameter_tolerance_mm)
            self.v_full_retract.set(job.drill_full_retract)
            self.v_retract.set(job.drill_retract_mm)

            m = self.project.machine
            self.v_asign.set("+1" if m.a_sign >= 0 else "-1")
            self.v_azero.set(m.a_zero_offset_deg)
            self.v_use_auto_radius.set(m.outer_radius_hint_mm is None)
            self.v_radius_hint.set(m.outer_radius_hint_mm if m.outer_radius_hint_mm is not None
                                    else t.outer_radius_mm)
            self.v_radius_tol.set(m.radius_match_tol_mm)
            self.v_extent_source.set(m.x_extent_source)

            self.v_source.set(self.project.source_mode)
            self.btn_browse.config(state=("disabled" if self.project.source_mode == "manual" else "normal"))
            self.lbl_model_path.config(
                text=self.project.source_model_path or "(brak wczytanego pliku)")

            self.refresh_holes_tree()
        finally:
            self._loading = False

    def set_project(self, project: AppProject):
        self.project = project
        self.refresh_from_project()
