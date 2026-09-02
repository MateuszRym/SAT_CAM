"""
main_window.py
---------------
Glowne okno aplikacji. Spina ze soba:
  - SettingsPanel (lewa strona) - wszystkie parametry z cam_core.config,
  - ToolpathViewer (prawa gora) - wizualizacja 2D/3D przygotowanej sciezki,
  - Log + podglad G-code (prawy dol),
  - Pasek narzedzi / menu z akcjami: projekt (nowy/otworz/zapisz), model
    (wczytaj STEP/mesh), sciezka narzedzia (zbuduj), G-code (generuj/zapisz).

Dlugotrwale operacje (parsowanie STEP/mesh, budowa sciezki dla wielu
otworow) ida przez task_runner.run_async, zeby UI sie nie zawieszalo.
"""

from __future__ import annotations

import datetime
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from .app_state import AppProject
from . import engine_bridge as eb
from .engine_bridge import EngineError
from .settings_panel import SettingsPanel
from .viewer import ToolpathViewer
from .task_runner import run_async


APP_TITLE = "Silesian Aerospace Technologies CAM"
APP_SHORT = "SAT CAM"

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _asset(name: str) -> str:
    return os.path.join(ASSETS_DIR, name)


class CamApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1500x950")
        self.minsize(1100, 700)
        self._load_icons()

        self.project = AppProject()
        self.loaded_holes: eb.LoadedHoles | None = None
        self.previews: list = []
        self.gcode_text: str = ""
        self._busy = False

        self._build_menu()
        self._build_brand_bar()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()

        self._log("Aplikacja gotowa. Ustaw parametry po lewej, wybierz zrodlo otworow "
                   "(zakladka 'Model'), a nastepnie kliknij 'Zbuduj sciezke narzedzia'.")
        self._update_action_states()

    def _load_icons(self):
        """Ikona okna/paska zadan - kilka rozdzielczosci na raz (iconphoto sam
        wybierze najlepsza dla danego kontekstu OS). Awaria (np. brak plikow
        assets/ w niestandardowej instalacji) nie powinna wywalac calej
        aplikacji - tylko brak ikony, dlatego owinięte w try/except."""
        self._icon_images = []  # referencje trzymane na zywo (Tk PhotoImage bez tego bywa zbierany przez GC)
        try:
            imgs = [tk.PhotoImage(file=_asset(f"icon_{s}.png")) for s in (16, 24, 32, 48, 64)]
            self.iconphoto(True, *imgs)
            self._icon_images = imgs
        except (tk.TclError, FileNotFoundError):
            pass

    # ------------------------------------------------------------------ #
    #  Pasek brandingowy (logo + nazwa)
    # ------------------------------------------------------------------ #

    def _build_brand_bar(self):
        bar = tk.Frame(self, bg="#1b2733")
        bar.pack(side="top", fill="x")

        inner = tk.Frame(bar, bg="#1b2733")
        inner.pack(side="left", padx=14, pady=8)

        try:
            logo_img = tk.PhotoImage(file=_asset("brand_bar_logo.png"))
            self._brand_logo_img = logo_img  # referencja na zywo
            tk.Label(inner, image=logo_img, bg="#1b2733").pack(side="left", padx=(0, 10))
        except (tk.TclError, FileNotFoundError):
            pass

        tk.Label(inner, text=APP_TITLE, bg="#1b2733", fg="#ffffff",
                 font=("", 13, "bold")).pack(side="left")

    # ------------------------------------------------------------------ #
    #  Menu / toolbar
    # ------------------------------------------------------------------ #

    def _build_menu(self):
        m = tk.Menu(self)
        self.config(menu=m)

        mfile = tk.Menu(m, tearoff=False)
        mfile.add_command(label="Nowy projekt", command=self.action_new_project)
        mfile.add_command(label="Otworz projekt...", command=self.action_open_project)
        mfile.add_command(label="Zapisz projekt", command=self.action_save_project)
        mfile.add_command(label="Zapisz projekt jako...", command=self.action_save_project_as)
        mfile.add_separator()
        mfile.add_command(label="Zapisz G-code...", command=self.action_save_gcode)
        mfile.add_separator()
        mfile.add_command(label="Zakoncz", command=self.destroy)
        m.add_cascade(label="Plik", menu=mfile)

        mrun = tk.Menu(m, tearoff=False)
        mrun.add_command(label="Zbuduj sciezke narzedzia", command=self.action_build_toolpath)
        mrun.add_command(label="Generuj G-code", command=self.action_generate_gcode)
        m.add_cascade(label="Uruchom", menu=mrun)

        mhelp = tk.Menu(m, tearoff=False)
        mhelp.add_command(label="O programie", command=self.action_about)
        m.add_cascade(label="Pomoc", menu=mhelp)

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side="top", fill="x")

        ttk.Button(bar, text="Nowy", command=self.action_new_project).pack(side="left", padx=2)
        ttk.Button(bar, text="Otworz...", command=self.action_open_project).pack(side="left", padx=2)
        ttk.Button(bar, text="Zapisz", command=self.action_save_project).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.btn_build = ttk.Button(bar, text="\u25b6 Zbuduj sciezke narzedzia",
                                     command=self.action_build_toolpath)
        self.btn_build.pack(side="left", padx=2)

        self.btn_gcode = ttk.Button(bar, text="\u25b6 Generuj G-code",
                                     command=self.action_generate_gcode)
        self.btn_gcode.pack(side="left", padx=2)

        self.btn_save_gcode = ttk.Button(bar, text="Zapisz G-code...",
                                          command=self.action_save_gcode)
        self.btn_save_gcode.pack(side="left", padx=2)

    def _build_statusbar(self):
        bar = ttk.Frame(self, relief="sunken")
        bar.pack(side="bottom", fill="x")
        self.lbl_statusbar = ttk.Label(bar, text="Gotowe.", anchor="w", padding=(6, 2))
        self.lbl_statusbar.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=8, pady=2)

    # ------------------------------------------------------------------ #
    #  Layout glowny
    # ------------------------------------------------------------------ #

    def _build_body(self):
        # UWAGA: outer split lewy panel/reszta okna celowo uzywa KLASYCZNEGO
        # tk.PanedWindow (nie ttk.PanedWindow) - ttk.PanedWindow NIE ma opcji
        # 'minsize' (tylko 'weight'), wiec nie moze dac twardej gwarancji, ze
        # lewy panel nigdy nie skurczy sie do 0px. Klasyczny tk.PanedWindow ma
        # prawdziwe 'minsize' na poziomie API, niezalezne od platformy/timingu
        # - to wlasciwa naprawa bledu "znikajacego" panelu ustawien (patrz
        # README.md, sekcja "Historia zmian").
        outer = tk.PanedWindow(self, orient="horizontal", sashwidth=6, sashrelief="flat",
                                bg="#d9d9d9", opaqueresize=True)
        outer.pack(side="top", fill="both", expand=True)

        left_wrap = ttk.Frame(outer, width=470)
        left_wrap.pack_propagate(False)  # patrz settings_panel.ScrollableFrame - naprawa
                                          # bledu "znikajacego" lewego panelu
        self.settings = SettingsPanel(
            left_wrap, self.project,
            on_change=self._on_project_changed,
            on_browse_model=self._on_model_file_chosen,
        )
        self.settings.pack(fill="both", expand=True)
        outer.add(left_wrap, minsize=360, width=470, stretch="never")

        right = ttk.PanedWindow(outer, orient="vertical")
        outer.add(right, minsize=500, stretch="always")

        viewer_wrap = ttk.Frame(right)
        self.viewer = ToolpathViewer(viewer_wrap)
        self.viewer.pack(fill="both", expand=True)
        right.add(viewer_wrap, weight=3)

        bottom = ttk.Notebook(right)
        right.add(bottom, weight=1)

        log_frame = ttk.Frame(bottom)
        bottom.add(log_frame, text="Log")
        self.txt_log = scrolledtext.ScrolledText(log_frame, height=10, state="disabled",
                                                   wrap="word", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True)
        self.txt_log.tag_config("info", foreground="#222222")
        self.txt_log.tag_config("warn", foreground="#a15c00")
        self.txt_log.tag_config("error", foreground="#b00020")
        self.txt_log.tag_config("ok", foreground="#1a7a2e")

        gcode_frame = ttk.Frame(bottom)
        bottom.add(gcode_frame, text="Podglad G-code")
        self.txt_gcode = scrolledtext.ScrolledText(gcode_frame, height=10, state="disabled",
                                                     wrap="none", font=("Consolas", 9))
        self.txt_gcode.pack(fill="both", expand=True)

        # UWAGA: w odroznieniu od ttk.PanedWindow, klasyczny tk.PanedWindow
        # honoruje 'width'/'minsize' podane wprost w add() (patrz wyzej) jako
        # RZECZYWISTY, natychmiastowy rozmiar poczatkowy panelu - nie trzeba
        # tu juz osobnego, opoznionego wywolania ustawiajacego pozycje sash.

    # ------------------------------------------------------------------ #
    #  Log
    # ------------------------------------------------------------------ #

    def _log(self, msg: str, level: str = "info"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"[{ts}] {msg}\n", level)
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _set_status(self, text: str):
        self.lbl_statusbar.config(text=text)

    # ------------------------------------------------------------------ #
    #  Stan zajetosci (blokada przyciskow podczas zadan w tle)
    # ------------------------------------------------------------------ #

    def _set_busy(self, busy: bool, status_text: str = ""):
        self._busy = busy
        if busy:
            self.progress.start(12)
            if status_text:
                self._set_status(status_text)
        else:
            self.progress.stop()
            self._set_status("Gotowe.")
        self._update_action_states()

    def _update_action_states(self):
        state = "disabled" if self._busy else "normal"
        self.btn_build.config(state=state)
        gcode_state = "disabled" if (self._busy or not self.previews) else "normal"
        self.btn_gcode.config(state=gcode_state)
        save_state = "disabled" if (self._busy or not self.gcode_text) else "normal"
        self.btn_save_gcode.config(state=save_state)

    # ------------------------------------------------------------------ #
    #  Callbacki z SettingsPanel
    # ------------------------------------------------------------------ #

    def _on_project_changed(self):
        # zmiana parametrow uniewaznia poprzednio zbudowana sciezke/gcode
        # (zeby nie pokazywac/zapisywac nieaktualnych wynikow)
        if self.previews:
            self.previews = []
            self.gcode_text = ""
            self.viewer.clear()
            self._clear_gcode_preview()
            self._update_action_states()

    def _on_model_file_chosen(self, path: str):
        self._log(f"Wybrano plik modelu: {path}")
        self._log("Kliknij 'Zbuduj sciezke narzedzia', aby go wczytac i policzyc sciezke.")

    # ------------------------------------------------------------------ #
    #  Akcje: projekt
    # ------------------------------------------------------------------ #

    def action_new_project(self):
        if not messagebox.askyesno("Nowy projekt", "Utworzyc nowy, pusty projekt? "
                                    "Niezapisane zmiany zostana utracone."):
            return
        self.project = AppProject()
        self.loaded_holes = None
        self.previews = []
        self.gcode_text = ""
        self.settings.set_project(self.project)
        self.viewer.clear()
        self._clear_gcode_preview()
        self._update_action_states()
        self._log("Utworzono nowy projekt.")

    def action_open_project(self):
        path = filedialog.askopenfilename(title="Otworz projekt", filetypes=[("Projekt CAM (JSON)", "*.json")])
        if not path:
            return
        try:
            self.project = AppProject.load(path)
        except Exception as e:
            messagebox.showerror("Blad wczytywania", f"Nie udalo sie wczytac projektu:\n{e}")
            self._log(f"Blad wczytywania projektu: {e}", "error")
            return
        self.loaded_holes = None
        self.previews = []
        self.gcode_text = ""
        self.settings.set_project(self.project)
        self.viewer.clear()
        self._clear_gcode_preview()
        self._update_action_states()
        self._log(f"Wczytano projekt: {path}", "ok")

    def action_save_project(self):
        if self.project.project_path:
            self._do_save_project(self.project.project_path)
        else:
            self.action_save_project_as()

    def action_save_project_as(self):
        path = filedialog.asksaveasfilename(title="Zapisz projekt jako", defaultextension=".json",
                                             filetypes=[("Projekt CAM (JSON)", "*.json")])
        if not path:
            return
        self._do_save_project(path)

    def _do_save_project(self, path: str):
        try:
            self.project.save(path)
        except Exception as e:
            messagebox.showerror("Blad zapisu", f"Nie udalo sie zapisac projektu:\n{e}")
            self._log(f"Blad zapisu projektu: {e}", "error")
            return
        self._log(f"Zapisano projekt: {path}", "ok")

    # ------------------------------------------------------------------ #
    #  Akcje: budowa sciezki narzedzia
    # ------------------------------------------------------------------ #

    def action_build_toolpath(self):
        if self._busy:
            return
        mode = self.project.source_mode
        path = self.project.source_model_path
        if mode != "manual" and not path:
            messagebox.showwarning("Brak pliku modelu",
                                    "W zakladce 'Model' wybierz plik STEP/mesh do wczytania.")
            return

        def work():
            loaded = eb.load_by_mode(self.project, path)
            # KRYTYCZNA KOLEJNOSC: geometria rury z modelu musi zostac zastosowana
            # PRZED zbudowaniem sciezki - patrz docstring apply_detected_tube_geometry.
            # To CZYSTA operacja na danych (bez Tk), wiec bezpieczna w watku w tle;
            # samo odswiezenie widgetow nastapi pozniej w on_done (glowny watek).
            geometry_log = eb.apply_detected_tube_geometry(self.project, loaded.tube_geometry)
            previews, warns = eb.build_toolpath(loaded, self.project)
            return loaded, previews, warns, geometry_log

        def on_done(result):
            loaded, previews, warns, geometry_log = result
            self.loaded_holes = loaded
            for line in geometry_log:
                self._log(line, "info")
            if geometry_log:
                self.settings.refresh_from_project()  # odswiez zakladke 'Rura' w UI (glowny watek)
            self.previews = previews
            self.gcode_text = ""
            self._clear_gcode_preview()
            for line in loaded.info_lines:
                self._log(line, "info")
            self.viewer.set_data(previews, self.project.tube, self.project.tool)
            if warns:
                for w in warns:
                    self._log(w, "warn")
                self._log(f"Zbudowano sciezke narzedzia z {len(warns)} ostrzezeniem(ami).", "warn")
            else:
                self._log("Zbudowano sciezke narzedzia bez ostrzezen.", "ok")
            self._set_busy(False)

        def on_error(msg, tb):
            self._set_busy(False)
            self._log(f"Blad budowy sciezki narzedzia: {msg}", "error")
            messagebox.showerror("Blad", msg)

        run_async(self, work, on_done, on_error,
                  on_start=lambda: self._set_busy(True, "Wczytywanie modelu i budowa sciezki..."))

    # ------------------------------------------------------------------ #
    #  Akcje: G-code
    # ------------------------------------------------------------------ #

    def action_generate_gcode(self):
        if not self.previews:
            messagebox.showinfo("Generuj G-code", "Najpierw zbuduj sciezke narzedzia.")
            return
        try:
            text, warns = eb.generate_gcode(self.previews, self.project)
        except EngineError as e:
            messagebox.showerror("Blad", str(e))
            self._log(f"Blad generowania G-code: {e}", "error")
            return
        self.gcode_text = text
        self._show_gcode_preview(text)
        for w in warns:
            self._log(f"[gcode] {w}", "warn")
        self._log(f"Wygenerowano G-code ({len(text.splitlines())} linii).", "ok")
        self._update_action_states()

    def action_save_gcode(self):
        if not self.gcode_text:
            messagebox.showinfo("Zapisz G-code", "Najpierw wygeneruj G-code.")
            return
        default_name = (self.project.job.program_name or "program") + ".nc"
        path = filedialog.asksaveasfilename(title="Zapisz G-code", defaultextension=".nc",
                                             initialfile=default_name,
                                             filetypes=[("G-code", "*.nc *.gcode *.tap"), ("Wszystkie", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.gcode_text)
        except Exception as e:
            messagebox.showerror("Blad zapisu", f"Nie udalo sie zapisac G-code:\n{e}")
            self._log(f"Blad zapisu G-code: {e}", "error")
            return
        self._log(f"Zapisano G-code: {path}", "ok")

    def _show_gcode_preview(self, text: str):
        self.txt_gcode.configure(state="normal")
        self.txt_gcode.delete("1.0", "end")
        self.txt_gcode.insert("1.0", text)
        self.txt_gcode.configure(state="disabled")

    def _clear_gcode_preview(self):
        self.txt_gcode.configure(state="normal")
        self.txt_gcode.delete("1.0", "end")
        self.txt_gcode.configure(state="disabled")

    # ------------------------------------------------------------------ #
    #  Pomoc
    # ------------------------------------------------------------------ #

    def action_about(self):
        win = tk.Toplevel(self)
        win.title(f"O programie \u2014 {APP_SHORT}")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = ttk.Frame(win, padding=20)
        pad.pack(fill="both", expand=True)

        header = ttk.Frame(pad)
        header.pack(fill="x")
        try:
            logo_img = tk.PhotoImage(file=_asset("about_logo.png"))
            win._logo_img = logo_img  # referencja na zywo
            ttk.Label(header, image=logo_img).pack(side="left", padx=(0, 16))
        except (tk.TclError, FileNotFoundError):
            pass
        title_box = ttk.Frame(header)
        title_box.pack(side="left", fill="both", expand=True)
        ttk.Label(title_box, text=APP_TITLE, font=("", 14, "bold")).pack(anchor="w")
        ttk.Label(title_box, text="Obrobka otworow w rurze \u2014 maszyna 3-osiowa X / Z / A",
                  foreground="#555555").pack(anchor="w", pady=(2, 0))

        ttk.Separator(pad, orient="horizontal").pack(fill="x", pady=14)

        body = (
            "Frontend UI dla silnika CAM (cam_core/: geometry_core, config, "
            "hole_shapes, model_io, toolpath, gcode_writer).\n\n"
            "Metoda: kazdy punkt konturu otworu jest laczony z osia obrotu rury, "
            "co daje kat osi A; kontur jest 'rozwijany' na plaszczyzne, offsetowany "
            "o promien narzedzia, i 'zwijany' z powrotem przed zapisem do G-code.\n\n"
            "UWAGA: kopia cam_core/ dolaczona do tej aplikacji zawiera 2 drobne "
            "poprawki zgodnosci wzgledem oryginalnych plikow (patrz PATCH w "
            "cam_core/geometry_core.py i cam_core/config.py oraz README.md)."
        )
        ttk.Label(pad, text=body, wraplength=440, justify="left").pack(anchor="w")

        ttk.Button(pad, text="Zamknij", command=win.destroy).pack(anchor="e", pady=(16, 0))

        win.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - win.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x,0)}+{max(y,0)}")
