"""
viewer.py
---------
Panel wizualizacji przygotowanej sciezki narzedzia. Dwa widoki (zakladki):

  1) "Rozwiniecie (X / A)" -- dokladny, 2D widok w ukladzie w ktorym
     faktycznie liczony jest offset narzedzia (patrz geometry_core.py):
     X [mm] wzdluz osi rury (pozioma), A [deg] kat obrotu (pionowa).
     To jest widok "co maszyna naprawde zrobi" -- wlasciwy do sprawdzania
     czy narzedzie zmiesci sie w naroznikach, czy offset jest poprawny itd.

  2) "Podglad 3D" -- poglądowy widok rury jako walca z konturami otworow
     naniesionymi na powierzchnie zewnetrzna (SCHEMATYCZNIE -- promieniowa
     glebokosc naciecia NIE jest tu pokazywana w skali, bo w porownaniu do
     srednicy rury byłaby wizualnie niezauwazalna; to podglad polozenia/
     ksztaltu, nie narzedzie pomiarowe).

WAZNA WLASCIWOSC ALGORYTMU (patrz toolpath.py:_build_contour_operation):
kontur po offsetcie jest liczony RAZ i uzywany na wszystkich glebokosciach
Z -- rozne ContourPass dla tego samego otworu maja WIZUALNIE IDENTYCZNA
sciezke X/A, rozni je tylko Z. Dlatego w obu widokach rysujemy tylko
przejscia z NAJGLEBSZEGO Z (finalny ksztalt), a liczbe przejsc pokazujemy
tekstowo -- rysowanie wszystkich nakladalyby sie idealnie na sobie.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - rejestruje projekcje '3d'
import matplotlib.cm as cm

from cam_core.config import TubeConfig, ToolConfig
from . import engine_bridge as eb
from .engine_bridge import HolePreview

ALL_HOLES = "(wszystkie otwory)"


class ToolpathViewer(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self._previews: List[HolePreview] = []
        self._tube: Optional[TubeConfig] = None
        self._tool: Optional[ToolConfig] = None

        self._build_controls()
        self._build_notebook()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #

    def _build_controls(self):
        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x", padx=4, pady=(4, 0))

        ttk.Label(bar, text="Otwor:").pack(side="left")
        self.var_hole = tk.StringVar(value=ALL_HOLES)
        self.cmb_hole = ttk.Combobox(bar, textvariable=self.var_hole, state="readonly", width=24)
        self.cmb_hole.pack(side="left", padx=(4, 12))
        self.cmb_hole.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        self.var_show_ref = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Kontur referencyjny (przed offsetem)",
                         variable=self.var_show_ref, command=self.redraw).pack(side="left", padx=6)

        self.var_show_toolpath = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Sciezka narzedzia (po offsecie)",
                         variable=self.var_show_toolpath, command=self.redraw).pack(side="left", padx=6)

        self.var_show_warnings = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Ostrzezenia (ciasne naroza)",
                         variable=self.var_show_warnings, command=self.redraw).pack(side="left", padx=6)

        self.lbl_status = ttk.Label(self, text="Brak danych - kliknij 'Zbuduj sciezke narzedzia'.",
                                     foreground="#666666")
        self.lbl_status.pack(side="top", fill="x", padx=8, pady=(2, 0))

    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="both", expand=True)

        tab2d = ttk.Frame(self.nb)
        tab3d = ttk.Frame(self.nb)
        self.nb.add(tab2d, text="Rozwiniecie (X / A)")
        self.nb.add(tab3d, text="Podglad 3D")
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.redraw())

        self.fig2d = Figure(figsize=(6, 4.5), dpi=100)
        self.ax2d = self.fig2d.add_subplot(111)
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=tab2d)
        self.canvas2d.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas2d, tab2d).pack(side="bottom", fill="x")

        self.fig3d = Figure(figsize=(6, 4.5), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection="3d")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=tab3d)
        self.canvas3d.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas3d, tab3d).pack(side="bottom", fill="x")

    # ------------------------------------------------------------------ #
    #  Dane
    # ------------------------------------------------------------------ #

    def set_data(self, previews: List[HolePreview], tube: TubeConfig, tool: ToolConfig):
        self._previews = previews
        self._tube = tube
        self._tool = tool

        names = [ALL_HOLES] + [p.name for p in previews]
        self.cmb_hole["values"] = names
        if self.var_hole.get() not in names:
            self.var_hole.set(ALL_HOLES)

        n_contour = sum(1 for p in previews if p.operation.mode == "contour")
        n_drill = sum(1 for p in previews if p.operation.mode == "drill")
        n_skip = sum(1 for p in previews if p.operation.mode == "skipped")
        self.lbl_status.config(
            text=(f"Otwory: {len(previews)}  |  frezowanie konturowe: {n_contour}  |  "
                  f"wiercenie: {n_drill}  |  pominiete: {n_skip}"),
            foreground="#666666" if n_skip == 0 else "#a15c00",
        )
        self.redraw()

    def clear(self):
        self._previews = []
        self.cmb_hole["values"] = [ALL_HOLES]
        self.var_hole.set(ALL_HOLES)
        self.lbl_status.config(text="Brak danych - kliknij 'Zbuduj sciezke narzedzia'.",
                                foreground="#666666")
        self.redraw()

    # ------------------------------------------------------------------ #
    #  Rysowanie
    # ------------------------------------------------------------------ #

    def redraw(self):
        current_tab = self.nb.index(self.nb.select())
        if current_tab == 0:
            self._draw_2d()
        else:
            self._draw_3d()

    def _selected_previews(self) -> List[HolePreview]:
        sel = self.var_hole.get()
        if sel == ALL_HOLES or not sel:
            return self._previews
        return [p for p in self._previews if p.name == sel]

    def _deepest_loops(self, op) -> List:
        """Zwraca kontur_passy z najglebszego Z (patrz docstring modulu) -- unikalne ksztalty."""
        if not op.contour_passes:
            return []
        min_z = min(p.z_mm for p in op.contour_passes)
        return [p for p in op.contour_passes if abs(p.z_mm - min_z) < 1e-9]

    def _draw_2d(self):
        ax = self.ax2d
        ax.clear()
        previews = self._selected_previews()

        if not previews:
            ax.text(0.5, 0.5, "Brak danych do wyswietlenia", ha="center", va="center",
                     transform=ax.transAxes, color="#888888")
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas2d.draw_idle()
            return

        cmap = cm.get_cmap("tab10")
        focused = len(previews) == 1

        for i, p in enumerate(previews):
            color = cmap(i % 10)
            op = p.operation

            if self.var_show_ref.get() and len(p.raw_dense_xa):
                ax.plot(p.raw_dense_xa[:, 0], p.raw_dense_xa[:, 1], "--", lw=1.0, color="#999999",
                        alpha=0.8, zorder=1)

            if op.mode == "contour" and self.var_show_toolpath.get():
                for loop_pass in self._deepest_loops(op):
                    ax.plot(loop_pass.path_x_mm, loop_pass.path_a_deg, "-", lw=1.8,
                            color=color, zorder=3)
                    ax.plot(loop_pass.path_x_mm[0], loop_pass.path_a_deg[0], "o",
                            color=color, ms=5, zorder=4)
            elif op.mode == "drill" and self.var_show_toolpath.get() and op.drill:
                ax.plot(op.drill.x_mm, op.drill.a_deg, "P", color=color, ms=10, zorder=4)
                if focused:
                    # zarys srednicy narzedzia wokol punktu wiercenia: promien fizyczny
                    # (mm) na osi X, ale na osi A trzeba przeliczyc luk->stopnie lokalnie
                    # (patrz geometry_core.unroll: s = ref_radius * theta)
                    t = np.linspace(0, 2 * np.pi, 60)
                    xs = op.drill.x_mm + self._tool.radius_mm * np.cos(t)
                    s_offset = self._tool.radius_mm * np.sin(t)
                    a_deg = op.drill.a_deg + np.degrees(s_offset / max(p.ref_radius_mm, 1e-6))
                    ax.plot(xs, a_deg, "-", color=color, lw=1.0, alpha=0.6)

            if self.var_show_warnings.get() and op.tight_corners and op.contour_passes:
                loops = self._deepest_loops(op)
                loop = loops[0] if loops else None
                if loop is not None:
                    idxs = [idx for idx, _ in op.tight_corners if idx < len(loop.path_x_mm)]
                    if idxs:
                        ax.scatter(loop.path_x_mm[idxs], loop.path_a_deg[idxs],
                                   marker="^", s=70, facecolors="none", edgecolors="red",
                                   linewidths=1.6, zorder=5, label=None)

            if not focused:
                cx = float(np.mean(p.x_mm_range))
                ax.annotate(p.name, xy=(cx, self._label_y(op)), color=color,
                            fontsize=8, ha="center", va="bottom", zorder=6)

        ax.set_xlabel("X - wzdluz osi rury [mm]")
        ax.set_ylabel("A - kat obrotu [deg]")
        title = "Wszystkie otwory" if not focused else previews[0].name
        ax.set_title(f"Rozwiniecie powierzchni rury - {title}", fontsize=10)
        ax.grid(True, alpha=0.3)
        self.fig2d.tight_layout()
        self.canvas2d.draw_idle()

    def _label_y(self, op) -> float:
        if op.mode == "drill" and op.drill:
            return op.drill.a_deg
        loops = self._deepest_loops(op)
        if loops:
            return float(np.max(loops[0].path_a_deg))
        return 0.0

    def _draw_3d(self):
        ax = self.ax3d
        ax.clear()
        previews = self._selected_previews()

        if self._tube is None:
            ax.text2D(0.5, 0.5, "Brak danych do wyswietlenia", ha="center", va="center",
                       transform=ax.transAxes, color="#888888")
            self.canvas3d.draw_idle()
            return

        r = self._tube.outer_radius_mm
        length = self._tube.length_mm
        theta = np.linspace(0, 2 * np.pi, 48)

        # Zarys walca: kilka linii wzdluznych + obreb na obu koncach (SCHEMATYCZNIE)
        for x in (0.0, length):
            ax.plot(np.full_like(theta, x), r * np.cos(theta), r * np.sin(theta),
                    color="#bbbbbb", lw=0.8)
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            ax.plot([0, length], [r * np.cos(a)] * 2, [r * np.sin(a)] * 2,
                    color="#dddddd", lw=0.5)

        cmap = cm.get_cmap("tab10")
        for i, p in enumerate(previews):
            color = cmap(i % 10)
            op = p.operation

            if self.var_show_ref.get() and len(p.raw_dense_xa):
                xyz = eb.contour_pass_to_xyz(p.raw_dense_xa[:, 0], p.raw_dense_xa[:, 1], r)
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "--", lw=0.9, color="#999999", alpha=0.7)

            if op.mode == "contour" and self.var_show_toolpath.get():
                for loop_pass in self._deepest_loops(op):
                    xyz = eb.contour_pass_to_xyz(loop_pass.path_x_mm, loop_pass.path_a_deg, r)
                    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-", lw=1.6, color=color)
            elif op.mode == "drill" and self.var_show_toolpath.get() and op.drill:
                xyz = eb.contour_pass_to_xyz(np.array([op.drill.x_mm]), np.array([op.drill.a_deg]), r)
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=40, marker="P")

        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Podglad 3D (schematyczny, glebokosc promieniowa nie w skali)", fontsize=9)
        try:
            ax.set_box_aspect((max(length, 1.0), 2 * r, 2 * r))
        except Exception:
            pass
        self.fig3d.tight_layout()
        self.canvas3d.draw_idle()
