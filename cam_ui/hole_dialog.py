"""
hole_dialog.py
---------------
Okno dialogowe dodawania / edycji pojedynczego otworu (cam_core.config.HoleDef)
w trybie recznym. Pola zmieniaja sie dynamicznie w zaleznosci od wybranego
ksztaltu (HoleShape).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List, Optional

from cam_core.config import HoleDef, HoleShape


SHAPE_LABELS = {
    HoleShape.CIRCLE: "Kolo",
    HoleShape.RECTANGLE: "Prostokat",
    HoleShape.ROUNDED_RECTANGLE: "Prostokat zaokraglony",
    HoleShape.POLYGON: "Dowolny wielokat (u,v)",
}
LABEL_TO_SHAPE = {v: k for k, v in SHAPE_LABELS.items()}


class HoleDialog(tk.Toplevel):
    def __init__(self, parent, hole: Optional[HoleDef], existing_names: List[str],
                 on_save: Callable[[HoleDef], None]):
        super().__init__(parent)
        self.title("Otwor" if hole is None else f"Edycja otworu: {hole.name}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._on_save = on_save
        self._is_new = hole is None
        self._original_name = None if hole is None else hole.name
        self._existing_names = set(existing_names)
        self._polygon_points: List[List[float]] = (
            [list(p) for p in hole.polygon_points_mm] if hole and hole.polygon_points_mm else []
        )

        src = hole if hole is not None else HoleDef(
            name="", shape=HoleShape.CIRCLE, center_x_mm=50.0, center_angle_deg=0.0,
            diameter_mm=8.0,
        )

        self.var_name = tk.StringVar(value=src.name)
        self.var_shape = tk.StringVar(value=SHAPE_LABELS[src.shape])
        self.var_enabled = tk.BooleanVar(value=src.enabled)
        self.var_center_x = tk.DoubleVar(value=src.center_x_mm)
        self.var_center_angle = tk.DoubleVar(value=src.center_angle_deg)
        self.var_diameter = tk.DoubleVar(value=src.diameter_mm or 8.0)
        self.var_width = tk.DoubleVar(value=src.width_mm or 10.0)
        self.var_height = tk.DoubleVar(value=src.height_mm or 10.0)
        self.var_corner_r = tk.DoubleVar(value=src.corner_radius_mm or 2.0)

        self._build_ui()
        self._on_shape_change()

    # ------------------------------------------------------------------ #

    def _build_ui(self):
        pad = dict(padx=8, pady=4)
        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        row = 0
        ttk.Label(frm, text="Nazwa:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_name, width=24).grid(row=row, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Aktywny", variable=self.var_enabled).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        ttk.Label(frm, text="Ksztalt:").grid(row=row, column=0, sticky="e", **pad)
        cb = ttk.Combobox(frm, textvariable=self.var_shape, state="readonly",
                           values=list(SHAPE_LABELS.values()), width=22)
        cb.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_shape_change())
        row += 1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        ttk.Label(frm, text="Polozenie srodka otworu", font=("", 9, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", **pad)
        row += 1
        ttk.Label(frm, text="X wzdluz osi rury [mm]:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_center_x, width=12).grid(row=row, column=1, sticky="w", **pad)
        row += 1
        ttk.Label(frm, text="Kat obwodowy [deg]:").grid(row=row, column=0, sticky="e", **pad)
        ttk.Entry(frm, textvariable=self.var_center_angle, width=12).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        row += 1

        self._shape_row_start = row
        self.lbl_diameter = ttk.Label(frm, text="Srednica [mm]:")
        self.ent_diameter = ttk.Entry(frm, textvariable=self.var_diameter, width=12)

        self.lbl_width = ttk.Label(frm, text="Szerokosc wzdluz X [mm]:")
        self.ent_width = ttk.Entry(frm, textvariable=self.var_width, width=12)
        self.lbl_height = ttk.Label(frm, text="Wysokosc obwodowo [mm]:")
        self.ent_height = ttk.Entry(frm, textvariable=self.var_height, width=12)
        self.lbl_corner = ttk.Label(frm, text="Promien naroznika [mm]:")
        self.ent_corner = ttk.Entry(frm, textvariable=self.var_corner_r, width=12)

        self.frm_polygon = ttk.Frame(frm)
        self._build_polygon_editor(self.frm_polygon)

        self._dynamic_row = row

        row2 = row + 5
        btns = ttk.Frame(frm)
        btns.grid(row=row2, column=0, columnspan=3, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="Anuluj", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Zapisz", command=self._save).pack(side="right", padx=4)

    def _build_polygon_editor(self, parent):
        ttk.Label(parent, text="Punkty (u wzdluz X, v obwodowo) [mm], wzgledem srodka:").pack(
            anchor="w")
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill="both", expand=True, pady=4)
        self.poly_list = tk.Listbox(list_frame, height=6, width=28)
        self.poly_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.poly_list.yview)
        sb.pack(side="left", fill="y")
        self.poly_list.configure(yscrollcommand=sb.set)
        self._refresh_polygon_list()

        row_frame = ttk.Frame(parent)
        row_frame.pack(fill="x", pady=2)
        self.var_poly_u = tk.DoubleVar(value=0.0)
        self.var_poly_v = tk.DoubleVar(value=0.0)
        ttk.Label(row_frame, text="u:").pack(side="left")
        ttk.Entry(row_frame, textvariable=self.var_poly_u, width=8).pack(side="left", padx=2)
        ttk.Label(row_frame, text="v:").pack(side="left")
        ttk.Entry(row_frame, textvariable=self.var_poly_v, width=8).pack(side="left", padx=2)
        ttk.Button(row_frame, text="Dodaj", command=self._poly_add).pack(side="left", padx=4)
        ttk.Button(row_frame, text="Usun zaznaczony", command=self._poly_remove).pack(side="left", padx=4)

    def _refresh_polygon_list(self):
        self.poly_list.delete(0, tk.END)
        for u, v in self._polygon_points:
            self.poly_list.insert(tk.END, f"u={u:.2f}  v={v:.2f}")

    def _poly_add(self):
        self._polygon_points.append([self.var_poly_u.get(), self.var_poly_v.get()])
        self._refresh_polygon_list()

    def _poly_remove(self):
        sel = self.poly_list.curselection()
        if sel:
            del self._polygon_points[sel[0]]
            self._refresh_polygon_list()

    # ------------------------------------------------------------------ #

    def _on_shape_change(self):
        shape = LABEL_TO_SHAPE[self.var_shape.get()]
        for w in (self.lbl_diameter, self.ent_diameter, self.lbl_width, self.ent_width,
                  self.lbl_height, self.ent_height, self.lbl_corner, self.ent_corner,
                  self.frm_polygon):
            w.grid_forget()

        r = self._dynamic_row
        if shape == HoleShape.CIRCLE:
            self.lbl_diameter.grid(row=r, column=0, sticky="e", padx=8, pady=4)
            self.ent_diameter.grid(row=r, column=1, sticky="w", padx=8, pady=4)
        elif shape == HoleShape.RECTANGLE:
            self.lbl_width.grid(row=r, column=0, sticky="e", padx=8, pady=4)
            self.ent_width.grid(row=r, column=1, sticky="w", padx=8, pady=4)
            self.lbl_height.grid(row=r + 1, column=0, sticky="e", padx=8, pady=4)
            self.ent_height.grid(row=r + 1, column=1, sticky="w", padx=8, pady=4)
        elif shape == HoleShape.ROUNDED_RECTANGLE:
            self.lbl_width.grid(row=r, column=0, sticky="e", padx=8, pady=4)
            self.ent_width.grid(row=r, column=1, sticky="w", padx=8, pady=4)
            self.lbl_height.grid(row=r + 1, column=0, sticky="e", padx=8, pady=4)
            self.ent_height.grid(row=r + 1, column=1, sticky="w", padx=8, pady=4)
            self.lbl_corner.grid(row=r + 2, column=0, sticky="e", padx=8, pady=4)
            self.ent_corner.grid(row=r + 2, column=1, sticky="w", padx=8, pady=4)
        elif shape == HoleShape.POLYGON:
            self.frm_polygon.grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=4)

    # ------------------------------------------------------------------ #

    def _save(self):
        name = self.var_name.get().strip()
        if not name:
            messagebox.showerror("Blad", "Nazwa otworu nie moze byc pusta.", parent=self)
            return
        if name != self._original_name and name in self._existing_names:
            messagebox.showerror("Blad", f"Otwor o nazwie '{name}' juz istnieje.", parent=self)
            return

        shape = LABEL_TO_SHAPE[self.var_shape.get()]
        if shape == HoleShape.POLYGON and len(self._polygon_points) < 3:
            messagebox.showerror("Blad", "Wielokat wymaga co najmniej 3 punktow.", parent=self)
            return

        try:
            hole = HoleDef(
                name=name,
                shape=shape,
                center_x_mm=float(self.var_center_x.get()),
                center_angle_deg=float(self.var_center_angle.get()),
                diameter_mm=float(self.var_diameter.get()),
                width_mm=float(self.var_width.get()),
                height_mm=float(self.var_height.get()),
                corner_radius_mm=float(self.var_corner_r.get()),
                polygon_points_mm=[list(p) for p in self._polygon_points],
                enabled=bool(self.var_enabled.get()),
            )
        except (tk.TclError, ValueError):
            messagebox.showerror("Blad", "Sprawdz, czy wszystkie pola liczbowe sa poprawne.", parent=self)
            return

        self._on_save(hole)
        self.destroy()
