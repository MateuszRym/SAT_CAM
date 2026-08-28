import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np

# Integracja Matplotlib z Tkinter (3D)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importy z Twoich plików
from config import Project, ToolKind
from model_io import load_step_holes, load_mesh_holes, canonicalize
from geometry_core import (
    axis_project, unwrap_theta, unroll, 
    offset_polygon_inward, roll_back, theta_to_deg_continuous
)

class CAMHoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNC Pipe Hole CAM - UI")
        self.project = Project()
        self.holes = []
        self.axis = None
        
        self.current_json_path = None
        self._auto_save_job = None
        
        # Zmienne do symulacji
        self.sim_points = []
        self.sim_job = None
        self.tool_marker = None
        self.sim_idx = 0

        # --- Główny układ okna ---
        left_frame = tk.Frame(root, width=280, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(root, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Przyciski (Lewy Panel) ---
        tk.Label(left_frame, text="Zarządzanie projektem", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(left_frame, text="Wczytaj projekt (JSON)", command=self.load_project, width=28).pack(pady=2)
        tk.Button(left_frame, text="Zapisz projekt jako...", command=self.save_project_as, width=28).pack(pady=2)
        
        # --- Parametry (Auto-Zapis) ---
        param_frame = tk.LabelFrame(left_frame, text="Parametry Narzędzia i Rury", padx=10, pady=10)
        param_frame.pack(fill=tk.X, pady=10)

        self.var_tool_kind = tk.StringVar(value=self.project.tool.kind.value)
        self.var_tool_dia = tk.StringVar(value=str(self.project.tool.diameter_mm))
        self.var_tube_od = tk.StringVar(value=str(self.project.tube.outer_diameter_mm))
        self.var_wall = tk.StringVar(value=str(self.project.tube.wall_thickness_mm))

        self.var_tool_kind.trace_add("write", self.schedule_apply_and_save)
        self.var_tool_dia.trace_add("write", self.schedule_apply_and_save)
        self.var_tube_od.trace_add("write", self.schedule_apply_and_save)
        self.var_wall.trace_add("write", self.schedule_apply_and_save)

        def add_param_row(parent, label_text, var):
            row = tk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label_text).pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, width=8).pack(side=tk.RIGHT)

        row_kind = tk.Frame(param_frame)
        row_kind.pack(fill=tk.X, pady=2)
        tk.Label(row_kind, text="Typ narz.:").pack(side=tk.LEFT)
        self.combo_kind = ttk.Combobox(row_kind, textvariable=self.var_tool_kind, values=["endmill", "drill"], width=7, state="readonly")
        self.combo_kind.pack(side=tk.RIGHT)

        add_param_row(param_frame, "Średnica narz. [mm]:", self.var_tool_dia)
        add_param_row(param_frame, "Średnica zewn. [mm]:", self.var_tube_od)
        add_param_row(param_frame, "Grubość ścianki [mm]:", self.var_wall)

        tk.Label(left_frame, text="Import geometrii", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        tk.Button(left_frame, text="Importuj model STEP", command=self.load_step, width=28).pack(pady=2)
        tk.Button(left_frame, text="Importuj siatkę (3MF/STL)", command=self.load_mesh, width=28).pack(pady=2)
        
        tk.Label(left_frame, text="Eksport i Symulacja", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        tk.Button(left_frame, text="Generuj G-Code", command=self.generate_gcode, width=28, bg="lightblue").pack(pady=5)
        tk.Button(left_frame, text="▶ Symuluj obróbkę", command=self.start_simulation, width=28, bg="lightgreen").pack(pady=5)

        # Sekcja paska postępu
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(left_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(15, 5))
        
        self.lbl_status = tk.Label(left_frame, text="Gotowy", font=("Arial", 9))
        self.lbl_status.pack()

        # --- Podgląd (Prawy Panel - Matplotlib 3D) ---
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.reset_plot_view()

    def schedule_apply_and_save(self, *args):
        if self._auto_save_job:
            self.root.after_cancel(self._auto_save_job)
        self._auto_save_job = self.root.after(500, self.apply_and_save)

    def apply_and_save(self):
        try:
            kind_val = self.var_tool_kind.get()
            tool_dia = float(self.var_tool_dia.get())
            tube_od = float(self.var_tube_od.get())
            wall = float(self.var_wall.get())

            if tool_dia <= 0 or tube_od <= 0 or wall <= 0:
                return 

            self.project.tool.kind = ToolKind(kind_val)
            self.project.tool.diameter_mm = tool_dia
            self.project.tube.outer_diameter_mm = tube_od
            self.project.tube.wall_thickness_mm = wall

            if self.current_json_path:
                with open(self.current_json_path, 'w', encoding='utf-8') as f:
                    f.write(self.project.to_json())

            self.update_preview()
        except ValueError:
            pass

    def update_ui_vars_from_project(self):
        self.var_tool_kind.set(self.project.tool.kind.value)
        self.var_tool_dia.set(str(self.project.tool.diameter_mm))
        self.var_tube_od.set(str(self.project.tube.outer_diameter_mm))
        self.var_wall.set(str(self.project.tube.wall_thickness_mm))

    def stop_simulation(self):
        """Zatrzymuje ewentualnie trwającą animację przy odświeżaniu widoku"""
        if self.sim_job:
            self.root.after_cancel(self.sim_job)
            self.sim_job = None
        if self.tool_marker:
            try:
                self.tool_marker.remove()
            except:
                pass
            self.tool_marker = None
        self.lbl_status.config(text="Gotowy")
        self.progress_var.set(0)

    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.project = Project.from_json(f.read())
                self.current_json_path = path
                self.update_ui_vars_from_project()
                messagebox.showinfo("Sukces", "Projekt załadowany. Parametry uaktualnione.")
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Błąd JSON", f"Nie udało się załadować: {e}")

    def save_project_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if path:
            self.current_json_path = path
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.project.to_json())
            messagebox.showinfo("Zapis", f"Projekt zapisany jako:\n{path}")

    def load_step(self):
        path = filedialog.askopenfilename(filetypes=[("STEP Files", "*.step *.stp")])
        if path:
            try:
                # ZMIANA: Dodano *_ na końcu, aby zignorować dodatkowe zwracane parametry (np. wymiary rury)
                self.holes, self.axis, *_ = load_step_holes(path, tolerance_mm=self.project.job.tolerance_mm)
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Błąd STEP", str(e))

    def load_mesh(self):
        path = filedialog.askopenfilename(filetypes=[("Mesh Files", "*.3mf *.stl *.obj")])
        if path:
            try:
                # ZMIANA: Analogiczne dodanie *_ dla importu siatek
                self.holes, self.axis, *_ = load_mesh_holes(path)
                self.update_preview()
            except Exception as e:
                messagebox.showerror("Błąd Siatki", str(e))

    def reset_plot_view(self):
        self.ax.clear()
        self.ax.set_title("Podgląd 3D (Zwijanie na walec)")
        self.ax.set_xlabel("Oś X [mm]")
        self.ax.set_ylabel("Oś Y [mm]")
        self.ax.set_zlabel("Oś Z [mm]")
        self.ax.grid(False)

    def draw_transparent_cylinder(self, ref_radius):
        if not self.holes:
            return
            
        all_x = []
        for hole in self.holes:
            can_pts = canonicalize(hole.points_xyz, self.axis)
            all_x.extend(can_pts[:, 0])
        x_min, x_max = min(all_x) - 10, max(all_x) + 10
        
        x_grid = np.linspace(x_min, x_max, 50)
        theta_grid = np.linspace(0, 2 * np.pi, 50)
        Xc, Thetac = np.meshgrid(x_grid, theta_grid)
        Yc = ref_radius * np.cos(Thetac)
        Zc = ref_radius * np.sin(Thetac)
        
        self.ax.plot_surface(Xc, Yc, Zc, alpha=0.15, color='cyan', edgecolor='none')
        
        try:
            self.ax.set_box_aspect((x_max - x_min, 2 * ref_radius, 2 * ref_radius))
        except AttributeError:
            pass

    def update_preview(self):
        self.stop_simulation()
        self.reset_plot_view()
        self.sim_points = [] # Zerowanie punktów dla nowej ścieżki narzędzia

        if not self.holes or not self.axis:
            self.canvas.draw()
            return

        tool_radius = self.project.tool.radius_mm
        ref_radius = self.project.tube.outer_radius_mm
        is_drill = self.project.tool.kind == ToolKind.DRILL

        self.draw_transparent_cylinder(ref_radius)

        for hole in self.holes:
            can_pts = canonicalize(hole.points_xyz, self.axis)
            axis_pts = axis_project(can_pts)
            thetas = unwrap_theta(axis_pts)
            xs = np.array([p.x_mm for p in axis_pts])
            
            xy_unrolled = unroll(xs, thetas, ref_radius)
            orig_theta = xy_unrolled[:, 1] / ref_radius
            orig_y = ref_radius * np.cos(orig_theta)
            orig_z = ref_radius * np.sin(orig_theta)
            self.ax.plot(xy_unrolled[:, 0], orig_y, orig_z, 'k-', alpha=0.5)

            # --- NOWA INTELIGENTNA LOGIKA DETEKCJI ---
            tool_dia = self.project.tool.diameter_mm
            tolerance = getattr(self.project.job, 'drill_diameter_tolerance_mm', 0.05)
            
            hole_w = np.max(xy_unrolled[:, 0]) - np.min(xy_unrolled[:, 0])
            hole_h = np.max(xy_unrolled[:, 1]) - np.min(xy_unrolled[:, 1])
            
            # Wymusza punktowanie jeśli narzędzie pokrywa wymiar otworu
            force_drill = is_drill or (hole_w <= tool_dia + tolerance and hole_h <= tool_dia + tolerance)

            if force_drill:
                center_x = (np.max(xy_unrolled[:, 0]) + np.min(xy_unrolled[:, 0])) / 2.0
                center_s = (np.max(xy_unrolled[:, 1]) + np.min(xy_unrolled[:, 1])) / 2.0
                c_theta = center_s / ref_radius
                cy = ref_radius * np.cos(c_theta)
                cz = ref_radius * np.sin(c_theta)
                
                self.ax.plot([center_x], [cy], [cz], 'rx', markersize=8, markeredgewidth=2)
                
                # Dodanie punktu dla symulacji wiercenia
                self.sim_points.append((center_x, cy, cz))
            else:
                toolpaths = offset_polygon_inward(xy_unrolled, tool_radius)
                for tp in toolpaths:
                    tp_draw = np.vstack([tp, tp[0]])
                    tp_x = tp_draw[:, 0]
                    tp_theta = tp_draw[:, 1] / ref_radius
                    tp_y = ref_radius * np.cos(tp_theta)
                    tp_z = ref_radius * np.sin(tp_theta)
                    
                    self.ax.plot(tp_x, tp_y, tp_z, 'r--', linewidth=2)
                    
                    # Zebranie punktów do animacji frezowania
                    self.sim_points.extend(list(zip(tp_x, tp_y, tp_z)))

        self.canvas.draw()

    def start_simulation(self):
        if not self.sim_points:
            messagebox.showwarning("Brak ścieżki", "Brak wygenerowanej ścieżki do symulacji. Sprawdź parametry.")
            return
            
        self.stop_simulation()
        
        # Przygotowanie wskaźnika (magenta kropka symbolizująca frez/wiertło)
        self.tool_marker, = self.ax.plot([], [], [], 'mo', markersize=8, label='Narzędzie')
        self.ax.legend()
        
        self.progress.config(maximum=len(self.sim_points))
        self.sim_idx = 0
        self.lbl_status.config(text="Symulacja w toku...")
        
        # Wywołanie pętli animacji
        self._animate_step()

    def _animate_step(self):
        if self.sim_idx >= len(self.sim_points):
            self.lbl_status.config(text="Symulacja zakończona!")
            self.progress_var.set(len(self.sim_points))
            self.sim_job = None
            return

        x, y, z = self.sim_points[self.sim_idx]
        
        # Aktualizacja pozycji w 3D
        self.tool_marker.set_data([x], [y])
        self.tool_marker.set_3d_properties([z])
        self.canvas.draw_idle()

        # Aktualizacja paska postępu
        self.sim_idx += 1
        self.progress_var.set(self.sim_idx)
        
        # Kolejny krok za 40 ms
        self.sim_job = self.root.after(40, self._animate_step)

    def generate_gcode(self):
        if not self.holes or not self.axis:
            messagebox.showwarning("Brak danych", "Załaduj najpierw model (STEP lub siatkę), aby wygenerować G-Code.")
            return

        out_path = filedialog.asksaveasfilename(defaultextension=".nc", filetypes=[("G-Code", "*.nc *.gcode")])
        if not out_path:
            return

        tool_radius = self.project.tool.radius_mm
        ref_radius = self.project.tube.outer_radius_mm
        safe_z = self.project.job.safe_z_mm
        cut_z = -self.project.tube.wall_thickness_mm - self.project.job.breakthrough_margin_mm
        feed = self.project.tool.feed_cut_mm_min
        is_drill = self.project.tool.kind == ToolKind.DRILL

        gcode_lines = [
            f"(Program: {self.project.job.program_name})",
            "(Maszyna XZA - Generacja Automatyczna)",
            f"(Tryb pracy: {'WIERCENIE' if is_drill else 'FREZOWANIE KONTUROW'})",
            "G21 (Jednostki: mm)",
            "G90 (Programowanie absolutne)",
            f"G0 Z{safe_z:.3f} (Podniesienie na bezpieczna wysokosc)",
        ]

        m_code = "M3" if self.project.tool.spindle_cw else "M4"
        gcode_lines.append(f"{m_code} S{int(self.project.tool.spindle_rpm)} (Start wrzeciona)")
        if self.project.job.coolant == "mist": gcode_lines.append("M7 (Chlodzenie mgla)")
        elif self.project.job.coolant == "flood": gcode_lines.append("M8 (Chlodzenie plynem)")

        try:
            for hole in self.holes:
                can_pts = canonicalize(hole.points_xyz, self.axis)
                axis_pts = axis_project(can_pts)
                thetas = unwrap_theta(axis_pts)
                xs = np.array([p.x_mm for p in axis_pts])
                xy_unrolled = unroll(xs, thetas, ref_radius)
                
                gcode_lines.append(f"\n(--- Start obrobki otworu ---)")
                
                # --- NOWA INTELIGENTNA LOGIKA DETEKCJI ---
                tool_dia = self.project.tool.diameter_mm
                tolerance = getattr(self.project.job, 'drill_diameter_tolerance_mm', 0.05)
                
                hole_w = np.max(xy_unrolled[:, 0]) - np.min(xy_unrolled[:, 0])
                hole_h = np.max(xy_unrolled[:, 1]) - np.min(xy_unrolled[:, 1])
                
                force_drill = is_drill or (hole_w <= tool_dia + tolerance and hole_h <= tool_dia + tolerance)
                
                if force_drill:
                    center_x = (np.max(xy_unrolled[:, 0]) + np.min(xy_unrolled[:, 0])) / 2.0
                    center_s = (np.max(xy_unrolled[:, 1]) + np.min(xy_unrolled[:, 1])) / 2.0
                    
                    _, center_theta = roll_back(np.array([[center_x, center_s]]), ref_radius)
                    center_a_deg = theta_to_deg_continuous(center_theta)[0]
                    
                    gcode_lines.append(f"G0 X{center_x:.3f} A{center_a_deg:.3f} (Nawigacja nad srodek)")
                    gcode_lines.append(f"G1 Z{cut_z:.3f} F{self.project.tool.feed_plunge_mm_min:.1f} (Wiercenie)")
                    gcode_lines.append(f"G0 Z{safe_z:.3f} (Wycofanie)")

                else:
                    toolpaths = offset_polygon_inward(xy_unrolled, tool_radius)
                    for tp in toolpaths:
                        x_path, theta_path = roll_back(tp, ref_radius)
                        a_deg_path = theta_to_deg_continuous(theta_path)
                        
                        gcode_lines.append(f"G0 X{x_path[0]:.3f} A{a_deg_path[0]:.3f}")
                        gcode_lines.append(f"G1 Z{cut_z:.3f} F{self.project.tool.feed_plunge_mm_min:.1f}")
                        
                        for x_val, a_val in zip(x_path[1:], a_deg_path[1:]):
                            gcode_lines.append(f"G1 X{x_val:.3f} A{a_val:.3f} F{feed:.1f}")
                        
                        gcode_lines.append(f"G1 X{x_path[0]:.3f} A{a_deg_path[0]:.3f} F{feed:.1f}")
                        gcode_lines.append(f"G0 Z{safe_z:.3f}")

            gcode_lines.append("\nM5 (Wylaczenie wrzeciona)")
            gcode_lines.append("M9 (Wylaczenie chlodzenia)")
            gcode_lines.append(f"G0 Z{safe_z + 10:.3f}")
            gcode_lines.append("M30 (Koniec programu)")

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(gcode_lines))
            
            messagebox.showinfo("Sukces", f"Zapisano G-Code do:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Błąd generacji", f"Wystąpił błąd podczas generowania ścieżki:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x630") # Lekko podniesiona ramka, by pomieścić pasek postępu
    app = CAMHoleApp(root)
    root.mainloop()
