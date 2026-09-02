#!/usr/bin/env python3
"""
main.py
-------
Punkt wejscia aplikacji. Uruchom:

    python3 main.py

Wymagania: patrz requirements.txt (numpy, pyclipper, matplotlib - obowiazkowe;
cadquery / trimesh - tylko jesli chcesz wczytywac pliki STEP / mesh).
"""

from __future__ import annotations

import sys
import os

# Pozwala uruchamiac `python3 main.py` bezposrednio z katalogu projektu,
# bez instalowania pakietu (cam_core/ i cam_ui/ sa obok tego pliku).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check_dependencies():
    missing = []
    for mod, hint in [("numpy", "numpy"), ("pyclipper", "pyclipper"),
                       ("matplotlib", "matplotlib")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(hint)
    if missing:
        print("Brakuje wymaganych pakietow: " + ", ".join(missing))
        print("Zainstaluj je poleceniem:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def main():
    _check_dependencies()
    from cam_ui.main_window import CamApp
    app = CamApp()
    app.mainloop()


if __name__ == "__main__":
    main()
