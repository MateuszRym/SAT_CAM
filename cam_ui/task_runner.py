"""
task_runner.py
---------------
Tkinter jest jednowatkowy - nie wolno dotykac widgetow z innego watku.
`run_async` odpala dlugotrwala funkcje (np. wczytanie STEP, budowa sciezki)
w tle (threading.Thread), a wynik/blad dostarcza z powrotem do glownego
watku przez `root.after(...)`, wiec callbacki `on_done`/`on_error` moga juz
bezpiecznie aktualizowac UI.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable, Any


def run_async(root, work: Callable[[], Any],
              on_done: Callable[[Any], None],
              on_error: Callable[[str, str], None],
              on_start: Callable[[], None] = None) -> None:
    """
    root: instancja Tk/Toplevel (potrzebna do .after)
    work: funkcja bezargumentowa wykonywana w tle, zwraca wynik
    on_done(result): wywolane w GLOWNYM watku po sukcesie
    on_error(message, traceback_text): wywolane w GLOWNYM watku po wyjatku
    on_start(): opcjonalnie wywolane w GLOWNYM watku od razu (np. pokaz progressbar)
    """
    if on_start is not None:
        on_start()

    def _worker():
        try:
            result = work()
        except Exception as e:  # noqa: BLE001 - celowo lapiemy wszystko, to watek w tle
            msg = str(e) or e.__class__.__name__
            tb = traceback.format_exc()
            root.after(0, lambda: on_error(msg, tb))
            return
        root.after(0, lambda: on_done(result))

    threading.Thread(target=_worker, daemon=True).start()
