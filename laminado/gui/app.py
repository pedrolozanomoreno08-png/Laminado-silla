"""Punto de entrada de la aplicación de escritorio."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="laminado-studio", description="Laminado Studio")
    ap.add_argument("archivo", nargs="?", help="Proyecto .lam o geometría (.wbpz, .dat, .cdb, .rst, .step, .stl)")
    ap.add_argument("--demo", action="store_true", help="Abrir la silla de demostración")
    ap.add_argument("--captura", help="(pruebas) guarda una captura de la ventana y sale")
    ap.add_argument("--simular", action="store_true", help="(pruebas) simula la capa seleccionada al abrir")
    ap.add_argument("--pestana", type=int, default=0, help="(pruebas) pestaña visible en la captura")
    ap.add_argument("--capa", help="(pruebas) capa a seleccionar")
    a = ap.parse_args(argv)
    from PySide6 import QtCore, QtWidgets
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Laminado Studio")
    app.setStyle("Fusion")
    from .ventana import VentanaPrincipal
    w = VentanaPrincipal()
    w.show()
    if a.demo:
        w.demo()
    elif a.archivo:
        if a.archivo.lower().endswith(".lam"):
            w.abrir(a.archivo)
        else:
            w.importar(a.archivo)
    if a.captura:
        def capturar():
            w.tabs.setCurrentIndex(a.pestana)
            app.processEvents()
            w.vista.plotter.render()
            app.processEvents()
            w.grab().save(a.captura)
            app.quit()

        def esperar():
            if w._ocupado:
                QtCore.QTimer.singleShot(500, esperar)
            else:
                QtCore.QTimer.singleShot(1500, capturar)
        if a.capa:
            w._seleccionar("capa", a.capa)
        if a.simular:
            QtCore.QTimer.singleShot(800, w.simular_sel)
        QtCore.QTimer.singleShot(2000, esperar)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
