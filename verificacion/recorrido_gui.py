"""Recorrido automático de toda la interfaz: cada menú, panel y opción. Anota excepciones."""
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, ".")
from PySide6 import QtCore, QtWidgets  # noqa: E402

errores = []
pasos = []


def gancho(t, v, tb):
    errores.append("".join(traceback.format_exception(t, v, tb))[-1500:])


sys.excepthook = gancho
QtCore.qInstallMessageHandler(lambda modo, ctx, msg: errores.append("Qt: " + msg)
                              if modo in (QtCore.QtMsgType.QtCriticalMsg, QtCore.QtMsgType.QtFatalMsg) else None)
app = QtWidgets.QApplication(sys.argv)
from laminado.gui import ventana as V  # noqa: E402

TMP = tempfile.mkdtemp()
avisos = []
# diálogos automáticos
QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: avisos.append(("info", a[1] if len(a) > 1 else "")))
QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: avisos.append(("aviso", str(a[2])[:300] if len(a) > 2 else "")))
QtWidgets.QMessageBox.about = staticmethod(lambda *a, **k: None)
QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: TMP)
QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(TMP, "proy.lam"), ""))
QtWidgets.QInputDialog.getText = staticmethod(lambda *a, **k: ("0, 45, -45, 90", True))
QtWidgets.QInputDialog.getDouble = staticmethod(lambda *a, **k: (5.0, True))

w = V.VentanaPrincipal()
w.show()


def esperar():
    for _ in range(12000):
        app.processEvents()
        if not w._ocupado and not w._pendiente:
            return
        QtCore.QThread.msleep(50)
    errores.append("tiempo agotado esperando un cálculo")


def paso(nombre, fn):
    n0 = len(errores)
    try:
        fn()
        esperar()
        app.processEvents()
    except Exception:  # noqa: BLE001
        errores.append(f"[{nombre}] " + traceback.format_exc()[-1500:])
    pasos.append((nombre, len(errores) == n0))


def sel(tipo, clave):
    w._seleccionar(tipo, clave)


# ------------------------------------------------------------------ recorrido
paso("abrir demo", w.demo)
w.a_live.setChecked(True)
paso("simular capa (F5)", w.simular_sel)
for modo in ("Producibilidad", "Cizalla", "Ángulo de fibra", "Espesores", "Sólo molde", "Producibilidad"):
    paso(f"color: {modo}", lambda m=modo: w.modo_color.setCurrentText(m))
for ch in (w.ch_red, w.ch_fib, w.ch_todas):
    paso(f"casilla {ch.text()}", lambda c=ch: (c.setChecked(not c.isChecked()), c.setChecked(not c.isChecked())))
for pest in range(4):
    paso(f"pestaña {pest}", lambda k=pest: w.tabs.setCurrentIndex(k))
paso("seleccionar proyecto", lambda: sel("proyecto", None))
paso("seleccionar molde", lambda: sel("molde", None))
paso("seleccionar selección ASIENTO", lambda: sel("seleccion", "ASIENTO"))
paso("seleccionar material", lambda: sel("material", next(iter(w.p.materiales))))
paso("seleccionar roseta", lambda: sel("roseta", "Roseta global"))
paso("seleccionar secuencia", lambda: sel("secuencia", "Refuerzos"))
paso("nuevo núcleo", w.nuevo_nucleo)
paso("seleccionar núcleo", lambda: sel("nucleo", 0))
sel("capa", "P003")
pan = w.panel
paso("orientación 30° (interactivo)", lambda: pan.ori.setValue(30))
for k in range(pan.estr.count()):
    paso(f"estrategia {pan.estr.itemData(k)}", lambda k=k: w.panel.estr.setCurrentIndex(k))
paso("estrategia geodésica", lambda: w.panel.estr.setCurrentIndex(0))
paso("contorno polígono", lambda: (w.panel.cont.tipo.setCurrentText("poligono"),
                                    w.panel.cont.puntos.setPlainText("-100,-60\n100,-60\n100,60\n-100,60")))
paso("contorno selección", lambda: w.panel.cont.tipo.setCurrentText("seleccion"))
paso("desfase y exceso", lambda: (w.panel.cont.desfase.setValue(-5), w.panel.cont.exceso.setValue(12)))
V_ = w.p.malla.V
pto = lambda x, y: V_[np.argmin(np.linalg.norm(V_[:, :2] - [x, y], axis=1))]  # noqa: E731
paso("picar semilla", lambda: (w._empezar_picado("semilla"), w._picado(pto(10, 5))))
paso("semilla automática", lambda: w.panel._semilla_auto())
sel("capa", "P001")
paso("dart manual", lambda: (w._empezar_picado("dart"), w._picado(pto(230, 120)), w._picado(pto(190, 110)),
                              w._picado(pto(150, 100)), w.terminar_picado()))
paso("curva de semilla", lambda: (w._empezar_picado("curva"), [w._picado(pto(x, 0)) for x in (-150, -50, 50, 150)],
                                   w.terminar_picado()))
paso("empalme manual", lambda: (w._empezar_picado("empalme"), w._picado(pto(60, 0))))
paso("empalmes por cizalla", lambda: w.panel.emp_ciz.setChecked(True))
paso("darts automáticos", lambda: w.panel.darts_auto.setChecked(True))
paso("borrar darts y empalmes", lambda: (w.panel._borrar_darts(), w.panel._borrar_emp()))
paso("cancelar picado", w.cancelar_picado)
paso("core sample", lambda: (w._modo_core(True), w._picado(pto(0, 0)), w._modo_core(False)))
paso("roseta: picar origen", lambda: (sel("roseta", "Roseta global"), w._empezar_picado("roseta"), w._picado(pto(0, 0))))
paso("nueva capa", lambda: (sel("capa", "P002"), w.nueva_capa()))
paso("duplicar capa", w.duplicar_capa)
paso("subir capa", lambda: w.mover(-1))
paso("bajar capa", lambda: w.mover(1))
paso("eliminar capa", w.eliminar)
paso("nuevo material", w.nuevo_material)
paso("nueva roseta", w.nueva_roseta)
paso("nueva secuencia", w.nueva_secuencia)
paso("asistente de laminado", lambda: (sel("capa", "P003"), w.asistente()))


def _cobertura():
    from laminado.gui import dialogos

    def ejecutar(self):
        self.ori.setText("0")
        for i in range(self.zona.count()):
            if self.zona.item(i).text() == "POMO":
                self.zona.item(i).setSelected(True)
        return True
    dialogos.DialogoCobertura.exec = ejecutar
    w.cobertura_auto()
    esperar()
    assert any(c.nombre.startswith("C0_") for c in w.p.capas), "sin parches de cobertura"
    w.modo_color.setCurrentText("Cobertura (nº de capas)")
    w._dibujar()


paso("cobertura completa automática", _cobertura)
paso("modo de color cobertura", lambda: (sel("capa", "P001"), w._dibujar()))
paso("laminado con espesor (panel del proyecto)", lambda: (sel("proyecto", None),
                                                          setattr(w.p, "drapeado_con_espesor", True)))
paso("simular todas", w.simular_todas)
paso("salida DXF", w.sal_dxf)
paso("salida anidado", w.sal_anidado)
paso("salida láser", w.sal_laser)
paso("salida ply book", w.sal_plybook)
paso("salida Excel", w.sal_excel)
paso("salida Ansys", w.sal_ansys)
paso("todas las salidas", w.sal_todo)
paso("guardar", lambda: w.guardar(True))
paso("abrir", lambda: w.abrir(os.path.join(TMP, "proy.lam")))
paso("importar .wbpz real (ACP)", lambda: w.importar(os.path.join("tests", "datos_ansys", "fatigue_example.wbpz")))
paso("importar .cdb real (class40)", lambda: w.importar(os.path.join("tests", "datos_ansys", "class40.cdb")))
paso("capa sobre selección DECK", lambda: (w.nueva_capa(seleccion="DECK"), w.simular_sel()))
paso("guía y acerca de", lambda: (w.ayuda(), w.acerca()))
paso("nuevo proyecto", w.nuevo)

w.grab().save(os.path.join(TMP, "final.png"))
print("PASOS:")
for n, ok in pasos:
    print(("  ✓ " if ok else "  ✗ ") + n)
print("AVISOS (mensajes mostrados):", [a for a in avisos if a[0] == "aviso"])
print("ERRORES:", len(errores))
for e in errores:
    print("-----\n" + e)
print("salidas:", sorted(os.listdir(TMP)))
