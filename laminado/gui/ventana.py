"""Ventana principal de Laminado Studio."""
from __future__ import annotations

import copy
import os
import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..ansys.lector import Molde
from ..drapeado.simular import Empalme
from ..geom.consulta import Consultor
from ..modelo.proyecto import Capa, Contorno, Nucleo, Proyecto, Roseta, Secuencia
from ..modelo.materiales import Material
from . import propiedades as P
from .vista3d import Vista3D

FILTRO_MOLDE = ("Geometría (*.wbpz *.dat *.cdb *.inp *.rst *.step *.stp *.iges *.igs *.stl *.obj *.ply *.vtk *.vtp);;"
                "Proyecto de Ansys Workbench (*.wbpz);;Entrada MAPDL (*.dat *.cdb *.inp);;Resultados (*.rst);;"
                "CAD (*.step *.stp *.iges *.igs);;Malla (*.stl *.obj *.ply *.vtk *.vtp);;Todos (*.*)")


class _Senales(QtCore.QObject):
    hecho = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progreso = QtCore.Signal(str)


class Trabajo(QtCore.QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args
        self.s = _Senales()

    def run(self):
        try:
            r = self.fn(*self.args)
            self.s.hecho.emit(r)
        except Exception as e:  # noqa: BLE001
            self.s.error.emit(f"{e}\n\n{traceback.format_exc()}")


class VentanaPrincipal(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Laminado Studio {__version__}")
        self.resize(1500, 920)
        self.p = Proyecto()
        self.sel = (None, None)              # (tipo, clave)
        self.modo_picado = None
        self._pts_picados = []
        self._ocupado = False
        self._pendiente = False
        self.pool = QtCore.QThreadPool.globalInstance()
        self._construir()
        self._refrescar_todo()

    # ================================================================== interfaz
    def _construir(self):
        # centro: pestañas
        self.tabs = QtWidgets.QTabWidget()
        self.vista = Vista3D(self)
        self.vista.punto_picado.connect(self._picado)
        self.tabs.addTab(self.vista.widget, "Vista 3D")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.fig2d = Figure(figsize=(6, 5), dpi=80)
        self.canvas2d = FigureCanvasQTAgg(self.fig2d)
        lay.addWidget(NavigationToolbar2QT(self.canvas2d, w))
        lay.addWidget(self.canvas2d)
        self.tabs.addTab(w, "Patrón plano")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.lab_core = QtWidgets.QLabel("Core sample: activa «Core sample» en la barra y haz clic derecho sobre el molde.")
        self.lab_core.setWordWrap(True)
        self.tabla_core = QtWidgets.QTableWidget(0, 8)
        self.tabla_core.setHorizontalHeaderLabels(["#", "Capa", "Material", "Nominal (°)", "Real (°)",
                                                   "Desviación (°)", "Cizalla (°)", "z (mm)"])
        self.tabla_core.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.lab_core)
        lay.addWidget(self.tabla_core)
        self.tabs.addTab(w, "Core sample")
        self.tabla_res = QtWidgets.QTableWidget(0, 11)
        self.tabla_res.setHorizontalHeaderLabels(["Capa", "Material", "Ori.", "Estrategia", "Cizalla máx",
                                                  "> aviso %", "> límite %", "Cubierto %", "Peso g", "Piezas", "Estado"])
        self.tabla_res.cellDoubleClicked.connect(lambda r, c: self._seleccionar("capa", self.tabla_res.item(r, 0).text()))
        self.tabs.addTab(self.tabla_res, "Tabla de capas")
        self.setCentralWidget(self.tabs)
        # árbol
        self.arbol = QtWidgets.QTreeWidget()
        self.arbol.setHeaderLabel("Laminado")
        self.arbol.itemSelectionChanged.connect(self._sel_arbol)
        self.arbol.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.arbol.customContextMenuRequested.connect(self._menu_arbol)
        d = QtWidgets.QDockWidget("Árbol del laminado", self)
        d.setWidget(self.arbol)
        d.setObjectName("arbol")
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, d)
        # propiedades
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(470)
        d = QtWidgets.QDockWidget("Propiedades", self)
        d.setWidget(self.scroll)
        d.setObjectName("prop")
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d)
        # resumen de producibilidad bajo propiedades
        self.lab_res = QtWidgets.QLabel()
        self.lab_res.setWordWrap(True)
        self.lab_res.setTextFormat(QtCore.Qt.RichText)
        d2 = QtWidgets.QDockWidget("Producibilidad", self)
        d2.setWidget(self.lab_res)
        d2.setObjectName("prod")
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d2)
        self._menus()
        self.barra = self.statusBar()
        self.progreso = QtWidgets.QProgressBar()
        self.progreso.setMaximumWidth(220)
        self.progreso.setRange(0, 0)
        self.progreso.hide()
        self.barra.addPermanentWidget(self.progreso)

    def _accion(self, texto, fn, atajo=None, icono=None, check=False):
        a = QtGui.QAction(texto, self)
        if atajo:
            a.setShortcut(QtGui.QKeySequence(atajo))
        if icono:
            a.setIcon(self.style().standardIcon(icono))
        a.setCheckable(check)
        a.triggered.connect(fn)
        return a

    def _menus(self):
        sp = QtWidgets.QStyle
        mb = self.menuBar()
        m = mb.addMenu("&Archivo")
        self.a_nuevo = self._accion("Nuevo proyecto", self.nuevo, "Ctrl+N", sp.SP_FileIcon)
        self.a_abrir = self._accion("Abrir proyecto…", self.abrir, "Ctrl+O", sp.SP_DialogOpenButton)
        self.a_guardar = self._accion("Guardar", self.guardar, "Ctrl+S", sp.SP_DialogSaveButton)
        a_guardar_c = self._accion("Guardar como…", lambda: self.guardar(True), "Ctrl+Shift+S")
        self.a_importar = self._accion("Importar molde (Ansys .wbpz, .dat, .cdb, .rst, STEP, STL)…",
                                       self.importar, "Ctrl+I", sp.SP_DirOpenIcon)
        a_demo = self._accion("Abrir la silla de demostración", self.demo)
        for a in (self.a_nuevo, self.a_abrir, self.a_guardar, a_guardar_c):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_importar)
        m.addAction(a_demo)
        m.addSeparator()
        m.addAction(self._accion("Salir", self.close, "Ctrl+Q"))
        m = mb.addMenu("&Laminado")
        m.addAction(self._accion("Nueva capa", self.nueva_capa, "Ctrl+L"))
        m.addAction(self._accion("Duplicar capa (girar +45°)", self.duplicar_capa, "Ctrl+D"))
        m.addAction(self._accion("Eliminar capa", self.eliminar))
        m.addAction(self._accion("Subir capa", lambda: self.mover(-1), "Ctrl+Up"))
        m.addAction(self._accion("Bajar capa", lambda: self.mover(1), "Ctrl+Down"))
        m.addSeparator()
        m.addAction(self._accion("Nuevo material", self.nuevo_material))
        m.addAction(self._accion("Nueva roseta", self.nueva_roseta))
        m.addAction(self._accion("Nueva secuencia", self.nueva_secuencia))
        m.addAction(self._accion("Nuevo núcleo", self.nuevo_nucleo))
        m.addAction(self._accion("Asistente de laminado (lista de orientaciones)…", self.asistente))
        m = mb.addMenu("&Drapeado")
        self.a_sim = self._accion("Simular capa seleccionada", self.simular_sel, "F5", sp.SP_MediaPlay)
        self.a_sim_todas = self._accion("Simular todas las capas", self.simular_todas, "Shift+F5", sp.SP_MediaSeekForward)
        m.addAction(self.a_sim)
        m.addAction(self.a_sim_todas)
        self.a_live = self._accion("Producibilidad interactiva (resimular al cambiar)", lambda: None, check=True)
        self.a_live.setChecked(True)
        m.addAction(self.a_live)
        self.a_terminar = self._accion("Terminar picado (dart / curva)", self.terminar_picado, "Return")
        m.addAction(self.a_terminar)
        self.a_cancelar = self._accion("Cancelar picado", self.cancelar_picado, "Esc")
        m.addAction(self.a_cancelar)
        m = mb.addMenu("&Salidas")
        m.addAction(self._accion("Patrones planos DXF…", self.sal_dxf))
        m.addAction(self._accion("Anidado en rollo (DXF)…", self.sal_anidado))
        m.addAction(self._accion("Proyección láser (DXF 3D + CSV)…", self.sal_laser))
        m.addAction(self._accion("Ply book PDF…", self.sal_plybook))
        m.addAction(self._accion("Tabla Excel…", self.sal_excel))
        m.addAction(self._accion("Exportar a Ansys ACP (direcciones de fibra)…", self.sal_ansys))
        m.addSeparator()
        m.addAction(self._accion("Todas las salidas…", self.sal_todo, "Ctrl+E"))
        m = mb.addMenu("A&yuda")
        m.addAction(self._accion("Guía rápida", self.ayuda, "F1"))
        m.addAction(self._accion("Acerca de", self.acerca))
        # barra de herramientas
        tb = self.addToolBar("Principal")
        tb.setObjectName("tb")
        for a in (self.a_nuevo, self.a_abrir, self.a_guardar, self.a_importar):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_sim)
        tb.addAction(self.a_sim_todas)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Color: "))
        self.modo_color = QtWidgets.QComboBox()
        self.modo_color.addItems(["Producibilidad", "Cizalla", "Ángulo de fibra", "Espesores", "Sólo molde"])
        self.modo_color.currentIndexChanged.connect(lambda *_: self._dibujar())
        tb.addWidget(self.modo_color)
        self.ch_red = QtWidgets.QCheckBox("Red")
        self.ch_fib = QtWidgets.QCheckBox("Fibras")
        self.ch_fib.setChecked(True)
        self.ch_todas = QtWidgets.QCheckBox("Capas anteriores")
        for c in (self.ch_red, self.ch_fib, self.ch_todas):
            c.toggled.connect(lambda *_: self._dibujar())
            tb.addWidget(c)
        tb.addSeparator()
        self.a_core = self._accion("Core sample", self._modo_core, check=True)
        tb.addAction(self.a_core)
        self.ch_live = QtWidgets.QCheckBox("Interactivo")
        self.ch_live.setChecked(True)
        self.ch_live.toggled.connect(self.a_live.setChecked)
        tb.addWidget(self.ch_live)
        a = self._accion("Encuadrar", self.vista.encuadrar)
        tb.addAction(a)

    # ================================================================== árbol
    def _item(self, padre, texto, tipo, clave, icono=None):
        it = QtWidgets.QTreeWidgetItem(padre, [texto])
        it.setData(0, QtCore.Qt.UserRole, (tipo, clave))
        if icono is not None:
            it.setIcon(0, self.style().standardIcon(icono))
        return it

    def _refrescar_arbol(self):
        self.arbol.blockSignals(True)
        self.arbol.clear()
        sp = QtWidgets.QStyle
        raiz = self._item(self.arbol, f"Proyecto: {self.p.nombre}", "proyecto", None, sp.SP_DirHomeIcon)
        mo = self._item(raiz, "Molde" + ("" if self.p.molde else " (sin cargar)"), "molde", None, sp.SP_DriveHDIcon)
        if self.p.molde:
            for n, caras in self.p.molde.selecciones.items():
                self._item(mo, f"{n}  ({len(caras)})", "seleccion", n)
        ma = self._item(raiz, "Materiales", "grupo", "materiales", sp.SP_DirIcon)
        for n in self.p.materiales:
            self._item(ma, n, "material", n)
        ro = self._item(raiz, "Rosetas", "grupo", "rosetas", sp.SP_DirIcon)
        for n in self.p.rosetas:
            self._item(ro, n, "roseta", n)
        se = self._item(raiz, "Secuencias", "grupo", "secuencias", sp.SP_DirIcon)
        for n in self.p.secuencias:
            self._item(se, n, "secuencia", n)
        ca = self._item(raiz, "Capas (de molde hacia fuera)", "grupo", "capas", sp.SP_DirIcon)
        for k, c in enumerate(self.p.capas, 1):
            rc = self.p.resultados.get(c.nombre)
            est = rc.resumen.get("estado", "") if rc else ""
            it = self._item(ca, f"{k:02d}  {c.nombre}   {c.orientacion:g}°", "capa", c.nombre)
            col = {"VERDE": "#2e7d32", "AMARILLO": "#b8860b", "ROJO": "#c62828"}.get(est)
            if col:
                it.setForeground(0, QtGui.QBrush(QtGui.QColor(col)))
                it.setToolTip(0, f"Producibilidad: {est}")
            if not c.activa:
                it.setForeground(0, QtGui.QBrush(QtGui.QColor("#999")))
        nu = self._item(raiz, "Núcleos", "grupo", "nucleos", sp.SP_DirIcon)
        for k, n in enumerate(self.p.nucleos):
            self._item(nu, n.nombre, "nucleo", k)
        self.arbol.expandAll()
        # reseleccionar
        if self.sel[0]:
            for it in self.arbol.findItems("*", QtCore.Qt.MatchWildcard | QtCore.Qt.MatchRecursive):
                if it.data(0, QtCore.Qt.UserRole) == self.sel:
                    it.setSelected(True)
                    break
        self.arbol.blockSignals(False)

    def _sel_arbol(self):
        its = self.arbol.selectedItems()
        if not its:
            return
        tipo, clave = its[0].data(0, QtCore.Qt.UserRole)
        self.sel = (tipo, clave)
        self._panel()
        self._dibujar()

    def _seleccionar(self, tipo, clave):
        self.sel = (tipo, clave)
        self._refrescar_arbol()
        self._panel()
        self._dibujar()
        self.tabs.setCurrentIndex(0)

    def _menu_arbol(self, pos):
        it = self.arbol.itemAt(pos)
        if not it:
            return
        tipo, clave = it.data(0, QtCore.Qt.UserRole)
        m = QtWidgets.QMenu(self)
        if tipo in ("capa",) or clave == "capas":
            m.addAction("Nueva capa", self.nueva_capa)
        if tipo == "capa":
            m.addAction("Simular", self.simular_sel)
            m.addAction("Duplicar (+45°)", self.duplicar_capa)
            m.addAction("Subir", lambda: self.mover(-1))
            m.addAction("Bajar", lambda: self.mover(1))
            m.addAction("Eliminar", self.eliminar)
        if tipo == "seleccion":
            m.addAction("Crear capa sobre esta selección", lambda: self.nueva_capa(seleccion=clave))
        if tipo == "material" or clave == "materiales":
            m.addAction("Nuevo material", self.nuevo_material)
            if tipo == "material":
                m.addAction("Eliminar material", self.eliminar)
        if tipo == "roseta" or clave == "rosetas":
            m.addAction("Nueva roseta", self.nueva_roseta)
        if tipo == "secuencia" or clave == "secuencias":
            m.addAction("Nueva secuencia", self.nueva_secuencia)
        if tipo == "nucleo" or clave == "nucleos":
            m.addAction("Nuevo núcleo", self.nuevo_nucleo)
        m.exec(self.arbol.viewport().mapToGlobal(pos))

    # ================================================================ paneles
    def _panel(self):
        tipo, clave = self.sel
        w = None
        try:
            if tipo == "material":
                w = P.PanelMaterial(self.p.materiales[clave])
            elif tipo == "roseta":
                w = P.PanelRoseta(self.p.rosetas[clave])
                w.picar.connect(self._empezar_picado)
            elif tipo == "secuencia":
                w = P.PanelSecuencia(self.p.secuencias[clave], self.p)
            elif tipo == "capa":
                w = P.PanelCapa(self.p.capa(clave), self.p)
                w.picar.connect(self._empezar_picado)
                w.simular.connect(self.simular_sel)
            elif tipo == "nucleo":
                w = P.PanelNucleo(self.p.nucleos[clave], self.p)
            elif tipo in ("proyecto", "molde", None):
                w = P.PanelProyecto(self.p)
        except KeyError:
            w = None
        if w is None:
            w = QtWidgets.QLabel("Selecciona un elemento del árbol.")
        else:
            if hasattr(w, "cambiado"):
                w.cambiado.connect(self._cambio_panel)
        self.panel = w
        self.scroll.setWidget(w)
        self._resumen_prod()

    def _cambio_panel(self, que):
        tipo, clave = self.sel
        if que == "nombre":
            self._renombrar()
            return
        if tipo == "capa":
            capa = self.p.capa(clave)
            if que == "geometria":
                self.p.resultados.pop(capa.nombre, None)
            if self.a_live.isChecked() and que in ("drapeado", "geometria"):
                self.simular_sel()
            else:
                self._dibujar()
        elif tipo in ("roseta", "secuencia", "material") and que == "geometria":
            self.p.resultados.clear()
            self._dibujar()
        elif tipo == "proyecto":
            self._dibujar()

    def _renombrar(self):
        tipo, clave = self.sel
        if tipo == "material":
            m = self.p.materiales.pop(clave)
            self.p.materiales[m.nombre] = m
            for c in self.p.capas:
                if c.material == clave:
                    c.material = m.nombre
            self.sel = ("material", m.nombre)
        elif tipo == "roseta":
            r = self.p.rosetas.pop(clave)
            self.p.rosetas[r.nombre] = r
            for c in self.p.capas:
                if c.roseta == clave:
                    c.roseta = r.nombre
            self.sel = ("roseta", r.nombre)
        elif tipo == "capa":
            for c in self.p.capas:
                if c.nombre != clave and getattr(self.panel, "c", None) is c:
                    pass
            capa = self.panel.c
            if clave in self.p.resultados:
                self.p.resultados[capa.nombre] = self.p.resultados.pop(clave)
            for s in self.p.secuencias.values():
                s.capas = [capa.nombre if x == clave else x for x in s.capas]
            self.sel = ("capa", capa.nombre)
        elif tipo == "secuencia":
            s = self.p.secuencias.pop(clave)
            self.p.secuencias[s.nombre] = s
            self.sel = ("secuencia", s.nombre)
        self._refrescar_arbol()

    def _resumen_prod(self):
        tipo, clave = self.sel
        if tipo != "capa" or clave not in self.p.resultados:
            self.lab_res.setText("<i>Sin simular. Pulsa F5 para drapear la capa.</i>" if tipo == "capa" else "")
            return
        s = self.p.resultados[clave].resumen
        col = {"VERDE": "#2e7d32", "AMARILLO": "#b8860b", "ROJO": "#c62828"}.get(s.get("estado"), "#000000")
        self.lab_res.setText(
            f"<b style='color:{col}; font-size:14px'>● {s.get('estado')}</b><br>"
            f"Cizalla máx: <b>{s['cizalla_max']:.1f}°</b> &nbsp; media: {s['cizalla_media']:.1f}°<br>"
            f"Área &gt; aviso: {s.get('area_aviso_%', 0):.2f} % &nbsp; &gt; límite: {s.get('area_limite_%', 0):.2f} %<br>"
            f"Cubierto por el tejido: {s['cubierto_%']:.1f} %<br>"
            f"Área neta: {s['area_neta_mm2'] / 1e6:.4f} m² &nbsp; corte: {s['area_corte_mm2'] / 1e6:.4f} m²<br>"
            f"Peso: {s['peso_g']:.0f} g &nbsp; Piezas: {s['piezas']} &nbsp; Darts: {s['n_darts']}")

    # ================================================================ dibujo
    def _dibujar(self):
        v = self.vista
        v.quitar_prefijo("capa")
        for n in ("borde", "red", "fibras", "semilla", "darts", "picados", "empalmes", "previas", "espesor"):
            v.quitar_prefijo(n)
        if self.p.molde is None:
            v.quitar("molde")
            v.render()
            self._dibujar_2d()
            return
        modo = self.modo_color.currentText()
        tipo, clave = self.sel
        rc = self.p.resultados.get(clave) if tipo == "capa" else None
        if modo == "Espesores" and self.p.resultados:
            esp, _ = self.p.espesores()
            v.quitar("molde")
            v.malla_coloreada("espesor", self.p.malla, esp, "Espesor (mm)")
        else:
            v.molde(self.p.malla, opacidad=1.0)
        if tipo == "seleccion" and self.p.molde:
            sub, _, _ = self.p.malla.submalla(np.isin(np.arange(self.p.malla.nf), self.p.molde.selecciones[clave]))
            v.lineas("borde", [sub.V[b] for b in sub.boundary_loops()], "#c62828", 4, True)
        if self.ch_todas.isChecked() and tipo == "capa":
            for c in self.p.capas:
                if c.nombre == clave:
                    break
                r = self.p.resultados.get(c.nombre)
                if r:
                    v.lineas(f"previas_{c.nombre}", [r.neto.V[b] for b in r.neto.boundary_loops()], "#555555", 2, True)
        if rc is not None and modo not in ("Sólo molde", "Espesores"):
            capa = self.p.capa(clave)
            mat = self.p.materiales[capa.material]
            r = rc.drapeado
            ref = None
            if modo == "Ángulo de fibra":
                ros = self.p.rosetas.get(capa.roseta) or next(iter(self.p.rosetas.values()))
                ref = ros.direccion(r.malla.face_centroids, r.malla.face_normals, 0.0)
            v.capa(r, {"Producibilidad": "producibilidad", "Cizalla": "cizalla",
                       "Ángulo de fibra": "desviacion"}[modo], mat.cizalla_aviso, mat.cizalla_limite,
                   rc.phi_neto, ref)
            v.lineas("borde", [rc.neto.V[b] + rc.neto.vertex_normals[b] * 0.6 for b in rc.neto.boundary_loops()],
                     "k", 3, True)
            if r.darts:
                v.lineas("darts", [d.puntos for d in r.darts], "#d81b60", 5)
            if rc.empalmes:
                lineas = []
                for pz in rc.piezas:
                    for b in pz.malla.boundary_loops():
                        lineas.append(pz.malla.V[b])
                v.lineas("empalmes", lineas, "#6a1b9a", 2, True)
            if self.ch_fib.isChecked():
                ea, eb = r.fibras()
                n = r.malla.nf
                sel = np.unique(np.linspace(0, n - 1, min(n, 220)).astype(int))
                esc = 0.05 * float(np.ptp(self.p.malla.V, axis=0).max())
                C = r.malla.face_centroids[sel] + r.malla.face_normals[sel] * 0.8
                v.flechas("fibras", C, ea[sel], esc, "#0d47a1")
            if self.ch_red.isChecked() and r.red is not None:
                v.red("red", r.red)
            s = r.malla.V[np.argmin(np.linalg.norm(r.uv, axis=1))]
            v.puntos("semilla", s[None], "#000000", 16)
        elif tipo == "capa":
            capa = self.p.capa(clave)
            if capa.semilla is not None:
                v.puntos("semilla", np.asarray(capa.semilla)[None], "#000000", 16)
            if capa.darts:
                v.lineas("darts", capa.darts, "#d81b60", 5)
        if self._pts_picados:
            v.puntos("picados", np.asarray(self._pts_picados), "#ff6f00", 12)
            if len(self._pts_picados) > 1:
                v.lineas("picados_l", [np.asarray(self._pts_picados)], "#ff6f00", 3)
        v.render()
        self._dibujar_2d()

    def _dibujar_2d(self):
        self.fig2d.clear()
        ax = self.fig2d.add_subplot(111)
        tipo, clave = self.sel
        rc = self.p.resultados.get(clave) if tipo == "capa" else None
        if rc is None:
            ax.text(0.5, 0.5, "Simula una capa para ver su patrón plano", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas2d.draw_idle()
            return
        capa = self.p.capa(clave)
        mat = self.p.materiales[capa.material]
        from ..salidas.patrones import contorno_xy, patrones_de
        for k, pz in enumerate(rc.piezas):
            g = np.abs(pz.cizalla())
            ax.tripcolor(pz.uv[:, 0], pz.uv[:, 1], pz.malla.F, facecolors=g, cmap="RdYlGn_r",
                         vmin=0, vmax=mat.cizalla_limite * 1.2, alpha=0.85)
        for pt in patrones_de(self.p, clave):
            for an in contorno_xy(pt.corte):
                ax.plot(an[:, 0], an[:, 1], "k-", lw=1.2)
            for nn in pt.neto:
                q = np.vstack([nn, nn[:1]])
                ax.plot(q[:, 0], q[:, 1], "--", color="#1b5e20", lw=1)
            for d in pt.darts:
                ax.plot(d[:, 0], d[:, 1], "-", color="#d81b60", lw=1.5)
            c = pt.corte.representative_point() if not pt.corte.is_empty else None
            if c is not None:
                L = 0.15 * np.sqrt(pt.area)
                ax.annotate("", xy=(c.x + L / 2, c.y), xytext=(c.x - L / 2, c.y),
                            arrowprops=dict(arrowstyle="->", color="#c62828", lw=2))
                ax.text(c.x, c.y - 0.05 * np.sqrt(pt.area), f"{pt.etiqueta}  ({pt.area / 1e6:.3f} m²)",
                        ha="center", va="top", fontsize=8)
        ext_v = max(np.ptp(pz.uv[:, 1]) for pz in rc.piezas)
        if ext_v > 0.7 * mat.ancho_rollo:
            vc = 0.5 * (min(pz.uv[:, 1].min() for pz in rc.piezas) + max(pz.uv[:, 1].max() for pz in rc.piezas))
            for sg in (-1, 1):
                ax.axhline(vc + sg * mat.ancho_rollo / 2, color="0.5", ls=":", lw=0.8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("urdimbre = largo del rollo (mm)")
        ax.set_ylabel("trama (mm)")
        ax.set_title(f"{capa.nombre}  {capa.orientacion:g}°  –  corte (—), neto (- -), darts (rosa)", fontsize=9)
        self.fig2d.tight_layout()
        self.canvas2d.draw_idle()

    def _tabla_resumen(self):
        t = self.tabla_res
        t.setRowCount(len(self.p.capas))
        for i, c in enumerate(self.p.capas):
            s = self.p.resultados[c.nombre].resumen if c.nombre in self.p.resultados else {}
            vals = [c.nombre, c.material, f"{c.orientacion:g}", c.estrategia,
                    f"{s.get('cizalla_max', float('nan')):.1f}", f"{s.get('area_aviso_%', float('nan')):.2f}",
                    f"{s.get('area_limite_%', float('nan')):.2f}", f"{s.get('cubierto_%', float('nan')):.1f}",
                    f"{s.get('peso_g', float('nan')):.0f}", str(s.get("piezas", "")), s.get("estado", "sin simular")]
            for j, x in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(x)
                if j == 10:
                    it.setBackground(QtGui.QColor({"VERDE": "#c8e6c9", "AMARILLO": "#fff59d",
                                                   "ROJO": "#ef9a9a"}.get(x, "#ffffff")))
                t.setItem(i, j, it)
        t.resizeColumnsToContents()

    def _refrescar_todo(self):
        self._refrescar_arbol()
        self._panel()
        self._dibujar()
        self._tabla_resumen()
        if self.p.molde is not None:
            self.vista.encuadrar()

    # ================================================================ picado
    def _empezar_picado(self, modo):
        if self.p.molde is None:
            return
        self.modo_picado = modo
        self._pts_picados = []
        self.vista.modo_picado = modo
        self.a_core.setChecked(False)
        txt = {"semilla": "Clic derecho sobre el molde para colocar la semilla.",
               "dart": "Clic derecho: puntos del dart desde el borde hacia dentro. Intro para terminar, Esc para cancelar.",
               "curva": "Clic derecho: puntos de la curva de semilla. Intro para terminar.",
               "empalme": "Clic derecho en el punto por donde pasa el empalme (corte perpendicular a la urdimbre).",
               "roseta": "Clic derecho para situar el origen de la roseta."}[modo]
        self.barra.showMessage(txt)
        self.tabs.setCurrentIndex(0)

    def _modo_core(self, activo):
        self.modo_picado = "core" if activo else None
        self.vista.modo_picado = self.modo_picado
        if activo:
            self.barra.showMessage("Core sample: clic derecho sobre el molde.")

    def _picado(self, P):
        modo = self.modo_picado
        tipo, clave = self.sel
        if modo == "core":
            self._core(P)
            return
        if modo == "roseta" and tipo == "roseta":
            self.p.rosetas[clave].origen = tuple(P)
            self.cancelar_picado()
            self._panel()
            return
        if tipo != "capa":
            return
        capa = self.p.capa(clave)
        if modo == "semilla":
            capa.semilla = tuple(float(x) for x in P)
            self.cancelar_picado()
            if isinstance(self.panel, P.PanelCapa):
                self.panel.refrescar()
            if self.a_live.isChecked():
                self.simular_sel()
            else:
                self._dibujar()
        elif modo in ("dart", "curva"):
            self._pts_picados.append(P)
            self._dibujar()
        elif modo == "empalme":
            rc = self.p.resultados.get(clave)
            if rc is None:
                QtWidgets.QMessageBox.information(self, "Empalme", "Simula primero la capa para conocer la urdimbre.")
                return
            r = rc.drapeado
            _, f, _, _ = Consultor(r.malla).mas_cercano(P[None])
            ea, _ = r.fibras()
            capa.empalmes.append({"tipo": "plano", "punto": tuple(map(float, P)), "normal": tuple(map(float, ea[f[0]])),
                                  "valor": 0.0, "solape": capa.solape_empalme, "nombre": f"Empalme {len(capa.empalmes) + 1}",
                                  "automatico": False})
            self.cancelar_picado()
            self.panel.refrescar()
            self.simular_sel()

    def terminar_picado(self):
        tipo, clave = self.sel
        if self.modo_picado in ("dart", "curva") and tipo == "capa" and len(self._pts_picados) >= 2:
            capa = self.p.capa(clave)
            pts = [tuple(map(float, p)) for p in self._pts_picados]
            if self.modo_picado == "dart":
                capa.darts.append(pts)
            else:
                capa.curva_semilla = pts
                capa.estrategia = "curva"
            self.cancelar_picado()
            self._panel()
            if self.a_live.isChecked():
                self.simular_sel()
        else:
            self.cancelar_picado()

    def cancelar_picado(self):
        self.modo_picado = None
        self.vista.modo_picado = None
        self._pts_picados = []
        self.barra.clearMessage()
        self._dibujar()

    def _core(self, P):
        filas = self.p.core_sample(P)
        t = self.tabla_core
        t.setRowCount(len(filas))
        for i, f in enumerate(filas):
            vals = [str(i + 1), f["capa"], f["material"], f"{f['nominal']:g}", f"{f['real']:.2f}",
                    f"{f['desviacion']:+.2f}", f"{f['cizalla']:.2f}", f"{f['z']:.2f}"]
            for j, x in enumerate(vals):
                t.setItem(i, j, QtWidgets.QTableWidgetItem(x))
        t.resizeColumnsToContents()
        txt = f"Punto ({P[0]:.1f}, {P[1]:.1f}, {P[2]:.1f}) – {len(filas)} capas"
        if filas:
            e = self.p.propiedades_en(P)
            txt += (f" – espesor {e['espesor']:.2f} mm – Ex {e['Ex']:.0f} MPa, Ey {e['Ey']:.0f} MPa, "
                    f"Gxy {e['Gxy']:.0f} MPa, νxy {e['nuxy']:.3f} (CLT con ángulos reales)")
        self.lab_core.setText(txt)
        self.vista.puntos("picados", P[None], "#ff6f00", 14)
        self.vista.render()
        self.tabs.setCurrentIndex(2)

    # ============================================================== simulación
    def _ejecutar(self, fn, al_terminar, *args, texto="Calculando…"):
        self._ocupado = True
        self.progreso.show()
        self.barra.showMessage(texto)
        t = Trabajo(fn, *args)
        t.s.hecho.connect(lambda r: self._fin_trabajo(al_terminar, r))
        t.s.error.connect(self._error_trabajo)
        self._trabajo = t
        self.pool.start(t)

    def _fin_trabajo(self, al_terminar, r):
        self._ocupado = False
        self.progreso.hide()
        self.barra.clearMessage()
        try:
            al_terminar(r)
        finally:
            if self._pendiente:
                self._pendiente = False
                self.simular_sel()

    def _error_trabajo(self, msg):
        self._ocupado = False
        self._pendiente = False
        self.progreso.hide()
        self.barra.showMessage("Error en el cálculo")
        QtWidgets.QMessageBox.warning(self, "Error", msg[:3000])

    def simular_sel(self):
        tipo, clave = self.sel
        if tipo != "capa" or self.p.molde is None:
            return
        if self._ocupado:
            self._pendiente = True
            return
        capa = copy.deepcopy(self.p.capa(clave))

        def fn(c):
            return self.p.simular(self.p.capa(c.nombre))

        def fin(rc):
            self._refrescar_arbol()
            self._resumen_prod()
            self._tabla_resumen()
            self._dibujar()
        self._ejecutar(fn, fin, capa, texto=f"Drapeando {clave}…")

    def simular_todas(self):
        if self.p.molde is None or self._ocupado:
            return

        def fin(_):
            self._refrescar_arbol()
            self._resumen_prod()
            self._tabla_resumen()
            self._dibujar()
        self._ejecutar(lambda: self.p.simular_todas(), fin, texto="Drapeando todas las capas…")

    # ============================================================ proyecto/E-S
    def nuevo(self):
        self.p = Proyecto()
        self.sel = ("proyecto", None)
        self._refrescar_todo()

    def abrir(self, ruta=None):
        if not ruta:
            ruta, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Abrir proyecto", "", "Laminado Studio (*.lam)")
        if not ruta:
            return
        self.p = Proyecto.abrir(ruta)
        self.sel = ("proyecto", None)
        self.setWindowTitle(f"Laminado Studio – {os.path.basename(ruta)}")
        self._refrescar_todo()

    def guardar(self, como=False):
        ruta = self.p.ruta
        if como or not ruta:
            ruta, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar proyecto", f"{self.p.nombre}.lam",
                                                            "Laminado Studio (*.lam)")
        if not ruta:
            return
        self.p.guardar(ruta)
        self.barra.showMessage(f"Guardado en {ruta}", 5000)

    def importar(self, ruta=None):
        if not ruta:
            ruta, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Importar molde", "", FILTRO_MOLDE)
        if not ruta:
            return

        def fin(molde):
            self.p.cargar_molde(molde)
            if not self.p.capas:
                self.nueva_capa()
            self.sel = ("proyecto", None)
            self._refrescar_todo()
            if molde.notas:
                self.barra.showMessage(" ".join(molde.notas), 10000)
        from ..ansys.lector import abrir_molde
        self._ejecutar(abrir_molde, fin, ruta, texto="Leyendo geometría…")

    def demo(self):
        from ..demo.silla import casco_silla, selecciones_silla
        m = casco_silla(6.0)
        p = Proyecto("Casco silla de montar")
        p.cargar_molde(Molde(m, selecciones_silla(m), origen="Silla de demostración (paramétrica)"))
        mat = next(k for k, v in p.materiales.items() if v.tipo == "tejido")
        for k, (ori, cont) in enumerate([(0, Contorno("molde", exceso=20)),
                                         (45, Contorno("molde", desfase=-8, exceso=20)),
                                         (0, Contorno("seleccion", ["ASIENTO"], exceso=15)),
                                         (45, Contorno("seleccion", ["POMO"], exceso=15)),
                                         (45, Contorno("seleccion", ["BORREN"], exceso=15)),
                                         (0, Contorno("molde", desfase=-16, exceso=20))], 1):
            p.capas.append(Capa(f"P{k:03d}", mat, ori, contorno=cont))
        p.capas[0].darts_auto = True
        p.secuencias["Refuerzos"] = Secuencia("Refuerzos", ["P003", "P004", "P005"], 0.0)
        for c in p.capas[2:5]:
            c.secuencia = "Refuerzos"
        self.p = p
        self.sel = ("capa", "P001")
        self._refrescar_todo()

    # ================================================================ edición
    def nueva_capa(self, *_, seleccion=None):
        mat = next(iter(self.p.materiales))
        if self.p.capas:
            mat = self.p.capas[-1].material
        cont = Contorno("seleccion", [seleccion]) if seleccion else Contorno("molde", exceso=20)
        c = Capa(self.p.nombre_libre("P"), mat, 0.0, contorno=cont)
        tipo, clave = self.sel
        if tipo == "capa":
            i = [x.nombre for x in self.p.capas].index(clave) + 1
            self.p.capas.insert(i, c)
        else:
            self.p.capas.append(c)
        self._seleccionar("capa", c.nombre)
        self._tabla_resumen()

    def duplicar_capa(self):
        tipo, clave = self.sel
        if tipo != "capa":
            return
        c = copy.deepcopy(self.p.capa(clave))
        c.nombre = self.p.nombre_libre("P")
        c.orientacion = (c.orientacion + 45 + 90) % 180 - 90
        i = [x.nombre for x in self.p.capas].index(clave) + 1
        self.p.capas.insert(i, c)
        self._seleccionar("capa", c.nombre)

    def mover(self, d):
        tipo, clave = self.sel
        if tipo != "capa":
            return
        i = [x.nombre for x in self.p.capas].index(clave)
        j = i + d
        if 0 <= j < len(self.p.capas):
            self.p.capas[i], self.p.capas[j] = self.p.capas[j], self.p.capas[i]
            self._refrescar_arbol()
            self._tabla_resumen()

    def eliminar(self):
        tipo, clave = self.sel
        if tipo == "capa":
            self.p.capas = [c for c in self.p.capas if c.nombre != clave]
            self.p.resultados.pop(clave, None)
        elif tipo == "material" and not any(c.material == clave for c in self.p.capas):
            self.p.materiales.pop(clave)
        elif tipo == "nucleo":
            self.p.nucleos.pop(clave)
        else:
            return
        self.sel = ("proyecto", None)
        self._refrescar_todo()

    def nuevo_material(self):
        n = "Material nuevo"
        k = 1
        while n in self.p.materiales:
            k += 1
            n = f"Material nuevo {k}"
        self.p.materiales[n] = Material(n)
        self._seleccionar("material", n)

    def nueva_roseta(self):
        n = f"Roseta {len(self.p.rosetas) + 1}"
        self.p.rosetas[n] = Roseta(n)
        self._seleccionar("roseta", n)

    def nueva_secuencia(self):
        n = f"Secuencia {len(self.p.secuencias) + 1}"
        self.p.secuencias[n] = Secuencia(n)
        self._seleccionar("secuencia", n)

    def nuevo_nucleo(self):
        mat = next((k for k, v in self.p.materiales.items() if v.tipo == "nucleo"), next(iter(self.p.materiales)))
        self.p.nucleos.append(Nucleo(f"Núcleo {len(self.p.nucleos) + 1}", mat, Contorno("molde", desfase=-30)))
        self._seleccionar("nucleo", len(self.p.nucleos) - 1)

    def asistente(self):
        txt, ok = QtWidgets.QInputDialog.getText(
            self, "Asistente de laminado",
            "Orientaciones separadas por comas (p.ej. 0, 45, 0, 45, 0). Se crean sobre el contorno de la capa "
            "seleccionada (o todo el molde) con escalonado opcional:")
        if not ok or not txt.strip():
            return
        esc, ok = QtWidgets.QInputDialog.getDouble(self, "Escalonado", "Escalonado entre capas (mm):", 0.0, 0, 200, 1)
        tipo, clave = self.sel
        base = self.p.capa(clave) if tipo == "capa" else None
        mat = base.material if base else next(iter(self.p.materiales))
        cont = copy.deepcopy(base.contorno) if base else Contorno("molde", exceso=20)
        sec = Secuencia(f"Secuencia {len(self.p.secuencias) + 1}", [], esc)
        for x in txt.replace(";", ",").split(","):
            c = Capa(self.p.nombre_libre("P"), mat, float(x), contorno=copy.deepcopy(cont), secuencia=sec.nombre)
            self.p.capas.append(c)
            sec.capas.append(c.nombre)
        self.p.secuencias[sec.nombre] = sec
        self._refrescar_todo()

    # ================================================================= salidas
    def _carpeta(self, titulo):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, titulo)
        return d or None

    def _asegurar_simulado(self):
        faltan = [c.nombre for c in self.p.capas if c.activa and c.nombre not in self.p.resultados]
        if faltan:
            r = QtWidgets.QMessageBox.question(self, "Capas sin simular",
                                               f"Hay {len(faltan)} capas sin simular. ¿Simularlas ahora?")
            if r == QtWidgets.QMessageBox.Yes:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    for n in faltan:
                        self.p.simular(n)
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                self._refrescar_arbol()
                self._tabla_resumen()
        return bool(self.p.resultados)

    def _salida(self, fn, titulo):
        if self.p.molde is None or not self._asegurar_simulado():
            return
        d = self._carpeta(titulo)
        if not d:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            rutas = fn(d)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Error", f"{e}\n\n{traceback.format_exc()[:2000]}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(self, titulo, "Generado:\n" + "\n".join(map(str, rutas))[:3000])

    def sal_dxf(self):
        from ..salidas import dxf
        from ..salidas.patrones import todos_los_patrones
        self._salida(lambda d: dxf.exportar_patrones(todos_los_patrones(self.p), os.path.join(d, "patrones")),
                     "Patrones DXF")

    def sal_anidado(self):
        from ..salidas import anidado, dxf
        from ..salidas.patrones import todos_los_patrones

        def fn(d):
            a = anidado.anidar(todos_los_patrones(self.p), self.p)
            r = dxf.exportar_anidado(a, os.path.join(d, "anidado.dxf"))
            return [r] + [f"{x['material']}: {x['largo_mm']:.0f} mm, {x['aprovechamiento_%']:.1f} %" for x in a.resumen()]
        self._salida(fn, "Anidado")

    def sal_laser(self):
        from ..salidas import dxf
        self._salida(lambda d: [dxf.exportar_laser(self.p, os.path.join(d, "laser.dxf"), os.path.join(d, "laser.csv")),
                                os.path.join(d, "laser.csv")], "Proyección láser")

    def sal_plybook(self):
        from ..salidas import anidado, informes
        from ..salidas.patrones import todos_los_patrones

        def fn(d):
            try:
                a = anidado.anidar(todos_los_patrones(self.p), self.p)
            except Exception:  # noqa: BLE001
                a = None
            return [informes.ply_book(self.p, os.path.join(d, f"plybook_{self.p.nombre}.pdf"), a)]
        self._salida(fn, "Ply book")

    def sal_excel(self):
        from ..salidas import informes
        self._salida(lambda d: [informes.excel(self.p, os.path.join(d, f"laminado_{self.p.nombre}.xlsx"))], "Excel")

    def sal_ansys(self):
        from ..salidas import informes
        self._salida(lambda d: informes.exportar_ansys(self.p, os.path.join(d, "ansys")), "Exportar a Ansys")

    def sal_todo(self):
        from ..salidas.todo import generar_todo
        self._salida(lambda d: generar_todo(self.p, d), "Todas las salidas")

    # =================================================================== ayuda
    def ayuda(self):
        QtWidgets.QMessageBox.information(self, "Guía rápida", GUIA)

    def acerca(self):
        QtWidgets.QMessageBox.about(self, "Laminado Studio",
                                    f"<b>Laminado Studio {__version__}</b><br>Diseño y fabricación de laminados "
                                    "de composite: drapeado, producibilidad, darts, empalmes, patrones planos y "
                                    "ply book.<br>Geometría desde Ansys Workbench, MAPDL, STEP y STL.")



GUIA = """1. Archivo › Importar molde: elige tu proyecto de Ansys (.wbpz) o el ds.dat / .cdb de Mechanical.
   Las selecciones con nombre (Named Selections) aparecen en el árbol bajo «Molde».
2. Revisa los materiales (ancho de rollo, cizalla de aviso y límite) y la roseta (dirección de 0°).
3. Crea capas (Ctrl+L) o usa Laminado › Asistente con la lista de orientaciones.
4. En cada capa: contorno (molde, selección o polígono), desfase y exceso de recorte, estrategia de drapeado
   y punto semilla (botón «Picar» + clic derecho en el molde).
5. F5 simula la capa. Con «Interactivo» activado, cada cambio vuelve a simular.
   Verde: correcto · amarillo: supera la cizalla de aviso · rojo: supera el límite o no se alcanza.
6. Darts: «Añadir dart» y clic derecho desde el borde hacia dentro (Intro para terminar), o activa
   «Darts automáticos». Empalmes: automáticos por ancho de rollo o «Añadir empalme» (clic derecho).
7. Core sample: botón de la barra + clic derecho → capas, ángulos reales y propiedades CLT.
8. Salidas › Todas las salidas: DXF de corte, anidado, láser, ply book PDF, Excel y tablas para Ansys ACP."""
