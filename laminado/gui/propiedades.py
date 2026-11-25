"""Paneles de propiedades (formularios) para materiales, rosetas, capas, secuencias y núcleos."""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..drapeado.red import ESTRATEGIAS

ESTR_CLAVES = list(ESTRATEGIAS)


def _num(valor, minimo=-1e6, maximo=1e6, dec=2, sufijo=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(minimo, maximo)
    w.setDecimals(dec)
    w.setValue(float(valor))
    w.setKeyboardTracking(False)
    if sufijo:
        w.setSuffix(" " + sufijo)
    return w


def _vec(v):
    w = QtWidgets.QLineEdit(", ".join(f"{x:g}" for x in v))
    return w


def _leer_vec(w):
    return tuple(float(x) for x in w.text().replace(";", ",").split(",")[:3])


class PanelBase(QtWidgets.QWidget):
    cambiado = QtCore.Signal(str)          # qué ha cambiado ('geometria', 'datos', 'nombre')

    def __init__(self, padre=None):
        super().__init__(padre)
        self.form = QtWidgets.QFormLayout(self)
        self.form.setLabelAlignment(QtCore.Qt.AlignRight)
        self._cargando = False

    def _conectar(self, w, que="datos"):
        def avisar(*_):
            if not self._cargando:
                self.aplicar()
                self.cambiado.emit(que)
        for sig in ("valueChanged", "currentIndexChanged", "editingFinished", "toggled", "itemSelectionChanged"):
            if hasattr(w, sig):
                getattr(w, sig).connect(avisar)
                break
        return w

    def aplicar(self):
        pass


class PanelMaterial(PanelBase):
    def __init__(self, mat, padre=None):
        super().__init__(padre)
        self.m = mat
        self.w = {}
        self._cargando = True
        self.w["nombre"] = self._conectar(QtWidgets.QLineEdit(mat.nombre), "nombre")
        tipo = QtWidgets.QComboBox()
        tipo.addItems(["tejido", "ud", "nucleo"])
        tipo.setCurrentText(mat.tipo)
        self.w["tipo"] = self._conectar(tipo)
        campos = [("espesor", "Espesor curado", "mm", 3), ("peso_area", "Peso del preimpregnado", "g/m²", 1),
                  ("gramaje_fibra", "Gramaje de fibra", "g/m²", 1), ("ancho_rollo", "Ancho de rollo", "mm", 0),
                  ("cizalla_aviso", "Cizalla de aviso", "°", 1), ("cizalla_limite", "Cizalla límite (bloqueo)", "°", 1),
                  ("E1", "E1", "MPa", 0), ("E2", "E2", "MPa", 0), ("G12", "G12", "MPa", 0), ("nu12", "ν12", "", 3),
                  ("Xt", "Xt", "MPa", 0), ("Xc", "Xc", "MPa", 0), ("Yt", "Yt", "MPa", 0), ("Yc", "Yc", "MPa", 0),
                  ("S", "S", "MPa", 0), ("coste_m2", "Coste", "€/m²", 2)]
        self.form.addRow("Nombre", self.w["nombre"])
        self.form.addRow("Tipo", tipo)
        for clave, etiqueta, uds, dec in campos:
            self.w[clave] = self._conectar(_num(getattr(mat, clave), 0, 1e7, dec, uds))
            self.form.addRow(etiqueta, self.w[clave])
        self.w["simetrico"] = self._conectar(QtWidgets.QCheckBox("Tejido equilibrado (giro de 90° al anidar)"))
        self.w["simetrico"].setChecked(mat.simetrico)
        self.form.addRow("", self.w["simetrico"])
        self._cargando = False

    def aplicar(self):
        m = self.m
        m.nombre = self.w["nombre"].text().strip() or m.nombre
        m.tipo = self.w["tipo"].currentText()
        for k, w in self.w.items():
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                setattr(m, k, w.value())
        m.simetrico = self.w["simetrico"].isChecked()


class PanelRoseta(PanelBase):
    picar = QtCore.Signal(str)

    def __init__(self, ros, padre=None):
        super().__init__(padre)
        self.r = ros
        self._cargando = True
        self.nombre = self._conectar(QtWidgets.QLineEdit(ros.nombre), "nombre")
        self.origen = self._conectar(_vec(ros.origen), "geometria")
        self.eje = self._conectar(_vec(ros.eje_x), "geometria")
        self.normal = self._conectar(_vec(ros.normal), "geometria")
        b = QtWidgets.QPushButton("Picar origen en el molde (clic derecho)")
        b.clicked.connect(lambda: self.picar.emit("roseta"))
        for et, w in (("Nombre", self.nombre), ("Origen", self.origen), ("Eje 0°", self.eje),
                      ("Normal (giro +)", self.normal)):
            self.form.addRow(et, w)
        self.form.addRow("", b)
        self._cargando = False

    def aplicar(self):
        try:
            self.r.nombre = self.nombre.text().strip() or self.r.nombre
            self.r.origen = _leer_vec(self.origen)
            self.r.eje_x = _leer_vec(self.eje)
            self.r.normal = _leer_vec(self.normal)
        except ValueError:
            pass


class PanelSecuencia(PanelBase):
    def __init__(self, sec, proyecto, padre=None):
        super().__init__(padre)
        self.s = sec
        self._cargando = True
        self.nombre = self._conectar(QtWidgets.QLineEdit(sec.nombre), "nombre")
        self.esc = self._conectar(_num(sec.escalonado, -500, 500, 1, "mm"), "geometria")
        self.form.addRow("Nombre", self.nombre)
        self.form.addRow("Escalonado entre capas", self.esc)
        self.lista = QtWidgets.QListWidget()
        self.lista.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for c in proyecto.capas:
            it = QtWidgets.QListWidgetItem(c.nombre)
            self.lista.addItem(it)
            it.setSelected(c.nombre in sec.capas)
        self._conectar(self.lista, "geometria")
        self.form.addRow("Capas (en orden)", self.lista)
        self.proyecto = proyecto
        self._cargando = False

    def aplicar(self):
        self.s.nombre = self.nombre.text().strip() or self.s.nombre
        self.s.escalonado = self.esc.value()
        sel = [self.lista.item(i).text() for i in range(self.lista.count()) if self.lista.item(i).isSelected()]
        self.s.capas = sel
        for c in self.proyecto.capas:
            if c.nombre in sel:
                c.secuencia = self.s.nombre
            elif c.secuencia == self.s.nombre:
                c.secuencia = ""


class EditorContorno(QtWidgets.QGroupBox):
    cambiado = QtCore.Signal()

    def __init__(self, contorno, proyecto, titulo="Contorno", padre=None):
        super().__init__(titulo, padre)
        self.c = contorno
        self.p = proyecto
        f = QtWidgets.QFormLayout(self)
        self.tipo = QtWidgets.QComboBox()
        self.tipo.addItems(["molde", "seleccion", "poligono"])
        self.tipo.setItemText(0, "molde")
        self.tipo.setCurrentText(contorno.tipo)
        f.addRow("Tipo", self.tipo)
        self.sel = QtWidgets.QListWidget()
        self.sel.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.sel.setMaximumHeight(110)
        if proyecto.molde is not None:
            for n in proyecto.molde.selecciones:
                it = QtWidgets.QListWidgetItem(n)
                self.sel.addItem(it)
                it.setSelected(n in contorno.selecciones)
        f.addRow("Selecciones Ansys", self.sel)
        self.puntos = QtWidgets.QPlainTextEdit("\n".join(", ".join(f"{x:g}" for x in p) for p in contorno.puntos))
        self.puntos.setPlaceholderText("x, y  (una línea por vértice, plano XY de la roseta)")
        self.puntos.setMaximumHeight(90)
        f.addRow("Polígono", self.puntos)
        self.desfase = _num(contorno.desfase, -500, 500, 1, "mm")
        self.exceso = _num(contorno.exceso, 0, 500, 1, "mm")
        f.addRow("Desfase del borde", self.desfase)
        f.addRow("Exceso de recorte", self.exceso)
        self.exceso_borde = _num(contorno.exceso_borde, 0, 200, 1, "mm")
        self.exceso_borde.setToolTip("Tejido que la selección sigue más allá de los bordes libres del molde")
        f.addRow("Exceso en borde de pieza", self.exceso_borde)
        self.tipo.currentIndexChanged.connect(self._cambio)
        self.sel.itemSelectionChanged.connect(self._cambio)
        self.puntos.textChanged.connect(self._cambio_pts)
        self.desfase.valueChanged.connect(self._cambio)
        self.exceso.valueChanged.connect(self._cambio)
        self.exceso_borde.valueChanged.connect(self._cambio)
        self._visibles()

    def _visibles(self):
        t = self.tipo.currentText()
        self.sel.setEnabled(t == "seleccion")
        self.puntos.setEnabled(t == "poligono")

    def _cambio_pts(self):
        try:
            pts = [tuple(float(x) for x in ln.replace(";", ",").split(",")[:3])
                   for ln in self.puntos.toPlainText().splitlines() if ln.strip()]
            if len(pts) >= 3:
                self.c.puntos = pts
                self.cambiado.emit()
        except ValueError:
            pass

    def _cambio(self, *_):
        self.c.tipo = self.tipo.currentText()
        self.c.selecciones = [self.sel.item(i).text() for i in range(self.sel.count()) if self.sel.item(i).isSelected()]
        self.c.desfase = self.desfase.value()
        self.c.exceso = self.exceso.value()
        self.c.exceso_borde = self.exceso_borde.value()
        self._visibles()
        self.cambiado.emit()


class PanelCapa(PanelBase):
    picar = QtCore.Signal(str)              # 'semilla' | 'dart' | 'empalme' | 'curva'
    simular = QtCore.Signal()

    def __init__(self, capa, proyecto, padre=None):
        super().__init__(padre)
        self.c = capa
        self.p = proyecto
        self._cargando = True
        self.nombre = self._conectar(QtWidgets.QLineEdit(capa.nombre), "nombre")
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(proyecto.materiales))
        self.material.setCurrentText(capa.material)
        self._conectar(self.material, "geometria")
        self.ori = self._conectar(_num(capa.orientacion, -360, 360, 1, "°"), "drapeado")
        self.roseta = QtWidgets.QComboBox()
        self.roseta.addItems(list(proyecto.rosetas))
        self.roseta.setCurrentText(capa.roseta)
        self._conectar(self.roseta, "drapeado")
        self.estr = QtWidgets.QComboBox()
        for k in ESTR_CLAVES:
            self.estr.addItem(ESTRATEGIAS[k], k)
        self.estr.setCurrentIndex(ESTR_CLAVES.index(capa.estrategia) if capa.estrategia in ESTR_CLAVES else 0)
        self._conectar(self.estr, "drapeado")
        self.h = self._conectar(_num(capa.h, 0, 100, 1, "mm"), "drapeado")
        self.h.setSpecialValueText("del proyecto")
        self.form.addRow("Nombre", self.nombre)
        self.form.addRow("Material", self.material)
        self.form.addRow("Orientación", self.ori)
        self.form.addRow("Roseta", self.roseta)
        self.form.addRow("Estrategia de drapeado", self.estr)
        self.form.addRow("Celda de la red", self.h)
        # semilla
        fila = QtWidgets.QHBoxLayout()
        self.lab_semilla = QtWidgets.QLabel(self._txt_semilla())
        b1 = QtWidgets.QPushButton("Picar")
        b1.setToolTip("Clic derecho sobre el molde para colocar la semilla")
        b1.clicked.connect(lambda: self.picar.emit("semilla"))
        b2 = QtWidgets.QPushButton("Auto")
        b2.clicked.connect(self._semilla_auto)
        fila.addWidget(self.lab_semilla, 1)
        fila.addWidget(b1)
        fila.addWidget(b2)
        self.form.addRow("Punto semilla", fila)
        b = QtWidgets.QPushButton("Definir curva de semilla (picar puntos)")
        b.clicked.connect(lambda: self.picar.emit("curva"))
        self.form.addRow("", b)
        # contorno
        self.cont = EditorContorno(capa.contorno, proyecto)
        self.cont.cambiado.connect(lambda: (self.cambiado.emit("geometria")))
        self.form.addRow(self.cont)
        # darts
        g = QtWidgets.QGroupBox("Darts")
        fl = QtWidgets.QFormLayout(g)
        self.lab_darts = QtWidgets.QLabel(f"{len(capa.darts)} manuales")
        bd = QtWidgets.QPushButton("Añadir dart (picar del borde hacia dentro)")
        bd.clicked.connect(lambda: self.picar.emit("dart"))
        bb = QtWidgets.QPushButton("Borrar darts manuales")
        bb.clicked.connect(self._borrar_darts)
        self.darts_auto = self._conectar(QtWidgets.QCheckBox("Darts automáticos donde la cizalla supera el límite\n"
                                                             "(sólo los que abren en V; en zonas en silla usa empalmes)"),
                                         "drapeado")
        self.darts_auto.setChecked(capa.darts_auto)
        fl.addRow(self.lab_darts, bd)
        fl.addRow("", bb)
        fl.addRow("", self.darts_auto)
        self.form.addRow(g)
        # empalmes
        g = QtWidgets.QGroupBox("Empalmes")
        fl = QtWidgets.QFormLayout(g)
        self.emp_auto = self._conectar(QtWidgets.QCheckBox("Automáticos por ancho de rollo"), "drapeado")
        self.emp_auto.setChecked(capa.empalme_auto)
        self.solape = self._conectar(_num(capa.solape_empalme, -100, 200, 1, "mm"), "drapeado")
        self.lab_emp = QtWidgets.QLabel(f"{len(capa.empalmes)} manuales")
        be = QtWidgets.QPushButton("Añadir empalme (picar punto: corte ⟂ urdimbre)")
        be.clicked.connect(lambda: self.picar.emit("empalme"))
        be2 = QtWidgets.QPushButton("Borrar empalmes manuales")
        be2.clicked.connect(self._borrar_emp)
        self.emp_ciz = self._conectar(QtWidgets.QCheckBox("Automáticos donde la cizalla supera el límite\n"
                                                          "(piezas drapeadas por separado)"), "drapeado")
        self.emp_ciz.setChecked(capa.empalmes_auto_cizalla)
        fl.addRow("", self.emp_auto)
        fl.addRow("", self.emp_ciz)
        fl.addRow("Solape (− = hueco)", self.solape)
        fl.addRow(self.lab_emp, be)
        fl.addRow("", be2)
        self.form.addRow(g)
        self.notas = self._conectar(QtWidgets.QLineEdit(capa.notas))
        self.form.addRow("Notas de laminado", self.notas)
        self.activa = self._conectar(QtWidgets.QCheckBox("Activa"), "geometria")
        self.activa.setChecked(capa.activa)
        self.form.addRow("", self.activa)
        bs = QtWidgets.QPushButton("Simular capa  (F5)")
        bs.setStyleSheet("font-weight: bold; padding: 6px;")
        bs.clicked.connect(self.simular.emit)
        self.form.addRow(bs)
        self._cargando = False

    def _txt_semilla(self):
        s = self.c.semilla
        return "automática (centro de la capa)" if s is None else f"({s[0]:.1f}, {s[1]:.1f}, {s[2]:.1f})"

    def refrescar(self):
        self.lab_semilla.setText(self._txt_semilla())
        self.lab_darts.setText(f"{len(self.c.darts)} manuales")
        self.lab_emp.setText(f"{len(self.c.empalmes)} manuales")

    def _semilla_auto(self):
        self.c.semilla = None
        self.refrescar()
        self.cambiado.emit("drapeado")

    def _borrar_darts(self):
        self.c.darts = []
        self.refrescar()
        self.cambiado.emit("drapeado")

    def _borrar_emp(self):
        self.c.empalmes = []
        self.refrescar()
        self.cambiado.emit("drapeado")

    def aplicar(self):
        c = self.c
        c.nombre = self.nombre.text().strip() or c.nombre
        c.material = self.material.currentText()
        c.orientacion = self.ori.value()
        c.roseta = self.roseta.currentText()
        c.estrategia = self.estr.currentData()
        c.h = self.h.value()
        c.darts_auto = self.darts_auto.isChecked()
        c.empalme_auto = self.emp_auto.isChecked()
        c.empalmes_auto_cizalla = self.emp_ciz.isChecked()
        c.solape_empalme = self.solape.value()
        c.notas = self.notas.text()
        c.activa = self.activa.isChecked()


class PanelNucleo(PanelBase):
    def __init__(self, nucleo, proyecto, padre=None):
        super().__init__(padre)
        self.n = nucleo
        self._cargando = True
        self.nombre = self._conectar(QtWidgets.QLineEdit(nucleo.nombre), "nombre")
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(proyecto.materiales))
        self.material.setCurrentText(nucleo.material)
        self._conectar(self.material)
        self.esp = self._conectar(_num(nucleo.espesor, 0, 200, 2, "mm"), "geometria")
        self.chaflan = self._conectar(_num(nucleo.chaflan, 0, 90, 1, "°"))
        self.despues = QtWidgets.QComboBox()
        self.despues.addItems([""] + [c.nombre for c in proyecto.capas])
        self.despues.setCurrentText(nucleo.despues_de)
        self._conectar(self.despues)
        for et, w in (("Nombre", self.nombre), ("Material", self.material), ("Espesor", self.esp),
                      ("Chaflán", self.chaflan), ("Colocar tras la capa", self.despues)):
            self.form.addRow(et, w)
        self.cont = EditorContorno(nucleo.contorno, proyecto)
        self.cont.cambiado.connect(lambda: self.cambiado.emit("geometria"))
        self.form.addRow(self.cont)
        self._cargando = False

    def aplicar(self):
        n = self.n
        n.nombre = self.nombre.text().strip() or n.nombre
        n.material = self.material.currentText()
        n.espesor = self.esp.value()
        n.chaflan = self.chaflan.value()
        n.despues_de = self.despues.currentText()


class PanelProyecto(PanelBase):
    def __init__(self, proyecto, padre=None):
        super().__init__(padre)
        self.p = proyecto
        self._cargando = True
        self.nombre = self._conectar(QtWidgets.QLineEdit(proyecto.nombre), "nombre")
        self.h = self._conectar(_num(proyecto.h, 1, 50, 1, "mm"), "drapeado")
        self.ext = self._conectar(_num(proyecto.extension, 0, 500, 0, "mm"), "geometria")
        self.espesor = self._conectar(QtWidgets.QCheckBox("Drapear cada capa sobre las anteriores (espesor real)"),
                                      "drapeado")
        self.espesor.setChecked(bool(proyecto.drapeado_con_espesor))
        self.espesor.setToolTip("Cada capa se drapea sobre la superficie desplazada por el espesor de las capas y "
                                "núcleos que tiene debajo: las capas exteriores desarrollan más y los patrones "
                                "planos salen a su medida real.")
        self.sentido = QtWidgets.QComboBox()
        self.sentido.addItems(["Según la normal del molde", "Contra la normal del molde"])
        self.sentido.setCurrentIndex(0 if proyecto.sentido_laminado >= 0 else 1)
        self.sentido = self._conectar(self.sentido, "drapeado")
        self.rampa = self._conectar(_num(proyecto.rampa, 1, 200, 0, "× espesor"), "drapeado")
        self.rampa.setToolTip("Longitud de la rampa en el borde de cada capa (terminación), en múltiplos de su espesor")
        self.form.addRow("Nombre", self.nombre)
        self.form.addRow("Celda de red por defecto", self.h)
        self.form.addRow("Superficie extendida", self.ext)
        self.form.addRow("", self.espesor)
        self.form.addRow("Sentido de crecimiento del laminado", self.sentido)
        self.form.addRow("Rampa de terminación de capa", self.rampa)
        if proyecto.molde is not None:
            m = proyecto.malla
            info = (f"{proyecto.molde.origen}\n{m.nv} nodos, {m.nf} triángulos, área {m.area / 1e6:.4f} m²\n"
                    f"Caja: {np.ptp(m.V, axis=0).round(1)} mm")
            lab = QtWidgets.QLabel(info)
            lab.setWordWrap(True)
            self.form.addRow("Molde", lab)
            for n in proyecto.molde.notas:
                self.form.addRow("", QtWidgets.QLabel(n))
        self._cargando = False

    def aplicar(self):
        self.p.nombre = self.nombre.text().strip() or self.p.nombre
        self.p.h = self.h.value()
        self.p.drapeado_con_espesor = self.espesor.isChecked()
        self.p.sentido_laminado = 1.0 if self.sentido.currentIndex() == 0 else -1.0
        self.p.rampa = self.rampa.value()
        if self.ext.value() != self.p.extension:
            self.p.extension = self.ext.value()
            self.p._ext = None
            self.p._superficies = {}
