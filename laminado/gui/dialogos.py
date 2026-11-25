"""Diálogos auxiliares de la ventana principal."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class DialogoCobertura(QtWidgets.QDialog):
    """Opciones de la cobertura completa automática."""

    def __init__(self, padre, proyecto):
        super().__init__(padre)
        self.p = proyecto
        self.setWindowTitle("Cobertura completa automática")
        f = QtWidgets.QFormLayout(self)
        expl = QtWidgets.QLabel(
            "Cubre toda la zona elegida con parches de tejido producibles (cada uno drapeado desde su "
            "propio centro, con la cizalla dentro del límite elegido) y solapados entre sí, hasta no dejar "
            "huecos. Se crea una secuencia por orientación. En piezas grandes puede tardar varios minutos.")
        expl.setWordWrap(True)
        f.addRow(expl)
        self.mat = QtWidgets.QComboBox()
        self.mat.addItems([k for k, v in proyecto.materiales.items() if v.tipo != "nucleo"])
        f.addRow("Material", self.mat)
        self.ori = QtWidgets.QLineEdit("0, 45, -45")
        self.ori.setToolTip("Una capa de cobertura completa por cada orientación")
        f.addRow("Orientaciones (°)", self.ori)
        self.ref = QtWidgets.QComboBox()
        self.ref.addItem("Eje local de cada tubo / zona", "eje")
        for n in proyecto.rosetas:
            if not n.startswith("Eje "):
                self.ref.addItem(f"Roseta: {n}", n)
        f.addRow("0° respecto a", self.ref)
        self.zona = QtWidgets.QListWidget()
        self.zona.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for n in proyecto.molde.selecciones:
            if not n.startswith("_"):
                self.zona.addItem(n)
        self.zona.setMaximumHeight(140)
        f.addRow("Zona (ninguna = molde entero)", self.zona)
        self.solape = QtWidgets.QDoubleSpinBox()
        self.solape.setRange(0, 100)
        self.solape.setValue(15)
        self.solape.setSuffix(" mm")
        f.addRow("Solape entre parches", self.solape)
        self.radio = QtWidgets.QDoubleSpinBox()
        self.radio.setRange(20, 5000)
        self.radio.setValue(320)
        self.radio.setSuffix(" mm")
        f.addRow("Tamaño máximo de parche (radio)", self.radio)
        self.crit = QtWidgets.QComboBox()
        self.crit.addItem("Cizalla ≤ aviso (todo VERDE)", "aviso")
        self.crit.addItem("Cizalla ≤ límite (admite AMARILLO, menos parches)", "limite")
        f.addRow("Criterio", self.crit)
        self.pref = QtWidgets.QLineEdit("C")
        f.addRow("Prefijo de las capas", self.pref)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.resize(520, 560)

    def opciones(self):
        from ..modelo.cobertura import OpcionesCobertura
        try:
            oris = [float(x) for x in self.ori.text().replace(";", ",").split(",") if x.strip()]
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Orientaciones", "Escribe números separados por comas.")
            return []
        sels = [i.text() for i in self.zona.selectedItems()]
        pref = self.pref.text().strip() or "C"
        out = []
        for o in oris:
            out.append(OpcionesCobertura(
                material=self.mat.currentText(), orientacion=o, referencia=self.ref.currentData(),
                solape=self.solape.value(), radio_max=self.radio.value(), criterio=self.crit.currentData(),
                selecciones=sels, prefijo=f"{pref}{o:g}_", secuencia=f"Cobertura {o:g}°"))
        return out


del QtCore
