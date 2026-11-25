"""Archivos reales escritos por Ansys (ver tests/datos_ansys/LEEME.md)."""
import os

import numpy as np
import pytest

from laminado.ansys.lector import abrir_molde
from laminado.modelo.proyecto import Capa, Contorno, Proyecto

D = os.path.join(os.path.dirname(__file__), "datos_ansys")


def test_cdb_mapdl_2022():
    m = abrir_molde(os.path.join(D, "minimal_plate.cdb"))
    assert abs(m.malla.area - 400.0) < 1e-6           # coincide con ansys-mapdl-reader
    assert {"FRONT", "MIDDLE", "TAIL"} <= set(m.selecciones)


def test_dsdat_workbench():
    m = abrir_molde(os.path.join(D, "flat_plate_input.dat"))
    assert m.malla.nf == 128 and abs(m.malla.area - 100.0) < 1e-6
    assert any("convertidas a mm" in n for n in m.notas)


def test_wbpz_con_acp_hdf5():
    m = abrir_molde(os.path.join(D, "fatigue_example.wbpz"))
    assert m.malla.nv == 85 and m.malla.nf == 128
    assert abs(m.malla.area - 400.0 * 100.0) < 1e-6
    p = Proyecto()
    p.cargar_molde(m)
    assert any(k.startswith("ACP:") for k in p.rosetas)


@pytest.fixture(scope="module")
def class40():
    p = Proyecto("class40")
    p.cargar_molde(os.path.join(D, "class40.cdb"))
    p.h = 25.0
    return p


def test_class40_ensamblaje(class40):
    m = class40.malla
    assert (m.edge_count > 2).sum() > 100              # uniones en T
    assert m.componentes()[0] == 6                      # paneles separados por las uniones
    sel = class40.molde.selecciones["BULKHEAD_ALL"]
    n = m.face_normals[sel]
    assert np.allclose(np.abs(n @ n[0]), 1) and (n @ n[0] > 0).all()   # mamparo bien orientado


@pytest.mark.parametrize("sel, ori", [("BULKHEAD_ALL", 0), ("HULL_PORT", 0), ("DECK", 45)])
def test_class40_drapeado(class40, sel, ori):
    c = Capa("C", "Vidrio 7781 satén 8H (prepreg epoxi)", ori, contorno=Contorno("seleccion", [sel], exceso=20))
    class40.capas = [c]
    s = class40.simular(c).resumen
    assert s["cubierto_%"] > 99.5
    assert s["area_limite_%"] < 0.5
    if sel == "BULKHEAD_ALL":
        assert s["cizalla_max"] < 0.5                  # pieza plana: sin cizalla
