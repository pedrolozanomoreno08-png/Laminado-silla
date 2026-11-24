import os
import zipfile

import numpy as np

from laminado.ansys.lector import Molde, abrir_molde, escribir_mapdl
from laminado.demo.silla import casco_silla, selecciones_silla
from laminado.modelo import clt
from laminado.modelo.materiales import biblioteca
from laminado.modelo.proyecto import Capa, Contorno, Proyecto, Secuencia


def _proyecto():
    m = casco_silla(10.0)
    p = Proyecto("Prueba")
    p.cargar_molde(Molde(m, selecciones_silla(m), origen="demo"))
    p.h = 10.0
    mat = "Carbono sarga 2x2 200 g/m² (prepreg epoxi)"
    p.capas = [Capa("P001", mat, 0, contorno=Contorno("molde", exceso=15)),
               Capa("P002", mat, 45, contorno=Contorno("seleccion", ["ASIENTO"], exceso=10)),
               Capa("P003", mat, 0, contorno=Contorno("seleccion", ["ASIENTO"], exceso=10), secuencia="S")]
    p.secuencias["S"] = Secuencia("S", ["P002", "P003"], 10.0)
    return p


def test_lector_mapdl_y_wbpz(tmp_path):
    m = casco_silla(12.0)
    sel = selecciones_silla(m)
    dat = tmp_path / "ds.dat"
    escribir_mapdl(str(dat), m, sel)
    mo = abrir_molde(str(dat))
    assert mo.malla.nf == m.nf and np.allclose(mo.malla.V, m.V, atol=1e-6)
    assert set(mo.selecciones) == set(sel)
    w = tmp_path / "proyecto.wbpz"
    with zipfile.ZipFile(w, "w") as z:
        z.write(dat, "proyecto_files/dp0/SYS/MECH/ds.dat")
    mw = abrir_molde(str(w))
    assert mw.malla.nf == m.nf


def test_proyecto_completo(tmp_path):
    p = _proyecto()
    p.simular_todas()
    r1, r2, r3 = (p.resultados[n] for n in ("P001", "P002", "P003"))
    assert abs(r1.neto.area - p.malla.area) / p.malla.area < 0.01
    assert r3.neto.area < r2.neto.area        # escalonado de la secuencia
    assert r1.resumen["area_corte_mm2"] > r1.resumen["area_neta_mm2"]
    P = p.malla.V[np.argmin(np.abs(p.malla.V[:, 0]) + np.abs(p.malla.V[:, 1]))]
    cs = p.core_sample(P)
    assert [f["capa"] for f in cs] == ["P001", "P002", "P003"]
    assert all(abs(f["desviacion"]) < 1.0 for f in cs)
    ruta = tmp_path / "x.lam"
    p.guardar(str(ruta))
    q = Proyecto.abrir(str(ruta))
    assert [c.nombre for c in q.capas] == ["P001", "P002", "P003"] and q.malla.nf == p.malla.nf
    from laminado.salidas.todo import generar_todo
    rutas = generar_todo(p, str(tmp_path / "salidas"))
    for r in rutas:
        if isinstance(r, str) and os.path.sep in r:
            assert os.path.exists(r), r


def test_clt_cuasi_isotropo():
    mat = biblioteca()["Carbono UD 150 g/m² (prepreg epoxi)"]
    lam = [(mat, a, 0.15) for a in (0, 45, -45, 90, 90, -45, 45, 0)]
    A, B, D, z = clt.abd(lam)
    assert np.abs(B).max() < 1e-6 * np.abs(A).max()
    e = clt.constantes_ingenieria(lam)
    assert abs(e["Ex"] - e["Ey"]) / e["Ex"] < 1e-6
    fi = clt.tsai_wu(lam, [100.0, 0, 0])
    assert fi.max() > 0
