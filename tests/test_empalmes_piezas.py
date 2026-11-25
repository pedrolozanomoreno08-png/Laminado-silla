"""Empalmes: piezas conexas, drapeado de cada pieza y patrones siempre válidos."""
import numpy as np

from laminado.ansys.lector import Molde
from laminado.demo.silla import casco_silla, selecciones_silla
from laminado.drapeado.simular import Empalme
from laminado.geom.malla import TriMesh
from laminado.modelo.proyecto import Capa, Contorno, Proyecto

MAT = "Carbono sarga 2x2 200 g/m² (prepreg epoxi)"


def _placa_u():
    """Placa plana en U de 200 × 200 mm con una ranura central de 80 mm de ancho."""
    n = 41
    x, y = np.meshgrid(np.linspace(-100, 100, n), np.linspace(-100, 100, n), indexing="ij")
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.vstack([np.c_[a, b, c], np.c_[a, c, d]])
    V = np.c_[x.ravel(), y.ravel(), np.zeros(n * n)]
    m = TriMesh(V, F)
    cg = m.face_centroids
    ranura = (np.abs(cg[:, 0]) < 40) & (cg[:, 1] > -20)
    return m.submalla(~ranura)[0]


def test_empalme_separa_trozos_conexos(tmp_path):
    m = _placa_u()
    p = Proyecto("U")
    p.cargar_molde(Molde(m, {}, origen="prueba"))
    p.h = 5.0
    c = Capa("P1", MAT, 0, contorno=Contorno("molde", exceso=0))
    c.empalme_auto = False
    # plano y = 40: corta las dos patas de la U → 3 piezas (base y dos puntas separadas)
    c.empalmes = [Empalme("plano", (0.0, 40.0, 0.0), (0.0, 1.0, 0.0), solape=0.0)]
    p.capas = [c]
    rc = p.simular("P1")
    assert len(rc.piezas) == 3
    for pz in rc.piezas:
        assert len(pz.malla.componentes()[1]) and pz.malla.componentes()[0] == 1
        assert pz.cara_cubierta().mean() > 0.99
        assert abs(pz.area_plana() - pz.malla.area) / pz.malla.area < 1e-3
    from laminado.salidas import dxf
    from laminado.salidas.patrones import todos_los_patrones
    pats = todos_los_patrones(p)
    assert len(pats) == 3 and all(not q.corte.is_empty for q in pats)
    assert abs(sum(q.area for q in pats) - m.area) / m.area < 0.01
    dxf.exportar_patrones(pats, str(tmp_path / "pat"))


def test_piezas_con_curva_de_semilla(tmp_path):
    """La curva de semilla de la capa no pasa por todas las piezas: cada pieza debe drapearse
    igualmente (antes quedaban piezas sin tejido y fallaban los DXF y el anidado)."""
    m = casco_silla(10.0)
    p = Proyecto("silla")
    p.cargar_molde(Molde(m, selecciones_silla(m), origen="demo"))
    p.h = 8.0
    c = Capa("P1", MAT, 0, contorno=Contorno("molde", exceso=10))
    c.empalme_auto = False
    c.estrategia = "curva"
    c.curva_semilla = [[x, 0.0, float(m.V[np.argmin(np.abs(m.V[:, 0] - x) + np.abs(m.V[:, 1])), 2])]
                       for x in (-120.0, -40.0, 40.0, 120.0)]
    c.empalmes = [Empalme("plano", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), solape=10.0)]
    p.capas = [c]
    rc = p.simular("P1")
    assert len(rc.piezas) >= 2
    assert rc.resumen["cubierto_%"] > 98
    from laminado.salidas import anidado, dxf
    from laminado.salidas.patrones import todos_los_patrones
    pats = todos_los_patrones(p)
    assert len(pats) == len(rc.piezas)
    dxf.exportar_patrones(pats, str(tmp_path / "pat"))
    a = anidado.anidar(pats, p)
    assert a.resumen()[0]["piezas"] == len(pats)
