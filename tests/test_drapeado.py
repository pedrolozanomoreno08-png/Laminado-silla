import numpy as np

from laminado.drapeado.energia import drapear_energia
from laminado.drapeado.red import Dart
from laminado.drapeado.simular import OpcionesDrapeado, dividir_en_piezas, empalmes_por_ancho, simular_capa
from laminado.geom.consulta import Consultor
from laminado.geom.cortes import extender_superficie


def test_cilindro_sin_cizalla_y_area(medio_cilindro):
    m = medio_cilindro
    c = Consultor(extender_superficie(m, 60))
    for est in ("geodesica", "proyeccion"):
        r = simular_capa(m, c, [0, 0, 100], [1, 0.4, 0] if est == "geodesica" else [1, 0, 0],
                         OpcionesDrapeado(estrategia=est, h=6.0))
        g = np.abs(r.cizalla())
        assert np.percentile(g, 99) < 1.5
        assert abs(r.area_plana() - m.area) / m.area < 0.005


def test_casquete_cizalla_y_energia(casquete):
    c = Consultor(extender_superficie(casquete, 40))
    r = simular_capa(casquete, c, [0, 0, 100], [1, 0, 0], OpcionesDrapeado(h=5.0))
    g = np.abs(r.cizalla())
    assert 10 < g.max() < 45          # doble curvatura: cizalla clara pero sin bloqueo
    assert r.cara_cubierta().mean() > 0.99
    re = simular_capa(casquete, c, [0, 0, 100], [1, 0, 0], OpcionesDrapeado(h=5.0, estrategia="energia"))
    a, b = re.deformaciones()
    assert np.abs(a).max() < 0.03 and np.abs(b).max() < 0.03
    assert np.abs(re.cizalla()).mean() <= g.mean() + 0.5


def test_jacobiano_energia(casquete):
    import sys
    m = casquete.submalla(casquete.face_centroids[:, 2] > 95)[0]
    uv = m.V[:, :2] + np.random.default_rng(1).normal(0, 0.2, (m.nv, 2))
    capt = {}

    def traza(frame, ev, arg):
        if ev == "return" and frame.f_code.co_name == "drapear_energia":
            capt.update(frame.f_locals)
        return traza
    sys.setprofile(traza)
    drapear_energia(m, uv, 0, [1, 0, 0], max_iter=0)
    sys.setprofile(None)
    res, jac = capt["residuos"], capt["jac"]
    x0 = uv.ravel()
    J = jac(x0).toarray()
    r0 = res(x0)
    for i in range(0, len(x0), 7):
        x = x0.copy()
        x[i] += 1e-7
        assert np.allclose((res(x) - r0) / 1e-7, J[:, i], atol=1e-3 * max(1, np.abs(J).max()))


def test_dart_reduce_cizalla(casquete):
    c = Consultor(extender_superficie(casquete, 40))
    op = OpcionesDrapeado(h=5.0)
    r0 = simular_capa(casquete, c, [0, 0, 100], [1, 0, 0], op)
    P = np.array([[np.cos(np.pi / 4) * 85, np.sin(np.pi / 4) * 85, 52.0], [40, 40, 80]])
    r1 = simular_capa(casquete, c, [0, 0, 100], [1, 0, 0], op, [Dart(P)])
    assert len(r1.darts) == 1
    assert r1.malla.nv > casquete.nv
    g0 = np.abs(r0.cizalla())
    g1 = np.abs(r1.cizalla())
    assert g1.max() < g0.max()


def test_empalmes_por_ancho(medio_cilindro):
    c = Consultor(extender_superficie(medio_cilindro, 60))
    r = simular_capa(medio_cilindro, c, [0, 0, 100], [1, 0, 0], OpcionesDrapeado(h=6.0))
    emp = empalmes_por_ancho(r, 200.0, solape=10.0)
    assert len(emp) >= 1
    piezas = dividir_en_piezas(r, emp)
    assert len(piezas) == len(emp) + 1
    assert all(np.ptp(p.uv[:, 1]) <= 200.0 for p in piezas)
    # el solape hace que la suma de áreas supere al total
    assert sum(p.area_plana() for p in piezas) > r.area_plana()
