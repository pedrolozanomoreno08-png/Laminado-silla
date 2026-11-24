import numpy as np

from laminado.geom.consulta import Consultor
from laminado.geom.cortes import abrir_rendija, cortar_por_nivel, extender_superficie, recortar
from laminado.geom.geodesia import camino_vertices, fast_marching


def test_proyeccion_exacta(medio_cilindro):
    c = Consultor(medio_cilindro)
    P = medio_cilindro.V[::37] + medio_cilindro.vertex_normals[::37] * 2.5
    Q, n, en, d = c.proyectar(P)
    assert np.allclose(d, 2.5, atol=0.05)


def test_fast_marching_borde_recto(plano):
    d = fast_marching(plano, plano.boundary_vertices_mask)
    V = plano.V
    ex = np.minimum.reduce([V[:, 0] + 100, V[:, 1] + 100, 100 - V[:, 0], 100 - V[:, 1]])
    banda = (V[:, 0] < -40) & (np.abs(V[:, 1]) < 30)
    assert np.abs(d - ex)[banda].max() < 1e-9


def test_corte_por_nivel_conserva_area(plano):
    phi = np.linalg.norm(plano.V[:, :2], axis=1) - 60.0
    sub, _, _, _ = recortar(plano, phi)
    assert abs(sub.area - np.pi * 60 ** 2) / (np.pi * 60 ** 2) < 0.01
    m2, phi2, _ = cortar_por_nivel(plano, phi)
    assert abs(m2.area - plano.area) < 1e-6


def test_rendija_y_extension(plano):
    b = int(np.argmin(np.abs(plano.V[:, 0] + 100) + np.abs(plano.V[:, 1])))
    c = int(np.argmin(np.linalg.norm(plano.V[:, :2] - [-40, 0], axis=1)))
    cam = camino_vertices(plano, b, c)
    m, org = abrir_rendija(plano, cam)
    assert m.nv == plano.nv + len(cam) - 1
    assert len(m.boundary_loops()) == 1
    e = extender_superficie(plano, 20)
    assert e.area > plano.area * 1.3
