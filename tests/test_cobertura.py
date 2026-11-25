"""Cobertura completa automática: sin huecos y con todos los parches producibles."""
import numpy as np

from laminado.ansys.lector import Molde
from laminado.geom.malla import TriMesh
from laminado.modelo.cobertura import OpcionesCobertura, cubrir, mapa_cobertura
from laminado.modelo.proyecto import Proyecto

MAT = "Carbono sarga 2x2 200 g/m² (prepreg epoxi)"


def _tubo_en_t():
    """Dos tubos unidos en T (superficie cerrada, unión suave): un nudo como los de un cuadro."""
    import pyvista as pv
    paso = 3.0
    lo, hi = np.array([-130.0, -40, -40]), np.array([130.0, 40, 150])
    dims = np.ceil((hi - lo) / paso).astype(int) + 1
    g = pv.ImageData(dimensions=dims, spacing=(paso,) * 3, origin=lo)
    P = np.asarray(g.points)

    def caps(A, B, r):
        A, B = np.asarray(A, float), np.asarray(B, float)
        ab = B - A
        t = np.clip(((P - A) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(P - (A + t[:, None] * ab), axis=1) - r
    a, b = caps((-110, 0, 0), (110, 0, 0), 22), caps((0, 0, 0), (0, 0, 130), 16)
    k = 10.0
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    g["d"] = b + (a - b) * h - k * h * (1 - h)
    s = g.contour([0.0], scalars="d").triangulate().clean().smooth_taubin(n_iter=30, pass_band=0.1)
    m = TriMesh(np.asarray(s.points, float), s.faces.reshape(-1, 4)[:, 1:]).orientar_coherente()
    vol = np.einsum("ij,ij->i", m.V[m.F[:, 0]], np.cross(m.V[m.F[:, 1]], m.V[m.F[:, 2]])).sum()
    return m if vol > 0 else m.invertida()


def test_cobertura_completa_nudo_en_t():
    m = _tubo_en_t()
    p = Proyecto("T")
    p.cargar_molde(Molde(m, {}, origen="prueba"))
    p.h = 4.0
    inf = cubrir(p, OpcionesCobertura(MAT, 0.0, solape=10.0, radio_max=200.0))
    assert inf.hueco_mm2 == 0.0 and inf.cobertura_pct == 100.0
    assert not inf.no_producibles
    for c in p.capas:
        s = p.resultados[c.nombre].resumen
        assert s["estado"] in ("VERDE", "AMARILLO"), (c.nombre, s["estado"])
        assert s["solape_patron_%"] < 0.5
    n = mapa_cobertura(p)
    assert (n >= 1).all()
    # tiene que haber solape entre parches contiguos (no a tope)
    assert m.face_areas[n >= 2].sum() > 0
