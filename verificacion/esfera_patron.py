"""Cizalla por triángulo y patrón plano frente a la referencia exacta de la esfera."""
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, ".")
sys.path.insert(0, "verificacion")
from esfera_analitica import cizalla_nodos, red_exacta  # noqa: E402
from laminado.drapeado.simular import OpcionesDrapeado, simular_capa  # noqa: E402
from laminado.geom.consulta import Consultor  # noqa: E402
from laminado.geom.cortes import extender_superficie  # noqa: E402
from laminado.geom.malla import TriMesh  # noqa: E402

R, h = 100.0, 5.0
N = int(np.pi / 2 * R / h) + 2
Ge = red_exacta(R, h, N)
ce = cizalla_nodos(Ge)
ok = np.isfinite(ce)
arbol = cKDTree(Ge[ok])
vals = ce[ok]
# coordenadas exactas de tejido de cada nodo: (i, j)·h
ii, jj = np.meshgrid(np.arange(2 * N + 1) - N, np.arange(2 * N + 1) - N, indexing="ij")
uv_ex = np.stack([ii, jj], -1)[ok] * h
for res in (120, 240):
    s = pv.Sphere(radius=R, theta_resolution=res, phi_resolution=res).triangulate().clean()
    m = TriMesh(np.asarray(s.points, float), s.faces.reshape(-1, 4)[:, 1:])
    m = m.submalla(m.face_centroids[:, 2] > np.cos(np.radians(70)) * R)[0]   # casquete de 70°
    if m.face_normals[:, 2].mean() < 0:
        m = m.invertida()
    c = Consultor(extender_superficie(m, 30))
    r = simular_capa(m, c, [0, 0, R], [1, 0, 0], OpcionesDrapeado(h=h, angulo_bloqueo=89))
    g = np.abs(r.cizalla())
    # cizalla exacta en el punto de tejido (u, v) de cada cara: interpolación del campo exacto
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(uv_ex, vals)
    uvc = r.uv[m.F].mean(1)
    gex = interp(uvc)
    cerca = np.isfinite(gex)
    dg = np.abs(g - gex)[cerca]
    print(f"malla {res}: caras comparadas {cerca.sum()}/{m.nf}  Δcizalla por cara: media {dg.mean():.3f}°  "
          f"p95 {np.percentile(dg, 95):.3f}°  máx {dg.max():.3f}°  (cizalla exacta máx {gex[cerca].max():.1f}°)")
    # consumo de tejido: una celda cizallada γ ocupa h²·cos γ en la pieza y h² en el plano
    print(f"          Σ área/cos γ = {np.sum(m.face_areas / np.cos(np.radians(g))):.0f} mm² (debe igualar al área del patrón)")
    # patrón plano: las coordenadas uv de los vértices deben coincidir con (i, j)·h exactos
    dv, kv = arbol.query(m.V)
    cv = dv < 0.05
    print(f"          vértices que caen sobre un nodo exacto: {cv.sum()}  Δuv máx {np.abs(r.uv[cv] - uv_ex[kv[cv]]).max():.3f} mm")
    print(f"          área superficie {m.area:.0f}  área patrón {r.area_plana():.0f} mm²")
