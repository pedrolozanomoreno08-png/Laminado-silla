"""Invariancias físicas: el resultado no puede depender de la posición, giro o escala de la
pieza (si la roseta y la semilla se mueven con ella), debe ser simétrico en piezas simétricas
y debe converger al afinar la malla."""
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, ".")
from laminado.ansys.lector import Molde  # noqa: E402
from laminado.demo.silla import casco_silla, selecciones_silla  # noqa: E402
from laminado.geom.malla import TriMesh  # noqa: E402
from laminado.modelo.proyecto import Capa, Contorno, Proyecto, Roseta  # noqa: E402

MAT = "Carbono sarga 2x2 200 g/m² (prepreg epoxi)"


def proyecto(m, sel, rot=np.eye(3), tras=np.zeros(3), esc=1.0, h=6.0):
    V = (m.V * esc) @ rot.T + tras
    p = Proyecto()
    p.cargar_molde(Molde(TriMesh(V, m.F), sel))
    p.h = h * esc
    p.extension = 60 * esc
    p.rosetas = {"R": Roseta("R", tuple(tras), tuple(rot @ [1, 0, 0]), tuple(rot @ [0, 0, 1]))}
    return p


def correr(p, capa, semilla, rot, tras, esc):
    c = Capa("C", MAT, capa["ori"], roseta="R", contorno=Contorno(**capa["cont"]),
             estrategia=capa["est"], semilla=tuple(rot @ (np.asarray(semilla) * esc) + tras))
    p.capas = [c]
    return p.simular(c)


m = casco_silla(6.0)
sel = selecciones_silla(m)
semilla = m.V[np.argmin(np.abs(m.V[:, 0]) + np.abs(m.V[:, 1]))]
casos = [
    {"nombre": "silla entera 0° geodésica", "ori": 0, "cont": {"tipo": "molde", "exceso": 15}, "est": "geodesica"},
    {"nombre": "asiento 30° geodésica", "ori": 30, "cont": {"tipo": "seleccion", "selecciones": ["ASIENTO"]}, "est": "geodesica"},
    {"nombre": "asiento 30° plano natural", "ori": 30, "cont": {"tipo": "seleccion", "selecciones": ["ASIENTO"]}, "est": "plano"},
    {"nombre": "silla entera 0° mínima energía", "ori": 0, "cont": {"tipo": "molde"}, "est": "energia"},
]
R = Rotation.from_euler("xyz", [37, -21, 64], degrees=True).as_matrix()
T = np.array([1234.5, -876.0, 432.1])
print("== Invariancia frente a giro + traslación y a escala ×2 ==")
for cs in casos:
    base = correr(proyecto(m, sel), cs, semilla, np.eye(3), np.zeros(3), 1.0)
    mov = correr(proyecto(m, sel, R, T), cs, semilla, R, T, 1.0)
    esc = correr(proyecto(m, sel, esc=2.0), cs, semilla, np.eye(3), np.zeros(3), 2.0)
    g0 = np.abs(base.drapeado.cizalla())
    for nombre, r, f in (("giro+traslación", mov, 1.0), ("escala ×2", esc, 2.0)):
        g1 = np.abs(r.drapeado.cizalla())
        # emparejar caras por posición (la malla recortada puede tener otro orden)
        C0 = base.drapeado.malla.face_centroids
        C1 = (r.drapeado.malla.face_centroids - (T if nombre.startswith("giro") else 0)) @ (R if nombre.startswith("giro") else np.eye(3)) / f
        d, k = cKDTree(C1).query(C0)
        ok = d < 0.5
        print(f"  {cs['nombre']:32s} {nombre:16s} Δcizalla media {np.mean(np.abs(g0[ok] - g1[k[ok]])):.4f}° "
              f"máx {np.max(np.abs(g0[ok] - g1[k[ok]])):.3f}°  área patrón {base.resumen['area_corte_mm2']:.0f} vs "
              f"{r.resumen['area_corte_mm2'] / f**2:.0f}  estado {base.resumen['estado']}/{r.resumen['estado']}")

print("== Simetría (silla simétrica respecto a y = 0, semilla en el plano, 0°) ==")
for cs in casos[:1] + casos[3:]:
    r = correr(proyecto(m, sel), cs, semilla, np.eye(3), np.zeros(3), 1.0).drapeado
    g = np.abs(r.cizalla())
    C = r.malla.face_centroids
    d, k = cKDTree(C * [1, -1, 1]).query(C)
    ok = d < 0.5
    print(f"  {cs['nombre']:32s} Δcizalla espejo media {np.mean(np.abs(g[ok] - g[k[ok]])):.4f}° máx {np.max(np.abs(g[ok] - g[k[ok]])):.3f}°")

print("== Convergencia con la malla (asiento, 30°) ==")
for tam in (8.0, 6.0, 4.0, 3.0):
    mm = casco_silla(tam)
    sm = selecciones_silla(mm)
    sd = mm.V[np.argmin(np.abs(mm.V[:, 0]) + np.abs(mm.V[:, 1]))]
    r = correr(proyecto(mm, sm, h=5.0), casos[1], sd, np.eye(3), np.zeros(3), 1.0)
    g = np.abs(r.drapeado.cizalla())
    print(f"  elemento {tam:.0f} mm ({mm.nf:6d} caras): cizalla máx {r.resumen['cizalla_max']:.2f}°  media "
          f"{r.resumen['cizalla_media']:.3f}°  área neta {r.resumen['area_neta_mm2']:.0f}  área corte {r.resumen['area_corte_mm2']:.0f}")
