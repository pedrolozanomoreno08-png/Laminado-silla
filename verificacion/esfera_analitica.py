"""Referencia exacta e independiente: red cinemática sobre una esfera analítica.

No usa nada del paquete `laminado` para el cálculo de referencia: generadores = círculos
máximos por el polo con cuerda h, y cada nodo se obtiene en forma cerrada como intersección
de la esfera con las dos esferas de radio h centradas en sus vecinos. Después se compara con
el programa, que trabaja sobre una malla triangulada de la misma esfera."""
import sys
import time

import numpy as np


def red_exacta(R, h, N):
    alfa = 2 * np.arcsin(h / (2 * R))
    G = np.full((2 * N + 1, 2 * N + 1, 3), np.nan)
    c0 = N
    for k in range(-N, N + 1):
        if abs(k * alfa) < np.pi / 2:
            G[c0 + k, c0] = [R * np.sin(k * alfa), 0, R * np.cos(k * alfa)]
            G[c0, c0 + k] = [0, R * np.sin(k * alfa), R * np.cos(k * alfa)]
    c = R * R - h * h / 2
    for si in (1, -1):
        for sj in (1, -1):
            for s in range(2, 4 * N):
                for a in range(1, s):
                    b = s - a
                    if a > N or b > N:
                        continue
                    i, j = c0 + si * a, c0 + sj * b
                    A, B, C = G[i - si, j], G[i, j - sj], G[i - si, j - sj]
                    if np.isnan(A).any() or np.isnan(B).any() or np.isnan(C).any():
                        continue
                    M = np.array([[A @ A, A @ B], [A @ B, B @ B]])
                    al, be = np.linalg.solve(M, [c, c])
                    p = al * A + be * B
                    n = np.cross(A, B)
                    n /= np.linalg.norm(n)
                    t2 = R * R - p @ p
                    if t2 < 0:
                        continue
                    X1, X2 = p + np.sqrt(t2) * n, p - np.sqrt(t2) * n
                    X = X1 if np.linalg.norm(X1 - C) > np.linalg.norm(X2 - C) else X2
                    if X[2] < 0:          # sólo el hemisferio superior
                        continue
                    G[i, j] = X
    return G


def cizalla_nodos(G):
    """Cizalla (°) en cada nodo interior: 90° − ángulo entre hilos (media de 4 esquinas)."""
    ni, nj, _ = G.shape
    out = np.full((ni, nj), np.nan)
    for i in range(1, ni - 1):
        for j in range(1, nj - 1):
            X = G[i, j]
            if np.isnan(X).any():
                continue
            vals = []
            for di, dj in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                A, B = G[i + di, j], G[i, j + dj]
                if np.isnan(A).any() or np.isnan(B).any():
                    continue
                a, b = A - X, B - X
                vals.append(np.degrees(np.arcsin(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b))))
            if len(vals) == 4:
                out[i, j] = np.mean(vals)
    return out


if __name__ == "__main__":
    import pyvista as pv
    sys.path.insert(0, ".")
    from laminado.geom.consulta import Consultor
    from laminado.geom.malla import TriMesh
    from laminado.drapeado.red import drapear_red
    R, h = 100.0, 5.0
    N = int(np.pi / 2 * R / h) + 2
    t = time.time()
    Ge = red_exacta(R, h, N)
    ce = cizalla_nodos(Ge)
    print(f"referencia exacta: {np.isfinite(ce).sum()} nodos, {time.time() - t:.1f} s")
    for res in (60, 120, 240, 400):
        s = pv.Sphere(radius=R, theta_resolution=res, phi_resolution=res).triangulate().clean()
        m = TriMesh(np.asarray(s.points, float), s.faces.reshape(-1, 4)[:, 1:])
        m = m.submalla(m.face_centroids[:, 2] > -5)[0]
        if m.face_normals[:, 2].mean() < 0:
            m = m.invertida()
        cons = Consultor(m)
        t = time.time()
        red = drapear_red(cons, [0, 0, R], [1, 0, 0], h, "geodesica", radio=200, angulo_max=89.0, tam_max=N)
        tt = time.time() - t
        # mismas posiciones de índice
        Gp = np.full_like(Ge, np.nan)
        off = red.i0 - N
        sl = slice(max(0, -off), min(Ge.shape[0], red.G.shape[0] - off))
        Gp[sl, sl] = np.where(red.ok[sl.start + off:sl.stop + off, sl.start + off:sl.stop + off, None],
                              red.G[sl.start + off:sl.stop + off, sl.start + off:sl.stop + off], np.nan)
        cp = cizalla_nodos(Gp)
        comun = np.isfinite(ce) & np.isfinite(cp)
        # zonas por altura: hasta 60° de colatitud y hasta el ecuador (80°)
        z = Ge[..., 2]
        colat = np.degrees(np.arccos(np.clip(z / R, -1, 1)))
        dpos = np.linalg.norm(Ge - Gp, axis=-1)
        for lim in (60, 80):
            msk = comun & (colat <= lim)
            print(f"malla {res:3d}x{res:3d} ({m.nf:6d} caras, arista {np.median(m.edge_lengths):.2f} mm) colatitud ≤{lim}°: "
                  f"nodos {msk.sum():5d}/{(np.isfinite(ce) & (colat <= lim)).sum():5d}  "
                  f"Δcizalla media {np.nanmean(np.abs(ce - cp)[msk]):.3f}° máx {np.nanmax(np.abs(ce - cp)[msk]):.3f}°  "
                  f"Δposición máx {np.nanmax(dpos[msk]):.3f} mm  | cizalla ref máx {np.nanmax(ce[msk]):.1f}°  [{tt:.1f} s]")
