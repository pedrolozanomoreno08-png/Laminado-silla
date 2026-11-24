"""Distancias y caminos geodésicos sobre mallas."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra

from .malla import TriMesh


def _gradiente(m: TriMesh):
    """Operador gradiente por cara (3*nf x nv) en forma dispersa."""
    V, F = m.V, m.F
    n = m.face_normals
    A2 = 2 * np.maximum(m.face_areas, 1e-300)
    filas, cols, vals = [], [], []
    for k in range(3):
        i = F[:, k]
        e = V[F[:, (k + 2) % 3]] - V[F[:, (k + 1) % 3]]       # arista opuesta
        g = np.cross(n, e) / A2[:, None]                      # gradiente de la función sombrero
        for c in range(3):
            filas.append(3 * np.arange(m.nf) + c)
            cols.append(i)
            vals.append(g[:, c])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                         shape=(3 * m.nf, m.nv))


class DistanciaGeodesica:
    """Método del calor (Crane et al. 2013) con matrices factorizadas para reutilizar."""

    def __init__(self, m: TriMesh, t_factor: float = 1.0):
        self.m = m
        h = float(np.mean(m.edge_lengths))
        t = t_factor * h * h
        L = m.laplaciano_cot
        M = m.masa
        self._calor = spla.factorized((M - t * L).tocsc())
        # Poisson con un ancla mediante regularización mínima
        self._poisson = spla.factorized((-L + 1e-10 * M).tocsc())
        self._G = _gradiente(m)
        self._Ga = self._G.T.multiply(np.repeat(m.face_areas, 3)[None, :]).tocsr()

    def desde(self, fuentes) -> np.ndarray:
        """Distancia geodésica a un conjunto de vértices (índices) o de pesos (array nv)."""
        u0 = np.zeros(self.m.nv)
        fuentes = np.asarray(fuentes)
        if fuentes.dtype == bool:
            u0[fuentes] = 1.0
        else:
            u0[fuentes] = 1.0
        u = self._calor(u0)
        g = (self._G @ u).reshape(-1, 3)
        X = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
        div = self._Ga @ X.ravel()
        phi = self._poisson(-div)
        idx = np.nonzero(u0)[0]
        phi -= phi[idx].mean()
        return phi


def camino_vertices(m: TriMesh, a: int, b: int) -> np.ndarray:
    """Camino más corto por aristas entre los vértices a y b (Dijkstra)."""
    dist, pred = dijkstra(m.adyacencia, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        raise ValueError("Los vértices no están conectados")
    camino = [b]
    while camino[-1] != a:
        camino.append(pred[camino[-1]])
    return np.array(camino[::-1])


def camino_por_puntos(m: TriMesh, indices) -> np.ndarray:
    """Concatena caminos más cortos entre una secuencia de vértices."""
    total = [indices[0]]
    for a, b in zip(indices[:-1], indices[1:]):
        c = camino_vertices(m, a, b)
        total.extend(c[1:].tolist())
    return np.array(total)


def trazar_geodesica(consultor, P0, d0, longitud, paso, max_pasos=100000, tol_borde=None):
    """Geodésica 'más recta' desde P0 con dirección d0 (proyecta y transporta la dirección).

    Devuelve los puntos (k,3) muestreados cada ``paso`` (en distancia de cuerda)
    hasta recorrer ``longitud`` o salir de la superficie."""
    sub = 8
    ds = paso / sub
    P = np.asarray(P0, float)[None]
    Q, n, _, _ = consultor.proyectar(P)
    P = Q
    d = d0 - (d0 @ n[0]) * n[0]
    d = d / np.linalg.norm(d)
    pts = [P[0].copy()]
    ultimo = P[0].copy()
    recorrido = 0.0
    tol = tol_borde if tol_borde is not None else 0.25 * ds
    for _ in range(max_pasos):
        # paso predictor-corrector en el plano tangente
        X = P[0] + d * ds
        Q, n2, en, dist = consultor.proyectar(X[None])
        if en[0] and dist[0] > tol:
            break
        nueva = Q[0] - P[0]
        L = np.linalg.norm(nueva)
        if L < 1e-12:
            break
        dn = nueva / L
        # transporte: quitar la componente normal manteniendo la dirección de avance
        d = dn - (dn @ n2[0]) * n2[0]
        d /= np.linalg.norm(d)
        P = Q
        recorrido += L
        if np.linalg.norm(P[0] - ultimo) >= paso:
            # ajustar para que la cuerda sea exactamente 'paso' (bisección en el último tramo)
            a, b = pts[-1], P[0]
            prev = P[0] - nueva
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(prev + mid * nueva - a) < paso:
                    lo = mid
                else:
                    hi = mid
            X = prev + 0.5 * (lo + hi) * nueva
            Xq, _, _, _ = consultor.proyectar(X[None])
            pts.append(Xq[0])
            ultimo = Xq[0]
            P = Xq
            if len(pts) * paso >= longitud:
                break
    return np.array(pts)
