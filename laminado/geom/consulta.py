"""Consultas geométricas vectorizadas sobre una malla: punto más cercano, normal suave, rayos."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .malla import TriMesh


def punto_triangulo(P, A, B, C):
    """Punto más cercano de P a los triángulos ABC (todo vectorizado, forma (...,3)).

    Devuelve (Q, baricéntricas (...,3)). Algoritmo de Ericson (Real-Time Collision Detection)."""
    ab = B - A
    ac = C - A
    ap = P - A
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = P - B
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = P - C
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-300, 1e-300, va + vb + vc)
    v = vb / den
    w = vc / den
    bary = np.stack([1 - v - w, v, w], -1)

    def poner(m, b):
        nonlocal bary
        bary = np.where(m[..., None], b, bary)
    hecho = np.zeros(d1.shape, bool)
    # Orden inverso de prioridad: las regiones más específicas se aplican al final.
    # Arista BC
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6)), 0, 1)
        poner(m, np.stack([np.zeros_like(t), 1 - t, t], -1))
        # Arista AC
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.clip(d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0, 1)
        poner(m, np.stack([1 - t, np.zeros_like(t), t], -1))
        # Arista AB
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.clip(d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0, 1)
        poner(m, np.stack([1 - t, t, np.zeros_like(t)], -1))
    # Vértices
    uno = np.ones_like(d1)
    cero = np.zeros_like(d1)
    poner((d6 >= 0) & (d5 <= d6), np.stack([cero, cero, uno], -1))
    poner((d3 >= 0) & (d4 <= d3), np.stack([cero, uno, cero], -1))
    poner((d1 <= 0) & (d2 <= 0), np.stack([uno, cero, cero], -1))
    del hecho
    Q = bary[..., :1] * A + bary[..., 1:2] * B + bary[..., 2:3] * C
    return Q, bary


class Consultor:
    """Punto más cercano sobre la malla para lotes de puntos.

    Candidatos: las ``k`` caras de centroide más cercano y las caras de los vértices más cercanos."""

    def __init__(self, malla: TriMesh, k: int = 12):
        self.m = malla
        self.k = min(k, malla.nf)
        self.arbol_c = cKDTree(malla.face_centroids)
        self.arbol_v = cKDTree(malla.V)
        # cara de cada vértice (una cualquiera) para ampliar candidatos
        vc = malla.vertice_caras.tocsr()
        self._vf_ptr = vc.indptr
        self._vf_idx = vc.indices
        self.A = malla.V[malla.F[:, 0]]
        self.B = malla.V[malla.F[:, 1]]
        self.C = malla.V[malla.F[:, 2]]
        self.vn = malla.vertex_normals
        self.fn = malla.face_normals
        bordes = malla.boundary_vertices_mask
        self._borde_cara = bordes[malla.F]          # (nf,3) vértice de borde
        self._arista_borde_cara = malla.boundary_edges_mask[malla.face_edges]  # (nf,3)

    def _candidatos(self, P):
        _, ic = self.arbol_c.query(P, k=self.k)
        ic = np.atleast_2d(ic).reshape(len(P), -1)
        _, iv = self.arbol_v.query(P, k=1)
        iv = np.atleast_1d(iv)
        # hasta 8 caras del vértice más cercano
        extra = np.full((len(P), 8), -1, np.int64)
        for j in range(8):
            pos = self._vf_ptr[iv] + j
            ok = pos < self._vf_ptr[iv + 1]
            extra[ok, j] = self._vf_idx[pos[ok]]
        extra = np.where(extra < 0, ic[:, :1], extra)
        return np.concatenate([ic, extra], axis=1)

    def mas_cercano(self, P):
        """(Q, cara, baricéntricas, distancia) para P (n,3)."""
        P = np.atleast_2d(np.asarray(P, float))
        cand = self._candidatos(P)
        Pq = P[:, None, :]
        Q, bar = punto_triangulo(Pq, self.A[cand], self.B[cand], self.C[cand])
        d = np.linalg.norm(Q - Pq, axis=-1)
        j = np.argmin(d, axis=1)
        r = np.arange(len(P))
        return Q[r, j], cand[r, j], bar[r, j], d[r, j]

    def normal(self, cara, bar):
        """Normal interpolada (Phong) en los puntos dados por cara + baricéntricas."""
        F = self.m.F[cara]
        n = (bar[:, :, None] * self.vn[F]).sum(1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    def proyectar(self, P):
        """(Q, normal, en_borde, distancia): proyecta sobre la superficie.

        ``en_borde`` es True si el punto más cercano está sobre una arista de borde
        (el punto original cae fuera de la superficie)."""
        Q, f, b, d = self.mas_cercano(P)
        n = self.normal(f, b)
        # está en borde si alguna baricéntrica es 0 y la arista opuesta es de borde
        ab = self._arista_borde_cara[f]   # aristas (0-1, 1-2, 2-0)
        eps = 1e-9
        en = ((b[:, 2] < eps) & ab[:, 0]) | ((b[:, 0] < eps) & ab[:, 1]) | ((b[:, 1] < eps) & ab[:, 2])
        return Q, n, en, d

    def dentro(self, P, tol):
        """True si P está a menos de ``tol`` de la superficie y no fuera del borde."""
        Q, n, en, d = self.proyectar(P)
        lateral = np.linalg.norm((P - Q) - ((P - Q) * n).sum(1, keepdims=True) * n, axis=1)
        return (d < tol) & ~(en & (lateral > 1e-6 * max(tol, 1e-9) + 1e-3 * tol))


def interseccion_rayo(malla: TriMesh, origen, direccion):
    """Primera intersección de un rayo con la malla (Möller–Trumbore, todas las caras).

    Devuelve (t, cara) o (None, None)."""
    o = np.asarray(origen, float)
    d = np.asarray(direccion, float)
    d = d / np.linalg.norm(d)
    A = malla.V[malla.F[:, 0]]
    e1 = malla.V[malla.F[:, 1]] - A
    e2 = malla.V[malla.F[:, 2]] - A
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    s = o - A
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q * d).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    if not hit.any():
        return None, None
    idx = np.nonzero(hit)[0]
    j = idx[np.argmin(t[idx])]
    return float(t[j]), int(j)


def base_tangente(n, ref=None):
    """Dos vectores tangentes ortonormales (t1, t2) para normales n (k,3)."""
    n = np.atleast_2d(n)
    if ref is None:
        ref = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    ref = np.broadcast_to(ref, n.shape)
    t1 = ref - (ref * n).sum(1, keepdims=True) * n
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-300)
    t2 = np.cross(n, t1)
    return t1, t2


def proyectar_tangente(v, n):
    """Proyecta vectores v sobre el plano tangente de normal n y normaliza."""
    v = np.atleast_2d(v)
    n = np.atleast_2d(n)
    t = v - (v * n).sum(1, keepdims=True) * n
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
