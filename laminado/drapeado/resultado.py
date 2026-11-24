"""Resultado del drapeado sobre la malla de la capa: coordenadas de tejido, fibras, cizalla y
patrón plano."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from ..geom.malla import TriMesh
from .red import Red


def _bilineal(P00, P10, P01, P11, s, t):
    return ((1 - s) * (1 - t))[:, None] * P00 + (s * (1 - t))[:, None] * P10 + \
        ((1 - s) * t)[:, None] * P01 + (s * t)[:, None] * P11


def invertir_bilineal(P00, P10, P01, P11, X, iters=8):
    """(s, t, residuo) tales que el parche bilineal pasa lo más cerca posible de X."""
    s = np.full(len(X), 0.5)
    t = np.full(len(X), 0.5)
    for _ in range(iters):
        B = _bilineal(P00, P10, P01, P11, s, t)
        r = B - X
        Bs = (1 - t)[:, None] * (P10 - P00) + t[:, None] * (P11 - P01)
        Bt = (1 - s)[:, None] * (P01 - P00) + s[:, None] * (P11 - P10)
        a = (Bs * Bs).sum(1)
        b = (Bs * Bt).sum(1)
        c = (Bt * Bt).sum(1)
        g1 = (Bs * r).sum(1)
        g2 = (Bt * r).sum(1)
        det = a * c - b * b
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        ds = (c * g1 - b * g2) / det
        dt = (a * g2 - b * g1) / det
        s = s - ds
        t = t - dt
    res = np.linalg.norm(_bilineal(P00, P10, P01, P11, s, t) - X, axis=1)
    return s, t, res


def mapear_red(red: Red, malla: TriMesh):
    """Coordenadas de tejido (u, v) de cada vértice de la malla de la capa.

    Cada cara se asocia a la celda de la red que contiene su centroide; sus vértices se
    expresan en esa celda (permite extrapolación ligera). Respeta las rendijas: los vértices
    duplicados de un dart sólo ven las caras de su lado. Devuelve (uv (nv,2), cubierto (nv,),
    cara_cubierta (nf,))."""
    h = red.h
    G = red.G
    co = red.celda_ok
    ci, cj = np.nonzero(co)
    if len(ci) == 0:
        raise RuntimeError("La red de drapeado está vacía: revisa la semilla y la dirección")
    centros = 0.25 * (G[ci, cj] + G[ci + 1, cj] + G[ci, cj + 1] + G[ci + 1, cj + 1])
    arbol = cKDTree(centros)
    X = malla.face_centroids
    k = min(8, len(ci))
    _, cand = arbol.query(X, k=k)
    cand = cand.reshape(len(X), -1)
    mejor = np.full(len(X), -1)
    mejor_r = np.full(len(X), np.inf)
    mejor_st = np.zeros((len(X), 2))
    for c in range(cand.shape[1]):
        q = cand[:, c]
        a, b = ci[q], cj[q]
        s, t, r = invertir_bilineal(G[a, b], G[a + 1, b], G[a, b + 1], G[a + 1, b + 1], X)
        fuera = np.maximum.reduce([-s, s - 1, -t, t - 1, np.zeros_like(s)])
        pen = r + fuera * h * 2
        mej = pen < mejor_r
        mejor[mej] = q[mej]
        mejor_r[mej] = pen[mej]
        mejor_st[mej] = np.c_[s, t][mej]
    cubierta = mejor_r < 0.25 * h
    # vértices: en la celda de cada cara incidente
    F = malla.F
    uv_acc = np.zeros((malla.nv, 2))
    w_acc = np.zeros(malla.nv)
    for kk in range(3):
        v = F[:, kk]
        q = mejor
        a, b = ci[q], cj[q]
        s, t, r = invertir_bilineal(G[a, b], G[a + 1, b], G[a, b + 1], G[a + 1, b + 1], malla.V[v])
        u = (a - red.i0 + s) * h
        w = (b - red.j0 + t) * h
        peso = np.where(cubierta, malla.face_areas, 0.0)
        np.add.at(uv_acc, v, np.c_[u, w] * peso[:, None])
        np.add.at(w_acc, v, peso)
    cubierto = w_acc > 0
    uv = uv_acc / np.maximum(w_acc, 1e-300)[:, None]
    if not cubierto.all() and cubierto.any():
        uv = _rellenar_laplaciano(malla, uv, cubierto)
    return uv, cubierto, cubierta


def _rellenar_laplaciano(malla, valores, conocido):
    """Extiende valores a los vértices desconocidos resolviendo Laplace (extrapolación suave)."""
    L = malla.laplaciano_cot
    u = ~conocido
    out = valores.copy()
    Luu = L[u][:, u].tocsc()
    Luk = L[u][:, conocido]
    try:
        sol = spla.factorized(Luu + 1e-9 * sp.eye(Luu.shape[0]))
        for c in range(valores.shape[1]):
            out[u, c] = sol(-(Luk @ valores[conocido, c]))
    except RuntimeError:
        pass
    return out


@dataclass
class ResultadoDrapeado:
    malla: TriMesh                       # malla de la capa (con rendijas si hay darts)
    uv: np.ndarray                       # (nv,2) coordenadas de tejido en mm
    cubierto: np.ndarray                 # (nv,) vértice alcanzado por el tejido
    estrategia: str = ""
    red: Red | None = None
    origen_vertice: np.ndarray | None = None   # vértice de la malla sin rendijas
    darts: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # ----------------------------------------------------------- campos por cara
    def gradientes(self):
        """(a, b): direcciones de urdimbre y trama en 3D por cara (∂X/∂u, ∂X/∂v)."""
        V, F, uv = self.malla.V, self.malla.F, self.uv
        x1 = V[F[:, 1]] - V[F[:, 0]]
        x2 = V[F[:, 2]] - V[F[:, 0]]
        u1 = uv[F[:, 1]] - uv[F[:, 0]]
        u2 = uv[F[:, 2]] - uv[F[:, 0]]
        det = u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        # [x1 x2] = [a b] [u1 u2]  →  [a b] = [x1 x2] [u1 u2]^-1
        a = (x1 * u2[:, 1:2] - x2 * u1[:, 1:2]) / det[:, None]
        b = (-x1 * u2[:, 0:1] + x2 * u1[:, 0:1]) / det[:, None]
        return a, b, det

    def fibras(self):
        a, b, _ = self.gradientes()
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        return a / np.maximum(na, 1e-300), b / np.maximum(nb, 1e-300)

    def cizalla(self):
        """Ángulo de cizalla por cara (grados, con signo): 90° − ángulo entre hilos."""
        a, b, det = self.gradientes()
        n = self.malla.face_normals
        ea, eb = self.fibras()
        cosg = (ea * eb).sum(1)
        g = np.degrees(np.arcsin(np.clip(cosg, -1, 1)))
        g = np.where(det <= 0, 90.0, g)          # cara invertida: pliegue
        del n
        return g

    def deformaciones(self):
        """Alargamiento de urdimbre y trama (adimensional) por cara."""
        a, b, _ = self.gradientes()
        return np.linalg.norm(a, axis=1) - 1, np.linalg.norm(b, axis=1) - 1

    def angulo_respecto(self, ref_dirs):
        """Ángulo (grados) de la urdimbre respecto a una dirección de referencia por cara."""
        ea, _ = self.fibras()
        n = self.malla.face_normals
        r = ref_dirs - (ref_dirs * n).sum(1, keepdims=True) * n
        r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-300)
        s = (np.cross(r, ea) * n).sum(1)
        c = (r * ea).sum(1)
        return np.degrees(np.arctan2(s, c))

    def cara_cubierta(self):
        return self.cubierto[self.malla.F].all(1)

    # ------------------------------------------------------------- patrón plano
    def contornos_planos(self):
        """Bucles de borde en el plano (lista de arrays (k,2)), el exterior primero."""
        bucles = self.malla.boundary_loops()
        polis = [self.uv[b] for b in bucles]
        areas = [abs(_area(p)) for p in polis]
        orden = np.argsort(areas)[::-1]
        return [polis[i] for i in orden], [bucles[i] for i in orden]

    def area_plana(self):
        """Área del patrón plano (suma de triángulos en uv)."""
        uv, F = self.uv, self.malla.F
        u1 = uv[F[:, 1]] - uv[F[:, 0]]
        u2 = uv[F[:, 2]] - uv[F[:, 0]]
        return float(0.5 * np.abs(u1[:, 0] * u2[:, 1] - u1[:, 1] * u2[:, 0]).sum())

    def resumen(self, aviso=None, limite=None):
        g = np.abs(self.cizalla())
        A = self.malla.face_areas
        out = {
            "estrategia": self.estrategia,
            "cizalla_max": float(np.nanmax(g)) if len(g) else 0.0,
            "cizalla_media": float(np.average(g, weights=A)) if len(g) else 0.0,
            "area_superficie_mm2": float(A.sum()),
            "area_plana_mm2": self.area_plana(),
            "cubierto_%": float(100 * A[self.cara_cubierta()].sum() / max(A.sum(), 1e-300)),
            "n_darts": len(self.darts),
        }
        if aviso is not None:
            out["area_aviso_%"] = float(100 * A[g > aviso].sum() / A.sum())
        if limite is not None:
            out["area_limite_%"] = float(100 * A[g > limite].sum() / A.sum())
        return out


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
