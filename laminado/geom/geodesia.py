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
        A = (M - t * L).tocsc()
        self._calor = spla.factorized(A)
        # condición de Dirichlet en el borde (se promedia con Neumann: mejor cerca del borde)
        self._borde = m.boundary_vertices_mask
        self._calor_d = None
        if self._borde.any():
            libre = ~self._borde
            self._libre = np.nonzero(libre)[0]
            self._calor_d = spla.factorized(A[self._libre][:, self._libre].tocsc())
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
        if self._calor_d is not None:
            ud = np.zeros_like(u)
            ud[self._libre] = self._calor_d(u0[self._libre])
            u = 0.5 * (u + ud)
        g = (self._G @ u).reshape(-1, 3)
        X = -g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
        div = self._Ga @ X.ravel()
        phi = self._poisson(div)
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


def fast_marching(m: TriMesh, fuentes, valores=None, limite: float = np.inf) -> np.ndarray:
    """Distancia geodésica por fast marching sobre triángulos (Kimmel–Sethian).

    Exacta para frentes planos (p.ej. distancia a un borde recto) y de primer orden en general.
    ``fuentes``: índices de vértice (o máscara); ``valores``: distancia inicial de cada fuente.
    ``limite``: detiene la propagación más allá de esa distancia (más rápido para desfases)."""
    import heapq
    V = m.V
    fuentes = np.asarray(fuentes)
    if fuentes.dtype == bool:
        fuentes = np.nonzero(fuentes)[0]
    T = np.full(m.nv, np.inf)
    T[fuentes] = 0.0 if valores is None else np.asarray(valores, float)
    estado = np.zeros(m.nv, np.int8)       # 0 lejano, 1 frente, 2 fijo
    vc = m.vertice_caras.tocsr()
    ptr, ind = vc.indptr, vc.indices
    F = m.F
    heap = [(T[i], int(i)) for i in fuentes]
    heapq.heapify(heap)
    estado[fuentes] = 1
    # una fuente aislada (punto): inicializa su primer anillo con la distancia euclídea exacta
    if valores is None and len(fuentes) <= 3:
        for s in fuentes:
            for f in ind[ptr[s]:ptr[s + 1]]:
                for v in F[f]:
                    d = float(np.linalg.norm(V[v] - V[s]))
                    if d < T[v]:
                        T[v] = d
                        estado[v] = 1
                        heapq.heappush(heap, (d, int(v)))

    def actualizar(c, a, b):
        ta, tb = T[a], T[b]
        if not np.isfinite(ta) and not np.isfinite(tb):
            return np.inf
        if not np.isfinite(ta):
            return tb + np.linalg.norm(V[c] - V[b])
        if not np.isfinite(tb):
            return ta + np.linalg.norm(V[c] - V[a])
        A, B, C = V[a], V[b], V[c]
        ab = B - A
        L = np.linalg.norm(ab)
        e1 = ab / L
        ac = C - A
        cx = ac @ e1
        cy = np.linalg.norm(ac - cx * e1)
        dt = tb - ta
        cand = min(ta + np.linalg.norm(ac), tb + np.linalg.norm(C - B))
        if abs(dt) < L and cy > 1e-12:
            gx = dt / L
            gy = np.sqrt(1 - gx * gx)
            tc = ta + gx * cx + gy * cy
            # el rayo característico que llega a C debe cruzar el segmento AB
            lam = cy / gy
            px = cx - lam * gx
            if 0 <= px <= L:
                cand = min(cand, tc)
        return cand

    while heap:
        t, i = heapq.heappop(heap)
        if estado[i] == 2 or t > T[i]:
            continue
        estado[i] = 2
        if t > limite:
            break
        for f in ind[ptr[i]:ptr[i + 1]]:
            tri = F[f]
            for c in tri:
                if estado[c] == 2:
                    continue
                a, b = [x for x in tri if x != c]
                if estado[a] != 2 and estado[b] != 2:
                    continue
                ta_ok = estado[a] == 2
                tb_ok = estado[b] == 2
                if ta_ok and tb_ok:
                    tn = actualizar(c, a, b)
                else:
                    k = a if ta_ok else b
                    tn = T[k] + np.linalg.norm(V[c] - V[k])
                if tn < T[c]:
                    T[c] = tn
                    estado[c] = 1
                    heapq.heappush(heap, (tn, int(c)))
    return T


def distancia_a_borde_region(m: TriMesh, mascara_caras, limite=np.inf):
    """Distancia con signo al contorno de una región de caras (negativa dentro).

    El contorno debe coincidir con aristas de la malla (p.ej. tras ``cortar_por_nivel``)."""
    F = m.F
    dentro_v = np.zeros(m.nv, bool)
    fuera_v = np.zeros(m.nv, bool)
    dentro_v[F[mascara_caras].ravel()] = True
    fuera_v[F[~mascara_caras].ravel()] = True
    contorno = dentro_v & fuera_v
    sub_borde = np.zeros(m.nv, bool)
    # el borde libre de la región que coincide con el borde de la malla no es contorno de desfase
    d = fast_marching(m, contorno, limite=limite) if contorno.any() else np.full(m.nv, np.inf)
    del sub_borde
    signo = np.where(dentro_v & ~fuera_v, -1.0, 1.0)
    return signo * d
