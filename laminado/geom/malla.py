"""Malla triangular con las operaciones topológicas y diferenciales que usa el programa."""
from __future__ import annotations

from functools import cached_property

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


class TriMesh:
    """Malla de triángulos (V: (n,3) float, F: (m,3) int).

    Las propiedades derivadas se calculan una vez (la malla se trata como inmutable:
    cualquier operación que la cambie devuelve una malla nueva)."""

    def __init__(self, V, F):
        self.V = np.ascontiguousarray(V, dtype=float)
        self.F = np.ascontiguousarray(F, dtype=np.int64).reshape(-1, 3)

    # ------------------------------------------------------------ tamaños
    @property
    def nv(self) -> int:
        return len(self.V)

    @property
    def nf(self) -> int:
        return len(self.F)

    def copia(self) -> "TriMesh":
        return TriMesh(self.V.copy(), self.F.copy())

    # --------------------------------------------------------- geometría
    @cached_property
    def _cruz(self):
        a, b, c = self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]
        return np.cross(b - a, c - a)

    @cached_property
    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self._cruz, axis=1)

    @property
    def area(self) -> float:
        return float(self.face_areas.sum())

    @cached_property
    def face_normals(self) -> np.ndarray:
        n = self._cruz
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    @cached_property
    def face_centroids(self) -> np.ndarray:
        return self.V[self.F].mean(axis=1)

    @cached_property
    def vertex_normals(self) -> np.ndarray:
        """Normales nodales ponderadas por ángulo (robustas en mallas irregulares)."""
        V, F = self.V, self.F
        n = np.zeros_like(V)
        fn = self.face_normals
        for k in range(3):
            a = V[F[:, (k + 1) % 3]] - V[F[:, k]]
            b = V[F[:, (k + 2) % 3]] - V[F[:, k]]
            ang = np.arccos(np.clip((a * b).sum(1) / np.maximum(
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300), -1, 1))
            np.add.at(n, F[:, k], fn * ang[:, None])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    @cached_property
    def vertex_areas(self) -> np.ndarray:
        a = np.zeros(self.nv)
        for k in range(3):
            np.add.at(a, self.F[:, k], self.face_areas / 3.0)
        return a

    # ---------------------------------------------------------- topología
    @cached_property
    def _aristas(self):
        F = self.F
        E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        Es = np.sort(E, axis=1)
        uniq, inv = np.unique(Es, axis=0, return_inverse=True)
        inv = inv.ravel()
        fe = inv.reshape(3, -1).T                         # aristas de cada cara (m,3)
        caras = np.tile(np.arange(len(F)), 3)
        ef = -np.ones((len(uniq), 2), np.int64)
        orden = np.argsort(inv, kind="stable")
        inv_s = inv[orden]
        car_s = caras[orden]
        primero = np.r_[True, inv_s[1:] != inv_s[:-1]]
        ef[inv_s[primero], 0] = car_s[primero]
        seg = ~primero
        ef[inv_s[seg], 1] = car_s[seg]
        return uniq, fe, ef

    @property
    def edges(self) -> np.ndarray:
        return self._aristas[0]

    @property
    def face_edges(self) -> np.ndarray:
        return self._aristas[1]

    @property
    def edge_faces(self) -> np.ndarray:
        return self._aristas[2]

    @cached_property
    def edge_lengths(self) -> np.ndarray:
        E = self.edges
        return np.linalg.norm(self.V[E[:, 1]] - self.V[E[:, 0]], axis=1)

    @cached_property
    def boundary_edges_mask(self) -> np.ndarray:
        return self.edge_faces[:, 1] < 0

    @cached_property
    def boundary_vertices_mask(self) -> np.ndarray:
        m = np.zeros(self.nv, bool)
        m[self.edges[self.boundary_edges_mask].ravel()] = True
        return m

    @cached_property
    def adyacencia(self) -> sp.csr_matrix:
        """Matriz de adyacencia de vértices con la longitud de arista como peso."""
        E = self.edges
        w = self.edge_lengths
        A = sp.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                          shape=(self.nv, self.nv))
        return A.tocsr()

    @cached_property
    def vertice_caras(self) -> sp.csr_matrix:
        """Incidencia vértice-cara (nv, nf)."""
        F = self.F
        return sp.csr_matrix((np.ones(3 * self.nf), (F.ravel(), np.repeat(np.arange(self.nf), 3))),
                             shape=(self.nv, self.nf))

    def boundary_loops(self) -> list[np.ndarray]:
        """Bucles de borde orientados como las caras (lista de arrays de vértices)."""
        F = self.F
        siguiente = {}
        # arista dirigida (a->b) de cada cara; es de borde si (b->a) no existe
        dirigidas = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        clave = set(map(tuple, dirigidas.tolist()))
        for a, b in dirigidas.tolist():
            if (b, a) not in clave:
                siguiente.setdefault(a, []).append(b)
        bucles = []
        usados = set()
        for ini in list(siguiente):
            for primero in siguiente[ini]:
                if (ini, primero) in usados:
                    continue
                bucle = [ini]
                a, b = ini, primero
                while True:
                    usados.add((a, b))
                    if b == ini:
                        break
                    bucle.append(b)
                    nxt = [c for c in siguiente.get(b, []) if (b, c) not in usados]
                    if not nxt:
                        break
                    a, b = b, nxt[0]
                    if len(bucle) > len(F) * 3:
                        break
                if len(bucle) >= 3:
                    bucles.append(np.array(bucle))
        return bucles

    def componentes(self):
        """(número, etiqueta por cara) de componentes conexas por aristas."""
        ef = self.edge_faces
        ok = ef[:, 1] >= 0
        A = sp.coo_matrix((np.ones(ok.sum()), (ef[ok, 0], ef[ok, 1])), shape=(self.nf, self.nf))
        from scipy.sparse.csgraph import connected_components
        return connected_components(A, directed=False)

    # ------------------------------------------------------------ edición
    def submalla(self, mascara_caras):
        """(malla, índice de vértice original de cada vértice nuevo, caras originales)."""
        caras = np.nonzero(mascara_caras)[0]
        Fs = self.F[caras]
        usados, inv = np.unique(Fs.ravel(), return_inverse=True)
        return TriMesh(self.V[usados], inv.reshape(-1, 3)), usados, caras

    def invertida(self) -> "TriMesh":
        return TriMesh(self.V.copy(), self.F[:, ::-1].copy())

    def orientar_coherente(self) -> "TriMesh":
        """Orienta las caras de forma coherente dentro de cada componente (BFS por aristas)."""
        F = self.F.copy()
        ef = self.edge_faces
        fe = self.face_edges
        visit = np.zeros(self.nf, bool)
        from collections import deque
        for s in range(self.nf):
            if visit[s]:
                continue
            visit[s] = True
            q = deque([s])
            while q:
                f = q.popleft()
                for e in fe[f]:
                    for g in ef[e]:
                        if g < 0 or g == f or visit[g]:
                            continue
                        a, b = self.edges[e]
                        fa = list(F[f])
                        ga = list(F[g])
                        df = fa[(fa.index(a) + 1) % 3] == b
                        dg = ga[(ga.index(a) + 1) % 3] == b
                        if df == dg:
                            F[g] = F[g][::-1]
                        visit[g] = True
                        q.append(g)
        return TriMesh(self.V, F)

    # ---------------------------------------------------- operadores FEM
    @cached_property
    def laplaciano_cot(self) -> sp.csr_matrix:
        """Laplaciano cotangente (semidefinido negativo)."""
        V, F = self.V, self.F
        I, J, W = [], [], []
        for k in range(3):
            i, j, o = F[:, (k + 1) % 3], F[:, (k + 2) % 3], F[:, k]
            a = V[i] - V[o]
            b = V[j] - V[o]
            cot = (a * b).sum(1) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-300)
            I += [i, j]
            J += [j, i]
            W += [0.5 * cot, 0.5 * cot]
        L = sp.coo_matrix((np.concatenate(W), (np.concatenate(I), np.concatenate(J))),
                          shape=(self.nv, self.nv)).tocsr()
        return L - sp.diags(np.asarray(L.sum(1)).ravel())

    @cached_property
    def masa(self) -> sp.dia_matrix:
        return sp.diags(self.vertex_areas)

    def arbol_vertices(self) -> cKDTree:
        return cKDTree(self.V)

    def interpolar_caras_a_vertices(self, valores):
        v = np.asarray(valores, float)
        w = self.face_areas
        out = np.zeros((self.nv,) + v.shape[1:])
        s = np.zeros(self.nv)
        for k in range(3):
            np.add.at(out, self.F[:, k], v * (w if v.ndim == 1 else w[:, None]))
            np.add.at(s, self.F[:, k], w)
        return out / (np.maximum(s, 1e-300) if v.ndim == 1 else np.maximum(s, 1e-300)[:, None])


def fusionar_vertices(V, F, tol=1e-6):
    """Une vértices a menos de ``tol``; devuelve (V, F, mapa viejo->nuevo)."""
    arbol = cKDTree(V)
    pares = arbol.query_pairs(tol, output_type="ndarray")
    n = len(V)
    padre = np.arange(n)

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    for a, b in pares:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)
    r = np.array([raiz(i) for i in range(n)])
    u, mapa = np.unique(r, return_inverse=True)
    F2 = mapa[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V[u], F2[ok], mapa
