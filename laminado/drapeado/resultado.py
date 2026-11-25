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
    with np.errstate(all="ignore"):
        return _invertir(P00, P10, P01, P11, X, s, t, iters)


def _invertir(P00, P10, P01, P11, X, s, t, iters):
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
        s = np.clip(s - ds, -3, 4)
        t = np.clip(t - dt, -3, 4)
    res = np.linalg.norm(_bilineal(P00, P10, P01, P11, s, t) - X, axis=1)
    res = np.where(np.isfinite(res), res, np.inf)
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
    X = malla.face_centroids
    mejor, mejor_r = _asignar_por_indices(red, malla)
    filas = np.arange(len(X))
    a_m, b_m = mejor[:, 0], mejor[:, 1]
    # tolerancia de media celda: una celda doblada sobre una arista viva no es plana
    cubierta = mejor_r < 0.5 * h
    # vértices: en la celda de cada cara incidente
    F = malla.F
    uv_acc = np.zeros((malla.nv, 2))
    w_acc = np.zeros(malla.nv)
    for kk in range(3):
        v = F[:, kk]
        a, b = a_m, b_m
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


def _asignar_por_indices(red: Red, malla: TriMesh, radio=None):
    """Celda (i, j) de la red para cada cara, siguiendo la continuidad del tejido.

    Recorre las caras en anchura desde la cara de la semilla; cada cara busca su celda sólo
    entre las vecinas (±radio en índices) de la celda de su cara padre. Así, en superficies
    cerradas donde la red da varias vueltas y se solapa, cada cara se queda en la misma
    «vuelta» que sus vecinas. Devuelve (celda (nf,2), penalización (nf,))."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order
    h = red.h
    G = red.G
    co = red.celda_ok
    ni, nj = co.shape
    X = malla.face_centroids
    nf = malla.nf
    if radio is None:
        # entre caras vecinas el salto en celdas es ~ tamaño de cara / h
        radio = int(np.clip(np.ceil(1.5 * float(np.percentile(malla.edge_lengths, 90)) / h) + 1, 2, 12))
    ef = malla.edge_faces
    ok = ef[:, 1] >= 0
    A = coo_matrix((np.ones(ok.sum()), (ef[ok, 0], ef[ok, 1])), shape=(nf, nf)).tocsr()
    A = A + A.T
    celda = np.full((nf, 2), -1, np.int64)
    pen = np.full(nf, np.inf)
    off = np.arange(-radio, radio + 1)
    di, dj = [x.ravel() for x in np.meshgrid(off, off, indexing="ij")]

    def evaluar(caras, base):
        """Mejor celda entre las vecinas de ``base`` (k,2) para las caras dadas."""
        I = np.clip(base[:, :1] + di[None], 0, ni - 1)
        J = np.clip(base[:, 1:] + dj[None], 0, nj - 1)
        valida = co[I, J]
        P = np.repeat(X[caras], I.shape[1], axis=0)
        Ir, Jr = I.ravel(), J.ravel()
        s_, t_, r = invertir_bilineal(G[Ir, Jr], G[Ir + 1, Jr], G[Ir, Jr + 1], G[Ir + 1, Jr + 1], P)
        fuera = np.maximum.reduce([-s_, s_ - 1, -t_, t_ - 1, np.zeros_like(s_)])
        p_ = (r + fuera * h * 2).reshape(I.shape)
        p_ = np.where(valida, p_, np.inf)
        k = np.argmin(p_, axis=1)
        fil = np.arange(len(caras))
        return np.c_[I[fil, k], J[fil, k]], p_[fil, k]

    sin_asignar = np.ones(nf, bool)
    # puede haber varias componentes (no debería): se arranca desde la cara más cercana a la
    # semilla en cada una
    P0 = G[red.i0, red.j0]
    while sin_asignar.any():
        cand = np.nonzero(sin_asignar)[0]
        f0 = int(cand[np.argmin(np.linalg.norm(X[cand] - P0, axis=1))])
        orden, pred = breadth_first_order(A, f0, directed=False, return_predecessors=True)
        # semilla: búsqueda más amplia alrededor de (i0, j0)
        base0 = np.array([[red.i0 - 1, red.j0 - 1]])
        c0, p0 = evaluar(np.array([f0]), base0)
        if not np.isfinite(p0[0]) or p0[0] > 0.5 * h:
            # la cara de la semilla no está en la red: celda válida más cercana en el espacio
            ii, jj = np.nonzero(co)
            cen = 0.25 * (G[ii, jj] + G[ii + 1, jj] + G[ii, jj + 1] + G[ii + 1, jj + 1])
            k = int(np.argmin(np.linalg.norm(cen - X[f0], axis=1)))
            c0, p0 = evaluar(np.array([f0]), np.array([[ii[k], jj[k]]]))
        celda[f0] = c0[0]
        pen[f0] = p0[0]
        sin_asignar[f0] = False
        # niveles de la búsqueda en anchura
        nivel = np.zeros(nf, np.int64)
        for f in orden[1:]:
            nivel[f] = nivel[pred[f]] + 1
        resto = orden[1:]
        if len(resto):
            niv = nivel[resto]
            for L in range(1, int(niv.max()) + 1):
                caras = resto[niv == L]
                padres = pred[caras]
                base = celda[padres]
                c, p_ = evaluar(caras, base)
                # si no hay celda válida se hereda la del padre (zona no alcanzada)
                malo = ~np.isfinite(p_)
                c[malo] = base[malo]
                celda[caras] = c
                pen[caras] = np.where(malo, np.inf, p_)
        sin_asignar[orden] = False
    pen = np.where(np.isfinite(pen), pen, 1e9)
    return celda, pen


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
    de_red: np.ndarray | None = None          # vértices cuyas (u, v) salen directamente de la red

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
        """Ángulo de cizalla por cara (grados, con signo): 90° − ángulo entre hilos.

        Donde el tejido viene directamente de la red cinemática, se interpola la cizalla de los
        nodos de la red en la posición de tejido (u, v) de la cara (exacta y sin el ruido que
        introduce derivar dentro de una celda). En el resto (mínima energía, zonas completadas,
        piezas) se usa el gradiente de (u, v) en el triángulo."""
        a, b, det = self.gradientes()
        ea, eb = self.fibras()
        cosg = (ea * eb).sum(1)
        g = np.degrees(np.arcsin(np.clip(cosg, -1, 1)))
        g = np.where(det <= 0, 90.0, g)          # cara invertida: pliegue
        if self.red is not None and self.de_red is not None:
            caras = self.de_red[self.malla.F].all(1) & (det > 0)
            if caras.any():
                gi = _interpolar_red(self.red, self.uv[self.malla.F[caras]].mean(1))
                ok = np.isfinite(gi)
                idx = np.nonzero(caras)[0][ok]
                g[idx] = np.sign(g[idx] + 1e-12) * gi[ok]
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

    def solape_plano(self, mascara=None):
        """Área (mm²) en la que el patrón plano se superpone consigo mismo: tela que haría falta
        dos veces en el mismo sitio del plano (p.ej. un dart en una zona en silla de montar, donde
        los bordes del corte se cruzan en vez de abrirse). Un patrón así no se puede cortar de una
        sola pieza."""
        from shapely import polygons, union_all
        uv, F = self.uv, self.malla.F
        if mascara is not None:
            F = F[mascara]
        T = uv[F]
        a = 0.5 * np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                         - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
        ok = a > 1e-9
        if not ok.any():
            return 0.0
        u = union_all(polygons(T[ok])).area
        return float(max(0.0, a[ok].sum() - u))

    def resumen(self, aviso=None, limite=None, mascara=None):
        """Estadísticas de producibilidad (sobre las caras de ``mascara`` si se da, p.ej. el neto)."""
        g = np.abs(self.cizalla())
        A = self.malla.face_areas
        cub = self.cara_cubierta()
        if mascara is not None and np.any(mascara):
            g, A, cub = g[mascara], A[mascara], cub[mascara]
        out = {
            "estrategia": self.estrategia,
            "cizalla_max": float(np.nanmax(g)) if len(g) else 0.0,
            "cizalla_media": float(np.average(g, weights=A)) if len(g) else 0.0,
            "area_superficie_mm2": float(A.sum()),
            "area_plana_mm2": self.area_plana(),
            "cubierto_%": float(100 * A[cub].sum() / max(A.sum(), 1e-300)),
            "n_darts": len(self.darts),
            "solape_patron_%": float(100 * self.solape_plano() / max(self.area_plana(), 1e-300)),
        }
        if aviso is not None:
            out["area_aviso_%"] = float(100 * A[g > aviso].sum() / A.sum())
        if limite is not None:
            out["area_limite_%"] = float(100 * A[g > limite].sum() / A.sum())
        return out


def _interpolar_red(red, uv):
    """Interpolación bilineal de la cizalla nodal de la red en coordenadas de tejido (mm)."""
    if not hasattr(red, "_cizalla_nodos"):
        red._cizalla_nodos = red.angulo_cizalla()
    gn = red._cizalla_nodos
    a = uv[:, 0] / red.h + red.i0
    b = uv[:, 1] / red.h + red.j0
    i = np.floor(a).astype(int)
    j = np.floor(b).astype(int)
    ni, nj = gn.shape
    dentro = (i >= 0) & (j >= 0) & (i < ni - 1) & (j < nj - 1)
    out = np.full(len(uv), np.nan)
    i, j, s, t = i[dentro], j[dentro], (a - np.floor(a))[dentro], (b - np.floor(b))[dentro]
    v = ((1 - s) * (1 - t) * gn[i, j] + s * (1 - t) * gn[i + 1, j] + (1 - s) * t * gn[i, j + 1]
         + s * t * gn[i + 1, j + 1])
    out[dentro] = v
    return out


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
