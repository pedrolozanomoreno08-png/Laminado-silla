"""Simulación cinemática del drapeado (red de pescador / *fishnet*).

El tejido se modela como una red de hilos inextensibles articulada en los cruces. A partir del
punto semilla se trazan los dos hilos generadores (urdimbre i, trama j) según la estrategia y el
resto de nodos se obtiene por barridos: el nodo (i, j) queda a distancia h de (i∓1, j) y de
(i, j∓1) y sobre la superficie. Cada barrido se resuelve vectorizado (Newton en el plano
tangente + proyección).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geom.consulta import Consultor, base_tangente, proyectar_tangente

ESTRATEGIAS = {
    "geodesica": "Geodésica (camino más recto)",
    "plano": "Plano natural (corte con plano)",
    "proyeccion": "Dirección fija proyectada",
    "curva": "Curva de semilla",
    "energia": "Mínima energía",
}


@dataclass
class Dart:
    """Corte en V (rendija) definido por una polilínea 3D sobre la superficie, del borde a la punta."""
    puntos: np.ndarray
    nombre: str = "Dart"
    automatico: bool = False


@dataclass
class Red:
    h: float
    G: np.ndarray            # (ni, nj, 3) posiciones
    ok: np.ndarray           # (ni, nj) nodo válido
    i0: int                  # índice de la semilla
    j0: int
    celda_ok: np.ndarray = field(default=None)   # (ni-1, nj-1)

    def nodos_validos(self):
        return self.G[self.ok]

    def coordenadas(self):
        """Coordenadas de tejido (u, v) en mm de cada nodo."""
        ii, jj = np.meshgrid(np.arange(self.G.shape[0]), np.arange(self.G.shape[1]), indexing="ij")
        return (ii - self.i0) * self.h, (jj - self.j0) * self.h

    def angulo_cizalla(self):
        """Cizalla en cada nodo (grados): 90° − ángulo entre hilos, promedio de sus celdas."""
        G, ok = self.G, self.ok
        ni, nj, _ = G.shape
        gam = np.full((ni, nj), np.nan)
        acum = np.zeros((ni, nj))
        cnt = np.zeros((ni, nj))
        co = self.celda_ok
        P00, P10, P01, P11 = G[:-1, :-1], G[1:, :-1], G[:-1, 1:], G[1:, 1:]
        for (O, A, B, oi, oj) in ((P00, P10, P01, 0, 0), (P10, P11, P00, 1, 0),
                                  (P11, P01, P10, 1, 1), (P01, P00, P11, 0, 1)):
            a = A - O
            b = B - O
            c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-300)
            ang = np.degrees(np.arcsin(np.clip(np.abs(c), 0, 1)))
            sl = (slice(oi, ni - 1 + oi), slice(oj, nj - 1 + oj))
            acum[sl] += np.where(co, ang, 0)
            cnt[sl] += co
        gam = np.where(cnt > 0, acum / np.maximum(cnt, 1), np.nan)
        gam[~ok] = np.nan
        return gam


class _Darts:
    """Prueba vectorizada de cruce de segmentos con las rendijas."""

    def __init__(self, darts, h, consultor):
        self.activo = bool(darts)
        if not self.activo:
            return
        pts, tans, punta, ext = [], [], [], []
        for d in darts:
            P = np.asarray(d.puntos, float)
            # remuestrear finamente
            L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            s = np.linspace(0, L[-1], max(2, int(L[-1] / (h / 10)) + 1))
            Q = np.stack([np.interp(s, L, P[:, k]) for k in range(3)], 1)
            T = np.gradient(Q, axis=0)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-300)
            pts.append(Q)
            tans.append(T)
            pu = np.zeros(len(Q), bool)
            pu[-1] = True
            punta.append(pu)
            e0 = np.zeros(len(Q), bool)
            e0[0] = True
            ext.append(e0)
        self.P = np.vstack(pts)
        self.T = np.vstack(tans)
        self.punta = np.concatenate(punta)
        self.inicio = np.concatenate(ext)
        _, f, b, _ = consultor.mas_cercano(self.P)
        self.N = consultor.normal(f, b)
        self.arbol = cKDTree(self.P)
        self.h = h

    def cruza(self, A, B):
        """True para cada segmento A[k]-B[k] que atraviesa una rendija."""
        if not self.activo or len(A) == 0:
            return np.zeros(len(A), bool)
        M = 0.5 * (A + B)
        d, k = self.arbol.query(M)
        cerca = d < 1.5 * self.h
        D = self.P[k]
        lado = np.cross(self.N[k], self.T[k])
        sa = ((A - D) * lado).sum(1)
        sb = ((B - D) * lado).sum(1)
        cambia = (sa * sb) < 0
        # más allá de la punta o antes del inicio no hay corte
        mas_alla = self.punta[k] & (((M - D) * self.T[k]).sum(1) > 0)
        antes = self.inicio[k] & (((M - D) * self.T[k]).sum(1) < 0)
        return cerca & cambia & ~mas_alla & ~antes


def _trazar(consultor, P0, d0, h, n_max, regla, zona=None, darts=None, plano_n=None,
            curva=None, dir_fija=None):
    """Traza un hilo generador desde P0 con nodos cada h (cuerda). Devuelve (k,3)."""
    sub = 8
    ds = h / sub
    P = P0.copy()
    d = d0.copy()
    nodos = [P.copy()]
    if regla == "curva":
        nodos = _seguir_curva(consultor, P0, d0, h, n_max, curva)
        if 2 <= len(nodos) <= n_max:
            # más allá del final de la curva el hilo continúa por el camino más recto
            dfin = nodos[-1] - nodos[-2]
            resto = _trazar(consultor, nodos[-1], dfin / np.linalg.norm(dfin), h, n_max - len(nodos) + 1,
                            "geodesica", zona=zona, darts=darts)
            nodos = np.vstack([nodos, resto[1:]])
        return nodos
    prev = P.copy()
    for _ in range(n_max * sub * 3):
        X = P + d * ds
        if regla == "plano":
            X = X - ((X - P0) @ plano_n) * plano_n
        Q, n, en, dist = consultor.proyectar(X[None])
        if regla == "plano":
            for _ in range(3):
                q = Q[0] - ((Q[0] - P0) @ plano_n) * plano_n
                Q, n, en, dist = consultor.proyectar(q[None])
        if en[0] and dist[0] > 0.02 * ds:
            break
        Q = Q[0]
        n = n[0]
        paso = Q - P
        Lp = np.linalg.norm(paso)
        if Lp < 1e-12:
            break
        if regla == "geodesica":
            dn = paso / Lp
        elif regla == "plano":
            dn = np.cross(plano_n, n)
            if dn @ d < 0:
                dn = -dn
        elif regla == "proyeccion":
            dn = dir_fija - (dir_fija @ n) * n
            if dn @ d < 0:
                dn = -dn
        else:
            dn = paso / Lp
        dn = dn - (dn @ n) * n
        nd = np.linalg.norm(dn)
        if nd < 1e-9:
            break
        d = dn / nd
        prev, P = P, Q
        if np.linalg.norm(P - nodos[-1]) >= h:
            # bisección en el último tramo para que la cuerda sea exactamente h
            a = nodos[-1]
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(prev + mid * (P - prev) - a) < h:
                    lo = mid
                else:
                    hi = mid
            X = prev + 0.5 * (lo + hi) * (P - prev)
            Xq, _, _, _ = consultor.proyectar(X[None])
            # corregir la cuerda sobre la superficie (pequeño Newton en la dirección de avance)
            Xn = Xq[0]
            for _ in range(3):
                err = np.linalg.norm(Xn - a) - h
                if abs(err) < 1e-9 * h:
                    break
                Xn = consultor.proyectar((Xn - err * (Xn - a) / np.linalg.norm(Xn - a))[None])[0][0]
            nodos.append(Xn)
            P = Xn
            if zona is not None and not zona(Xn[None])[0]:
                break
            if darts is not None and darts.activo and darts.cruza(nodos[-2][None], Xn[None])[0]:
                nodos.pop()
                break
            if len(nodos) > n_max:
                break
    return np.array(nodos)


def _seguir_curva(consultor, P0, d0, h, n_max, curva):
    """Nodos a cuerda h a lo largo de una polilínea 3D (proyectada), en el sentido de d0."""
    C = consultor.proyectar(np.asarray(curva, float))[0]
    # parámetro de longitud de arco y punto de la curva más cercano a la semilla
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    k = np.argmin(np.linalg.norm(C - P0, axis=1))
    tang = C[min(k + 1, len(C) - 1)] - C[max(k - 1, 0)]
    sentido = 1 if tang @ d0 >= 0 else -1
    dens = np.linspace(0, L[-1], max(2, int(L[-1] / (h / 20)) + 1))
    D = np.stack([np.interp(dens, L, C[:, c]) for c in range(3)], 1)
    kd = np.argmin(np.linalg.norm(D - P0, axis=1))
    nodos = [P0.copy()]
    idx = kd
    while len(nodos) <= n_max:
        a = nodos[-1]
        j = idx
        while 0 <= j < len(D) and np.linalg.norm(D[j] - a) < h:
            j += sentido
        if not (0 <= j < len(D)):
            break
        # interpolar entre j-sentido y j
        p, q = D[j - sentido], D[j]
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(p + mid * (q - p) - a) < h:
                lo = mid
            else:
                hi = mid
        nodos.append(p + 0.5 * (lo + hi) * (q - p))
        idx = j
    return np.array(nodos)


def resolver_nodos(consultor, A, B, C, h, iters=20):
    """Resuelve vectorizado X sobre la superficie con |X−A| = |X−B| = h, cerca de A+B−C."""
    X0 = A + B - C
    X, n, _, _ = consultor.proyectar(X0)
    conv = np.zeros(len(A), bool)
    for _ in range(iters):
        t1, t2 = base_tangente(n)
        ra = X - A
        rb = X - B
        la = np.linalg.norm(ra, axis=1)
        lb = np.linalg.norm(rb, axis=1)
        f = np.stack([la - h, lb - h], 1)
        conv = np.abs(f).max(1) < 5e-3 * h      # 25 µm con h = 5 mm (sobre aristas vivas no se
        #                                         puede exigir más: la superficie es facetada)
        if conv.all():
            break
        ga = ra / la[:, None]
        gb = rb / lb[:, None]
        J = np.stack([np.stack([(ga * t1).sum(1), (ga * t2).sum(1)], 1),
                      np.stack([(gb * t1).sum(1), (gb * t2).sum(1)], 1)], 1)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        du = (-f[:, 0] * J[:, 1, 1] + f[:, 1] * J[:, 0, 1]) / det
        dv = (-f[:, 1] * J[:, 0, 0] + f[:, 0] * J[:, 1, 0]) / det
        paso = du[:, None] * t1 + dv[:, None] * t2
        # limitar el paso para estabilidad
        lp = np.linalg.norm(paso, axis=1, keepdims=True)
        paso = np.where(lp > 0.5 * h, paso * (0.5 * h / np.maximum(lp, 1e-300)), paso)
        X, n, _, _ = consultor.proyectar(X + paso)
    _, _, en, dist = consultor.proyectar(X)
    return X, n, conv, en & (dist > 1e-3 * h)


def drapear_red(consultor: Consultor, semilla, direccion, h, estrategia="geodesica",
                radio=None, zona=None, darts=None, curva_semilla=None,
                angulo_max=None, tam_max=600):
    """Red cinemática. Devuelve un objeto ``Red``.

    semilla: punto 3D; direccion: dirección de urdimbre (se proyecta al plano tangente).
    radio: extensión máxima (mm) desde la semilla; zona: función P->bool (zona de interés);
    angulo_max: cizalla máxima admisible para aceptar un nodo (bloqueo del tejido)."""
    P0, n0, _, _ = consultor.proyectar(np.asarray(semilla, float)[None])
    P0 = P0[0]
    n0 = n0[0]
    d0 = proyectar_tangente(np.asarray(direccion, float)[None], n0[None])[0]
    if estrategia == "curva" and curva_semilla is not None and len(curva_semilla) >= 2:
        # la curva de semilla define la urdimbre: su tangente en la semilla manda sobre la orientación
        C = np.asarray(curva_semilla, float)
        # la semilla debe estar sobre la curva: se lleva al punto más cercano de la curva
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        dens = np.linspace(0, L[-1], max(2, int(L[-1] / (h / 20)) + 1))
        D = np.stack([np.interp(dens, L, C[:, q]) for q in range(3)], 1)
        Pq, nq, _, _ = consultor.proyectar(D[[int(np.argmin(np.linalg.norm(D - P0, axis=1)))]])
        P0, n0 = Pq[0], nq[0]
        d0 = proyectar_tangente(np.asarray(direccion, float)[None], n0[None])[0]
        k = int(np.argmin(np.linalg.norm(C - P0, axis=1)))
        t = C[min(k + 1, len(C) - 1)] - C[max(k - 1, 0)]
        t = proyectar_tangente(t[None], n0[None])[0]
        d0 = t if t @ d0 >= 0 else -t
    d1 = np.cross(n0, d0)
    N = int(np.ceil((radio if radio else 1000.0) / h)) + 2
    N = min(N, tam_max)
    ni = nj = 2 * N + 1
    G = np.zeros((ni, nj, 3))
    ok = np.zeros((ni, nj), bool)
    excl = np.zeros((ni, nj), bool)
    i0 = j0 = N
    G[i0, j0] = P0
    ok[i0, j0] = True
    dz = _Darts(darts or [], h, consultor)
    if estrategia == "energia" or (estrategia == "curva" and (curva_semilla is None or len(curva_semilla) < 2)):
        estrategia = "geodesica"
    regla = estrategia
    for eje, d in ((0, d0), (1, d1)):
        for sg in (1, -1):
            pn = np.cross(d, n0)
            pn /= np.linalg.norm(pn)
            if regla == "curva" and eje == 1:
                r = "geodesica"
            else:
                r = regla
            nodos = _trazar(consultor, P0, sg * d, h, N, r, zona=None, darts=dz, plano_n=pn,
                            curva=curva_semilla, dir_fija=sg * d)
            for k, X in enumerate(nodos[1:N + 1], start=1):
                if eje == 0:
                    G[i0 + sg * k, j0] = X
                    ok[i0 + sg * k, j0] = True
                else:
                    G[i0, j0 + sg * k] = X
                    ok[i0, j0 + sg * k] = True
    cos_lim = np.sin(np.radians(angulo_max)) if angulo_max else np.sin(np.radians(80))
    # signo de referencia de la orientación de las celdas
    ref = -1.0
    # barridos por cuadrantes hasta que no haya progreso
    cuadrantes = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    for _ in range(8 * N):
        progreso = False
        for si, sj in cuadrantes:
            A_ok = np.zeros_like(ok)
            B_ok = np.zeros_like(ok)
            C_ok = np.zeros_like(ok)
            # A = (i - si, j), B = (i, j - sj), C = (i - si, j - sj)
            A_ok = np.roll(ok, si, axis=0)
            B_ok = np.roll(ok, sj, axis=1)
            C_ok = np.roll(np.roll(ok, si, axis=0), sj, axis=1)
            cand = A_ok & B_ok & C_ok & ~ok & ~excl
            # evitar el efecto de bordes del roll
            if si == 1:
                cand[0, :] = False
            else:
                cand[-1, :] = False
            if sj == 1:
                cand[:, 0] = False
            else:
                cand[:, -1] = False
            ii, jj = np.nonzero(cand)
            if len(ii) == 0:
                continue
            A = G[ii - si, jj]
            B = G[ii, jj - sj]
            C = G[ii - si, jj - sj]
            X, n, conv, fuera = resolver_nodos(consultor, A, B, C, h)
            # validación
            a = A - X
            b = B - X
            cang = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
            # la celda debe estar bien orientada respecto a la normal en X y en C: así se admiten
            # celdas dobladas sobre una arista viva (hasta ~90°) pero no celdas plegadas
            nC = consultor.proyectar(C)[1]
            cruz = np.cross(A - C, B - C) * (si * sj)
            orient = np.minimum((cruz * n).sum(1) * ref, (cruz * nC).sum(1) * ref) * ref
            # la celda debe conservar la orientación de la red (sin pliegues)
            # orientación respecto a la normal local (no a la de la semilla: en tubos o faldones
            # la superficie gira más de 90° y la normal de la semilla ya no sirve de referencia)
            bien = conv & ~fuera & (np.abs(cang) < cos_lim) & (orient * ref > 0)
            bien &= ((X - C) * (A + B - 2 * C)).sum(1) > 0
            if zona is not None:
                bien &= zona(X)
            if dz.activo:
                bien &= ~dz.cruza(A, X) & ~dz.cruza(B, X)
            G[ii[bien], jj[bien]] = X[bien]
            ok[ii[bien], jj[bien]] = True
            # nodos que no convergen o salen de la zona se excluyen
            malos = ~bien
            excl[ii[malos], jj[malos]] = True
            progreso |= bool(bien.any())
        if not progreso:
            break
        # los excluidos pueden alcanzarse desde otro cuadrante en la siguiente vuelta
        excl[:] = False
    red = Red(h, G, ok, i0, j0)
    co = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    if dz.activo:
        for (sa, sb) in (((slice(None, -1), slice(None, -1)), (slice(1, None), slice(None, -1))),
                         ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None))),
                         ((slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None))),
                         ((slice(None, -1), slice(1, None)), (slice(1, None), slice(1, None)))):
            A = G[sa].reshape(-1, 3)
            B = G[sb].reshape(-1, 3)
            m = co.ravel().copy()
            cr = np.zeros(len(A), bool)
            cr[m] = dz.cruza(A[m], B[m])
            co &= ~cr.reshape(co.shape)
    red.celda_ok = co
    return red
