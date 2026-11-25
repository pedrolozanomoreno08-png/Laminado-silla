"""Geometrías de prueba paramétricas: casco de ciclismo, tabla de surf y casco de kayak.

Todas en mm, superficie de molde con la normal hacia la cara de laminado, y con zonas con
nombre como las Named Selections de Ansys."""
from __future__ import annotations

import numpy as np

from ..geom.cortes import recortar
from ..geom.malla import TriMesh


def _rejilla(P, nu, nv):
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    par = ((np.add.outer(np.arange(nu - 1), np.arange(nv - 1))) % 2 == 0).ravel()
    F1 = np.where(par[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    F2 = np.where(par[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    return TriMesh(P.reshape(-1, 3), np.vstack([F1, F2]))


def _hacia(m: TriMesh, direccion):
    """Orienta las normales para que, de media, apunten hacia ``direccion``."""
    if (m.face_normals @ np.asarray(direccion, float) * m.face_areas).sum() < 0:
        return m.invertida()
    return m


# ----------------------------------------------------------------------------- casco ciclismo
def casco_ciclismo(tam=5.0):
    """Carcasa de casco de ciclismo (~270 x 210 x 150 mm), molde macho: normal hacia fuera.

    Elipsoide alargado hacia atrás, con el borde inferior más alto en la frente que en la nuca."""
    import pyvista as pv
    a, b, c = 135.0, 105.0, 150.0
    # icosfera: triángulos casi equiláteros y de tamaño uniforme (sin concentración en el polo)
    nsub = int(np.clip(np.ceil(np.log2(1.6 * np.pi * a / tam / 5)), 3, 7))
    ico = pv.Icosphere(radius=1.0, nsub=nsub)
    V = np.asarray(ico.points, float) * [a, b, c]
    V[:, 0] = np.where(V[:, 0] < 0, V[:, 0] * 1.18, V[:, 0])      # cola trasera alargada
    F = ico.faces.reshape(-1, 4)[:, 1:]
    m = _hacia(TriMesh(V, F).orientar_coherente(), [0, 0, 1])
    # borde inferior: frente a z = +25, nuca a z = -55, laterales sobre las orejas
    X = m.V
    zc = -15.0 + 40.0 * np.tanh(X[:, 0] / 60.0) - 25.0 * np.exp(-((X[:, 0] + 60) / 70) ** 2)
    m, _, _, _ = recortar(m, zc - X[:, 2])
    C = m.face_centroids
    sel = {
        "CORONA": np.nonzero(C[:, 2] > 0.72 * c)[0],
        "FRONTAL": np.nonzero(C[:, 0] > 0.55 * a)[0],
        "NUCA": np.nonzero(C[:, 0] < -0.6 * a)[0],
        "LATERAL_IZQ": np.nonzero(C[:, 1] > 0.6 * b)[0],
        "LATERAL_DER": np.nonzero(C[:, 1] < -0.6 * b)[0],
    }
    return m, sel


# --------------------------------------------------------------------------------- tabla surf
def tabla_surf(tam=8.0, largo=1830.0, ancho=520.0):
    """Cubierta de una tabla de surf de 6'0" (molde de la cubierta, normal hacia arriba).

    Planta con nariz en punta y cola cuadrada redondeada, rocker en nariz y cola y cubierta
    abombada hacia los cantos."""
    nx = int(largo / tam) + 1
    ny = int(ancho / tam) + 1
    xs = np.linspace(0, largo, nx)
    ys = np.linspace(-ancho / 2, ancho / 2, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    t = X / largo

    def semiancho(t):
        return ancho / 2 * np.clip(np.sin(np.pi * np.clip(t, 0, 1) ** 0.85) ** 0.55 * (1 - 0.18 * t), 0, None) \
            + 70.0 * (t < 0.08) * (1 - t / 0.08) ** 2 * 0      # (cola sin ensanchar)
    w = np.maximum(semiancho(t), 1e-3)
    rocker = 125.0 * t ** 4.2 + 38.0 * (1 - t) ** 3.5
    espesor = 58.0 * np.clip(np.sin(np.pi * np.clip(t * 0.96 + 0.02, 0, 1)), 0, None) ** 0.55
    u = np.clip(np.abs(Y) / w, 0, 1)
    # cubierta abombada con canto redondeado (pendiente finita en el borde)
    Z = rocker + espesor * (1 - u ** 3)
    m = _hacia(_rejilla(np.stack([X, Y, Z], -1), nx, ny), [0, 0, 1])
    phi = np.abs(m.V[:, 1]) - semiancho(m.V[:, 0] / largo) * 0.985
    # cola recortada (cola cuadrada) y nariz
    phi = np.maximum(phi, 12.0 - m.V[:, 0])
    m, _, _, _ = recortar(m, phi)
    C = m.face_centroids
    tc = C[:, 0] / largo
    wc = semiancho(tc)
    sel = {
        "PAD_COLA": np.nonzero(tc < 0.25)[0],
        "NARIZ": np.nonzero(tc > 0.8)[0],
        "CENTRO": np.nonzero((tc > 0.3) & (tc < 0.7) & (np.abs(C[:, 1]) < 0.55 * wc))[0],
        "CANTOS": np.nonzero(np.abs(C[:, 1]) > 0.85 * wc)[0],
    }
    return m, sel


# ------------------------------------------------------------------------------- casco kayak
def casco_kayak(tam=20.0, largo=2400.0, manga=580.0, puntal=280.0):
    """Casco de kayak (molde hembra: la normal apunta hacia dentro del barco, cara de laminado).

    Secciones superelípticas que se afinan hacia proa y popa; la borda queda en z = 0."""
    nx = int(largo / tam) + 1
    perim = np.pi * (manga / 2 + puntal)
    na = int(perim / tam) + 1
    xs = np.linspace(-0.485 * largo, 0.485 * largo, nx)
    al = np.linspace(-np.pi / 2, np.pi / 2, na)
    X, A = np.meshgrid(xs, al, indexing="ij")
    s = 2 * X / largo
    asim = np.where(s > 0, 1.0, 0.85)                   # proa más fina que popa
    b = manga / 2 * np.clip(1 - np.abs(s) ** (2.2 * asim), 0, None) ** 0.75
    d = puntal * np.clip(1 - np.abs(s) ** 3.0, 0, None) ** 0.5 + 25.0
    n = 3.0
    Y = b * np.sign(np.sin(A)) * np.abs(np.sin(A)) ** (2 / n)
    Z = -d * np.abs(np.cos(A)) ** (2 / n) + 40.0 * s ** 4         # arrufo: proa y popa suben
    m = _hacia(_rejilla(np.stack([X, Y, Z], -1), nx, na), [0, 0, 1])
    C = m.face_centroids
    tc = 2 * C[:, 0] / largo
    ang = np.arctan2(C[:, 1], -(C[:, 2] - 40.0 * tc ** 4))
    sel = {
        "QUILLA": np.nonzero(np.abs(ang) < 0.18)[0],
        "FONDO": np.nonzero(np.abs(ang) < 0.6)[0],
        "COSTADO_IZQ": np.nonzero(ang > 0.6)[0],
        "COSTADO_DER": np.nonzero(ang < -0.6)[0],
        "PROA": np.nonzero(tc > 0.6)[0],
        "POPA": np.nonzero(tc < -0.6)[0],
    }
    return m, sel


# ------------------------------------------------------------------- formas de aristas vivas
def piramide(tam=5.0, lado=300.0, altura=200.0):
    """Pirámide de base cuadrada sin base (molde macho, normal hacia fuera). Aristas vivas."""
    a = lado / 2
    n = int(lado / tam) // 2 * 2 + 1
    xs = np.linspace(-a, a, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = altura * (1 - np.maximum(np.abs(X), np.abs(Y)) / a)
    idx = np.arange(n * n).reshape(n, n)
    i0, j0 = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    p, q, r, s = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    # diagonal de cada celda alineada con las aristas |x| = |y| de la pirámide
    cx = xs[i0] + xs[1] - xs[0]
    cy = xs[j0] + xs[1] - xs[0]
    diag_xy = (cx - (xs[1] - xs[0]) / 2) * (cy - (xs[1] - xs[0]) / 2) > 0
    F1 = np.where(diag_xy[..., None], np.stack([p, q, r], -1), np.stack([p, q, s], -1)).reshape(-1, 3)
    F2 = np.where(diag_xy[..., None], np.stack([p, r, s], -1), np.stack([q, r, s], -1)).reshape(-1, 3)
    m = _hacia(TriMesh(np.stack([X, Y, Z], -1).reshape(-1, 3), np.vstack([F1, F2])), [0, 0, 1])
    C = m.face_centroids
    sel = {"CARA_XPOS": np.nonzero((C[:, 0] > np.abs(C[:, 1])))[0],
           "CARA_XNEG": np.nonzero((-C[:, 0] > np.abs(C[:, 1])))[0],
           "CARA_YPOS": np.nonzero((C[:, 1] > np.abs(C[:, 0])))[0],
           "CARA_YNEG": np.nonzero((-C[:, 1] > np.abs(C[:, 0])))[0],
           "VERTICE": np.nonzero(C[:, 2] > 0.7 * altura)[0]}
    return m, sel


def caja_abierta(tam=5.0, lado=200.0, alto=200.0):
    """Cubo sin tapa, como molde macho (caja boca abajo: techo + 4 paredes, abierto abajo).

    Aristas y esquinas vivas a 90°: el caso clásico donde el tejido necesita darts en las
    esquinas."""
    from ..geom.malla import fusionar_vertices
    a = lado / 2
    n = int(lado / tam) + 1
    nz = int(alto / tam) + 1
    t = np.linspace(-a, a, n)
    z = np.linspace(0, alto, nz)
    partes = []

    def cara(P):
        nu, nv = P.shape[:2]
        idx = np.arange(nu * nv).reshape(nu, nv)
        p, q, r, s = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        return P.reshape(-1, 3), np.vstack([np.c_[p, q, r], np.c_[p, r, s]])
    U, W = np.meshgrid(t, t, indexing="ij")
    partes.append(cara(np.stack([U, W, np.full_like(U, alto)], -1)))            # techo
    U, Z = np.meshgrid(t, z, indexing="ij")
    partes.append(cara(np.stack([U, np.full_like(U, a), Z], -1)))
    partes.append(cara(np.stack([U, np.full_like(U, -a), Z], -1)))
    partes.append(cara(np.stack([np.full_like(U, a), U, Z], -1)))
    partes.append(cara(np.stack([np.full_like(U, -a), U, Z], -1)))
    Vs, Fs, off = [], [], 0
    for P, F in partes:
        Vs.append(P)
        Fs.append(F + off)
        off += len(P)
    V, F, _ = fusionar_vertices(np.vstack(Vs), np.vstack(Fs), 1e-6)
    m = TriMesh(V, F).orientar_coherente()
    # normal hacia fuera: en el techo debe apuntar a +z
    techo = m.face_centroids[:, 2] > alto - 1e-6
    if (m.face_normals[techo, 2]).mean() < 0:
        m = m.invertida()
    C = m.face_centroids
    sel = {"TECHO": np.nonzero(C[:, 2] > alto - 1e-6)[0],
           "PARED_XPOS": np.nonzero(C[:, 0] > a - 1e-6)[0], "PARED_XNEG": np.nonzero(C[:, 0] < -a + 1e-6)[0],
           "PARED_YPOS": np.nonzero(C[:, 1] > a - 1e-6)[0], "PARED_YNEG": np.nonzero(C[:, 1] < -a + 1e-6)[0]}
    return m, sel


def cono(tam=5.0, radio=150.0, altura=250.0):
    """Cono circular sin base (molde macho, normal hacia fuera)."""
    gen = np.hypot(radio, altura)
    ns = int(gen / tam) + 1
    nt = int(2 * np.pi * radio / tam)
    s = np.linspace(0.0, 1.0, ns)                # 0 = vértice, 1 = base
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    S, T = np.meshgrid(s, th, indexing="ij")
    P = np.stack([radio * S * np.cos(T), radio * S * np.sin(T), altura * (1 - S)], -1)
    idx = np.arange(ns * nt).reshape(ns, nt)
    a_, b_ = idx[:-1], idx[1:]
    a1, b1 = np.roll(a_, -1, 1), np.roll(b_, -1, 1)
    F = np.vstack([np.c_[a_.ravel(), b_.ravel(), b1.ravel()], np.c_[a_.ravel(), b1.ravel(), a1.ravel()]])
    from ..geom.malla import fusionar_vertices
    V, F, _ = fusionar_vertices(P.reshape(-1, 3), F, 1e-6)
    m = _hacia(TriMesh(V, F).orientar_coherente(), [0, 0, 1])
    C = m.face_centroids
    sel = {"PUNTA": np.nonzero(C[:, 2] > 0.6 * altura)[0],
           "FALDA": np.nonzero(C[:, 2] < 0.35 * altura)[0],
           "MITAD_XPOS": np.nonzero(C[:, 0] > 0)[0]}
    return m, sel
