"""Casco de silla de montar paramétrico para demostración y pruebas.

Ejes: X a lo largo del lomo del caballo (positivo hacia la cruz/pomo), Y lateral,
Z hacia arriba. Unidades: mm. La superficie es la cara exterior (molde macho).

El casco es un arco sobre el lomo cuyo radio varía a lo largo (más cerrado bajo el pomo),
con el asiento hundido, el borrén (trasera) y el pomo (delantera) levantados y los faldones
laterales cayendo hasta ~95° respecto a la horizontal.
"""
from __future__ import annotations

import numpy as np

from ..geom.malla import TriMesh

LARGO = 500.0      # mm a lo largo del lomo
ARCO = 230.0       # mm de desarrollo del arco a cada lado del eje


def _radio(x):
    t = x / (LARGO / 2)
    return 150.0 - 35.0 * np.exp(-((t - 0.85) / 0.25) ** 2) - 10.0 * np.exp(-((t + 0.9) / 0.3) ** 2)


def _altura(x):
    t = x / (LARGO / 2)
    asiento = 35.0 * t ** 2
    borren = 55.0 * np.exp(-((t + 0.92) / 0.18) ** 2)
    pomo = 45.0 * np.exp(-((t - 0.88) / 0.16) ** 2)
    return asiento + borren + pomo


def punto(x, s):
    R = _radio(x)
    h = _altura(x)
    a = s / R
    # el faldón se abre ligeramente hacia fuera abajo (curvatura de las costillas del caballo)
    abre = 0.12 * np.clip(np.abs(a) - 1.0, 0, None) ** 2 * np.sign(a)
    y = R * np.sin(a + abre)
    z = h - R * (1 - np.cos(a + abre))
    return np.stack([x, y, z], -1)


def casco_silla(tam: float = 6.0, contorno: bool = True) -> TriMesh:
    """Malla del casco con elementos de ~``tam`` mm.

    Si ``contorno`` es True el borde inferior de los faldones y las puntas se redondean
    (contorno real de un casco), si no se devuelve el rectángulo paramétrico completo."""
    nx = int(np.ceil(LARGO / tam)) + 1
    # número impar de líneas: hay una línea de malla en y = 0 y el patrón de diagonales
    # alternas queda simétrico respecto al plano medio (silla exactamente simétrica)
    ns = 2 * int(np.ceil(ARCO / tam)) + 1
    xs = np.linspace(-LARGO / 2, LARGO / 2, nx)
    ss = np.linspace(-ARCO, ARCO, ns)
    X, S = np.meshgrid(xs, ss, indexing="ij")
    P = punto(X, S).reshape(-1, 3)
    idx = np.arange(nx * ns).reshape(nx, ns)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    par = ((np.add.outer(np.arange(nx - 1), np.arange(ns - 1))) % 2 == 0).ravel()
    F1 = np.where(par[:, None], np.stack([a, b, c], 1), np.stack([a, b, d], 1))
    F2 = np.where(par[:, None], np.stack([a, c, d], 1), np.stack([b, c, d], 1))
    F = np.vstack([F1, F2])
    m = TriMesh(P, F)
    # normal hacia fuera (+Z en la cumbrera)
    top = np.argmax(m.face_centroids[:, 2])
    if m.face_normals[top, 2] < 0:
        m = m.invertida()
    if contorno:
        from ..geom.cortes import recortar
        u = X.ravel() / (LARGO / 2)
        v = S.ravel() / ARCO
        # superelipse en el dominio paramétrico: puntas redondeadas
        phi = (np.abs(u) ** 4 + np.abs(v) ** 4) ** 0.25 - 0.995
        m, _, _, _ = recortar(m, phi)
    return m


def selecciones_silla(m: TriMesh) -> dict:
    """Zonas con nombre como las tendría un modelo de Ansys (componentes de elementos)."""
    c = m.face_centroids
    x = c[:, 0]
    t = x / (LARGO / 2)
    sel = {
        "ASIENTO": np.nonzero((np.abs(t) < 0.55) & (c[:, 2] > _altura(x) - 60))[0],
        "POMO": np.nonzero(t > 0.6)[0],
        "BORREN": np.nonzero(t < -0.6)[0],
        "FALDON_IZQ": np.nonzero(c[:, 1] > 90)[0],
        "FALDON_DER": np.nonzero(c[:, 1] < -90)[0],
    }
    return {k: v for k, v in sel.items() if len(v)}
