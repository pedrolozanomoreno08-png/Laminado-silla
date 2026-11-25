"""Cuadro de bicicleta de carretera paramétrico (talla ~54) para pruebas.

Superficie exterior cerrada obtenida como isosuperficie de la unión suave de tubos
(cápsulas). Ejes: X hacia delante, Y a la izquierda, Z arriba; origen en el pedalier. mm.
"""
from __future__ import annotations

import numpy as np

from ..geom.malla import TriMesh, fusionar_vertices

# (nombre, punto A, punto B, radio A, radio B)
TUBOS = [
    ("TUBO_DIAGONAL", (0, 0, 0), (424, 0, 428), 27, 22),
    ("TUBO_SUPERIOR", (-140, 0, 495), (392, 0, 535), 16, 18),
    ("TUBO_SILLIN", (0, 0, 0), (-152, 0, 515), 17, 15),
    ("TUBO_DIRECCION", (432, 0, 405), (385, 0, 560), 25, 20),
    ("VAINA_IZQ", (-25, 36, 8), (-405, 64, 70), 12, 8),
    ("VAINA_DER", (-25, -36, 8), (-405, -64, 70), 12, 8),
    ("TIRANTE_IZQ", (-132, 12, 455), (-400, 62, 80), 9, 7),
    ("TIRANTE_DER", (-132, -12, 455), (-400, -62, 80), 9, 7),
    ("CAJA_PEDALIER", (0, -36, 0), (0, 36, 0), 23, 23),
]


def _sdf_capsula(P, A, B, ra, rb):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    ab = B - A
    t = np.clip(((P - A) @ ab) / (ab @ ab), 0, 1)
    Q = A + t[:, None] * ab
    r = ra + (rb - ra) * t
    return np.linalg.norm(P - Q, axis=1) - r, t, Q


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


# extremos abiertos de un cuadro real: (tubo, extremo 0=A / 1=B)
BOCAS = [("TUBO_DIRECCION", 0), ("TUBO_DIRECCION", 1), ("TUBO_SILLIN", 1),
         ("CAJA_PEDALIER", 0), ("CAJA_PEDALIER", 1)]


def cuadro_bicicleta(paso=3.5, k=10.0, abierto=False):
    """Superficie exterior del cuadro. ``abierto``: con las bocas de la dirección, del tubo de
    sillín y de la caja de pedalier abiertas (como el molde de un cuadro real), en vez de
    cerradas con casquetes semiesféricos."""
    import pyvista as pv
    lo = np.array([-440.0, -95.0, -45.0])
    hi = np.array([480.0, 95.0, 600.0])
    dims = np.ceil((hi - lo) / paso).astype(int) + 1
    g = pv.ImageData(dimensions=dims, spacing=(paso,) * 3, origin=lo)
    P = np.asarray(g.points)
    d = None
    for _, A, B, ra, rb in TUBOS:
        s, _, _ = _sdf_capsula(P, A, B, ra, rb)
        d = s if d is None else _smin(d, s, k)
    g["sdf"] = d
    s = g.contour([0.0], scalars="sdf").triangulate()
    F0 = s.faces.reshape(-1, 4)[:, 1:]
    V0, F0, _ = fusionar_vertices(np.asarray(s.points, float), F0, 1e-6)
    s = pv.PolyData(V0, np.c_[np.full(len(F0), 3), F0].ravel())
    # sin diezmar: los triángulos de marching cubes tienen tamaño acotado (~paso), lo que
    # mantiene la superficie fiel para el drapeado; sólo se suaviza el escalonado
    s = s.smooth_taubin(n_iter=40, pass_band=0.08)
    F = s.faces.reshape(-1, 4)[:, 1:]
    V, F, _ = fusionar_vertices(np.asarray(s.points, float), F, 1e-6)
    m = TriMesh(V, F).orientar_coherente()
    # normales hacia fuera: volumen con signo positivo
    vol = np.einsum("ij,ij->i", m.V[m.F[:, 0]], np.cross(m.V[m.F[:, 1]], m.V[m.F[:, 2]])).sum() / 6
    if vol < 0:
        m = m.invertida()
    if abierto:
        m = _abrir_bocas(m)
    return m


def _abrir_bocas(m: TriMesh):
    """Quita los casquetes de los extremos libres (corte plano perpendicular al eje)."""
    from ..geom.cortes import cortar_por_nivel
    nombres = [t[0] for t in TUBOS]
    for nombre, extremo in BOCAS:
        _, A, B, ra, rb = TUBOS[nombres.index(nombre)]
        A, B = np.asarray(A, float), np.asarray(B, float)
        eje = (B - A) / np.linalg.norm(B - A)
        P = B if extremo else A
        sentido = eje if extremo else -eje
        r = rb if extremo else ra
        # distancia con signo al plano del extremo, sólo cerca del extremo (cilindro de radio 1.6 r)
        d = (m.V - P) @ sentido
        radial = np.linalg.norm((m.V - P) - np.outer((m.V - P) @ eje, eje), axis=1)
        phi = np.where(radial < 1.6 * r, d, -1e3)
        m2, phi2, _ = cortar_por_nivel(m, phi, 0.0)
        m = m2.submalla(phi2[m2.F].mean(1) <= 0)[0]
    nc, et = m.componentes()
    if nc > 1:
        m = m.submalla(et == np.argmax(np.bincount(et, weights=m.face_areas)))[0]
    return m


def selecciones_bicicleta(m: TriMesh, margen_nudo=45.0):
    """Zonas con nombre, como en un molde de dos conchas (izquierda/derecha):

    * ``<TUBO>``: caras cuyo tubo más cercano es ese (incluye las uniones).
    * ``<TUBO>_IZQ`` / ``_DER``: media caña del tramo recto del tubo, sin los últimos
      ``margen_nudo`` mm de cada extremo unido a otro tubo (las uniones se laminan aparte).
    * Vainas y tirantes: ``_EXTERIOR`` / ``_INTERIOR`` (media caña hacia fuera / hacia dentro).
    * ``NUDO_PEDALIER_IZQ`` / ``NUDO_DIRECCION_IZQ`` / ``NUDO_SILLIN_IZQ``: refuerzos de las uniones.
    * ``MITAD_IZQ`` / ``MITAD_DER``: cada concha completa.
    """
    C = m.face_centroids
    dist, ymed, tpar = [], [], []
    for _, A, B, ra, rb in TUBOS:
        s, t, Q = _sdf_capsula(C, A, B, ra, rb)
        dist.append(np.abs(s))
        ymed.append(Q[:, 1])
        tpar.append(t)
    dist = np.array(dist)
    cerca = np.argmin(dist, axis=0)
    # distancia a la superficie del segundo tubo más cercano: lejos de cualquier unión
    segunda = np.sort(dist, axis=0)[1]
    libre = segunda > 22.0
    idx = np.arange(len(C))
    yc = np.array(ymed)[cerca, idx]
    tc = np.array(tpar)[cerca, idx]
    n_bb = len(TUBOS) - 1
    izq = C[:, 1] >= np.where(cerca == n_bb, 0.0, yc)
    sel = {"MITAD_IZQ": np.nonzero(izq)[0], "MITAD_DER": np.nonzero(~izq)[0]}
    for k, (nombre, A, B, *_r) in enumerate(TUBOS):
        caras = cerca == k
        sel[nombre] = np.nonzero(caras)[0]
        L = float(np.linalg.norm(np.subtract(B, A)))
        recto = caras & libre & (tc * L > margen_nudo) & ((1 - tc) * L > margen_nudo)
        if nombre == "CAJA_PEDALIER":
            continue
        if nombre.endswith(("_IZQ", "_DER")):
            lado = izq if nombre.endswith("_IZQ") else ~izq
            sel[f"{nombre}_EXTERIOR"] = np.nonzero(recto & lado)[0]
            sel[f"{nombre}_INTERIOR"] = np.nonzero(recto & ~lado)[0]
        else:
            sel[f"{nombre}_IZQ"] = np.nonzero(recto & izq)[0]
            sel[f"{nombre}_DER"] = np.nonzero(recto & ~izq)[0]
    nudos = {"NUDO_PEDALIER": (np.array([-10.0, 0, 15.0]), 95.0),
             "NUDO_DIRECCION": (np.array([405.0, 0, 485.0]), 115.0),
             "NUDO_SILLIN": (np.array([-140.0, 0, 480.0]), 75.0)}
    for nombre, (P, r) in nudos.items():
        cerca_n = np.linalg.norm(C - P, axis=1) < r
        sel[f"{nombre}_IZQ"] = np.nonzero(cerca_n & izq)[0]
        sel[f"{nombre}_DER"] = np.nonzero(cerca_n & ~izq)[0]
    return {k: v for k, v in sel.items() if len(v)}
