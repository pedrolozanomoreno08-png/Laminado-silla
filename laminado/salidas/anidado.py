"""Anidado de piezas en el rollo (ráster + correlación por FFT, colocación abajo-izquierda).

La urdimbre de cada pieza (eje u del patrón) debe ir a lo largo del rollo (eje x); se permiten
giros de 180° (y de 90°/270° si el tejido es equilibrado)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from shapely import affinity
from shapely.geometry import Polygon

from .patrones import Patron


@dataclass
class Colocada:
    patron: Patron
    giro: float
    x: float
    y: float


@dataclass
class Hoja:
    material: str
    ancho: float
    largo: float = 0.0
    colocadas: list = field(default_factory=list)
    y0: float = 0.0

    @property
    def aprovechamiento(self):
        a = sum(c.patron.area for c in self.colocadas)
        return a / max(self.ancho * self.largo, 1e-9)


@dataclass
class Anidado:
    hojas: list

    def resumen(self):
        return [{"material": h.material, "ancho_mm": h.ancho, "largo_mm": h.largo,
                 "piezas": len(h.colocadas), "aprovechamiento_%": 100 * h.aprovechamiento}
                for h in self.hojas]


def _rasterizar(poli: Polygon, res):
    from matplotlib.path import Path
    x0, y0, x1, y1 = poli.bounds
    nx = int(np.ceil((x1 - x0) / res)) + 1
    ny = int(np.ceil((y1 - y0) / res)) + 1
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel()]
    dentro = np.zeros(len(pts), bool)
    geoms = poli.geoms if poli.geom_type == "MultiPolygon" else [poli]
    for g in geoms:
        d = Path(np.asarray(g.exterior.coords)).contains_points(pts, radius=res * 0.5)
        for i in g.interiors:
            d &= ~Path(np.asarray(i.coords)).contains_points(pts, radius=-res * 0.5)
        dentro |= d
    return dentro.reshape(nx, ny), (x0, y0)


def anidar(patrones, proyecto, separacion=5.0, margen=10.0, res=2.0, largo_max=20000.0):
    """Anida por material. Devuelve ``Anidado`` (una hoja/tramo de rollo por material)."""
    por_mat = {}
    for p in patrones:
        por_mat.setdefault(p.material, []).append(p)
    hojas = []
    y0 = 0.0
    for mat, lista in por_mat.items():
        m = proyecto.materiales[mat]
        W = m.ancho_rollo
        giros = [0, 180] + ([90, 270] if (m.tipo == "tejido" and m.simetrico) else [])
        ny = int((W - 2 * margen) / res)
        nx = int(largo_max / res)
        ocup = np.zeros((nx, ny), bool)
        hoja = Hoja(mat, W, y0=y0)
        sep = max(1, int(np.ceil(separacion / res)))
        for p in sorted(lista, key=lambda q: -q.area):
            mejor = None
            for g in giros:
                pol = affinity.rotate(p.corte, g, origin=(0, 0))
                mask, (bx, by) = _rasterizar(pol, res)
                dil = binary_dilation(np.pad(mask, sep), iterations=sep)
                if dil.shape[1] > ny:
                    continue
                # posiciones sin solape: correlación de la ocupación con la máscara
                usado = int(np.nonzero(ocup.any(1))[0].max()) + 1 if ocup.any() else 0
                lim = min(nx, usado + mask.shape[0] + 2)
                region = ocup[:lim].astype(float)
                corr = fftconvolve(region, dil[::-1, ::-1].astype(float), mode="valid") if \
                    region.shape[0] >= dil.shape[0] else np.zeros((0, 0))
                libres = np.argwhere(corr < 0.5) if corr.size else np.zeros((0, 2), int)
                if len(libres):
                    i, j = min(libres, key=lambda t: (t[0], t[1]))
                else:
                    i, j = usado, 0
                fin = i + mask.shape[0]
                clave = (fin, i, j)
                if mejor is None or clave < mejor[0]:
                    mejor = (clave, g, i, j, mask, dil, bx, by)
            if mejor is None:
                raise ValueError(f"La pieza {p.etiqueta} no cabe en el ancho de rollo de {mat} ({W} mm)")
            _, g, i, j, mask, dil, bx, by = mejor
            ocup[i:i + dil.shape[0], j:j + dil.shape[1]] |= dil
            x = margen + (i + sep) * res - bx
            y = margen + (j + sep) * res - by
            hoja.colocadas.append(Colocada(p, g, x, y))
        usado = int(np.nonzero(ocup.any(1))[0].max()) + 1 if ocup.any() else 0
        hoja.largo = usado * res + 2 * margen
        hojas.append(hoja)
        y0 += W + 150
    return Anidado(hojas)
