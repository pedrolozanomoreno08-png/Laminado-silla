"""Geometría 2D de los patrones planos (piezas de corte) a partir de los resultados."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


@dataclass
class Patron:
    capa: str
    pieza: int
    material: str
    orientacion: float
    corte: Polygon                         # contorno de corte (neto + exceso)
    neto: list = field(default_factory=list)          # polilíneas (k,2) del contorno neto (marcado)
    darts: list = field(default_factory=list)         # polilíneas (k,2)
    etiqueta: str = ""
    estado: str = ""

    @property
    def area(self):
        return self.corte.area

    def transformado(self, ang_deg, dx, dy):
        """Copia girada (respecto al origen) y trasladada."""
        from shapely import affinity
        c = np.cos(np.radians(ang_deg))
        s = np.sin(np.radians(ang_deg))

        def tr(p):
            p = np.asarray(p)
            return np.c_[c * p[:, 0] - s * p[:, 1] + dx, s * p[:, 0] + c * p[:, 1] + dy]
        return Patron(self.capa, self.pieza, self.material, self.orientacion,
                      affinity.translate(affinity.rotate(self.corte, ang_deg, origin=(0, 0)), dx, dy),
                      [tr(p) for p in self.neto], [tr(p) for p in self.darts], self.etiqueta, self.estado)


def _poligono(bucles):
    """Polígono válido a partir de bucles (el mayor es el exterior)."""
    polis = [Polygon(b).buffer(0) for b in bucles if len(b) >= 3]
    polis = [p for p in polis if p.area > 1e-6]
    if not polis:
        return Polygon()
    polis.sort(key=lambda p: -p.area)
    ext = polis[0]
    for h in polis[1:]:
        if ext.contains(h.representative_point()):
            ext = ext.difference(h)
    if ext.geom_type == "MultiPolygon":
        ext = max(ext.geoms, key=lambda g: g.area)
    return ext


def patrones_de(proyecto, nombre_capa):
    rc = proyecto.resultados[nombre_capa]
    capa = proyecto.capa(nombre_capa)
    out = []
    for k, pz in enumerate(rc.piezas):
        bucles, _ = pz.contornos_planos()
        corte = _poligono(bucles)
        neto = rc.contornos_netos_uv(pz)
        darts = []
        for d in pz.darts:
            # posición plana de los puntos del dart: vértices más cercanos de la pieza
            from scipy.spatial import cKDTree
            arbol = cKDTree(pz.malla.V)
            _, idx = arbol.query(np.asarray(d.puntos))
            darts.append(pz.uv[idx])
        etiqueta = f"{nombre_capa}" + (f"-{k + 1}" if len(rc.piezas) > 1 else "")
        out.append(Patron(nombre_capa, k + 1, capa.material, capa.orientacion, corte, neto, darts,
                          etiqueta, rc.resumen.get("estado", "")))
    return out


def todos_los_patrones(proyecto):
    out = []
    for c in proyecto.capas:
        if c.activa and c.nombre in proyecto.resultados:
            out += patrones_de(proyecto, c.nombre)
    return out


def contorno_xy(poligono):
    """Coordenadas del exterior y de los huecos de un polígono shapely."""
    if poligono.is_empty:
        return []
    geoms = poligono.geoms if poligono.geom_type == "MultiPolygon" else [poligono]
    out = []
    for g in geoms:
        out.append(np.asarray(g.exterior.coords))
        out += [np.asarray(i.coords) for i in g.interiors]
    return out


def union(patrones):
    return unary_union([p.corte for p in patrones])


del LineString
