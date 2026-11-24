"""Exportación DXF: patrones para la mesa de corte, anidado y proyección láser (3D)."""
from __future__ import annotations

import os

import ezdxf
import numpy as np

from .patrones import Patron, contorno_xy

CAPAS_DXF = {
    "CORTE": 7,        # blanco/negro: contorno de corte
    "NETO": 3,         # verde: contorno neto (marcado con rotulador / tinta)
    "FIBRA": 1,        # rojo: flecha de urdimbre (0° del tejido)
    "TEXTO": 5,        # azul: rótulos
    "DARTS": 6,        # magenta
    "ROLLO": 8,        # gris: borde del rollo
}


def _doc():
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    for nombre, color in CAPAS_DXF.items():
        if nombre not in doc.layers:
            doc.layers.add(nombre, color=color)
    return doc


def dibujar_patron(msp, p: Patron, flecha=True, texto=True):
    for anillo in contorno_xy(p.corte):
        msp.add_lwpolyline([tuple(x) for x in anillo], close=True, dxfattribs={"layer": "CORTE"})
    for n in p.neto:
        msp.add_lwpolyline([tuple(x) for x in n], close=True, dxfattribs={"layer": "NETO"})
    for d in p.darts:
        msp.add_lwpolyline([tuple(x) for x in d], dxfattribs={"layer": "DARTS"})
    if p.corte.is_empty:
        return
    c = p.corte.representative_point()
    tam = max(8.0, min(40.0, np.sqrt(p.corte.area) / 12))
    if flecha:
        # flecha de urdimbre: el eje u del patrón es la dirección de 0° del tejido
        ang = getattr(p, "_giro", 0.0)
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        L = 4 * tam
        a = np.array([c.x, c.y]) - d * L / 2
        b = a + d * L
        perp = np.array([-d[1], d[0]])
        msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "FIBRA"})
        msp.add_line(tuple(b), tuple(b - d * tam * 0.6 + perp * tam * 0.3), dxfattribs={"layer": "FIBRA"})
        msp.add_line(tuple(b), tuple(b - d * tam * 0.6 - perp * tam * 0.3), dxfattribs={"layer": "FIBRA"})
    if texto:
        t = msp.add_text(f"{p.etiqueta}  {p.orientacion:g}°", dxfattribs={"layer": "TEXTO", "height": tam * 0.6})
        t.set_placement((c.x, c.y - 1.2 * tam))


def exportar_patrones(patrones, carpeta, combinado=True):
    """Un DXF por pieza y, opcionalmente, uno con todas las piezas en fila."""
    os.makedirs(carpeta, exist_ok=True)
    rutas = []
    for p in patrones:
        doc = _doc()
        dibujar_patron(doc.modelspace(), p)
        r = os.path.join(carpeta, f"{p.etiqueta}.dxf")
        doc.saveas(r)
        rutas.append(r)
    if combinado and patrones:
        doc = _doc()
        msp = doc.modelspace()
        x = 0.0
        for p in patrones:
            b = p.corte.bounds
            q = p.transformado(0, x - b[0], -b[1])
            dibujar_patron(msp, q)
            x += (b[2] - b[0]) + 30
        r = os.path.join(carpeta, "todas_las_piezas.dxf")
        doc.saveas(r)
        rutas.append(r)
    return rutas


def exportar_anidado(anidado, ruta):
    doc = _doc()
    msp = doc.modelspace()
    for hoja in anidado.hojas:
        oy = hoja.y0
        msp.add_lwpolyline([(0, oy), (hoja.largo, oy), (hoja.largo, oy + hoja.ancho), (0, oy + hoja.ancho)],
                           close=True, dxfattribs={"layer": "ROLLO"})
        for c in hoja.colocadas:
            q = c.patron.transformado(c.giro, c.x, c.y + oy)
            q._giro = c.giro
            dibujar_patron(msp, q)
        t = msp.add_text(f"{hoja.material} – {hoja.largo:.0f} mm – aprovechamiento {100 * hoja.aprovechamiento:.1f} %",
                         dxfattribs={"layer": "TEXTO", "height": 15})
        t.set_placement((0, oy - 30))
    doc.saveas(ruta)
    return ruta


def exportar_laser(proyecto, ruta_dxf, ruta_csv=None, matriz=None):
    """Contornos netos 3D de cada capa (coordenadas del molde o transformadas con ``matriz`` 4x4)
    para proyección láser."""
    doc = _doc()
    msp = doc.modelspace()
    filas = []
    M = np.eye(4) if matriz is None else np.asarray(matriz, float)
    for c in proyecto.capas:
        rc = proyecto.resultados.get(c.nombre)
        if rc is None or not c.activa:
            continue
        if c.nombre not in doc.layers:
            doc.layers.add(c.nombre)
        for k, b in enumerate(rc.neto.boundary_loops()):
            P = rc.neto.V[b]
            P = (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]
            msp.add_polyline3d([tuple(x) for x in P], close=True, dxfattribs={"layer": c.nombre})
            for j, x in enumerate(P):
                filas.append((c.nombre, k + 1, j + 1, *x))
    doc.saveas(ruta_dxf)
    if ruta_csv:
        with open(ruta_csv, "w", encoding="utf-8") as f:
            f.write("capa;contorno;punto;x;y;z\n")
            for r in filas:
                f.write(f"{r[0]};{r[1]};{r[2]};{r[3]:.3f};{r[4]:.3f};{r[5]:.3f}\n")
    return ruta_dxf
