"""Genera todas las salidas de fabricación en una carpeta."""
from __future__ import annotations

import os

from . import anidado, dxf, informes, superficie
from .patrones import todos_los_patrones


def generar_todo(proyecto, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    rutas = []
    pats = todos_los_patrones(proyecto)
    rutas += dxf.exportar_patrones(pats, os.path.join(carpeta, "patrones"))
    try:
        a = anidado.anidar(pats, proyecto)
        rutas.append(dxf.exportar_anidado(a, os.path.join(carpeta, "anidado.dxf")))
    except ValueError as e:
        a = None
        rutas.append(f"Anidado no generado: {e}")
    rutas.append(dxf.exportar_laser(proyecto, os.path.join(carpeta, "laser.dxf"), os.path.join(carpeta, "laser.csv")))
    rutas.append(informes.ply_book(proyecto, os.path.join(carpeta, f"plybook_{proyecto.nombre}.pdf"), a))
    rutas.append(informes.excel(proyecto, os.path.join(carpeta, f"laminado_{proyecto.nombre}.xlsx"), a))
    rutas += informes.exportar_ansys(proyecto, os.path.join(carpeta, "ansys"))
    try:
        rutas.append(superficie.exportar_superficie_laminado(proyecto, os.path.join(carpeta, "superficie_laminado.stl")))
    except Exception as e:  # noqa: BLE001
        rutas.append(f"Superficie del laminado no generada: {e}")
    return rutas
