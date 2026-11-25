"""Lectura de archivos reales escritos por Ansys MAPDL (los de ejemplo que acompañan a
ansys-mapdl-reader) comparada con la lectura de referencia de esa librería."""
import sys

import numpy as np

sys.path.insert(0, ".")
from ansys.mapdl import reader as pr  # noqa: E402
from ansys.mapdl.reader import examples  # noqa: E402

from laminado.ansys.lector import abrir_molde  # noqa: E402

for nombre, ruta in (("HexBeam.cdb", examples.hexarchivefile), ("TetBeam.cdb", examples.tetarchivefile),
                     ("sector.cdb", examples.sector_archive_file), ("file.rst", examples.rstfile)):
    try:
        mo = abrir_molde(ruta)
    except Exception as e:  # noqa: BLE001
        print(f"{nombre:12s} ERROR: {e}")
        continue
    if ruta.endswith(".rst"):
        g = pr.read_binary(ruta).grid
    else:
        g = pr.Archive(ruta).grid
    piel = g.extract_surface().triangulate()
    area_ref = piel.area
    caja_ref = np.ptp(np.asarray(g.points), axis=0)
    # el lector convierte a mm si deduce metros: comparar en las unidades del archivo
    f = np.ptp(mo.malla.V, axis=0).max() / caja_ref.max()
    print(f"{nombre:12s} triángulos {mo.malla.nf:6d} (ref {piel.n_cells:6d})  área {mo.malla.area / f**2:.4g} "
          f"(ref {area_ref:.4g}, dif {100 * (mo.malla.area / f**2 - area_ref) / area_ref:+.3f} %)  "
          f"escala ×{f:g}  cerrada {len(mo.malla.boundary_loops()) == 0}  selecciones {list(mo.selecciones)[:4]}  {mo.notas}")
