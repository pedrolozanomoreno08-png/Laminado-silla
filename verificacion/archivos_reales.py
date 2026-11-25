"""Lectura de archivos reales de Ansys (repositorio público ansys/example-data) con el lector
del programa, comparada con la librería oficial ansys-mapdl-reader cuando ésta los lee."""
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, ".")
from laminado.ansys.lector import abrir_molde  # noqa: E402

BASE = "/home/claude/ansys/example-data"
archivos = ["pyacp/class40/class40.cdb", "pyacp/basic_flat_plate_example/flat_plate_input.dat",
            "pyacp/getting_started/minimal_plate.cdb", "pyacp/optimization_example/optimization_model.dat",
            "pydpf-composites/short_fiber/ds.dat",
            "pydpf-composites/fatigue/fatigue_example.wbpz",
            "pydpf-composites/non_layered_structure/demo_dpf_composites_without_ACP.wbpz",
            "pydpf-composites/random_vibration/Random_Vibration_Basic_Sandwich_Panel_26R1.wbpz",
            "result_files/average_per_body/two_cubes/two_cubes.wbpz"]
for a in archivos:
    ruta = os.path.join(BASE, a)
    nombre = os.path.basename(a)
    if ruta.endswith(".wbpz"):
        with zipfile.ZipFile(ruta) as z:
            dentro = [n for n in z.namelist() if n.lower().endswith((".dat", ".cdb", ".rst", ".step", ".stp", ".stl"))]
        print(f"{nombre}: contiene {dentro[:6]}{' …' if len(dentro) > 6 else ''}")
    try:
        mo = abrir_molde(ruta)
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {nombre}: {str(e)[:300]}")
        continue
    m = mo.malla
    ref = ""
    if not ruta.endswith(".wbpz"):
        try:
            from ansys.mapdl import reader as pr
            g = pr.Archive(ruta).grid
            import pyvista as pv
            celdas = g.extract_surface(algorithm="dataset_surface").triangulate()
            ref = f" | ref. ansys-mapdl-reader: {g.n_points} nodos, {g.n_cells} elementos, área {celdas.area:.6g}"
        except Exception as e:  # noqa: BLE001
            ref = f" | ref. no disponible ({str(e)[:60]})"
    print(f"  ✓ {nombre}: {m.nv} nodos, {m.nf} triángulos, área {m.area:.6g} mm², caja {np.ptp(m.V, 0).round(1)}, "
          f"bordes {len(m.boundary_loops())}, comps {m.componentes()[0]}, selecciones {len(mo.selecciones)} "
          f"{list(mo.selecciones)[:5]}{ref}")
    for n in mo.notas:
        print("     nota:", n)
