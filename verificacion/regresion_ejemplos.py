"""Regresión: simula todas las capas de los proyectos de ejemplo y resume la producibilidad.

Uso: python verificacion/regresion_ejemplos.py carpeta_o_archivo.lam [...]"""
import glob
import os
import sys
import time

from laminado.modelo.proyecto import Proyecto

rutas = []
for a in sys.argv[1:]:
    rutas += sorted(glob.glob(os.path.join(a, "**", "*.lam"), recursive=True)) if os.path.isdir(a) else [a]
for r in rutas:
    p = Proyecto.abrir(r)
    print("==", os.path.basename(r), p.molde.malla.nf, "caras", flush=True)
    for c in p.capas:
        if not c.activa:
            continue
        t = time.time()
        try:
            rc = p.simular(c)
        except Exception as e:  # noqa: BLE001
            print(f"  {c.nombre} ERROR {e}", flush=True)
            continue
        s = rc.resumen
        print(f"  {c.nombre:5s} {c.contorno.tipo[:4]} {str(c.contorno.selecciones)[:14]:14s} {c.orientacion:5.0f} "
              f"{c.estrategia:10s} {s['estado']:8s} >lim {s.get('area_limite_%', 0):5.2f} "
              f"cub {s['cubierto_%']:5.1f} piezas {s.get('piezas', 1)} darts {s.get('n_darts', 0)} "
              f"solape {s.get('solape_patron_%', 0):.2f} nodos {rc.drapeado.info.get('nodos_red', 0)} "
              f"{time.time() - t:.0f} s", flush=True)
