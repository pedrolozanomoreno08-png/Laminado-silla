"""Drapeado sobre el modelo real del tutorial «class40» de Ansys ACP (casco de velero de 4 m,
ensamblaje de láminas con uniones en T)."""
import sys
import time

sys.path.insert(0, ".")
from laminado.modelo.proyecto import Capa, Contorno, Proyecto  # noqa: E402

p = Proyecto("class40")
p.cargar_molde("/home/claude/ansys/example-data/pyacp/class40/class40.cdb")
p.h = 25.0
vid = "Vidrio 7781 satén 8H (prepreg epoxi)"
plan = [("HULL_ALL", 0, "geodesica"), ("HULL_ALL", 45, "geodesica"), ("HULL_ALL", 45, "energia"),
        ("DECK", 0, "geodesica"), ("DECK", 45, "geodesica"), ("BULKHEAD_ALL", 0, "geodesica"),
        ("HULL_PORT", 0, "geodesica"), ("KEELTOWER_PORT", 0, "geodesica")]
for k, (sel, ori, est) in enumerate(plan, 1):
    c = Capa(f"P{k}", vid, ori, contorno=Contorno("seleccion", [sel], exceso=20), estrategia=est)
    p.capas.append(c)
    t = time.time()
    s = p.simular(c).resumen
    print(f"{c.nombre} {sel:15s} {ori:3d}° {est:9s} {s['estado']:8s} máx {s['cizalla_max']:5.1f}  >aviso "
          f"{s['area_aviso_%']:5.2f}%  >lím {s['area_limite_%']:5.2f}%  cubierto {s['cubierto_%']:5.1f}%  "
          f"piezas {s['piezas']}  {time.time() - t:.1f} s", flush=True)
