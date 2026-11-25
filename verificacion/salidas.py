"""Verificación de las salidas releyéndolas: DXF, anidado, láser, PDF, Excel y CSV de ACP."""
import csv
import glob
import os
import sys
import tempfile

import ezdxf
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, ".")
from laminado.ansys.lector import Molde  # noqa: E402
from laminado.demo.silla import casco_silla, selecciones_silla  # noqa: E402
from laminado.geom.consulta import Consultor  # noqa: E402
from laminado.modelo.proyecto import Capa, Contorno, Proyecto  # noqa: E402
from laminado.salidas import anidado as an  # noqa: E402
from laminado.salidas.patrones import todos_los_patrones  # noqa: E402
from laminado.salidas.todo import generar_todo  # noqa: E402

fallos = []


def comprobar(cond, texto):
    print(("  ✓ " if cond else "  ✗ ") + texto)
    if not cond:
        fallos.append(texto)


m = casco_silla(8.0)
p = Proyecto("Verif")
p.cargar_molde(Molde(m, selecciones_silla(m)))
p.h = 8.0
tej = "Carbono sarga 2x2 200 g/m² (prepreg epoxi)"
ud = "Carbono UD 150 g/m² (prepreg epoxi)"
p.materiales[tej].ancho_rollo = 350.0          # fuerza empalmes automáticos
p.capas = [Capa("P001", tej, 0, contorno=Contorno("molde", exceso=15), darts_auto=True),
           Capa("P002", tej, 45, contorno=Contorno("seleccion", ["ASIENTO"], exceso=10)),
           Capa("P003", ud, 0, contorno=Contorno("seleccion", ["ASIENTO"], exceso=10))]
p.simular_todas()
dirn = tempfile.mkdtemp()
rutas = generar_todo(p, dirn)
pats = todos_los_patrones(p)
print(f"Patrones: {[(x.etiqueta, round(x.area)) for x in pats]}")

print("== DXF de piezas ==")
for pt in pats:
    doc = ezdxf.readfile(os.path.join(dirn, "patrones", f"{pt.etiqueta}.dxf"))
    msp = doc.modelspace()
    capas = {e.dxf.layer for e in msp}
    corte = [Polygon(np.array(e.get_points("xy"))) for e in msp.query("LWPOLYLINE[layer=='CORTE']")]
    neto = [Polygon(np.array(e.get_points("xy"))) for e in msp.query("LWPOLYLINE[layer=='NETO']")]
    a_dxf = max(c.area for c in corte) - sum(sorted(c.area for c in corte)[:-1])
    rc = p.resultados[pt.capa]
    pz = rc.piezas[pt.pieza - 1]
    comprobar(abs(a_dxf - pt.area) / pt.area < 1e-6 and abs(pz.area_plana() - pt.area) / pt.area < 0.01,
              f"{pt.etiqueta}: área DXF {a_dxf:.0f} = patrón {pt.area:.0f} ≈ malla plana {pz.area_plana():.0f} mm²")
    comprobar({"CORTE", "FIBRA", "TEXTO"} <= capas, f"{pt.etiqueta}: capas {sorted(capas)}")
    ext = max(corte, key=lambda c: c.area).buffer(0.01)
    comprobar(all(ext.contains(n.buffer(0)) for n in neto), f"{pt.etiqueta}: el contorno neto está dentro del de corte")
    # fibra a lo largo de +x (urdimbre = eje u del patrón)
    lin = list(msp.query("LINE[layer=='FIBRA']"))[0]
    d = np.array(lin.dxf.end) - np.array(lin.dxf.start)
    comprobar(abs(d[1]) < 1e-9 and d[0] > 0, f"{pt.etiqueta}: flecha de urdimbre según +x")

print("== Empalmes ==")
r1 = p.resultados["P001"]
comprobar(len(r1.piezas) >= 2, f"P001 dividida en {len(r1.piezas)} piezas por ancho de rollo 350 mm")
comprobar(all(np.ptp(pz.uv[:, 1]) <= 350 - 2 * 10 + 1e-6 for pz in r1.piezas),
          f"cada pieza cabe en el rollo: anchos {[round(np.ptp(pz.uv[:, 1])) for pz in r1.piezas]} mm")
solape = sum(pz.area_plana() for pz in r1.piezas) - r1.drapeado.area_plana()
comprobar(solape > 0, f"los solapes añaden material: {solape:.0f} mm²")

print("== Anidado ==")
A = an.anidar(pats, p)
for h in A.hojas:
    polis = [c.patron.transformado(c.giro, c.x, c.y).corte for c in h.colocadas]
    solapes = max((polis[i].intersection(polis[j]).area for i in range(len(polis)) for j in range(i + 1, len(polis))),
                  default=0)
    dentro = all(q.bounds[1] >= -1e-6 and q.bounds[3] <= h.ancho + 1e-6 and q.bounds[0] >= -1e-6 and q.bounds[2] <= h.largo + 1e-6
                 for q in polis)
    giros = {c.giro for c in h.colocadas}
    mat = p.materiales[h.material]
    permitidos = {0, 180} | ({90, 270} if mat.tipo == "tejido" and mat.simetrico else set())
    dmin = min((polis[i].distance(polis[j]) for i in range(len(polis)) for j in range(i + 1, len(polis))), default=99)
    comprobar(solapes < 1e-6 and dentro and giros <= permitidos and dmin >= 4.0,
              f"{h.material[:25]}: {len(polis)} piezas, sin solapes ({solapes:.2g}), dentro del rollo {dentro}, "
              f"giros {sorted(giros)} ⊆ {sorted(permitidos)}, separación mín {dmin:.1f} mm, aprovechamiento {100 * h.aprovechamiento:.0f} %")

print("== Láser ==")
doc = ezdxf.readfile(os.path.join(dirn, "laser.dxf"))
cons = Consultor(p.malla)
pts = np.array([v.dxf.location for e in doc.modelspace().query("POLYLINE") for v in e.vertices])
d = cons.mas_cercano(pts)[3]
comprobar(d.max() < 0.05, f"{len(pts)} puntos láser sobre la superficie del molde (distancia máx {d.max():.4f} mm)")
with open(os.path.join(dirn, "laser.csv"), encoding="utf-8") as f:
    filas = list(csv.reader(f, delimiter=";"))[1:]
comprobar(len(filas) == len(pts), f"CSV láser con {len(filas)} filas = puntos del DXF")

print("== PDF / Excel ==")
import pypdf  # noqa: E402
pdf = glob.glob(os.path.join(dirn, "plybook_*.pdf"))[0]
npag = len(pypdf.PdfReader(pdf).pages)
comprobar(npag == 1 + len(p.capas) + 1, f"ply book con {npag} páginas (portada + {len(p.capas)} capas + anidado)")
texto = pypdf.PdfReader(pdf).pages[1].extract_text()
comprobar("P001" in texto and "Cizalla" in texto, "la página de la capa P001 contiene sus datos")
import openpyxl  # noqa: E402
wb = openpyxl.load_workbook(glob.glob(os.path.join(dirn, "laminado_*.xlsx"))[0])
ws = wb["Capas"]
pesos = [ws.cell(r, 15).value for r in range(2, ws.max_row + 1)]
comprobar(ws.max_row - 1 == len(p.capas) and abs(sum(pesos) - sum(p.resultados[c.nombre].resumen["peso_g"] for c in p.capas)) < 1e-6,
          f"Excel: {ws.max_row - 1} capas, peso total {sum(pesos):.1f} g coincide con el cálculo")
mat_ok = abs(sum(wb["Materiales"].cell(r, 4).value for r in range(2, wb["Materiales"].max_row + 1)) * 1000 - sum(pesos)) < 1e-3
comprobar(mat_ok, "Excel: el consumo por material suma lo mismo que las capas")

print("== Tablas para ACP ==")
with open(os.path.join(dirn, "ansys", "P002_fibras.csv"), encoding="utf-8") as f:
    lineas = [l for l in f if not l.startswith("#")]
cab = lineas[0].strip().split(",")
datos = np.array([[float(x) for x in l.split(",")] for l in lineas[1:]])
U = datos[:, 4:7]
comprobar(np.allclose(np.linalg.norm(U, axis=1), 1, atol=1e-5), "direcciones de urdimbre unitarias")
nrm = p.malla.face_normals[datos[:, 0].astype(int) - 1]
comprobar(np.abs((U * nrm).sum(1)).max() < 1e-3, "direcciones tangentes a la superficie")
# ángulo real del CSV frente al core sample en 5 elementos
dif = []
for k in np.linspace(0, len(datos) - 1, 5).astype(int):
    P = datos[k, 1:4]
    cs = [f for f in p.core_sample(P) if f["capa"] == "P002"]
    if cs:
        dif.append(abs(((cs[0]["real"] - datos[k, 10]) + 90) % 180 - 90))
comprobar(len(dif) >= 3 and max(dif) < 0.5, f"ángulo real del CSV = core sample (dif. máx {max(dif) if dif else float('nan'):.3f}°)")

print("== Guardar / abrir ==")
ruta = os.path.join(dirn, "x.lam")
p.guardar(ruta)
q = Proyecto.abrir(ruta)
q.h = p.h
r_q = q.simular("P002").resumen
comprobar(abs(r_q["cizalla_max"] - p.resultados["P002"].resumen["cizalla_max"]) < 1e-9 and
          abs(r_q["area_corte_mm2"] - p.resultados["P002"].resumen["area_corte_mm2"]) < 1e-6,
          "al reabrir el proyecto y simular se obtiene exactamente el mismo resultado")
print("\nFALLOS:", fallos if fallos else "ninguno")
