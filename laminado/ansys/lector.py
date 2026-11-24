"""Lectura de la geometría del molde desde Ansys y otros formatos.

Formatos:
  * ``.wbpz``  – archivo de proyecto Workbench (zip). Se busca dentro un ``ds.dat``, ``.cdb``,
    ``.dat`` o ``.rst`` (del sistema Mechanical o de ACP) y se lee el primero útil.
  * ``.dat`` / ``.cdb`` / ``.inp`` – entrada de MAPDL (bloques NBLOCK / EBLOCK / CMBLOCK).
  * ``.rst`` / ``.rth`` – resultados binarios (requiere ``ansys-mapdl-reader``).
  * ``.stl``, ``.obj``, ``.ply``, ``.vtk``, ``.vtp`` – mallas (pyvista).
  * ``.step``, ``.stp``, ``.iges``, ``.igs`` – CAD (requiere ``gmsh``; se malla la superficie).

El resultado es un ``Molde``: malla triangular en milímetros + selecciones con nombre
(componentes de Ansys) como conjuntos de caras.
"""
from __future__ import annotations

import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field

import numpy as np

from ..geom.malla import TriMesh, fusionar_vertices

TIPOS_LAMINA = {181, 281, 131, 132, 63, 41, 43, 93, 91, 99, 208, 209, 28, 57, 157}
TIPOS_SOLIDO = {185, 186, 187, 285, 190, 45, 95, 92, 72, 73, 215, 216, 217, 226, 227}


@dataclass
class Molde:
    malla: TriMesh
    selecciones: dict[str, np.ndarray] = field(default_factory=dict)   # nombre -> índices de cara
    origen: str = ""
    unidades: str = "mm"
    id_elemento: np.ndarray | None = None      # id de elemento Ansys de cada cara (para exportar)
    notas: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- MAPDL
def _anchos(fmt: str):
    """Anchos de campo de un formato Fortran como (1i9,3e20.9e3) o (19i9)."""
    fmt = fmt.strip().strip("()")
    out = []
    for parte in fmt.split(","):
        m = re.match(r"\s*(\d*)\s*([a-zA-Z])\s*(\d+)", parte)
        if not m:
            continue
        rep = int(m.group(1) or 1)
        out += [int(m.group(3))] * rep
    return out


def _cortar(linea: str, anchos):
    vals = []
    pos = 0
    for w in anchos:
        trozo = linea[pos:pos + w]
        pos += w
        if not trozo.strip():
            break
        vals.append(trozo)
    return vals


def _num(s):
    s = s.strip().replace("D", "E").replace("d", "e")
    # números como 1.234-005 (exponente sin E)
    if re.match(r"^[-+]?\d*\.\d+[-+]\d+$", s):
        s = re.sub(r"([0-9.])([-+]\d+)$", r"\1E\2", s)
    return float(s)


def leer_mapdl(ruta: str) -> Molde:
    with open(ruta, "r", errors="replace") as fh:
        lineas = fh.read().splitlines()
    nodos = {}
    elementos = []          # (id, tipo, nodos)
    tipos = {}              # número de tipo -> número de elemento Ansys (181…)
    tipo_actual = 1
    comps = {}
    unidades = None
    i = 0
    n = len(lineas)
    while i < n:
        L = lineas[i]
        low = L.strip().lower()
        if low.startswith("/units"):
            unidades = low.split(",")[1].strip() if "," in low else None
        elif low.startswith("et,"):
            p = [x.strip() for x in low.split(",")]
            try:
                tipos[int(p[1])] = int(re.sub(r"\D", "", p[2]) or 0)
            except (ValueError, IndexError):
                pass
        elif low.startswith("type,"):
            try:
                tipo_actual = int(low.split(",")[1])
            except ValueError:
                pass
        elif low.startswith("nblock"):
            fmt = lineas[i + 1]
            anchos = _anchos(fmt)
            m_int = re.match(r"\s*\(\s*(\d*)\s*i", fmt.lower())
            n_ent = int(m_int.group(1) or 1) if m_int else 1
            i += 2
            while i < n:
                L = lineas[i]
                if L.strip().startswith("-1") or L.strip().upper().startswith("N,R5") or not L.strip():
                    break
                v = _cortar(L, anchos)
                nid = int(v[0])
                nodos[nid] = [_num(x) for x in v[n_ent:n_ent + 3]]
                i += 1
        elif low.startswith("eblock"):
            p = [x.strip() for x in low.split(",")]
            solido = len(p) > 2 and p[2] == "solid"
            fmt = lineas[i + 1]
            anchos = _anchos(fmt)
            i += 2
            while i < n:
                L = lineas[i]
                if L.strip().startswith("-1") or not L.strip():
                    break
                v = [int(x) for x in _cortar(L, anchos)]
                if solido:
                    nn = v[8]
                    eid = v[10]
                    tipo = v[1]
                    nd = v[11:]
                    while len(nd) < nn and i + 1 < n:
                        i += 1
                        nd += [int(x) for x in _cortar(lineas[i], anchos)]
                    nd = nd[:nn]
                else:
                    eid = v[0]
                    tipo = v[2] if len(v) > 2 else tipo_actual
                    nd = v[5:]
                elementos.append((eid, tipo, nd))
                i += 1
        elif low.startswith("cmblock"):
            p = [x.strip() for x in L.split(",")]
            nombre = p[1]
            clase = p[2].upper()
            cuantos = int(re.sub(r"\D", "", p[3]) or 0)
            fmt = lineas[i + 1]
            anchos = _anchos(fmt)
            i += 2
            vals = []
            while len(vals) < cuantos and i < n:
                vals += [int(x) for x in _cortar(lineas[i], anchos)]
                i += 1
            i -= 1
            ids = []
            for k, x in enumerate(vals):
                if x < 0:
                    ids.extend(range(ids[-1] + 1, -x + 1))
                else:
                    ids.append(x)
            comps[(nombre, clase)] = np.array(ids)
        i += 1
    if not nodos or not elementos:
        raise ValueError(f"No se han encontrado nodos/elementos en {os.path.basename(ruta)}")
    # nodos con formato de 3 enteros (id, sólido, línea) antes de las coordenadas
    ejemplo = next(iter(nodos.values()))
    ids_n = np.array(list(nodos.keys()))
    coords = np.zeros((len(ids_n), 3))
    for k, nid in enumerate(ids_n):
        c = nodos[nid]
        c = (c + [0.0, 0.0, 0.0])[:3]
        coords[k] = c
    del ejemplo
    return _ensamblar(ids_n, coords, elementos, tipos, comps, unidades, ruta)


def _ensamblar(ids_n, coords, elementos, tipos, comps, unidades, ruta):
    mapa = {int(nid): k for k, nid in enumerate(ids_n)}
    tris, eid_tri, solidos = [], [], []
    for eid, tipo, nd in elementos:
        et = tipos.get(tipo, 181 if len(nd) in (3, 4, 6, 8) else 0)
        if et in TIPOS_SOLIDO or (et not in TIPOS_LAMINA and len(nd) in (10, 20)):
            solidos.append((eid, nd))
            continue
        if et and et not in TIPOS_LAMINA:
            continue
        esq = nd[:4] if len(nd) >= 4 else nd[:3]
        if len(nd) == 6:      # triángulo cuadrático
            esq = nd[:3]
        try:
            q = [mapa[x] for x in esq if x > 0]
        except KeyError:
            continue
        if len(q) == 4 and q[2] == q[3]:
            q = q[:3]
        if len(q) == 3:
            tris.append(q)
            eid_tri.append(eid)
        elif len(q) == 4:
            tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            eid_tri += [eid, eid]
    notas = []
    if not tris and solidos:
        tris, eid_tri = _piel_solidos(solidos, mapa)
        notas.append("Modelo sólido: se usa la piel exterior como superficie de molde.")
    F = np.array(tris, np.int64)
    eid_tri = np.array(eid_tri)
    usados, inv = np.unique(F.ravel(), return_inverse=True)
    V = coords[usados]
    F = inv.reshape(-1, 3)
    factor, uds = _factor_unidades(unidades, V)
    V = V * factor
    if factor != 1.0:
        notas.append(f"Unidades del modelo: {uds} → convertidas a mm (×{factor:g}).")
    m = TriMesh(V, F).orientar_coherente()
    sel = {}
    for (nombre, clase), ids in comps.items():
        if clase.startswith("ELEM"):
            caras = np.nonzero(np.isin(eid_tri, ids))[0]
            if len(caras):
                sel[nombre] = caras
        elif clase.startswith("NODE"):
            nodos_ok = np.isin(ids_n[usados], ids)
            caras = np.nonzero(nodos_ok[F].all(1))[0]
            if len(caras):
                sel[nombre + " (nodos)"] = caras
    return Molde(m, sel, origen=ruta, id_elemento=eid_tri, notas=notas)


def _piel_solidos(solidos, mapa):
    caras_hex = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    caras_tet = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    cuenta = {}
    for eid, nd in solidos:
        nd = [mapa.get(x, -1) for x in nd]
        if len(nd) in (4, 10) or (len(nd) >= 8 and nd[2] == nd[3] and nd[4] == nd[5] == nd[6] == nd[7]):
            base = [nd[0], nd[1], nd[2], nd[4] if len(nd) >= 8 and len(nd) != 10 else nd[3]]
            lista = [tuple(base[k] for k in c) for c in caras_tet]
        else:
            lista = [tuple(nd[k] for k in c) for c in caras_hex]
        for c in lista:
            cl = tuple(sorted(set(c)))
            cuenta.setdefault(cl, []).append((c, eid))
    tris, eids = [], []
    for cl, lst in cuenta.items():
        if len(lst) != 1:
            continue
        c, eid = lst[0]
        c = list(dict.fromkeys(c))
        if len(c) == 3:
            tris.append(c)
            eids.append(eid)
        elif len(c) == 4:
            tris += [[c[0], c[1], c[2]], [c[0], c[2], c[3]]]
            eids += [eid, eid]
    return tris, eids


def _factor_unidades(unidades, V):
    if unidades:
        u = unidades.lower()
        if u in ("si", "mks"):
            return 1000.0, "m"
        if u in ("mpa", "nmm", "umks"):
            return 1.0, "mm"
        if u in ("bin", "bft"):
            return 25.4, "pulgadas"
        if u == "cgs":
            return 10.0, "cm"
    # sin indicación: se deduce del tamaño
    tam = float(np.ptp(V, axis=0).max()) if len(V) else 0
    if 0 < tam < 5:
        return 1000.0, "m (deducido)"
    return 1.0, "mm"


# --------------------------------------------------------------------------- RST
def leer_rst(ruta: str) -> Molde:
    try:
        from ansys.mapdl import reader as pymapdl_reader
    except ImportError as e:  # pragma: no cover
        raise ImportError("Para leer .rst instala ansys-mapdl-reader") from e
    r = pymapdl_reader.read_binary(ruta)
    g = r.mesh
    ids_n = np.asarray(g.nnum)
    coords = np.asarray(g.nodes)[:, :3]
    elementos = []
    tipos = {}
    for k, (eid, et) in enumerate(zip(g.enum, g.etype)):
        nd = [int(x) for x in g.elem[k][10:] if x > 0]
        elementos.append((int(eid), int(et), nd))
    for ti, num in zip(g.ekey[:, 0], g.ekey[:, 1]):
        tipos[int(ti)] = int(num)
    comps = {(nombre, "ELEM"): np.asarray(v) for nombre, v in getattr(g, "element_components", {}).items()}
    unidades = None
    try:
        unidades = {0: "mks", 6: "mpa", 1: "si"}.get(int(r._resultheader.get("units", -1)))
    except Exception:
        pass
    return _ensamblar(ids_n, coords, elementos, tipos, comps, unidades, ruta)


# --------------------------------------------------------------------------- otros
def leer_malla_generica(ruta: str) -> Molde:
    import pyvista as pv
    s = pv.read(ruta)
    s = s.extract_surface().triangulate().clean()
    F = s.faces.reshape(-1, 4)[:, 1:]
    V, F, _ = fusionar_vertices(np.asarray(s.points, float), F, tol=1e-9)
    factor, uds = _factor_unidades(None, V)
    m = TriMesh(V * factor, F).orientar_coherente()
    notas = [f"Unidades deducidas: {uds}"] if factor != 1 else []
    return Molde(m, {}, origen=ruta, notas=notas)


def leer_cad(ruta: str, tam_elemento: float = 5.0) -> Molde:
    try:
        import gmsh
    except Exception as e:  # pragma: no cover
        raise ImportError("Para leer STEP/IGES instala gmsh (pip install gmsh)") from e
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(ruta)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", tam_elemento)
        gmsh.option.setNumber("Mesh.MeshSizeMin", tam_elemento * 0.3)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        P = xyz.reshape(-1, 3)
        idx = {int(t): k for k, t in enumerate(tags)}
        tris = []
        cara_sup = []
        sel = {}
        for dim, sup in gmsh.model.getEntities(2):
            et, _, nod = gmsh.model.mesh.getElements(dim, sup)
            for t, nn in zip(et, nod):
                if t == 2:
                    q = np.array([idx[int(x)] for x in nn]).reshape(-1, 3)
                    ini = len(tris)
                    tris.extend(q.tolist())
                    cara_sup.extend([sup] * len(q))
                    sel[f"Superficie {sup}"] = np.arange(ini, len(tris))
        unidad = gmsh.option.getString("Geometry.OCCTargetUnit") if hasattr(gmsh.option, "getString") else ""
    finally:
        gmsh.finalize()
    V, F, _ = fusionar_vertices(P, np.array(tris), tol=1e-6)
    m = TriMesh(V, F).orientar_coherente()
    del unidad
    return Molde(m, sel, origen=ruta)


def _buscar_en_wbpz(ruta: str, destino: str):
    with zipfile.ZipFile(ruta) as z:
        nombres = z.namelist()
        prioridad = []
        for nm in nombres:
            b = nm.lower()
            if b.endswith("ds.dat"):
                prioridad.append((0, nm))
            elif b.endswith(".cdb"):
                prioridad.append((1, nm))
            elif b.endswith(".dat") and "mech" in b:
                prioridad.append((2, nm))
            elif b.endswith(".rst"):
                prioridad.append((3, nm))
            elif b.endswith((".stp", ".step", ".igs", ".iges")):
                prioridad.append((4, nm))
            elif b.endswith(".stl"):
                prioridad.append((5, nm))
        prioridad.sort()
        extraidos = []
        for _, nm in prioridad:
            z.extract(nm, destino)
            extraidos.append(os.path.join(destino, nm))
        return extraidos, nombres


def leer_wbpz(ruta: str) -> Molde:
    tmp = tempfile.mkdtemp(prefix="wbpz_")
    candidatos, nombres = _buscar_en_wbpz(ruta, tmp)
    errores = []
    for c in candidatos:
        try:
            m = abrir_molde(c)
            m.origen = f"{os.path.basename(ruta)} → {os.path.relpath(c, tmp)}"
            return m
        except Exception as e:
            errores.append(f"{os.path.basename(c)}: {e}")
    raise ValueError(
        "El .wbpz no contiene una malla legible (ds.dat, .cdb, .rst, STEP o STL).\n"
        "En Mechanical: con el modelo mallado, botón derecho en 'Solution' → 'Write Input File' "
        "o resuelve una vez y vuelve a archivar el proyecto.\n"
        + ("Detalles: " + "; ".join(errores) if errores else f"Archivos: {len(nombres)}"))


def abrir_molde(ruta: str, tam_elemento: float = 5.0) -> Molde:
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".wbpz":
        return leer_wbpz(ruta)
    if ext in (".dat", ".cdb", ".inp", ".ans", ".mac"):
        return leer_mapdl(ruta)
    if ext in (".rst", ".rth", ".rmg"):
        return leer_rst(ruta)
    if ext in (".step", ".stp", ".iges", ".igs", ".brep"):
        return leer_cad(ruta, tam_elemento)
    return leer_malla_generica(ruta)


def escribir_mapdl(ruta: str, malla: TriMesh, selecciones: dict | None = None):
    """Escribe la malla como entrada MAPDL (SHELL181 triangulares) — útil para pruebas e intercambio."""
    with open(ruta, "w") as f:
        f.write("/prep7\n/units,MPA\net,1,181\n")
        f.write(f"nblock,3,,{malla.nv}\n(1i9,3e20.9e3)\n")
        for k, p in enumerate(malla.V):
            f.write(f"{k + 1:9d}{p[0]:20.9e}{p[1]:20.9e}{p[2]:20.9e}\n")
        f.write("-1\n")
        f.write(f"eblock,19,solid,,{malla.nf}\n(19i9)\n")
        for k, t in enumerate(malla.F):
            campos = [1, 1, 1, 1, 0, 0, 0, 0, 4, 0, k + 1, t[0] + 1, t[1] + 1, t[2] + 1, t[2] + 1]
            f.write("".join(f"{x:9d}" for x in campos) + "\n")
        f.write("       -1\n")
        for nombre, caras in (selecciones or {}).items():
            ids = np.asarray(caras) + 1
            f.write(f"CMBLOCK,{nombre},ELEM,{len(ids):8d}\n(8i10)\n")
            for j in range(0, len(ids), 8):
                f.write("".join(f"{x:10d}" for x in ids[j:j + 8]) + "\n")
        f.write("finish\n")
