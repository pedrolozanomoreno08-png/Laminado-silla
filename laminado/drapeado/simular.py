"""Simulación completa de una capa: darts, red, mapeo, energía, darts y empalmes automáticos."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from ..geom.consulta import Consultor, proyectar_tangente
from ..geom.cortes import abrir_rendija, cortar_por_nivel
from ..geom.geodesia import camino_por_puntos, camino_vertices, fast_marching
from ..geom.malla import TriMesh
from .energia import drapear_energia
from .red import Dart, drapear_red
from .resultado import ResultadoDrapeado, mapear_red


@dataclass
class OpcionesDrapeado:
    estrategia: str = "geodesica"
    h: float = 5.0                       # tamaño de celda de la red (mm)
    angulo_bloqueo: float = 60.0         # cizalla a partir de la cual la red no admite nodos
    curva_semilla: np.ndarray | None = None
    unidireccional: bool = False
    k_t: float = 200.0
    k_s: float = 1.0
    max_iter_energia: int = 60
    relajar_darts: bool = True


def _camino_dart(m: TriMesh, puntos):
    """Convierte la polilínea de un dart en un camino de vértices que empieza en el borde."""
    arbol = cKDTree(m.V)
    _, idx = arbol.query(np.asarray(puntos, float))
    idx = [int(i) for k, i in enumerate(idx) if k == 0 or i != idx[k - 1]]
    borde = np.nonzero(m.boundary_vertices_mask)[0]
    if not m.boundary_vertices_mask[idx[0]]:
        # empezar en el vértice de borde más cercano al primer punto
        b = borde[np.argmin(np.linalg.norm(m.V[borde] - m.V[idx[0]], axis=1))]
        idx = [int(b)] + idx
    if len(idx) < 2:
        return None
    cam = camino_por_puntos(m, idx)
    # quitar vértices de borde intermedios (el dart no debe recorrer el borde)
    ultimo_borde = 0
    for k, v in enumerate(cam):
        if m.boundary_vertices_mask[v]:
            ultimo_borde = k
        else:
            break
    return cam[ultimo_borde:]


def preparar_darts(malla: TriMesh, darts):
    """Abre las rendijas de los darts. Devuelve (malla con rendijas, origen, darts ajustados)."""
    m = malla
    origen = np.arange(malla.nv)
    ajustados = []
    for d in darts:
        cam = _camino_dart(m, d.puntos)
        if cam is None or len(cam) < 2:
            continue
        m2, org = abrir_rendija(m, cam)
        origen = origen[org]
        ajustados.append(Dart(m.V[cam].copy(), d.nombre, d.automatico))
        m = m2
    return m, origen, ajustados


def simular_capa(malla_capa: TriMesh, consultor_molde: Consultor, semilla, direccion,
                 opciones: OpcionesDrapeado, darts=None) -> ResultadoDrapeado:
    """Drapea una capa (malla ya recortada a su contorno) sobre el molde (superficie extendida)."""
    darts = list(darts or [])
    m, origen, darts_aj = preparar_darts(malla_capa, darts) if darts else (malla_capa, np.arange(malla_capa.nv), [])
    h = opciones.h
    P0 = consultor_molde.proyectar(np.asarray(semilla, float)[None])[0][0]
    radio = float(np.linalg.norm(m.V - P0, axis=1).max()) * 1.25 + 3 * h
    zc = Consultor(malla_capa)

    def zona(P):
        return zc.mas_cercano(P)[3] < 2.5 * h

    estr = opciones.estrategia
    # con relajación, la red se calcula continua (sin rendijas) y la energía abre los darts
    darts_red = [] if (darts_aj and opciones.relajar_darts) else darts_aj
    red = drapear_red(consultor_molde, P0, direccion, h, estr if estr != "energia" else "geodesica",
                      radio=radio, zona=zona, darts=darts_red,
                      curva_semilla=opciones.curva_semilla, angulo_max=opciones.angulo_bloqueo)
    if darts_red is darts_aj:
        uv, cubierto, _ = mapear_red(red, m)
    else:
        uv0, cub0, _ = mapear_red(red, malla_capa)
        uv, cubierto = uv0[origen], cub0[origen]
    info = {"nodos_red": int(red.ok.sum())}
    sv = int(np.argmin(np.linalg.norm(m.V - P0, axis=1)))
    n0 = consultor_molde.proyectar(P0[None])[1]
    d0 = proyectar_tangente(np.asarray(direccion, float)[None], n0)[0]
    kw = dict(k_t=opciones.k_t, k_s=opciones.k_s,
              k_t_trama=opciones.k_t * (0.05 if opciones.unidireccional else 1.0))
    if estr == "energia" or (darts_aj and opciones.relajar_darts):
        # mínima energía (o relajación de la malla con rendijas para que los darts se abran)
        uv, inf = drapear_energia(m, uv, sv, d0, max_iter=opciones.max_iter_energia, **kw)
        uv -= uv[sv]
        info.update(inf)
        # la solución por energía cubre toda la capa (la producibilidad la da la cizalla)
        cubierto = np.ones(m.nv, bool)
    elif not cubierto.all():
        # zonas que la red no alcanza (bloqueo del tejido): se completan por energía con la
        # zona alcanzada fija, para que el patrón plano sea continuo
        uv, inf = drapear_energia(m, uv, sv, d0, max_iter=15, fijos=np.nonzero(cubierto)[0], **kw)
        info["relleno_no_alcanzado"] = inf["estado"]
    return ResultadoDrapeado(m, uv, cubierto, estr, red, origen, darts_aj, info)


# --------------------------------------------------------------------------- darts automáticos
def proponer_darts(res: ResultadoDrapeado, semilla, limite: float, aviso: float, n_max: int = 6,
                   h: float = 5.0, existentes=()):
    """Propone darts en las zonas con cizalla superior al límite.

    Cada zona conexa con cizalla > límite recibe un dart que parte del punto de borde más
    cercano a la zona y avanza hacia la semilla hasta que la cizalla baja del aviso."""
    m = res.malla
    g = np.abs(res.cizalla())
    gv = m.interpolar_caras_a_vertices(g)
    altas = g > limite
    if not altas.any():
        return []
    sub, ids, caras = m.submalla(altas)
    ncomp, etiqueta = sub.componentes()
    zonas = sorted(range(ncomp), key=lambda c: -m.face_areas[caras[etiqueta == c]].sum())
    sv = int(np.argmin(np.linalg.norm(m.V - np.asarray(semilla, float), axis=1)))
    borde = m.boundary_vertices_mask
    darts = []
    for c in zonas[:n_max]:
        fz = caras[etiqueta == c]
        if m.face_areas[fz].sum() < 4 * h * h:
            continue
        vz = np.unique(m.F[fz].ravel())
        # punto de borde de la zona (o el más cercano a ella) con mayor cizalla
        vb = vz[borde[vz]]
        if len(vb) == 0:
            d = fast_marching(m, vz, limite=50 * h)
            cand = np.nonzero(borde & np.isfinite(d))[0]
            if len(cand) == 0:
                continue
            vb = cand[np.argsort(d[cand])[:5]]
        b = int(vb[np.argmax(gv[vb])])
        # no repetir un dart que ya arranca cerca
        if any(np.linalg.norm(np.asarray(d.puntos)[0] - m.V[b]) < 8 * h for d in list(existentes) + darts):
            continue
        cam = camino_vertices(m, b, sv)
        # recortar donde la cizalla cae por debajo del aviso tras haber atravesado la zona
        largo = np.r_[0, np.cumsum(np.linalg.norm(np.diff(m.V[cam], axis=0), axis=1))]
        fin = None
        paso_zona = False
        for k, v in enumerate(cam):
            if gv[v] > limite:
                paso_zona = True
            if paso_zona and gv[v] < 0.6 * aviso:
                fin = k
                break
        if fin is None:
            fin = int(np.searchsorted(largo, 10 * h))
        fin = max(fin, int(np.searchsorted(largo, 6 * h)))
        # nunca más allá de la mitad del camino hacia la semilla
        fin = min(fin, int(np.searchsorted(largo, 0.5 * largo[-1])))
        if fin < 2:
            continue
        darts.append(Dart(m.V[cam[:fin + 1]].copy(), f"Dart auto {len(existentes) + len(darts) + 1}", True))
    return darts


def simular_con_darts_auto(malla_capa, consultor, semilla, direccion, opciones, limite, aviso,
                           darts=None, n_max=6, iter_max=3):
    darts = list(darts or [])
    res = simular_capa(malla_capa, consultor, semilla, direccion, opciones, darts)
    for _ in range(iter_max):
        nuevos = proponer_darts(res, semilla, limite, aviso, n_max - len(darts), opciones.h, darts)
        if not nuevos:
            break
        darts += nuevos
        res = simular_capa(malla_capa, consultor, semilla, direccion, opciones, darts)
        if len(darts) >= n_max:
            break
    return res


# --------------------------------------------------------------------------- empalmes
@dataclass
class Empalme:
    """División de una capa en piezas.

    tipo: 'plano' (punto + normal), 'urdimbre' (u = valor), 'trama' (v = valor).
    solape > 0 solapa las piezas, 0 a tope, < 0 deja hueco."""
    tipo: str = "plano"
    punto: tuple = (0.0, 0.0, 0.0)
    normal: tuple = (1.0, 0.0, 0.0)
    valor: float = 0.0
    solape: float = 10.0
    nombre: str = "Empalme"
    automatico: bool = False


def _phi_empalme(res: ResultadoDrapeado, e: Empalme):
    if e.tipo == "urdimbre":
        return res.uv[:, 0] - e.valor
    if e.tipo == "trama":
        return res.uv[:, 1] - e.valor
    n = np.asarray(e.normal, float)
    n /= np.linalg.norm(n)
    return (res.malla.V - np.asarray(e.punto, float)) @ n


def dividir_en_piezas(res: ResultadoDrapeado, empalmes):
    """Divide el resultado en piezas según los empalmes; cada pieza conserva la continuidad de
    fibra del tejido (las coordenadas uv del drapeado de la capa completa).

    Devuelve lista de ResultadoDrapeado (una por pieza)."""
    piezas = [res]
    for e in empalmes:
        nuevas = []
        for p in piezas:
            phi = _phi_empalme(p, e)
            if phi.min() >= 0 or phi.max() <= 0:
                nuevas.append(p)
                continue
            for lado, nivel in ((-1, e.solape / 2), (1, -e.solape / 2)):
                extra = np.c_[p.uv, p.cubierto.astype(float)]
                m2, phi2, ex2 = cortar_por_nivel(p.malla, phi, nivel, extra)
                pc = phi2[m2.F].mean(1)
                mask = pc <= 0 if lado < 0 else pc >= 0
                sub, ids, _ = m2.submalla(mask)
                if sub.nf == 0:
                    continue
                nuevas.append(ResultadoDrapeado(sub, ex2[ids, :2], ex2[ids, 2] > 0.5, p.estrategia,
                                                p.red, None, p.darts, dict(p.info)))
        piezas = nuevas
    return piezas


def empalmes_por_ancho(res: ResultadoDrapeado, ancho_rollo: float, solape: float = 10.0,
                       margen: float = 10.0):
    """Empalmes automáticos a lo largo de la trama para que cada pieza quepa en el rollo.

    La urdimbre (0°) va a lo largo del rollo, así que el ancho limita la extensión en v."""
    v = res.uv[:, 1]
    lo, hi = float(v.min()), float(v.max())
    util = ancho_rollo - 2 * margen - solape
    if hi - lo <= ancho_rollo - 2 * margen or util <= 0:
        return []
    n = int(np.ceil((hi - lo) / util))
    paso = (hi - lo) / n
    return [Empalme("trama", valor=lo + k * paso, solape=solape, nombre=f"Empalme auto {k}",
                    automatico=True) for k in range(1, n)]
