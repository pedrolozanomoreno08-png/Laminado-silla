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
    # margen de la zona de interés: la red necesita nodos fuera de la capa para llegar a las
    # esquinas y, en tubos con el hilo en hélice, para avanzar a lo largo del tubo (los solapes
    # de vueltas sucesivas los resuelve la asignación por continuidad de mapear_red)
    # (acotado a 100 celdas: en piezas grandes y planas no hace falta más y la red crecería mucho)
    margen_zona = max(3 * h, min(0.6 * float(np.linalg.norm(np.ptp(malla_capa.V, axis=0))), 100 * h))

    def zona(P):
        return zc.mas_cercano(P)[3] < margen_zona

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
    de_red = cubierto.copy()
    if not cubierto.all() and cubierto.sum() >= 3:
        # punto de partida de las zonas no alcanzadas: el tejido se prolonga desde la zona
        # cubierta con la dirección local de los hilos (luego la energía lo ajusta)
        from ..geom.parametrizacion import prolongar
        try:
            uv = prolongar(m, uv, cubierto)
        except Exception:  # noqa: BLE001
            pass
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
        de_red = None
    elif not cubierto.all():
        # zonas que la red no alcanza (bloqueo del tejido): se completan por energía con la
        # zona alcanzada fija, para que el patrón plano sea continuo
        uv, inf = drapear_energia(m, uv, sv, d0, max_iter=60, fijos=np.nonzero(cubierto)[0], **kw)
        info["relleno_no_alcanzado"] = inf["estado"]
        # las zonas completadas cuentan como cubiertas si el tejido no ha tenido que estirarse
        # (solución físicamente válida: hilos inextensibles, sólo cizalla)
        tmp = ResultadoDrapeado(m, uv, cubierto)
        ea, eb = tmp.deformaciones()
        ok_cara = (np.abs(ea) < 0.02) & (np.abs(eb) < 0.02) & (tmp.gradientes()[2] > 0)
        malos = np.zeros(m.nv, bool)
        malos[m.F[~ok_cara].ravel()] = True
        info["relleno_valido_%"] = float(100 * ok_cara.mean())
        cubierto = cubierto | ~malos
    return ResultadoDrapeado(m, uv, cubierto, estr, red, origen, darts_aj, info, de_red)


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


def _solape_cerca(res, dart, radio):
    """Solape del patrón plano en el entorno de un dart (mm²)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(dart.puntos)).query(res.malla.face_centroids)
    return res.solape_plano(d < radio)


def simular_con_darts_auto(malla_capa, consultor, semilla, direccion, opciones, limite, aviso,
                           darts=None, n_max=6, iter_max=3):
    """Añade darts automáticos donde la cizalla supera el límite.

    Sólo se conservan los darts que se abren en V en el patrón plano: en zonas en silla de
    montar los bordes del corte se cruzarían (sobra tela en vez de faltar) y el dart no sirve;
    ahí hace falta un empalme o dividir la capa."""
    darts = list(darts or [])
    res = simular_capa(malla_capa, consultor, semilla, direccion, opciones, darts)
    descartados = []
    for _ in range(iter_max):
        nuevos = proponer_darts(res, semilla, limite, aviso, n_max - len(darts), opciones.h, darts + descartados)
        if not nuevos:
            break
        prueba = simular_capa(malla_capa, consultor, semilla, direccion, opciones, darts + nuevos)
        utiles = []
        for d in nuevos:
            # el solape aceptable en el entorno del dart: el que ya había sin él
            antes = _solape_cerca(res, d, 4 * opciones.h)
            despues = _solape_cerca(prueba, d, 4 * opciones.h)
            if despues <= antes + 0.5 * opciones.h ** 2:
                utiles.append(d)
            else:
                descartados.append(d)
        if not utiles:
            break
        darts += utiles
        res = prueba if len(utiles) == len(nuevos) else simular_capa(malla_capa, consultor, semilla,
                                                                         direccion, opciones, darts)
        if len(darts) >= n_max:
            break
    res.info["darts_descartados_por_solape"] = len(descartados)
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


def proponer_empalmes_cizalla(piezas, semilla, limite, h, solape=15.0, n_existentes=0, n_max=4,
                              fracciones=None):
    """Empalmes que separan las zonas con cizalla por encima del límite de la semilla de su
    pieza: cada zona queda en una pieza nueva que se drapea desde su propio centro (el camino
    del tejido es más corto y la cizalla acumulada, menor). Es la solución para zonas en silla
    de montar, donde un dart no sirve.

    Sin ``fracciones`` devuelve una lista de empalmes (plano a medio camino semilla→zona). Con
    ``fracciones`` (p.ej. (0.5, 0.35, 0.65)) devuelve, por zona, la lista de alternativas con el
    plano en esas posiciones del camino."""
    from ..geom.malla import TriMesh  # noqa: F401
    fr = (0.5,) if fracciones is None else tuple(fracciones)
    propuestas = []
    for pz in piezas:
        g = np.abs(pz.cizalla())
        phi = getattr(pz, "phi_neto", None)
        nz = (phi[pz.malla.F].mean(1) <= 1e-9) if phi is not None else np.ones(pz.malla.nf, bool)
        altas = (g > limite) & nz
        if not altas.any():
            continue
        sub, _, caras = pz.malla.submalla(altas)
        nc, et = sub.componentes()
        S = pz.malla.V[int(np.argmin(np.linalg.norm(pz.uv, axis=1)))]
        zonas = sorted(range(nc), key=lambda c: -pz.malla.face_areas[caras[et == c]].sum())
        for c in zonas:
            fz = caras[et == c]
            A = pz.malla.face_areas[fz]
            if A.sum() < 25 * h * h:
                continue
            Z = (pz.malla.face_centroids[fz] * A[:, None]).sum(0) / A.sum()
            d = Z - S
            L = np.linalg.norm(d)
            if L < 6 * h:
                continue
            alt = [Empalme("plano", tuple(S + t * d), tuple(d / L), solape=solape,
                           nombre=f"Empalme por cizalla {n_existentes + len(propuestas) + 1}",
                           automatico=True) for t in fr]
            propuestas.append((A.sum(), alt))
    propuestas.sort(key=lambda t: -t[0])
    propuestas = propuestas[:n_max]
    if fracciones is None:
        return [alt[0] for _, alt in propuestas]
    return [alt for _, alt in propuestas]
