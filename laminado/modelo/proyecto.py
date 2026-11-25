"""Modelo del laminado y proyecto: rosetas, contornos, capas, secuencias, núcleos, simulación,
core sample, espesores y guardado/apertura (.lam)."""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..ansys.lector import Molde, abrir_molde
from ..drapeado.red import Dart
from ..drapeado.resultado import ResultadoDrapeado
from ..drapeado.simular import (Empalme, OpcionesDrapeado, dividir_en_piezas, empalmes_por_ancho,
                                proponer_empalmes_cizalla, simular_capa, simular_con_darts_auto)
from ..geom.consulta import Consultor, proyectar_tangente
from ..geom.cortes import cortar_por_nivel, extender_superficie
from ..geom.geodesia import distancia_a_borde_region, fast_marching
from ..geom.malla import TriMesh
from . import clt
from .materiales import Material, biblioteca


# ------------------------------------------------------------------------------------ rosetas
@dataclass
class Roseta:
    nombre: str = "Roseta global"
    origen: tuple = (0.0, 0.0, 0.0)
    eje_x: tuple = (1.0, 0.0, 0.0)          # dirección de 0°
    normal: tuple = (0.0, 0.0, 1.0)          # sentido de giro positivo (regla de la mano derecha)

    def referencia(self, n):
        """Dirección de 0° proyectada sobre planos tangentes de normales n (k,3)."""
        x = np.broadcast_to(np.asarray(self.eje_x, float), np.shape(n))
        r = proyectar_tangente(x, n)
        # si el eje x es casi normal a la superficie se usa la proyección del eje y
        malo = np.abs((np.asarray(self.eje_x, float)[None] * n).sum(1)) > 0.97
        if malo.any():
            y = np.cross(np.asarray(self.normal, float), np.asarray(self.eje_x, float))
            r[malo] = proyectar_tangente(np.broadcast_to(y, n[malo].shape), n[malo])
        return r

    def direccion(self, P, n, angulo):
        """Dirección de la capa (ángulo en grados respecto a 0°) en el punto de normal n."""
        n = np.atleast_2d(n)
        r = self.referencia(n)
        # el giro positivo sigue la normal de la roseta; si la superficie mira al revés se invierte
        sg = np.sign((n * np.asarray(self.normal, float)).sum(1, keepdims=True))
        sg[sg == 0] = 1
        t = np.cross(n, r) * sg
        a = np.radians(angulo)
        return np.cos(a) * r + np.sin(a) * t


# ---------------------------------------------------------------------------------- contornos
@dataclass
class Contorno:
    """Contorno neto de una capa.

    tipo: 'molde' (toda la superficie), 'seleccion' (componentes de Ansys), 'poligono'
    (polígono en el plano XY de la roseta, proyectado sobre la superficie)."""
    tipo: str = "molde"
    selecciones: list = field(default_factory=list)
    puntos: list = field(default_factory=list)          # [(x, y), ...] o [(x, y, z), ...]
    roseta: str = ""                                    # roseta que define el plano del polígono
    desfase: float = 0.0                                # mm (+ crece, − encoge)
    exceso: float = 0.0                                 # mm de material de recorte fuera del neto
    suavizado: float = 0.0                              # mm: suaviza el borde escalonado de una selección


@dataclass
class Capa:
    nombre: str
    material: str
    orientacion: float = 0.0
    roseta: str = "Roseta global"
    contorno: Contorno = field(default_factory=Contorno)
    semilla: tuple | None = None
    estrategia: str = "geodesica"
    h: float = 0.0                       # 0 → tamaño de celda del proyecto
    darts: list = field(default_factory=list)          # listas de puntos 3D
    darts_auto: bool = False
    empalmes: list = field(default_factory=list)       # dicts de Empalme
    empalme_auto: bool = True
    solape_empalme: float = 15.0
    empalmes_auto_cizalla: bool = False   # dividir la capa donde la cizalla supera el límite
    curva_semilla: list = field(default_factory=list)
    relleno: bool = True                  # completar por energía lo que la red no alcanza
    secuencia: str = ""
    activa: bool = True
    notas: str = ""


@dataclass
class Secuencia:
    nombre: str
    capas: list = field(default_factory=list)          # nombres de capa
    escalonado: float = 0.0                            # mm de desfase adicional entre capas sucesivas


@dataclass
class Nucleo:
    nombre: str
    material: str
    contorno: Contorno = field(default_factory=Contorno)
    espesor: float = 5.0
    chaflan: float = 30.0          # grados
    despues_de: str = ""           # capa sobre la que se coloca


@dataclass
class ResultadoCapa:
    capa: str
    drapeado: ResultadoDrapeado
    piezas: list                          # ResultadoDrapeado por pieza
    neto: TriMesh                         # malla de la región neta (sin exceso)
    phi_neto: np.ndarray                  # en los vértices de drapeado.malla (≤0 dentro del neto)
    empalmes: list = field(default_factory=list)
    resumen: dict = field(default_factory=dict)
    neto_molde: TriMesh | None = None      # región neta sobre el molde (sin desplazar por espesor)
    desplazamiento: float = 0.0            # desplazamiento máximo de la superficie de drapeado (mm)

    @property
    def neto_en_molde(self) -> TriMesh:
        return self.neto_molde if self.neto_molde is not None else self.neto

    def contornos_netos_uv(self, pieza=None):
        """Bucles del contorno neto en el plano para el resultado completo o una pieza."""
        r = self.drapeado if pieza is None else pieza
        m = r.malla
        phi = self._phi_de(r)
        if phi is None:
            return []
        dentro = phi[m.F].mean(1) <= 1e-9
        if not dentro.any():
            return []
        sub, ids, _ = m.submalla(dentro)
        out = []
        for b in sub.boundary_loops():
            out.append(r.uv[ids[b]])
        return out

    def localizar(self, X):
        """(resultado de pieza o de la capa, cara) más cercanos a cada punto X (k,3)."""
        cands = [pz for pz in self.piezas if getattr(pz, "redrapeada", False)] or [self.drapeado]
        mejor_d = np.full(len(X), np.inf)
        mejor_r = np.zeros(len(X), int)
        mejor_f = np.zeros(len(X), int)
        for k, r in enumerate(cands):
            _, f, _, d = Consultor(r.malla).mas_cercano(X)
            m = d < mejor_d
            mejor_d[m], mejor_r[m], mejor_f[m] = d[m], k, f[m]
        return [cands[k] for k in mejor_r], mejor_f

    def _phi_de(self, r):
        if r is self.drapeado:
            return self.phi_neto
        return getattr(r, "phi_neto", None)


class Proyecto:
    def __init__(self, nombre="Proyecto"):
        self.nombre = nombre
        self.molde: Molde | None = None
        self.materiales: dict[str, Material] = biblioteca()
        self.rosetas: dict[str, Roseta] = {"Roseta global": Roseta()}
        self.capas: list[Capa] = []
        self.secuencias: dict[str, Secuencia] = {}
        self.nucleos: list[Nucleo] = []
        self.h = 6.0
        self.extension = 60.0
        # laminado con espesor: cada capa se drapea sobre la superficie desplazada por el espesor
        # de las capas (y núcleos) que tiene debajo
        self.drapeado_con_espesor = True
        self.sentido_laminado = 1.0          # +1: el laminado crece según la normal del molde
        self.rampa = 20.0                    # longitud de la terminación de capa / espesor
        self.resultados: dict[str, ResultadoCapa] = {}
        self.ruta = None
        self._ext = None
        self._cons = None

    # ------------------------------------------------------------------ molde
    def cargar_molde(self, ruta_o_molde):
        self.molde = abrir_molde(ruta_o_molde) if isinstance(ruta_o_molde, str) else ruta_o_molde
        # rosetas definidas en ACP (si el modelo viene de un proyecto con composites)
        for nombre, o, d1, d2 in getattr(self.molde, "rosetas", []) or []:
            nmb = f"ACP: {nombre}"
            if nmb not in self.rosetas and np.linalg.norm(d1) > 0 and np.linalg.norm(np.cross(d1, d2)) > 0:
                self.rosetas[nmb] = Roseta(nmb, tuple(map(float, o)), tuple(map(float, d1)),
                                           tuple(map(float, np.cross(d1, d2))))
        self._ext = None
        self._cons = None
        self._superficies = {}
        self.resultados.clear()

    @property
    def malla(self) -> TriMesh:
        return self.molde.malla

    def superficie(self, componentes=None):
        """Superficie de drapeado para un conjunto de paneles (componentes conexas del molde por
        aristas de variedad): sus caras + tapas de agujeros + superficie extendida.

        En un ensamblaje (casco, cubierta, mamparos unidos en T) cada capa se drapea sólo sobre
        sus paneles: la red de tejido no «cae» sobre otros paneles al cruzar una abertura.
        Devuelve (malla, consultor, cara original del molde de cada cara o −1)."""
        m = self.malla
        nc, etiqueta = m.componentes()
        if componentes is None:
            componentes = range(nc)
        clave = tuple(sorted(set(int(c) for c in componentes)))
        if not hasattr(self, "_superficies"):
            self._superficies = {}
        if clave not in self._superficies:
            from ..geom.cortes import rellenar_agujeros
            mask = np.isin(etiqueta, clave)
            base, _, caras = m.submalla(mask)
            diag = float(np.linalg.norm(np.ptp(base.V, axis=0)))
            # superficie extendida amplia: los hilos generadores necesitan seguir más allá del
            # borde para que la red llegue a los extremos de piezas largas con el tejido a 45°
            ext = max(self.extension, 0.35 * diag)
            paso = 2.5 * float(np.median(base.edge_lengths))
            con_tapas = rellenar_agujeros(base)
            S = extender_superficie(con_tapas, ext, anillos=max(2, int(np.ceil(ext / paso))))
            mapa = -np.ones(S.nf, np.int64)
            mapa[:base.nf] = caras            # las caras del molde van primero (tapas y extensión después)
            # panel plano (mamparo, costilla, placa): el tejido se drapea sobre su plano infinito,
            # sin depender de la forma del contorno (entrantes, aberturas en U…)
            plano = _plano_si_plano(base, ext)
            self._superficies[clave] = (S, Consultor(plano if plano is not None else S), mapa)
        return self._superficies[clave]

    def extendida(self):
        S, cons, _ = self.superficie()
        return S, cons

    def componentes_de(self, contorno: "Contorno"):
        """Paneles del molde afectados por un contorno."""
        m = self.malla
        nc, etiqueta = m.componentes()
        if contorno.tipo == "seleccion":
            caras = np.concatenate([np.asarray(self.molde.selecciones.get(n, []), int) for n in contorno.selecciones] or
                                   [np.array([], int)])
            if len(caras) == 0:
                raise ValueError("La selección del contorno está vacía")
            return sorted(set(etiqueta[caras].tolist()))
        if contorno.tipo == "poligono":
            phi = self._phi_contorno(contorno, m, np.arange(m.nf))
            dentro = phi[m.F].min(1) <= contorno.desfase + contorno.exceso
            return sorted(set(etiqueta[dentro].tolist())) or list(range(nc))
        return list(range(nc))

    # -------------------------------------------------------------- utilidades
    def capa(self, nombre) -> Capa:
        for c in self.capas:
            if c.nombre == nombre:
                return c
        raise KeyError(nombre)

    def nombre_libre(self, base):
        nombres = {c.nombre for c in self.capas}
        k = 1
        while f"{base}{k:03d}" in nombres:
            k += 1
        return f"{base}{k:03d}"

    def desfase_total(self, capa: Capa) -> float:
        d = capa.contorno.desfase
        if capa.secuencia and capa.secuencia in self.secuencias:
            s = self.secuencias[capa.secuencia]
            if capa.nombre in s.capas:
                d -= s.escalonado * s.capas.index(capa.nombre)
        return d

    def semilla_de(self, capa: Capa, neto: TriMesh):
        if capa.semilla is not None:
            return np.asarray(capa.semilla, float)
        # centro de la capa: entre los puntos bien alejados del borde, el más próximo al
        # centroide de área (en piezas alargadas, como medias cañas, queda a mitad de largo)
        d = fast_marching(neto, neto.boundary_vertices_mask)
        d = np.where(np.isfinite(d), d, 0.0)
        cand = np.nonzero(d >= 0.7 * d.max())[0]
        cg = (neto.face_centroids * neto.face_areas[:, None]).sum(0) / neto.area
        return neto.V[cand[int(np.argmin(np.linalg.norm(neto.V[cand] - cg, axis=1)))]]

    # ---------------------------------------------------------------- regiones
    def _phi_contorno(self, contorno: Contorno, ext: TriMesh, mapa=None):
        """Función con signo (≤0 dentro) en los vértices de la superficie de drapeado.

        ``mapa``: cara original del molde de cada cara de ``ext`` (−1 en tapas y extensión)."""
        if mapa is None:
            mapa = np.arange(ext.nf)
            mapa[self.malla.nf:] = -1
        if contorno.tipo == "molde":
            # distancia al borde del molde (positiva en tapas y extensión)
            return distancia_a_borde_region(ext, mapa >= 0)
        if contorno.tipo == "seleccion":
            sel_total = np.zeros(self.malla.nf + 1, bool)
            for nombre in contorno.selecciones:
                sel = self.molde.selecciones.get(nombre)
                if sel is not None:
                    sel_total[np.asarray(sel, int)] = True
            caras = sel_total[np.where(mapa >= 0, mapa, self.malla.nf)]
            if not caras.any():
                raise ValueError("La selección del contorno está vacía")
            # la distancia sólo hace falta cerca del borde (desfases, excesos, rampas, suavizado)
            lim = 300.0 + abs(contorno.desfase) + contorno.exceso
            phi = distancia_a_borde_region(ext, caras, limite=lim)
            sv = contorno.suavizado
            a0 = float(ext.face_areas[caras].sum())
            while sv > 0.25:
                # el suavizado no debe comerse la selección (zonas estrechas): se reduce si hace falta
                ps = suavizar_campo(ext, phi, sv)
                a1 = float(ext.face_areas[ps[ext.F].mean(1) <= 0].sum())
                if a1 >= 0.9 * a0:
                    return ps
                sv /= 2
            return phi
        if contorno.tipo == "poligono":
            from shapely.geometry import Point, Polygon
            ros = self.rosetas.get(contorno.roseta) or next(iter(self.rosetas.values()))
            o = np.asarray(ros.origen, float)
            ex = np.asarray(ros.eje_x, float)
            ex /= np.linalg.norm(ex)
            nz = np.asarray(ros.normal, float)
            nz /= np.linalg.norm(nz)
            ey = np.cross(nz, ex)
            P = np.asarray(contorno.puntos, float)
            if P.shape[1] == 3:
                P = np.c_[(P - o) @ ex, (P - o) @ ey]
            poli = Polygon(P).buffer(0)
            Q = np.c_[(ext.V - o) @ ex, (ext.V - o) @ ey]
            import shapely
            pts = shapely.points(Q)
            d = shapely.distance(poli.exterior, pts)
            dentro = shapely.contains(poli, pts)
            return np.where(dentro, -d, d)
        raise ValueError(f"Tipo de contorno desconocido: {contorno.tipo}")

    def _clave_contorno(self, contorno: Contorno):
        sel = tuple((n, hash(np.asarray(self.molde.selecciones.get(n, []), np.int64).tobytes()))
                    for n in contorno.selecciones) if contorno.tipo == "seleccion" else ()
        return (contorno.tipo, sel, json.dumps(contorno.puntos, default=_json_def), contorno.roseta,
                round(contorno.suavizado, 6))

    def _phi_cache(self, contorno, ext, mapa, clave_sup):
        if not hasattr(self, "_phis"):
            self._phis = {}
        k = (clave_sup, id(ext), self._clave_contorno(contorno))
        if k not in self._phis:
            try:
                self._phis[k] = self._phi_contorno(contorno, ext, mapa)
            except ValueError:
                self._phis[k] = None
        return self._phis[k]

    def espesor_debajo(self, capa, ext, mapa, clave_sup):
        """Espesor (mm) en los vértices de la superficie ``ext`` de las capas y núcleos colocados
        antes que ``capa`` (todas si capa es None). Las terminaciones de capa y los chaflanes
        de núcleo son rampas (longitud = rampa·espesor o espesor/tan(chaflán))."""
        t = np.zeros(ext.nv)
        antes = []
        for c in self.capas:
            if capa is not None and c is capa:
                break
            if c.activa:
                antes.append(c)
        nombres = {c.nombre for c in antes}
        for c in antes:
            phi = self._phi_cache(c.contorno, ext, mapa, clave_sup)
            if phi is None:
                continue
            e = self.materiales[c.material].espesor
            L = max(self.rampa * e, 1e-6)
            t += e * np.clip(0.5 - (phi - self.desfase_total(c)) / L, 0.0, 1.0)
        for nu in self.nucleos:
            if capa is not None and nu.despues_de and nu.despues_de not in nombres:
                continue
            if capa is not None and not nu.despues_de and not antes:
                continue
            phi = self._phi_cache(nu.contorno, ext, mapa, clave_sup)
            if phi is None:
                continue
            L = max(nu.espesor / max(np.tan(np.radians(nu.chaflan)), 1e-3), 1e-6)
            # chaflán hacia dentro del contorno: espesor completo a L del borde
            t += nu.espesor * np.clip(-(phi - nu.contorno.desfase) / L, 0.0, 1.0)
        return t

    def region(self, capa: Capa | Nucleo):
        """(malla de corte = neto + exceso, phi neto en sus vértices, malla neta).

        Se trabaja sobre la superficie de los paneles afectados (ver ``superficie``). Con
        ``drapeado_con_espesor`` esa superficie se desplaza según la normal por el espesor de lo
        que hay debajo de la capa. El consultor de la superficie de drapeado queda en
        ``self._cons_capa``, la malla neta sobre el molde en ``self._neto_molde`` y el
        desplazamiento máximo en ``self._desplazamiento``."""
        c = capa.contorno
        comps = self.componentes_de(c)
        ext, cons, mapa = self.superficie(comps)
        clave_sup = tuple(comps)
        phi = self._phi_cache(c, ext, mapa, clave_sup)
        if phi is None:
            raise ValueError("La selección del contorno está vacía")
        phi = np.where(np.isfinite(phi), phi, np.sign(phi) * 1e6)
        desp = np.zeros((ext.nv, 3))
        if self.drapeado_con_espesor and isinstance(capa, Capa):
            t = self.espesor_debajo(capa, ext, mapa, clave_sup)
            if t.max() > 1e-9:
                desp = (self.sentido_laminado * t)[:, None] * ext.vertex_normals
                cons = Consultor(TriMesh(ext.V + desp, ext.F))
        self._cons_capa = cons
        self._desplazamiento = float(np.linalg.norm(desp, axis=1).max())
        desf = self.desfase_total(capa) if isinstance(capa, Capa) else c.desfase
        # corte por el contorno neto (desfase): phi' = phi - desfase
        m1, phi1, d1 = cortar_por_nivel(ext, phi, desf, extra=desp)
        neto_caras = phi1[m1.F].mean(1) <= 0
        if c.exceso > 0:
            d = distancia_a_borde_region(m1, neto_caras, limite=3 * c.exceso)
            d = np.where(np.isinf(d), np.sign(d) * 1e9, d)
            m2, d2, ex = cortar_por_nivel(m1, d, c.exceso, extra=np.c_[phi1, d1])
            corte = d2[m2.F].mean(1) <= 0
            sub, ids, _ = m2.submalla(corte)
            phi_sub, desp_sub = ex[ids, 0], ex[ids, 1:]
        else:
            sub, ids, _ = m1.submalla(neto_caras)
            phi_sub, desp_sub = phi1[ids], d1[ids]
        if sub.nf == 0:
            import os
            if os.environ.get("LAMINADO_DEBUG"):
                print("REGION VACIA", c.tipo, c.selecciones, [len(self.molde.selecciones.get(n, [])) for n in c.selecciones],
                      "phi min", float(np.min(phi)), "desf", desf, "suav", c.suavizado, "ext", ext.nf, flush=True)
            raise ValueError("El contorno de la capa no contiene superficie")
        # quedarse con la componente conexa mayor
        nc, et = sub.componentes()
        if nc > 1:
            mayor = np.argmax(np.bincount(et, weights=sub.face_areas))
            sub2, ids2, _ = sub.submalla(et == mayor)
            sub, phi_sub, desp_sub = sub2, phi_sub[ids2], desp_sub[ids2]
        neto_molde, idn, _ = sub.submalla(phi_sub[sub.F].mean(1) <= 1e-9)
        self._neto_molde = neto_molde
        sub_d = TriMesh(sub.V + desp_sub, sub.F)
        neto = TriMesh(neto_molde.V + desp_sub[idn], neto_molde.F)
        return sub_d, phi_sub, neto

    def superficie_laminado(self, componentes=None):
        """Superficie del molde desplazada por el espesor total del laminado (la otra cara de la
        pieza: interior si se lamina sobre un molde hembra, exterior si es macho)."""
        if componentes is None:
            componentes = range(self.malla.componentes()[0])
        comps = list(componentes)
        ext, _, mapa = self.superficie(comps)
        t = self.espesor_debajo(None, ext, mapa, tuple(comps))
        caras = mapa >= 0
        sub, ids, _ = ext.submalla(caras)
        V = sub.V + (self.sentido_laminado * t[ids])[:, None] * sub.vertex_normals
        return TriMesh(V, sub.F), t[ids]

    # -------------------------------------------------------------- simulación
    def opciones(self, capa: Capa) -> OpcionesDrapeado:
        mat = self.materiales[capa.material]
        return OpcionesDrapeado(estrategia=capa.estrategia, h=capa.h or self.h,
                                angulo_bloqueo=min(80.0, mat.cizalla_limite * 1.2),
                                curva_semilla=np.asarray(capa.curva_semilla, float) if capa.curva_semilla else None,
                                unidireccional=mat.unidireccional, rellenar=capa.relleno)

    def simular(self, capa: Capa | str, progreso=None) -> ResultadoCapa:
        if isinstance(capa, str):
            capa = self.capa(capa)
        if self.molde is None:
            raise RuntimeError("No hay molde cargado")
        mat = self.materiales[capa.material]
        malla_c, phi_c, neto = self.region(capa)
        cons = self._cons_capa
        semilla = self.semilla_de(capa, neto)
        n0 = cons.proyectar(semilla[None])[1]
        ros = self.rosetas.get(capa.roseta) or next(iter(self.rosetas.values()))
        direccion = ros.direccion(semilla[None], n0, capa.orientacion)[0]
        op = self.opciones(capa)
        if not capa.h:
            # celda adaptada a la curvatura: como mucho ~1/3 del radio de curvatura mínimo
            # (percentil 95) para que la red siga fielmente tubos finos y radios pequeños
            # (no por debajo de ~0.4 del tamaño de elemento: la malla no resuelve más detalle)
            k95 = _curvatura_p95(neto)
            h_curv = 0.3 / k95
            h0 = min(op.h, max(1.0, 0.4 * float(np.median(neto.edge_lengths)), h_curv))
            # piezas muy grandes y poco curvas (cubiertas, paneles de casco): limitar la red a
            # ~200 000 nodos; sólo donde la curvatura lo permite, para no perder precisión
            h_n = float(np.sqrt(malla_c.area / 2.0e5))
            op.h = max(h0, min(h_n, max(self.h, h_curv)))
        darts = [Dart(np.asarray(p, float), f"Dart {k + 1}") for k, p in enumerate(capa.darts)]
        if capa.darts_auto:
            res = simular_con_darts_auto(malla_c, cons, semilla, direccion, op,
                                         mat.cizalla_limite, mat.cizalla_aviso, darts)
        else:
            res = simular_capa(malla_c, cons, semilla, direccion, op, darts)
        # phi del contorno neto en la malla drapeada (posibles rendijas: usar origen)
        org = res.origen_vertice if res.origen_vertice is not None else np.arange(res.malla.nv)
        phi_r = phi_c[org]
        empalmes = [Empalme(**e) if isinstance(e, dict) else e for e in capa.empalmes]
        if capa.empalme_auto and mat.ancho_rollo > 0:
            empalmes += empalmes_por_ancho(res, mat.ancho_rollo, capa.solape_empalme)
        res.phi_neto = phi_r
        cache = {}
        piezas = self._piezas(res, empalmes, cons, op, cache)
        if capa.empalmes_auto_cizalla:
            n_auto_max = 4
            n_auto = 0
            diag = []
            res.info["empalmes_auto"] = diag
            roja = _area_roja(piezas, mat.cizalla_limite)
            probados = []
            for _ in range(4):
                grupos = proponer_empalmes_cizalla(piezas, semilla, mat.cizalla_limite, op.h,
                                                   capa.solape_empalme, len(empalmes),
                                                   n_max=n_auto_max - n_auto, fracciones=(0.5, 0.35, 0.65))
                # no repetir propuestas ya probadas
                grupos = [[e for e in alt if not _ya_probado(e, probados, op.h)] for alt in grupos]
                grupos = [alt for alt in grupos if alt]
                if not grupos:
                    break
                aceptados = []
                for alt in grupos:
                    for e in alt:
                        probados.append(e)
                        e.nombre = f"Empalme por cizalla {len(empalmes) + len(aceptados) + 1}"
                        prueba = self._piezas(res, empalmes + aceptados + [e], cons, op, cache)
                        r2 = _area_roja(prueba, mat.cizalla_limite)
                        # ni retales: cada pieza debe tener un tamaño razonable de laminado neto
                        amin = min(_area_neta(q) for q in prueba)
                        if amin < 25 * op.h ** 2:
                            diag.append(f"{e.nombre}: descartado (deja un retal de {amin:.0f} mm²)")
                            continue
                        if r2 < roja - max(1e-9, 0.02 * roja):
                            diag.append(f"{e.nombre}: aceptado (área sobre el límite {roja:.0f} → {r2:.0f} mm²)")
                            aceptados.append(e)
                            piezas, roja = prueba, r2
                            break
                        diag.append(f"{e.nombre}: descartado (no reduce la cizalla: {roja:.0f} → {r2:.0f} mm²)")
                if not aceptados:
                    break
                empalmes += aceptados
                n_auto += len(aceptados)
                if n_auto >= n_auto_max or roja <= 0:
                    break
        rc = ResultadoCapa(capa.nombre, res, piezas, neto, phi_r, empalmes)
        rc.neto_molde = self._neto_molde
        rc.desplazamiento = self._desplazamiento
        rc.resumen = self._resumen(capa, rc)
        self.resultados[capa.nombre] = rc
        return rc

    def _piezas(self, res, empalmes, cons, op, cache=None):
        """Divide la capa por los empalmes y drapea cada pieza por separado (con su propia
        semilla en el centro de la pieza y la dirección de fibra que tenía allí la capa entera):
        así un empalme alivia la cizalla igual que en el taller."""
        if not empalmes:
            return [res]
        piezas = _dividir_con_phi(res, empalmes)
        # trozos que sólo son exceso (sin laminado neto) no se cortan
        utiles = [q for q in piezas if _area_neta(q) > 1.0]
        piezas = utiles or piezas
        if len(piezas) <= 1:
            return piezas
        ea, _ = res.fibras()
        cr = Consultor(res.malla)
        # la curva de semilla de la capa entera no suele pasar por cada pieza: las piezas se
        # drapean geodésicamente desde su centro con la dirección local de la capa
        op_p = replace(op, estrategia="geodesica", curva_semilla=None) if op.estrategia == "curva" else op
        out = []
        for pz in piezas:
            clave = (pz.malla.nv, pz.malla.nf, hash(pz.malla.V.tobytes()), hash(pz.malla.F.tobytes()))
            if cache is not None and clave in cache:
                out.append(cache[clave])
                continue
            try:
                d = fast_marching(pz.malla, pz.malla.boundary_vertices_mask)
                d = np.where(np.isfinite(d), d, 0.0)
                cand = np.nonzero(d >= 0.7 * d.max())[0]
                cg = (pz.malla.face_centroids * pz.malla.face_areas[:, None]).sum(0) / pz.malla.area
                sd = pz.malla.V[cand[int(np.argmin(np.linalg.norm(pz.malla.V[cand] - cg, axis=1)))]]
                _, f, _, _ = cr.mas_cercano(sd[None])
                rp = simular_capa(pz.malla, cons, sd, ea[f[0]], op_p)
                rp.phi_neto = pz.phi_neto[rp.origen_vertice] if rp.origen_vertice is not None else pz.phi_neto
                rp.redrapeada = True
                # nunca peor que la pieza recortada de la capa entera
                elegida = pz if _cubierta(rp) + 1e-9 < _cubierta(pz) else rp
            except Exception:  # noqa: BLE001
                elegida = pz
            if cache is not None:
                cache[clave] = elegida
            out.append(elegida)
        return out

    def simular_todas(self, progreso=None):
        for k, c in enumerate(self.capas):
            if not c.activa:
                continue
            if progreso:
                progreso(k, len(self.capas), c.nombre)
            self.simular(c)

    def _resumen(self, capa, rc: ResultadoCapa):
        mat = self.materiales[capa.material]
        r = rc.drapeado
        neto = rc.phi_neto[r.malla.F].mean(1) <= 1e-9
        s = r.resumen(mat.cizalla_aviso, mat.cizalla_limite, neto)
        if len(rc.piezas) > 1 and any(getattr(pz, "redrapeada", False) for pz in rc.piezas):
            # con piezas drapeadas por separado, la producibilidad es la de las piezas
            g_l, a_l, c_l = [], [], []
            for pz in rc.piezas:
                nz = pz.phi_neto[pz.malla.F].mean(1) <= 1e-9
                g_l.append(np.abs(pz.cizalla())[nz])
                a_l.append(pz.malla.face_areas[nz])
                c_l.append(pz.cara_cubierta()[nz])
            g, A, cub = np.concatenate(g_l), np.concatenate(a_l), np.concatenate(c_l)
            if len(g):
                s.update({"cizalla_max": float(g.max()), "cizalla_media": float(np.average(g, weights=A)),
                          "cubierto_%": float(100 * A[cub].sum() / A.sum()),
                          "area_aviso_%": float(100 * A[g > mat.cizalla_aviso].sum() / A.sum()),
                          "area_limite_%": float(100 * A[g > mat.cizalla_limite].sum() / A.sum())})
        s["solape_patron_%"] = float(100 * sum(pz.solape_plano() for pz in rc.piezas)
                                     / max(sum(pz.area_plana() for pz in rc.piezas), 1e-300))
        neto_area = rc.neto.area
        area_corte = sum(p.area_plana() for p in rc.piezas)
        s.update({
            "material": mat.nombre, "orientacion": capa.orientacion,
            "area_neta_mm2": neto_area, "area_corte_mm2": area_corte,
            "peso_g": area_corte * 1e-6 * mat.peso_area,
            "piezas": len(rc.piezas), "empalmes": len(rc.empalmes),
            # ROJO: cizalla por encima del límite o zonas que el tejido no alcanza (> 2 %)
            # AMARILLO: cizalla por encima del aviso o pequeñas zonas no alcanzadas (bordes)
            "estado": ("ROJO" if s.get("area_limite_%", 0) > 0.5 or s["cubierto_%"] < 98.0
                       or s.get("solape_patron_%", 0) > 0.5 else
                       "AMARILLO" if s.get("area_aviso_%", 0) > 0.5 or s["cubierto_%"] < 99.5 else "VERDE"),
        })
        return s

    # ---------------------------------------------------- core sample y espesor
    def core_sample(self, P, tol=1.0):
        """Capas que cubren el punto P con su ángulo real de fibra, cizalla y espesor."""
        P = np.asarray(P, float)[None]
        filas = []
        z = 0.0
        for c in self.capas:
            rc = self.resultados.get(c.nombre)
            if rc is None or not c.activa:
                continue
            cons = Consultor(rc.neto_en_molde)
            Q, f, b, d = cons.mas_cercano(P)
            if d[0] > tol:
                continue
            # ángulo real en la cara del drapeado más cercana (sobre la superficie desplazada)
            Qd = Q if rc.desplazamiento == 0 else Consultor(rc.neto).mas_cercano(Q)[0]
            rs, fr = rc.localizar(Qd)
            r = rs[0]
            n = r.malla.face_normals[fr]
            ros = self.rosetas.get(c.roseta) or next(iter(self.rosetas.values()))
            ref = ros.direccion(Q, n, 0.0)
            ea = r.fibras()[0][fr]
            sg = float(np.sign((n[0] * np.asarray(ros.normal)).sum()) or 1)
            ang = float(np.degrees(np.arctan2((np.cross(ref, ea) * n).sum(1), (ref * ea).sum(1)))[0]) * sg
            mat = self.materiales[c.material]
            z += mat.espesor
            filas.append({"capa": c.nombre, "material": c.material, "nominal": c.orientacion,
                          "real": ang, "desviacion": _dif_ang(ang, c.orientacion, mat),
                          "cizalla": float(abs(r.cizalla()[fr[0]])), "espesor": mat.espesor, "z": z})
        return filas

    def laminado_en(self, P):
        """Lista (material, ángulo real, espesor) para CLT en el punto P."""
        return [(self.materiales[f["material"]], f["real"], f["espesor"]) for f in self.core_sample(P)]

    def propiedades_en(self, P):
        lam = self.laminado_en(P)
        if not lam:
            return {}
        return clt.constantes_ingenieria(lam)

    def espesores(self):
        """Espesor curado (mm, con las rampas de terminación de capa y chaflanes de núcleo) y
        número de capas en cada cara del molde."""
        m = self.malla
        X = m.face_centroids
        esp = np.zeros(m.nf)
        n = np.zeros(m.nf, int)
        nc, et = m.componentes()
        for comp in range(nc):
            ext, _, mapa = self.superficie([comp])
            t = self.espesor_debajo(None, ext, mapa, (comp,))
            caras = np.nonzero(mapa >= 0)[0]
            esp[mapa[caras]] = t[ext.F[caras]].mean(1)
        for c in self.capas:
            rc = self.resultados.get(c.nombre)
            if rc is None or not c.activa:
                continue
            d = Consultor(rc.neto_en_molde).mas_cercano(X)[3]
            n[d < 0.5] += 1
        return esp, n

    # --------------------------------------------------------- guardar / abrir
    def a_dict(self):
        buf = io.BytesIO()
        extra = {}
        if self.molde is not None:
            for k, v in self.molde.selecciones.items():
                extra["sel__" + k] = v
            np.savez_compressed(buf, V=self.malla.V, F=self.malla.F, **extra)
        return {
            "formato": "laminado-studio/1",
            "nombre": self.nombre,
            "h": self.h, "extension": self.extension,
            "drapeado_con_espesor": self.drapeado_con_espesor, "sentido_laminado": self.sentido_laminado,
            "rampa": self.rampa,
            "molde": {"origen": self.molde.origen if self.molde else "",
                      "datos": base64.b64encode(buf.getvalue()).decode() if self.molde else ""},
            "materiales": {k: m.a_dict() for k, m in self.materiales.items()},
            "rosetas": {k: asdict(r) for k, r in self.rosetas.items()},
            "capas": [_capa_a_dict(c) for c in self.capas],
            "secuencias": {k: asdict(s) for k, s in self.secuencias.items()},
            "nucleos": [asdict(n) for n in self.nucleos],
        }

    def guardar(self, ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.a_dict(), f, ensure_ascii=False, indent=1, default=_json_def)
        self.ruta = ruta

    @staticmethod
    def abrir(ruta) -> "Proyecto":
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        p = Proyecto(d.get("nombre", os.path.basename(ruta)))
        p.h = d.get("h", 6.0)
        p.extension = d.get("extension", 60.0)
        p.drapeado_con_espesor = d.get("drapeado_con_espesor", False)
        p.sentido_laminado = d.get("sentido_laminado", 1.0)
        p.rampa = d.get("rampa", 20.0)
        if d["molde"]["datos"]:
            z = np.load(io.BytesIO(base64.b64decode(d["molde"]["datos"])))
            sel = {k[5:]: z[k] for k in z.files if k.startswith("sel__")}
            p.molde = Molde(TriMesh(z["V"], z["F"]), sel, origen=d["molde"]["origen"])
        p.materiales = {k: Material.de_dict(v) for k, v in d["materiales"].items()}
        p.rosetas = {k: Roseta(**{kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()})
                     for k, v in d["rosetas"].items()}
        p.capas = [_capa_de_dict(c) for c in d["capas"]]
        p.secuencias = {k: Secuencia(**v) for k, v in d.get("secuencias", {}).items()}
        p.nucleos = [Nucleo(**{**n, "contorno": Contorno(**n["contorno"])}) for n in d.get("nucleos", [])]
        p.ruta = ruta
        return p


def _plano_si_plano(m: TriMesh, margen: float):
    """Si la malla es plana (desviación < 0,05 mm + 1e-4·tamaño) devuelve una malla rectangular
    de ese plano que la cubre con ``margen``; si no, None."""
    c = m.V.mean(0)
    _, sv, vt = np.linalg.svd(m.V - c, full_matrices=False)
    n = vt[2]
    diag = float(np.linalg.norm(np.ptp(m.V, axis=0)))
    if np.abs((m.V - c) @ n).max() > 0.05 + 1e-4 * diag:
        return None
    e1, e2 = vt[0], vt[1]
    if (np.cross(e1, e2) @ m.face_normals.mean(0)) < 0:
        e2 = -e2
    u = (m.V - c) @ e1
    v = (m.V - c) @ e2
    paso = max(float(np.median(m.edge_lengths)), 1e-3)
    us = np.arange(u.min() - margen, u.max() + margen + paso, paso)
    vs = np.arange(v.min() - margen, v.max() + margen + paso, paso)
    U, W = np.meshgrid(us, vs, indexing="ij")
    P = c + U[..., None] * e1 + W[..., None] * e2
    nu, nv = U.shape
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, cc, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    return TriMesh(P.reshape(-1, 3), np.vstack([np.c_[a, b, cc], np.c_[a, cc, d]]))


def _curvatura_p95(m: TriMesh) -> float:
    """Curvatura normal aproximada (1/mm) por arista, percentil 95."""
    E = m.edges
    n = m.vertex_normals
    interior = ~m.boundary_vertices_mask[E].any(1)        # las normales del borde son menos fiables
    largas = m.edge_lengths > 0.3 * np.median(m.edge_lengths)   # aristas diminutas de los cortes
    sel = interior & largas
    if not sel.any():
        return 1e-6
    ang = np.arccos(np.clip((n[E[sel, 0]] * n[E[sel, 1]]).sum(1), -1, 1))
    k = ang / np.maximum(m.edge_lengths[sel], 1e-9)
    return float(max(np.percentile(k, 95), 1e-6))


def _por_cara(ref, caras, nf):
    out = np.zeros((nf, 3))
    out[:] = ref[0]
    out[caras] = ref
    return out


def _dif_ang(real, nominal, mat):
    """Desviación angular mínima teniendo en cuenta la simetría del material."""
    periodo = 90.0 if (mat.tipo == "tejido" and mat.simetrico) else 180.0
    d = (real - nominal + periodo / 2) % periodo - periodo / 2
    return float(d)


def suavizar_campo(m: TriMesh, phi, s):
    """Difusión implícita (M − s²·L) φ' = M φ: el nivel 0 de una distancia a un borde escalonado
    (caras de una selección) pasa a ser una curva suave, sin desplazarla en promedio."""
    import scipy.sparse.linalg as spla
    lim = 50.0 * s
    phi = np.clip(np.where(np.isfinite(phi), phi, np.sign(phi) * lim), -lim, lim)
    A = (m.masa - (s * s) * m.laplaciano_cot).tocsc()
    return spla.spsolve(A, m.masa @ phi)


def _cubierta(r):
    """Fracción de área neta cubierta por el tejido."""
    phi = getattr(r, "phi_neto", None)
    nz = (phi[r.malla.F].mean(1) <= 1e-9) if phi is not None else np.ones(r.malla.nf, bool)
    A = r.malla.face_areas[nz]
    return float(A[r.cara_cubierta()[nz]].sum() / max(A.sum(), 1e-300))


def _ya_probado(e, probados, h):
    for q in probados:
        if (np.linalg.norm(np.subtract(e.punto, q.punto)) < 2 * h
                and abs(float(np.dot(e.normal, q.normal))) > 0.98):
            return True
    return False


def _area_neta(r):
    phi = getattr(r, "phi_neto", None)
    if phi is None:
        return float(r.malla.area)
    return float(r.malla.face_areas[phi[r.malla.F].mean(1) <= 1e-9].sum())


def _area_roja(piezas, limite):
    """Área neta por encima del límite de cizalla o sin cubrir."""
    tot = 0.0
    for r in piezas:
        phi = getattr(r, "phi_neto", None)
        nz = (phi[r.malla.F].mean(1) <= 1e-9) if phi is not None else np.ones(r.malla.nf, bool)
        mal = (np.abs(r.cizalla()) > limite) | ~r.cara_cubierta()
        tot += float(r.malla.face_areas[nz & mal].sum())
    return tot


def _dividir_con_phi(res, empalmes):
    """Divide en piezas manteniendo phi_neto en cada pieza."""
    from ..drapeado.simular import _phi_empalme
    piezas = [res]
    for e in empalmes:
        nuevas = []
        for p in piezas:
            phi = _phi_empalme(p, e)
            if phi.min() >= 0 or phi.max() <= 0:
                nuevas.append(p)
                continue
            for lado, nivel in ((-1, e.solape / 2), (1, -e.solape / 2)):
                extra = np.c_[p.uv, p.cubierto.astype(float), p.phi_neto]
                m2, phi2, ex2 = cortar_por_nivel(p.malla, phi, nivel, extra)
                pc = phi2[m2.F].mean(1)
                mask = pc <= 0 if lado < 0 else pc >= 0
                sub, ids, _ = m2.submalla(mask)
                if sub.nf == 0:
                    continue
                # cada trozo conexo es una pieza de tejido distinta
                nc, et = sub.componentes()
                for c in range(nc):
                    s2, ids2, _ = sub.submalla(et == c)
                    if s2.nf == 0 or s2.area < 1e-6:
                        continue
                    jj = ids[ids2]
                    q = ResultadoDrapeado(s2, ex2[jj, :2], ex2[jj, 2] > 0.5, p.estrategia, p.red,
                                          None, p.darts, dict(p.info))
                    q.phi_neto = ex2[jj, 3]
                    nuevas.append(q)
        piezas = nuevas
    return piezas


def _json_def(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    raise TypeError(type(o))


def _capa_a_dict(c: Capa):
    d = asdict(c)
    d["darts"] = [np.asarray(p, float).tolist() for p in c.darts]
    d["empalmes"] = [asdict(e) if hasattr(e, "__dataclass_fields__") else e for e in c.empalmes]
    return d


def _capa_de_dict(d):
    d = dict(d)
    d["contorno"] = Contorno(**d["contorno"])
    if d.get("semilla") is not None:
        d["semilla"] = tuple(d["semilla"])
    return Capa(**d)
