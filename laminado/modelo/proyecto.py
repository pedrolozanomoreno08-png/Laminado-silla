"""Modelo del laminado y proyecto: rosetas, contornos, capas, secuencias, núcleos, simulación,
core sample, espesores y guardado/apertura (.lam)."""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..ansys.lector import Molde, abrir_molde
from ..drapeado.red import Dart
from ..drapeado.resultado import ResultadoDrapeado
from ..drapeado.simular import (Empalme, OpcionesDrapeado, dividir_en_piezas, empalmes_por_ancho,
                                simular_capa, simular_con_darts_auto)
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
    curva_semilla: list = field(default_factory=list)
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
        self.resultados: dict[str, ResultadoCapa] = {}
        self.ruta = None
        self._ext = None
        self._cons = None

    # ------------------------------------------------------------------ molde
    def cargar_molde(self, ruta_o_molde):
        self.molde = abrir_molde(ruta_o_molde) if isinstance(ruta_o_molde, str) else ruta_o_molde
        self._ext = None
        self._cons = None
        self.resultados.clear()

    @property
    def malla(self) -> TriMesh:
        return self.molde.malla

    def extendida(self):
        if self._ext is None:
            m = self.malla
            diag = float(np.linalg.norm(np.ptp(m.V, axis=0)))
            ext = max(self.extension, 0.12 * diag)
            self._ext = extender_superficie(m, ext)
            self._cons = Consultor(self._ext)
        return self._ext, self._cons

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
        # punto de la región más alejado del borde (centro geodésico)
        d = fast_marching(neto, neto.boundary_vertices_mask)
        return neto.V[int(np.argmax(d))]

    # ---------------------------------------------------------------- regiones
    def _phi_contorno(self, contorno: Contorno, ext: TriMesh):
        """Función con signo (≤0 dentro) en los vértices de la malla extendida."""
        nv_molde = self.malla.nv
        if contorno.tipo == "molde":
            # distancia al borde del molde (positiva en la extensión)
            mask = np.zeros(ext.nf, bool)
            mask[:self.malla.nf] = True
            return distancia_a_borde_region(ext, mask)
        if contorno.tipo == "seleccion":
            caras = np.zeros(ext.nf, bool)
            for nombre in contorno.selecciones:
                sel = self.molde.selecciones.get(nombre)
                if sel is not None:
                    caras[sel] = True
            if not caras.any():
                raise ValueError("La selección del contorno está vacía")
            return distancia_a_borde_region(ext, caras)
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
            frontera = poli.exterior
            d = np.array([frontera.distance(Point(q)) for q in Q])
            dentro = np.array([poli.contains(Point(q)) for q in Q])
            del nv_molde
            return np.where(dentro, -d, d)
        raise ValueError(f"Tipo de contorno desconocido: {contorno.tipo}")

    def region(self, capa: Capa | Nucleo):
        """(malla de corte = neto + exceso, phi neto en sus vértices, malla neta)."""
        ext, _ = self.extendida()
        c = capa.contorno
        phi = self._phi_contorno(c, ext)
        desf = self.desfase_total(capa) if isinstance(capa, Capa) else c.desfase
        # corte por el contorno neto (desfase): phi' = phi - desfase
        m1, phi1, _ = cortar_por_nivel(ext, phi, desf)
        neto_caras = phi1[m1.F].mean(1) <= 0
        if c.exceso > 0:
            d = distancia_a_borde_region(m1, neto_caras, limite=3 * c.exceso)
            d = np.where(np.isinf(d), np.sign(d) * 1e9, d)
            m2, d2, ex = cortar_por_nivel(m1, d, c.exceso, extra=phi1)
            corte = d2[m2.F].mean(1) <= 0
            sub, ids, _ = m2.submalla(corte)
            phi_sub = ex[ids]
        else:
            sub, ids, _ = m1.submalla(neto_caras)
            phi_sub = phi1[ids]
        # quedarse con la componente conexa mayor
        nc, et = sub.componentes()
        if nc > 1:
            mayor = np.argmax(np.bincount(et, weights=sub.face_areas))
            sub2, ids2, _ = sub.submalla(et == mayor)
            sub, phi_sub = sub2, phi_sub[ids2]
        neto, _, _ = sub.submalla(phi_sub[sub.F].mean(1) <= 1e-9)
        return sub, phi_sub, neto

    # -------------------------------------------------------------- simulación
    def opciones(self, capa: Capa) -> OpcionesDrapeado:
        mat = self.materiales[capa.material]
        return OpcionesDrapeado(estrategia=capa.estrategia, h=capa.h or self.h,
                                angulo_bloqueo=min(80.0, mat.cizalla_limite * 1.8),
                                curva_semilla=np.asarray(capa.curva_semilla, float) if capa.curva_semilla else None,
                                unidireccional=mat.unidireccional)

    def simular(self, capa: Capa | str, progreso=None) -> ResultadoCapa:
        if isinstance(capa, str):
            capa = self.capa(capa)
        if self.molde is None:
            raise RuntimeError("No hay molde cargado")
        mat = self.materiales[capa.material]
        _, cons = self.extendida()
        malla_c, phi_c, neto = self.region(capa)
        semilla = self.semilla_de(capa, neto)
        n0 = cons.proyectar(semilla[None])[1]
        ros = self.rosetas.get(capa.roseta) or next(iter(self.rosetas.values()))
        direccion = ros.direccion(semilla[None], n0, capa.orientacion)[0]
        op = self.opciones(capa)
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
        piezas = _dividir_con_phi(res, empalmes) if empalmes else [res]
        rc = ResultadoCapa(capa.nombre, res, piezas, neto, phi_r, empalmes)
        rc.resumen = self._resumen(capa, rc)
        self.resultados[capa.nombre] = rc
        return rc

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
        neto_area = rc.neto.area
        area_corte = sum(p.area_plana() for p in rc.piezas)
        s.update({
            "material": mat.nombre, "orientacion": capa.orientacion,
            "area_neta_mm2": neto_area, "area_corte_mm2": area_corte,
            "peso_g": area_corte * 1e-6 * mat.peso_area,
            "piezas": len(rc.piezas), "empalmes": len(rc.empalmes),
            "estado": ("ROJO" if s.get("area_limite_%", 0) > 0.5 or s["cubierto_%"] < 99.5 else
                       "AMARILLO" if s.get("area_aviso_%", 0) > 0.5 else "VERDE"),
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
            cons = Consultor(rc.neto)
            Q, f, b, d = cons.mas_cercano(P)
            if d[0] > tol:
                continue
            # ángulo real en la cara del drapeado más cercana
            r = rc.drapeado
            cr = Consultor(r.malla)
            _, fr, _, _ = cr.mas_cercano(Q)
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
        """Espesor curado (mm) y número de capas en cada cara del molde."""
        m = self.malla
        X = m.face_centroids
        esp = np.zeros(m.nf)
        n = np.zeros(m.nf, int)
        for c in self.capas:
            rc = self.resultados.get(c.nombre)
            if rc is None or not c.activa:
                continue
            d = Consultor(rc.neto).mas_cercano(X)[3]
            dentro = d < 0.5
            esp[dentro] += self.materiales[c.material].espesor
            n[dentro] += 1
        for nu in self.nucleos:
            try:
                _, _, neto = self.region(nu)
            except Exception:
                continue
            d = Consultor(neto).mas_cercano(X)[3]
            esp[d < 0.5] += nu.espesor
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
                q = ResultadoDrapeado(sub, ex2[ids, :2], ex2[ids, 2] > 0.5, p.estrategia, p.red,
                                      None, p.darts, dict(p.info))
                q.phi_neto = ex2[ids, 3]
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
