"""Cobertura completa automática de un molde con parches de tejido producibles.

Una capa «de cobertura» (p.ej. «todo el cuadro a 0°») no puede ser una sola pieza de tejido en
una pieza cerrada y con uniones (cuadro de bicicleta, casco, carenado): se divide en parches,
como en el taller. El algoritmo:

1. Parte de la zona sin cubrir más alejada de lo ya cubierto (al principio, la zona más
   desarrollable: la más alejada de las uniones de doble curvatura).
2. Toma como 0° el eje local de la superficie (dirección de mínima curvatura: el eje del tubo) o
   una roseta dada, y drapea desde ese punto sobre un entorno de radio ``radio_max``.
3. Se queda con las caras bien drapeadas (cubiertas, cizalla ≤ admisible, sin estirar el hilo,
   sin pliegue) conexas con la semilla y que cubren hueco (más ``solape`` mm sobre lo ya cubierto).
4. Crea la capa del parche (selección propia, semilla y roseta propias), la simula otra vez sola y
   marca como cubierto lo que realmente cubre.
5. Repite hasta que no queden huecos. Si en una zona no sale ningún parche válido, reduce el radio;
   en el radio mínimo acepta el mejor parche posible y lo informa.

Resultado: lista de capas creadas (con su selección y roseta) y el mapa de cobertura."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..geom.consulta import Consultor
from ..geom.geodesia import fast_marching
from ..geom.malla import TriMesh


@dataclass
class OpcionesCobertura:
    material: str
    orientacion: float = 0.0
    referencia: str = "eje"          # "eje" (eje local de tubo) o nombre de una roseta
    solape: float = 15.0             # mm de solape entre parches contiguos
    radio_max: float = 320.0         # mm (radio geodésico del entorno de drapeado de cada parche)
    radio_min: float = 30.0
    criterio: str = "aviso"          # cizalla admisible en el parche: "aviso" o "limite"
    estrategia: str = "geodesica"
    selecciones: list = field(default_factory=list)   # zona a cubrir (vacía = todo el molde)
    prefijo: str = "C"
    secuencia: str = ""
    max_parches: int = 120
    hueco_admisible: float = 0.0     # mm² de hueco total tolerado (0: ninguno)


@dataclass
class InformeCobertura:
    capas: list = field(default_factory=list)
    cobertura_pct: float = 0.0
    hueco_mm2: float = 0.0
    no_producibles: list = field(default_factory=list)
    notas: list = field(default_factory=list)


# ------------------------------------------------------------------------- utilidades de malla
def _adyacencia_caras(m: TriMesh):
    ef = m.edge_faces
    ok = (ef[:, 1] >= 0) & m.manifold_edges_mask
    a, b = ef[ok, 0], ef[ok, 1]
    A = sp.coo_matrix((np.ones(len(a) * 2), (np.r_[a, b], np.r_[b, a])), shape=(m.nf, m.nf)).tocsr()
    return A


def _componente_con(A, mascara, cara):
    """Caras de ``mascara`` conexas (por aristas) con ``cara``."""
    if not mascara[cara]:
        return np.zeros_like(mascara)
    sub = A[mascara][:, mascara]
    ids = np.nonzero(mascara)[0]
    from scipy.sparse.csgraph import connected_components
    _, et = connected_components(sub, directed=False)
    k = et[np.searchsorted(ids, cara)]
    out = np.zeros_like(mascara)
    out[ids[et == k]] = True
    return out


def _cerrar_agujeros(A, m: TriMesh, mascara, area_max):
    """Añade a ``mascara`` las islas de caras no marcadas rodeadas por ella y menores que area_max."""
    from scipy.sparse.csgraph import connected_components
    fuera = ~mascara
    if not fuera.any():
        return mascara
    ids = np.nonzero(fuera)[0]
    sub = A[fuera][:, fuera]
    n, et = connected_components(sub, directed=False)
    areas = np.bincount(et, weights=m.face_areas[ids], minlength=n)
    out = mascara.copy()
    mayor = int(np.argmax(areas))
    for k in range(n):
        if k != mayor and areas[k] <= area_max:
            out[ids[et == k]] = True
    return out


def curvatura_gauss(m: TriMesh):
    """Curvatura de Gauss por vértice (defecto angular / área de Voronoi aproximada)."""
    V, F = m.V, m.F
    ang = np.zeros(m.nv)
    for k in range(3):
        a = V[F[:, (k + 1) % 3]] - V[F[:, k]]
        b = V[F[:, (k + 2) % 3]] - V[F[:, k]]
        c = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        np.add.at(ang, F[:, k], np.arccos(np.clip(c, -1, 1)))
    K = (2 * np.pi - ang) / np.maximum(m.vertex_areas, 1e-300)
    K[m.boundary_vertices_mask] = 0.0
    return K


def eje_local(m: TriMesh, P, radio):
    """Dirección de mínima curvatura cerca de P (eje de un tubo) y calidad (0 = tubo perfecto,
    1 = sin dirección privilegiada). Se obtiene de la covarianza de las normales: en un tubo
    todas son perpendiculares al eje."""
    C = m.face_centroids
    cerca = np.linalg.norm(C - P, axis=1) < radio
    if cerca.sum() < 6:
        return None, 1.0
    n = m.face_normals[cerca]
    w = m.face_areas[cerca]
    M = (n * w[:, None]).T @ n / w.sum()
    val, vec = np.linalg.eigh(M)
    calidad = float(val[0] / max(val[1], 1e-12))
    eje = vec[:, 0]
    # en una zona casi plana la normal domina y la dirección mínima no es un eje de tubo
    if val[2] > 0.97:
        return None, 1.0
    # sentido estable: el de la componente global mayor, positiva
    if eje[np.argmax(np.abs(eje))] < 0:
        eje = -eje
    return eje, calidad


# ----------------------------------------------------------------------------------- algoritmo
class Cubridor:
    def __init__(self, proyecto, op: OpcionesCobertura, progreso=None):
        self.p = proyecto
        self.op = op
        self.progreso = progreso
        m = proyecto.malla
        self.m = m
        self.A = _adyacencia_caras(m)
        if op.selecciones:
            obj = np.zeros(m.nf, bool)
            for n in op.selecciones:
                obj[np.asarray(proyecto.molde.selecciones.get(n, []), int)] = True
        else:
            obj = np.ones(m.nf, bool)
        self.objetivo = obj
        self.cubierto = ~obj            # lo que no hay que cubrir cuenta como cubierto
        self.cons_molde = Consultor(m)
        mat = proyecto.materiales[op.material]
        self.mat = mat
        self.adm = mat.cizalla_aviso if op.criterio == "aviso" else mat.cizalla_limite
        self.K = None

    # ------------------------------------------------------------------ semillas
    def _semilla(self, excluidas):
        m = self.m
        hueco = ~self.cubierto
        hueco[excluidas] = False
        if not hueco.any():
            return None
        vh = np.zeros(m.nv, bool)
        vh[m.F[hueco].ravel()] = True
        vc = np.zeros(m.nv, bool)
        vc[m.F[~hueco].ravel()] = True
        borde = (vh & vc) | (vh & m.boundary_vertices_mask)
        sub, ids, _ = m.submalla(hueco)
        if borde.any():
            fuentes = np.nonzero(borde[ids])[0]
            d = fast_marching(sub, fuentes)
        else:
            # superficie cerrada sin nada cubierto: empezar lo más lejos posible de las zonas de
            # doble curvatura (uniones, cúpulas): allí el tejido se adapta peor
            if self.K is None:
                self.K = curvatura_gauss(m)
            Ks = np.abs(self.K[ids])
            umbral = np.percentile(Ks, 90)
            d = fast_marching(sub, np.nonzero(Ks >= umbral)[0])
        d = np.where(np.isfinite(d), d, -1)
        return int(ids[int(np.argmax(d))])

    # ------------------------------------------------------------------ drapeado de prueba
    def _crear_capa(self, nombre, caras, semilla_v, eje_dir, normal):
        from .proyecto import Capa, Contorno, Roseta
        p = self.p
        sel = f"_{nombre}"
        p.molde.selecciones[sel] = np.nonzero(caras)[0]
        ros_n = p.rosetas.get(self.op.referencia)
        if ros_n is None:
            ros_n = f"Eje {nombre}"
            p.rosetas[ros_n] = Roseta(ros_n, tuple(map(float, self.m.V[semilla_v])), tuple(map(float, eje_dir)),
                                      tuple(map(float, normal)))
            ros_name = ros_n
        else:
            ros_name = self.op.referencia
        c = Capa(nombre, self.op.material, self.op.orientacion, roseta=ros_name,
                 contorno=Contorno("seleccion", [sel], exceso=0.0), semilla=tuple(map(float, self.m.V[semilla_v])),
                 estrategia=self.op.estrategia, secuencia=self.op.secuencia)
        c.empalme_auto = False
        c.notas = "Parche de cobertura automática"
        if self.op.secuencia:
            from .proyecto import Secuencia
            sec = p.secuencias.setdefault(self.op.secuencia, Secuencia(self.op.secuencia, [], 0.0))
            sec.capas.append(nombre)
        return c

    def _quitar_capa(self, c):
        p = self.p
        p.capas = [x for x in p.capas if x is not c]
        p.resultados.pop(c.nombre, None)
        for s in c.contorno.selecciones:
            p.molde.selecciones.pop(s, None)
        if c.roseta.startswith("Eje "):
            p.rosetas.pop(c.roseta, None)
        sec = p.secuencias.get(c.secuencia)
        if sec is not None and c.nombre in sec.capas:
            sec.capas.remove(c.nombre)

    def _caras_bien(self, rc, caras_molde):
        """Caras del molde (de ``caras_molde``) bien cubiertas por el tejido de ``rc``."""
        m = self.m
        idx = np.nonzero(caras_molde)[0]
        bien = np.zeros(m.nf, bool)
        if not len(idx):
            return bien
        C = m.face_centroids[idx]
        mejor = np.full(len(idx), np.inf)
        ok = np.zeros(len(idx), bool)
        for r in (rc.piezas if rc.piezas else [rc.drapeado]):
            _, f, _, d = Consultor(r.malla).mas_cercano(C)
            g = np.abs(r.cizalla())
            ea, eb = r.deformaciones()
            phi = getattr(r, "phi_neto", None)
            neto = (phi[r.malla.F].mean(1) <= 1e-9) if phi is not None else np.ones(r.malla.nf, bool)
            buena = (r.cara_cubierta() & (g <= self.adm) & (np.abs(ea) < 0.02) & (np.abs(eb) < 0.02) & neto)
            mejor_aqui = d < mejor
            ok = np.where(mejor_aqui, buena[f] & (d < 0.5), ok)
            mejor = np.minimum(mejor, d)
        bien[idx] = ok
        return bien

    def _probar(self, nombre, semilla_v, radio):
        """Drapea un parche desde la semilla; devuelve (capa, caras útiles) o (None, None)."""
        m = self.m
        op = self.op
        dist = fast_marching(m, [semilla_v], limite=radio * 1.05)
        dcara = dist[m.F].max(1)
        entorno = dcara <= radio
        if entorno.sum() < 4:
            return None, None
        P = m.V[semilla_v]
        nrm = m.vertex_normals[semilla_v]
        ros = self.p.rosetas.get(op.referencia)
        if ros is not None:
            eje = np.asarray(ros.eje_x, float)
        else:
            r_eje = max(20.0, min(radio, 60.0))
            eje, cal = eje_local(m, P, r_eje)
            if eje is None or cal > 0.35:
                eje = np.asarray(next(iter(self.p.rosetas.values())).eje_x, float)
        eje = eje - (eje @ nrm) * nrm
        if np.linalg.norm(eje) < 1e-6:
            eje = np.cross(nrm, [0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.cross(nrm, [1.0, 0, 0])
        eje /= np.linalg.norm(eje)
        c = self._crear_capa(nombre, entorno, semilla_v, eje, nrm)
        self.p.capas.append(c)
        try:
            rc = self.p.simular(c)
        except Exception:  # noqa: BLE001
            self._quitar_capa(c)
            return None, None
        bien = self._caras_bien(rc, entorno)
        f0 = int(self.m.vertice_caras.tocsr()[semilla_v].indices[0])
        caras_semilla = self.m.vertice_caras.tocsr()[semilla_v].indices
        f0 = next((int(f) for f in caras_semilla if bien[f]), f0)
        util = _componente_con(self.A, bien, f0)
        # parche: hueco + solape sobre lo ya cubierto
        hueco = ~self.cubierto
        vh = np.zeros(m.nv, bool)
        vh[m.F[hueco].ravel()] = True
        if op.solape > 0 and (~hueco).any():
            dh = fast_marching(m, vh, limite=op.solape * 1.2)
            cerca_hueco = dh[m.F].max(1) <= op.solape
        else:
            cerca_hueco = hueco
        parche = util & (cerca_hueco | hueco)
        if not parche.any():
            self._quitar_capa(c)
            return None, None
        parche = _componente_con(self.A, parche, next((int(f) for f in np.nonzero(parche & hueco)[0]), int(np.argmax(parche))))
        parche = _cerrar_agujeros(self.A, m, parche, area_max=0.02 * m.face_areas[parche].sum() + 50.0)
        # la capa definitiva: sólo el parche
        self.p.molde.selecciones[c.contorno.selecciones[0]] = np.nonzero(parche)[0]
        try:
            rc = self.p.simular(c)
        except Exception:  # noqa: BLE001
            self._quitar_capa(c)
            return None, None
        bien2 = self._caras_bien(rc, parche)
        return c, bien2

    # ------------------------------------------------------------------ bucle
    def ejecutar(self) -> InformeCobertura:
        m, op = self.m, self.op
        inf = InformeCobertura()
        A_obj = float(m.face_areas[self.objetivo].sum())
        excluidas = np.zeros(m.nf, bool)
        radio = {}
        k = 0
        while k < op.max_parches:
            hueco_area = float(m.face_areas[~self.cubierto].sum())
            if hueco_area <= op.hueco_admisible:
                break
            sv = self._semilla(excluidas)
            if sv is None:
                break
            r = radio.get(sv, op.radio_max)
            nombre = self._nombre(k)
            if self.progreso:
                self.progreso(k, hueco_area / A_obj, nombre)
            c, bien = self._probar(nombre, sv, r)
            nuevo = 0.0 if bien is None else float(m.face_areas[bien & ~self.cubierto].sum())
            minimo = min(0.25 * np.pi * r * r * 0.05, 0.5 * hueco_area)
            if c is None or nuevo < max(minimo, 1e-9):
                if c is not None:
                    self._quitar_capa(c)
                if r > op.radio_min:
                    radio[sv] = max(op.radio_min, r / 2)
                    continue
                # ni con el radio mínimo: se acepta el mejor parche posible (o se excluye la zona)
                c, bien = self._forzar(nombre, sv)
                if c is None:
                    caras = m.vertice_caras.tocsr()[sv].indices
                    excluidas[caras] = True
                    inf.notas.append(f"Zona en {np.round(m.V[sv], 1).tolist()} sin parche posible")
                    continue
                inf.no_producibles.append(nombre)
            self.cubierto |= bien
            # lo que la capa declara (su selección) también deja de buscarse como semilla
            inf.capas.append(c.nombre)
            k += 1
        hueco = ~self.cubierto
        inf.hueco_mm2 = float(m.face_areas[hueco].sum())
        inf.cobertura_pct = 100.0 * (1 - inf.hueco_mm2 / max(A_obj, 1e-300))
        return inf

    def _nombre(self, k):
        existentes = {c.nombre for c in self.p.capas}
        j = k + 1
        while f"{self.op.prefijo}{j:02d}" in existentes:
            j += 1
        return f"{self.op.prefijo}{j:02d}"

    def _forzar(self, nombre, sv):
        """Parche pequeño alrededor de una zona imposible (se informará como no producible)."""
        m = self.m
        dist = fast_marching(m, [sv], limite=self.op.radio_min * 1.05)
        parche = (dist[m.F].max(1) <= self.op.radio_min) & ~self.cubierto
        if not parche.any():
            parche = np.zeros(m.nf, bool)
            parche[m.vertice_caras.tocsr()[sv].indices] = True
        P = m.V[sv]
        nrm = m.vertex_normals[sv]
        eje = np.cross(nrm, [0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.cross(nrm, [1.0, 0, 0])
        c = self._crear_capa(nombre, parche, sv, eje / np.linalg.norm(eje), nrm)
        c.estrategia = "energia"
        self.p.capas.append(c)
        try:
            self.p.simular(c)
        except Exception:  # noqa: BLE001
            self._quitar_capa(c)
            return None, None
        del P
        return c, parche


def cubrir(proyecto, op: OpcionesCobertura, progreso=None) -> InformeCobertura:
    return Cubridor(proyecto, op, progreso).ejecutar()


def mapa_cobertura(proyecto, capas=None):
    """Número de capas que cubren realmente (tejido drapeado, sin pliegue) cada cara del molde."""
    m = proyecto.malla
    C = m.face_centroids
    n = np.zeros(m.nf, int)
    for c in proyecto.capas:
        if not c.activa or c.nombre not in proyecto.resultados:
            continue
        if capas is not None and c.nombre not in capas:
            continue
        rc = proyecto.resultados[c.nombre]
        lo = rc.neto.V.min(0) - 1.0
        hi = rc.neto.V.max(0) + 1.0
        cand = np.nonzero(((C >= lo) & (C <= hi)).all(1))[0]
        if not len(cand):
            continue
        cub = np.zeros(len(cand), bool)
        dmin = np.full(len(cand), np.inf)
        for r in (rc.piezas if rc.piezas else [rc.drapeado]):
            _, f, _, d = Consultor(r.malla).mas_cercano(C[cand])
            phi = getattr(r, "phi_neto", None)
            neto = (phi[r.malla.F].mean(1) <= 1e-9) if phi is not None else np.ones(r.malla.nf, bool)
            ok = r.cara_cubierta() & neto
            mejor = d < dmin
            cub = np.where(mejor, ok[f] & (d < 0.5), cub)
            dmin = np.minimum(dmin, d)
        n[cand[cub]] += 1
    return n
