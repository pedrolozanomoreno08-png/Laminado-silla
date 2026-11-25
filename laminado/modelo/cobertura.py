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
    regiones: list = field(default_factory=list)      # selecciones que se intentan cubrir con un
                                                      # parche cada una (p.ej. cada tubo) antes de
                                                      # rellenar lo que quede con parches automáticos
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
        self.errores = []
        self.creadas = []

    # ------------------------------------------------------------------ semillas
    def _curvatura(self):
        """|K| suavizada (media en el entorno) para no confundir ruido de malla con curvatura."""
        if self.K is None:
            m = self.m
            K = np.abs(curvatura_gauss(m))
            A = (m.adyacencia > 0).astype(float) + sp.eye(m.nv)
            deg = np.asarray(A.sum(1)).ravel()
            for _ in range(6):
                K = (A @ K) / deg
            self.K = K
        return self.K

    def _semilla(self, excluidas):
        """Centro de la zona sin cubrir: el punto más alejado de su borde y de las zonas de doble
        curvatura (uniones, cúpulas), donde el tejido se adapta peor; así cada parche empieza en
        la parte más desarrollable que queda (el centro de un tubo, no la unión)."""
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
        K = self._curvatura()[ids]
        # curvatura de Gauss que con un parche de 50 mm de radio daría ~la cizalla de aviso
        k_umbral = 2 * np.tan(np.radians(self.adm)) / 50.0 ** 2
        curvo = K > k_umbral
        arista = float(np.median(m.edge_lengths))
        d = None
        fuentes = np.nonzero(borde[ids] | curvo)[0]
        if len(fuentes) and len(fuentes) < sub.nv:
            d = fast_marching(sub, fuentes)
            if not np.isfinite(d).any() or np.nanmax(np.where(np.isfinite(d), d, 0)) < 3 * arista:
                d = None
        if d is None:
            fb = np.nonzero(borde[ids])[0]
            if len(fb):
                d = fast_marching(sub, fb)
            else:
                d = -K           # superficie cerrada y toda curva: el punto menos curvo
        d = np.where(np.isfinite(d), d, -1)
        return int(ids[int(np.argmax(d))])

    def _area_componente_hueco(self, cara):
        hueco = ~self.cubierto
        comp = _componente_con(self.A, hueco, cara)
        return float(self.m.face_areas[comp].sum())

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
            ok = np.where(mejor_aqui, buena[f] & (d < 0.5 + getattr(rc, "desplazamiento", 0.0)), ok)
            mejor = np.minimum(mejor, d)
        bien[idx] = ok
        return bien

    def _distancia_hueco(self, limite):
        """Distancia geodésica (por cara, la máxima de sus vértices) a la zona sin cubrir."""
        m = self.m
        hueco = ~self.cubierto
        vh = np.zeros(m.nv, bool)
        vh[m.F[hueco].ravel()] = True
        if not (~hueco).any():
            return np.zeros(m.nf)
        d = fast_marching(m, vh, limite=limite)
        return d[m.F].max(1)

    def _abrir(self, mascara, r):
        """Apertura morfológica (erosión + dilatación de radio r): quita salientes y tiras más
        estrechas que 2r, que darían recortes imposibles de manejar en el taller."""
        m = self.m
        if r <= 0 or not mascara.any():
            return mascara
        vin = np.zeros(m.nv, bool)
        vin[m.F[mascara].ravel()] = True
        vout = np.zeros(m.nv, bool)
        vout[m.F[~mascara].ravel()] = True
        borde = (vin & vout) | (vin & m.boundary_vertices_mask)
        if not borde.any():
            return mascara
        d = fast_marching(m, borde, limite=r * 1.5)
        nucleo = mascara & (d[m.F].min(1) >= r)
        if not nucleo.any():
            return mascara
        vn = np.zeros(m.nv, bool)
        vn[m.F[nucleo].ravel()] = True
        d2 = fast_marching(m, vn, limite=r * 1.5)
        return mascara & (d2[m.F].max(1) <= r * 1.05)

    def _dilatar(self, mascara, r):
        m = self.m
        v = np.zeros(m.nv, bool)
        v[m.F[mascara].ravel()] = True
        d = fast_marching(m, v, limite=r * 1.2)
        return d[m.F].max(1) <= r

    def _semilla_en(self, zona):
        """Punto de la zona más alejado de su borde (centro del tubo o del panel)."""
        m = self.m
        sub, ids, _ = m.submalla(zona)
        b = sub.boundary_vertices_mask
        if not b.any():
            return int(ids[0])
        d = fast_marching(sub, b)
        d = np.where(np.isfinite(d), d, -1)
        return int(ids[int(np.argmax(d))])

    def _probar(self, nombre, semilla_v, radio, zona=None):
        """Drapea un parche desde la semilla; devuelve (capa, caras bien cubiertas) o (None, None).

        Sin ``zona``: entorno geodésico de radio ``radio``. Con ``zona`` (máscara de caras): el
        parche es esa zona más el solape."""
        m = self.m
        op = self.op
        arista = float(np.median(m.edge_lengths))
        margen = max(4 * arista, 6.0)
        dh = self._distancia_hueco(op.solape + margen * 1.5)
        if zona is None:
            dist = fast_marching(m, [semilla_v], limite=radio * 1.05)
            entorno = (dist[m.F].max(1) <= radio) & (dh <= op.solape + margen)
        else:
            zona_s = self._dilatar(zona, op.solape)
            entorno = self._dilatar(zona, op.solape + margen) & (dh <= op.solape + margen)
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
        c.relleno = False            # drapeado de prueba: sólo interesa lo que la red alcanza
        self.p.capas.append(c)
        try:
            rc = self.p.simular(c)
        except Exception as e:  # noqa: BLE001
            import os
            import traceback
            if os.environ.get("LAMINADO_DEBUG"):
                traceback.print_exc()
            tb = traceback.extract_tb(e.__traceback__)[-1]
            self.errores.append(f"{nombre}: {type(e).__name__}: {e} ({tb.filename.split('/')[-1]}:{tb.lineno})")
            self._quitar_capa(c)
            return None, None
        bien = self._caras_bien(rc, entorno)
        caras_semilla = m.vertice_caras.tocsr()[semilla_v].indices
        f0 = next((int(f) for f in caras_semilla if bien[f]), int(caras_semilla[0]))
        util = _componente_con(self.A, bien, f0)
        # parche: hueco + solape sobre lo ya cubierto, sin salientes ni tiras finas
        parche = util & (dh <= op.solape)
        if zona is not None:
            parche &= zona_s
        parche = self._abrir(parche, max(0.4 * op.solape, 2 * arista))
        hueco = ~self.cubierto
        if not (parche & hueco).any():
            self._quitar_capa(c)
            return None, None
        parche = _componente_con(self.A, parche, int(np.nonzero(parche & hueco)[0][0]))
        parche = _cerrar_agujeros(self.A, m, parche, area_max=0.02 * m.face_areas[parche].sum() + 50.0)
        # la capa definitiva: sólo el parche, con el borde suavizado (sin escalones de malla)
        self.p.molde.selecciones[c.contorno.selecciones[0]] = np.nonzero(parche)[0]
        c.contorno.suavizado = max(1.0 * arista, 2.0)
        c.relleno = True
        try:
            rc = self.p.simular(c)
        except Exception as e:  # noqa: BLE001
            import os
            import traceback
            if os.environ.get("LAMINADO_DEBUG"):
                traceback.print_exc()
            tb = traceback.extract_tb(e.__traceback__)[-1]
            self.errores.append(f"{nombre}: {type(e).__name__}: {e} ({tb.filename.split('/')[-1]}:{tb.lineno})")
            self._quitar_capa(c)
            return None, None
        bien2 = self._caras_bien(rc, parche | (dh <= op.solape))
        return c, bien2

    # ------------------------------------------------------------------ bucle
    def ejecutar(self) -> InformeCobertura:
        m, op = self.m, self.op
        inf = InformeCobertura()
        A_obj = float(m.face_areas[self.objetivo].sum())
        excluidas = np.zeros(m.nf, bool)
        radio = {}
        k = 0
        arista = float(np.median(m.edge_lengths))
        # 1) un parche por región (tubo, panel…) donde sea producible
        for nreg in op.regiones:
            caras = np.asarray(self.p.molde.selecciones.get(nreg, []), int)
            zona = np.zeros(m.nf, bool)
            zona[caras] = True
            zona &= ~self.cubierto
            if m.face_areas[zona].sum() < 25 * arista ** 2:
                continue
            # la mayor componente de la región dentro de la zona a cubrir
            from scipy.sparse.csgraph import connected_components
            ids = np.nonzero(zona)[0]
            ncomp, et = connected_components(self.A[zona][:, zona], directed=False)
            for kc in np.argsort(-np.bincount(et, weights=m.face_areas[ids], minlength=ncomp)):
                zc = np.zeros(m.nf, bool)
                zc[ids[et == kc]] = True
                a_zc = float(m.face_areas[zc].sum())
                if a_zc < 25 * arista ** 2:
                    break
                sv = self._semilla_en(zc)
                nombre = self._nombre(k)
                if self.progreso:
                    self.progreso(k, float(m.face_areas[~self.cubierto].sum()) / A_obj, f"{nombre} ({nreg})")
                c, bien = self._probar(nombre, sv, np.inf, zona=zc)
                nuevo = 0.0 if bien is None else float(m.face_areas[bien & zc & ~self.cubierto].sum())
                if c is None or nuevo < 0.25 * a_zc:
                    if c is not None:
                        self._quitar_capa(c)
                    continue
                c.notas = f"Parche de cobertura automática (región {nreg})"
                self.cubierto |= bien
                inf.capas.append(c.nombre)
                self.creadas.append(c)
                k += 1
        if op.regiones:
            self._absorber_migas()
        # 2) lo que quede, con parches automáticos
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
            caras_sv = m.vertice_caras.tocsr()[sv].indices
            f_h = next((int(f) for f in caras_sv if not self.cubierto[f]), int(caras_sv[0]))
            minimo = min(0.25 * np.pi * r * r * 0.05, 0.5 * self._area_componente_hueco(f_h))
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
            inf.capas.append(c.nombre)
            self.creadas.append(c)
            k += 1
            self._absorber_migas()
        hueco = ~self.cubierto
        inf.hueco_mm2 = float(m.face_areas[hueco].sum())
        inf.cobertura_pct = 100.0 * (1 - inf.hueco_mm2 / max(A_obj, 1e-300))
        inf.notas += [f"Error de cálculo en {e}" for e in self.errores[:5]]
        return inf

    def _absorber_migas(self):
        """Las islas pequeñas sin cubrir que quedan entre parches se añaden al parche vecino
        (ampliándolo un poco) en vez de crear parches diminutos."""
        from scipy.sparse.csgraph import connected_components
        m, op = self.m, self.op
        arista = float(np.median(m.edge_lengths))
        a_miga = max(2.0 * op.solape ** 2, 25 * arista ** 2)
        hueco = ~self.cubierto
        if not hueco.any() or not self.creadas:
            return
        ids = np.nonzero(hueco)[0]
        n, et = connected_components(self.A[hueco][:, hueco], directed=False)
        areas = np.bincount(et, weights=m.face_areas[ids], minlength=n)
        for k in np.argsort(areas):
            if areas[k] > a_miga:
                break
            comp = np.zeros(m.nf, bool)
            comp[ids[et == k]] = True
            if not (comp & ~self.cubierto).any():
                continue
            vz = np.zeros(m.nv, bool)
            vz[m.F[comp].ravel()] = True
            d = fast_marching(m, vz, limite=op.solape * 1.2)
            dc = d[m.F].max(1)
            cerca = dc <= op.solape
            mejor, mejor_a = None, 0.0
            for c in self.creadas:
                sel = np.zeros(m.nf, bool)
                sel[self.p.molde.selecciones[c.contorno.selecciones[0]]] = True
                a = float(m.face_areas[sel & cerca].sum())
                if a > mejor_a:
                    mejor, mejor_a = c, a
            if mejor is None:
                continue
            nsel = mejor.contorno.selecciones[0]
            vieja = self.p.molde.selecciones[nsel]
            viejo_rc = self.p.resultados.get(mejor.nombre)
            nueva = np.zeros(m.nf, bool)
            nueva[vieja] = True
            nueva |= comp | (dc <= 0.5 * op.solape)
            nueva = _cerrar_agujeros(self.A, m, nueva, area_max=a_miga)
            self.p.molde.selecciones[nsel] = np.nonzero(nueva)[0]
            ok = False
            try:
                rc = self.p.simular(mejor)
                bien = self._caras_bien(rc, nueva)
                ok = (rc.resumen.get("estado") != "ROJO"
                      and m.face_areas[bien & comp].sum() >= 0.98 * m.face_areas[comp].sum())
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                self.cubierto |= bien
            else:
                self.p.molde.selecciones[nsel] = vieja
                if viejo_rc is not None:
                    self.p.resultados[mejor.nombre] = viejo_rc

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
        lo = rc.neto.V.min(0) - 1.0 - rc.desplazamiento
        hi = rc.neto.V.max(0) + 1.0 + rc.desplazamiento
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
            cub = np.where(mejor, ok[f] & (d < 0.5 + getattr(rc, "desplazamiento", 0.0)), cub)
            dmin = np.minimum(dmin, d)
        n[cand[cub]] += 1
    return n
