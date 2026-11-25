"""Cortes exactos de mallas por curvas de nivel, rendijas (darts) y extensión de superficie."""
from __future__ import annotations

import numpy as np

from .malla import TriMesh


def cortar_por_nivel(m: TriMesh, phi, nivel: float = 0.0, extra=None):
    """Divide los triángulos que cruzan la curva phi = nivel.

    Devuelve (malla nueva, phi nuevo, extra nuevo) — ``extra`` es un array por vértice
    (p.ej. coordenadas) que se interpola linealmente en los vértices nuevos.
    Tras el corte cada triángulo queda entero a un lado (phi<=nivel o phi>=nivel)."""
    phi = np.asarray(phi, float) - nivel
    h = float(np.mean(m.edge_lengths)) if m.nf else 1.0
    # evitar vértices exactamente sobre el nivel (degeneran triángulos): se ajustan al nivel
    cerca = np.abs(phi) < 1e-7 * h
    phi = np.where(cerca, 0.0, phi)
    V = m.V
    F = m.F
    s = np.sign(phi)
    E = m.edges
    cruza = (s[E[:, 0]] * s[E[:, 1]]) < 0
    idx_e = np.nonzero(cruza)[0]
    a, b = E[idx_e, 0], E[idx_e, 1]
    t = phi[a] / (phi[a] - phi[b])
    nuevos = V[a] + t[:, None] * (V[b] - V[a])
    id_nuevo = -np.ones(len(E), np.int64)
    id_nuevo[idx_e] = m.nv + np.arange(len(idx_e))
    V2 = np.vstack([V, nuevos])
    phi2 = np.r_[phi, np.zeros(len(idx_e))]
    ex2 = None
    if extra is not None:
        extra = np.asarray(extra, float)
        exn = extra[a] + t.reshape((-1,) + (1,) * (extra.ndim - 1)) * (extra[b] - extra[a])
        ex2 = np.concatenate([extra, exn])
    fe = m.face_edges  # aristas en orden (0-1, 1-2, 2-0)
    Fn = []
    ncruz = cruza[fe].sum(1)
    Fn.append(F[ncruz == 0])
    for f in np.nonzero(ncruz > 0)[0]:
        tri = F[f]
        ef = fe[f]
        # recorre el triángulo insertando los puntos de corte
        poli = []
        for k in range(3):
            poli.append(tri[k])
            if cruza[ef[k]]:
                poli.append(id_nuevo[ef[k]])
        # poli tiene 4 (corte por un vértice) o 5 vértices; se triangula por lados
        lados = [phi2[v] for v in poli]
        Fn.extend(_triangular_poligono_cortado(poli, lados, V2))
    F2 = np.vstack([x if isinstance(x, np.ndarray) else np.array(x).reshape(-1, 3) for x in Fn if len(x)])
    m2 = TriMesh(V2, F2)
    return m2, phi2, ex2


def _triangular_poligono_cortado(poli, lados, V=None):
    """Triangula un polígono (lista de vértices) separando la parte negativa de la positiva.

    Los cuadriláteros se parten por su diagonal más corta (mejores triángulos y resultado
    independiente del orden de los vértices: un corte simétrico da una malla simétrica)."""
    n = len(poli)
    ceros = [i for i in range(n) if lados[i] == 0]
    out = []
    if len(ceros) < 2:
        # no debería ocurrir; abanico
        for i in range(1, n - 1):
            out.append([poli[0], poli[i], poli[i + 1]])
        return out
    i0, i1 = ceros[0], ceros[1]
    for parte in ([poli[(i0 + k) % n] for k in range((i1 - i0) % n + 1)],
                  [poli[(i1 + k) % n] for k in range((i0 - i1) % n + 1)]):
        if len(parte) == 4 and V is not None:
            p0, p1, p2, p3 = parte
            if np.linalg.norm(V[p1] - V[p3]) < np.linalg.norm(V[p0] - V[p2]):
                out += [[p0, p1, p3], [p1, p2, p3]]
            else:
                out += [[p0, p1, p2], [p0, p2, p3]]
            continue
        for k in range(1, len(parte) - 1):
            out.append([parte[0], parte[k], parte[k + 1]])
    return out


def recortar(m: TriMesh, phi, nivel: float = 0.0, lado: str = "menor", extra=None):
    """Corta por phi = nivel y se queda con la parte phi <= nivel (o >= si lado='mayor').

    Devuelve (submalla, índice en la malla cortada de cada vértice, phi, extra)."""
    m2, phi2, ex2 = cortar_por_nivel(m, phi, nivel, extra)
    pc = phi2[m2.F].mean(1)
    mask = pc <= 0 if lado == "menor" else pc >= 0
    sub, ids, _ = m2.submalla(mask)
    return sub, ids, phi2[ids], (ex2[ids] if ex2 is not None else None)


def abrir_rendija(m: TriMesh, camino) -> tuple[TriMesh, np.ndarray]:
    """Abre una rendija (dart) a lo largo de un camino de vértices conectados por aristas.

    El primer vértice debe estar en el borde (o el camino se abre en ambos extremos si los dos
    están en el borde). Los vértices interiores del camino se duplican; la punta (último
    vértice interior) no. Devuelve (malla, mapa nuevo→viejo de vértices)."""
    camino = list(map(int, camino))
    F = m.F.copy()
    V = m.V.copy()
    origen = list(range(m.nv))
    bordes = m.boundary_vertices_mask
    aristas_camino = set()
    for a, b in zip(camino[:-1], camino[1:]):
        aristas_camino.add((min(a, b), max(a, b)))
    vc = m.vertice_caras.tocsr()
    ultimo_dup = len(camino) - 1 if bordes[camino[-1]] else len(camino) - 2
    for k in range(0, ultimo_dup + 1):
        v = camino[k]
        caras = vc.indices[vc.indptr[v]:vc.indptr[v + 1]]
        # agrupar caras alrededor de v sin cruzar aristas del camino
        grupo = {}
        g = 0
        for c0 in caras:
            if c0 in grupo:
                continue
            pila = [c0]
            grupo[c0] = g
            while pila:
                c = pila.pop()
                for o in m.F[c]:
                    if o == v:
                        continue
                    e = (min(v, o), max(v, o))
                    if e in aristas_camino:
                        continue
                    for c2 in caras:
                        if c2 not in grupo and o in m.F[c2]:
                            grupo[c2] = g
                            pila.append(c2)
            g += 1
        if g < 2:
            continue
        # el grupo "izquierdo" es el que contiene la arista dirigida (v -> siguiente) o (anterior -> v)
        sig = camino[k + 1] if k + 1 < len(camino) else camino[k - 1]
        izq = None
        for c in caras:
            t = list(m.F[c])
            i = t.index(v)
            if t[(i + 1) % 3] == sig:
                izq = grupo[c]
                break
        if izq is None:
            izq = 0
        nuevo = len(V)
        V = np.vstack([V, m.V[v]])
        origen.append(v)
        for c in caras:
            if grupo[c] == izq:
                F[c][F[c] == v] = nuevo
    return TriMesh(V, F), np.array(origen)


def extender_superficie(m: TriMesh, distancia: float, anillos: int | None = None) -> TriMesh:
    """Extiende la superficie más allá de su borde de forma tangente (superficie extendida).

    Cada vértice de borde se desplaza en la dirección tangente hacia fuera, perpendicular al
    borde, con una ligera continuación de la curvatura normal."""
    if distancia <= 0:
        return m
    h = float(np.mean(m.edge_lengths))
    if anillos is None:
        anillos = max(1, int(np.ceil(distancia / h)))
    paso = distancia / anillos
    V = m.V.copy()
    Fs = [m.F]
    vn = m.vertex_normals
    for bucle in m.boundary_loops():
        P = m.V[bucle]
        n = vn[bucle]
        tang = np.roll(P, -1, 0) - np.roll(P, 1, 0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-300)
        # hacia fuera: bucles orientados como las caras → fuera = tangente × normal
        fuera = np.cross(tang, n)
        fuera /= np.maximum(np.linalg.norm(fuera, axis=1, keepdims=True), 1e-300)
        prev = bucle
        nprev = n
        base = len(V)
        nb = len(bucle)
        filas = [np.asarray(prev)]
        Pk = P.copy()
        dk = fuera.copy()
        for r in range(anillos):
            Pk = Pk + dk * paso
            nuevos = np.arange(len(V), len(V) + nb)
            V = np.vstack([V, Pk])
            filas.append(nuevos)
        for r in range(anillos):
            a = filas[r]
            b = filas[r + 1]
            a1 = np.roll(a, -1)
            b1 = np.roll(b, -1)
            # orientación coherente con la malla original: el borde (a -> a1) es dirigido
            # como las caras, así que las caras nuevas recorren (a1 -> a). Cada cuadrilátero se
            # parte por su diagonal más corta (regla independiente del sentido de recorrido:
            # una pieza simétrica tiene una extensión simétrica)
            d1 = np.linalg.norm(V[a1] - V[b], axis=1)      # diagonal a1-b
            d2 = np.linalg.norm(V[a] - V[b1], axis=1)      # diagonal a-b1
            corta1 = d1 <= d2
            Fs.append(np.where(corta1[:, None], np.stack([a1, a, b], 1), np.stack([a1, a, b1], 1)))
            Fs.append(np.where(corta1[:, None], np.stack([a1, b, b1], 1), np.stack([a, b, b1], 1)))
        del base, nprev
    return TriMesh(V, np.vstack(Fs))


def rellenar_agujeros(m: TriMesh) -> TriMesh:
    """Tapa los agujeros interiores (bucles de borde que no son el contorno exterior de su panel)
    para que la red de tejido pueda pasar por encima; el agujero se recorta después igual que
    el resto del contorno. Cada tapa se hace con dos anillos concéntricos y un abanico."""
    nc, etiqueta = m.componentes()
    bucles = m.boundary_loops()
    if len(bucles) <= 1:
        return m
    vc = m.vertice_caras.tocsr()
    comp_bucle = []
    for b in bucles:
        caras = vc.indices[vc.indptr[b[0]]:vc.indptr[b[0] + 1]]
        comp_bucle.append(etiqueta[caras[0]] if len(caras) else -1)
    perim = [float(np.linalg.norm(np.diff(m.V[np.r_[b, b[:1]]], axis=0), axis=1).sum()) for b in bucles]
    V = [m.V]
    F = [m.F]
    n = m.nv
    for c in set(comp_bucle):
        idx = [k for k in range(len(bucles)) if comp_bucle[k] == c]
        if len(idx) <= 1:
            continue
        exterior = max(idx, key=lambda k: perim[k])
        for k in idx:
            if k == exterior:
                continue
            b = bucles[k]
            P = m.V[b]
            cen = P.mean(0)
            anillo = cen + 0.5 * (P - cen)
            nb = len(b)
            ids_an = np.arange(n, n + nb)
            id_c = n + nb
            V += [anillo, cen[None]]
            n += nb + 1
            b1 = np.roll(b, -1)
            a1 = np.roll(ids_an, -1)
            # el borde (b -> b1) va en el sentido de las caras vecinas: las tapas lo recorren al revés
            F.append(np.stack([b1, b, ids_an], 1))
            F.append(np.stack([b1, ids_an, a1], 1))
            F.append(np.stack([a1, ids_an, np.full(nb, id_c)], 1))
    if len(V) == 1:
        return m
    return TriMesh(np.vstack(V), np.vstack(F))
