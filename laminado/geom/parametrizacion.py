"""Aplanado conformal (LSCM, Lévy et al. 2002) y alineación con coordenadas de referencia.

Se usa como punto de partida robusto del drapeado por mínima energía en zonas que la red
cinemática no alcanza."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .malla import TriMesh


def lscm(m: TriMesh, fijos=None):
    """Coordenadas planas (nv,2) que minimizan la distorsión angular.

    ``fijos``: dos índices de vértice a fijar (por defecto los dos más alejados del borde)."""
    V, F = m.V, m.F
    x1 = V[F[:, 1]] - V[F[:, 0]]
    x2 = V[F[:, 2]] - V[F[:, 0]]
    e1 = x1 / np.linalg.norm(x1, axis=1, keepdims=True)
    e2 = np.cross(m.face_normals, e1)
    # coordenadas locales de los tres vértices
    P = np.zeros((m.nf, 3, 2))
    P[:, 1, 0] = (x1 * e1).sum(1)
    P[:, 2, 0] = (x2 * e1).sum(1)
    P[:, 2, 1] = (x2 * e2).sum(1)
    raiz = np.sqrt(np.maximum(2 * m.face_areas, 1e-300))
    filas, cols, vr, vi = [], [], [], []
    for j in range(3):
        a = P[:, (j + 2) % 3] - P[:, (j + 1) % 3]          # W_j = x_{j+2} - x_{j+1}
        filas.append(np.arange(m.nf))
        cols.append(F[:, j])
        vr.append(a[:, 0] / raiz)
        vi.append(a[:, 1] / raiz)
    I = np.concatenate(filas)
    J = np.concatenate(cols)
    Mr = sp.csr_matrix((np.concatenate(vr), (I, J)), shape=(m.nf, m.nv))
    Mi = sp.csr_matrix((np.concatenate(vi), (I, J)), shape=(m.nf, m.nv))
    A = sp.bmat([[Mr, -Mi], [Mi, Mr]]).tocsc()
    if fijos is None:
        # dos vértices alejados entre sí
        a = int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))
        b = int(np.argmax(np.linalg.norm(V - V[a], axis=1)))
        fijos = (a, b)
    a, b = fijos
    L = float(np.linalg.norm(V[b] - V[a]))
    pos = {a: (0.0, 0.0), b: (L, 0.0)}
    libres = np.setdiff1d(np.arange(m.nv), [a, b])
    cols_l = np.r_[libres, libres + m.nv]
    cols_f = np.array([a, b, a + m.nv, b + m.nv])
    xf = np.array([pos[a][0], pos[b][0], pos[a][1], pos[b][1]])
    Al = A[:, cols_l]
    rhs = -(A[:, cols_f] @ xf)
    sol = spla.spsolve((Al.T @ Al).tocsc(), Al.T @ rhs)
    uv = np.zeros((m.nv, 2))
    uv[libres, 0] = sol[:len(libres)]
    uv[libres, 1] = sol[len(libres):]
    uv[a] = pos[a]
    uv[b] = pos[b]
    # escala para conservar el área total (LSCM no fija la escala)
    Fu = uv[F]
    area_p = 0.5 * np.abs((Fu[:, 1, 0] - Fu[:, 0, 0]) * (Fu[:, 2, 1] - Fu[:, 0, 1])
                          - (Fu[:, 2, 0] - Fu[:, 0, 0]) * (Fu[:, 1, 1] - Fu[:, 0, 1])).sum()
    uv *= np.sqrt(m.area / max(area_p, 1e-300))
    return uv


def alinear(uv, ref, mascara):
    """Giro + traslación (y simetría si hace falta) que lleva ``uv`` sobre ``ref`` en la máscara."""
    A = uv[mascara]
    B = ref[mascara]
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = (U @ Vt).T
    out = (uv - ca) @ R.T + cb
    return out


def prolongar(m: TriMesh, uv, conocido):
    """Extiende las coordenadas planas a los vértices desconocidos avanzando desde la zona
    conocida: cada vértice nuevo se coloca con el jacobiano local (∇u, ∇v en 3D) de su vecino
    conocido, como si el tejido continuara sin deformarse. Buen punto de partida para la energía."""
    from collections import deque
    V, F = m.V, m.F
    uv = np.asarray(uv, float).copy()
    conocido = np.asarray(conocido, bool).copy()
    # jacobiano 2x3 por cara a partir de las caras totalmente conocidas
    x1 = V[F[:, 1]] - V[F[:, 0]]
    x2 = V[F[:, 2]] - V[F[:, 0]]
    n = m.face_normals
    J = np.zeros((m.nf, 2, 3))

    def jac_caras(caras):
        u1 = uv[F[caras, 1]] - uv[F[caras, 0]]
        u2 = uv[F[caras, 2]] - uv[F[caras, 0]]
        # base dual: g1·x1 = 1, g1·x2 = 0 ; g2·x1 = 0, g2·x2 = 1 (en el plano de la cara)
        a, b, nn = x1[caras], x2[caras], n[caras]
        den = (np.cross(a, b) * nn).sum(1)[:, None]
        g1 = np.cross(b, nn) / den
        g2 = np.cross(nn, a) / den
        return (u1[:, :, None] * g1[:, None, :] + u2[:, :, None] * g2[:, None, :])

    caras_ok = conocido[F].all(1)
    J[caras_ok] = jac_caras(np.nonzero(caras_ok)[0])
    Jv = np.zeros((m.nv, 2, 3))
    cnt = np.zeros(m.nv)
    for k in range(3):
        np.add.at(Jv, F[caras_ok, k], J[caras_ok])
        np.add.at(cnt, F[caras_ok, k], 1)
    tiene = cnt > 0
    Jv[tiene] /= cnt[tiene, None, None]
    vc = m.adyacencia.tocsr()
    q = deque(np.nonzero(conocido & tiene)[0].tolist())
    while q:
        k = q.popleft()
        for j in vc.indices[vc.indptr[k]:vc.indptr[k + 1]]:
            if conocido[j]:
                continue
            # promedio de las predicciones de los vecinos conocidos
            vec = vc.indices[vc.indptr[j]:vc.indptr[j + 1]]
            vec = vec[conocido[vec] & tiene[vec]]
            pred = np.array([uv[i] + Jv[i] @ (V[j] - V[i]) for i in vec])
            uv[j] = pred.mean(0)
            Jv[j] = Jv[vec].mean(0)
            tiene[j] = True
            conocido[j] = True
            q.append(j)
    return uv
