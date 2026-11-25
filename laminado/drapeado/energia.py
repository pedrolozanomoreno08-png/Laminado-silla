"""Drapeado por mínima energía.

Incógnitas: coordenadas de tejido (u, v) de cada vértice de la malla de la capa. En cada
triángulo (coordenadas locales 2D) los gradientes ∇u, ∇v son lineales en las incógnitas y las
direcciones de hilo son a = ∂X/∂u = rot(∇v)/det, b = ∂X/∂v = −rot(∇u)/det con
det = ∇u × ∇v. Por tanto

    |a| = 1  ⇔  |∇v| = det        |b| = 1  ⇔  |∇u| = det        sen γ = −∇u·∇v / (|∇u||∇v|)

y la energía (hilos inextensibles, mínima cizalla)

    Σ A·[ k_t·(|∇v| − det)² + k_t'·(|∇u| − det)² + k_s·(∇u·∇v / |∇u||∇v|)² ]

no tiene divisiones por det: es robusta aunque haya triángulos degenerados o invertidos al
inicio (la propia energía los desdobla). Se resuelve con Levenberg–Marquardt y factorización
dispersa directa; el jacobiano es analítico.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..geom.malla import TriMesh


def _coeficientes(m: TriMesh):
    """c[f, k] (2-vector): ∇u = Σ_k u_k c[f,k] en coordenadas locales (e1, e2) de cada cara."""
    V, F = m.V, m.F
    x1 = V[F[:, 1]] - V[F[:, 0]]
    x2 = V[F[:, 2]] - V[F[:, 0]]
    e1 = x1 / np.linalg.norm(x1, axis=1, keepdims=True)
    e2 = np.cross(m.face_normals, e1)
    X = np.zeros((m.nf, 2, 2))
    X[:, 0, 0] = (x1 * e1).sum(1)
    X[:, 1, 0] = (x1 * e2).sum(1)
    X[:, 0, 1] = (x2 * e1).sum(1)
    X[:, 1, 1] = (x2 * e2).sum(1)
    Xi = np.linalg.inv(X)            # (u1-u0, u2-u0) = ∇u · X  →  ∇u = (u1-u0, u2-u0) · Xi
    c = np.zeros((m.nf, 3, 2))
    c[:, 1] = Xi[:, 0, :]
    c[:, 2] = Xi[:, 1, :]
    c[:, 0] = -(c[:, 1] + c[:, 2])
    return c, e1, e2


def drapear_energia(m: TriMesh, uv0, semilla_v: int, dir_semilla, k_t=200.0, k_s=1.0,
                    k_t_trama=None, max_iter=40, tol=1e-7, fijos=None, pesos_cara=None):
    """Optimiza las coordenadas de tejido partiendo de ``uv0``.

    dir_semilla: dirección 3D de la urdimbre en la semilla. ``fijos``: vértices con uv
    impuesto. Devuelve (uv, info)."""
    if k_t_trama is None:
        k_t_trama = k_t
    F = m.F
    nf, nv = m.nf, m.nv
    c, e1, e2 = _coeficientes(m)
    w = np.sqrt(m.face_areas) * (1.0 if pesos_cara is None else np.asarray(pesos_cara))
    vc = m.vertice_caras.tocsr()
    fs = vc.indices[vc.indptr[semilla_v]]
    ds = np.asarray(dir_semilla, float)
    dloc = np.array([ds @ e1[fs], ds @ e2[fs]])
    dloc /= np.linalg.norm(dloc)
    W = 30.0 * np.sqrt(m.face_areas.mean())
    fijos = np.array([], int) if fijos is None else np.asarray(fijos, int)
    uv_fijos = np.asarray(uv0)[fijos].copy() if len(fijos) else None
    st, stt, ss = np.sqrt(k_t), np.sqrt(k_t_trama), np.sqrt(k_s)

    def grads(x):
        uv = x.reshape(nv, 2)
        gu = np.einsum("fk,fkd->fd", uv[F, 0], c)
        gv = np.einsum("fk,fkd->fd", uv[F, 1], c)
        return uv, gu, gv

    def residuos(x):
        uv, gu, gv = grads(x)
        det = gu[:, 0] * gv[:, 1] - gu[:, 1] * gv[:, 0]
        nu = np.linalg.norm(gu, axis=1)
        nvv = np.linalg.norm(gv, axis=1)
        cs = (gu * gv).sum(1) / np.maximum(nu * nvv, 1e-24)
        r = [st * w * (nvv - det), stt * w * (nu - det), ss * w * cs,
             W * uv[semilla_v], [W * (gv[fs] @ dloc)]]
        if len(fijos):
            r.append(W * (uv[fijos] - uv_fijos).ravel())
        return np.concatenate([np.ravel(a) for a in r])

    def jac(x):
        uv, gu, gv = grads(x)
        nu = np.maximum(np.linalg.norm(gu, axis=1), 1e-12)
        nvv = np.maximum(np.linalg.norm(gv, axis=1), 1e-12)
        p = (gu * gv).sum(1)
        s = p / (nu * nvv)
        filas, cols, vals = [], [], []
        ar = np.arange(nf)
        for k in range(3):
            ck = c[:, k]
            ddet_u = ck[:, 0] * gv[:, 1] - ck[:, 1] * gv[:, 0]
            dnu_u = (gu * ck).sum(1) / nu
            ds_u = (gv * ck).sum(1) / (nu * nvv) - s * (gu * ck).sum(1) / nu ** 2
            ddet_v = gu[:, 0] * ck[:, 1] - gu[:, 1] * ck[:, 0]
            dnv_v = (gv * ck).sum(1) / nvv
            ds_v = (gu * ck).sum(1) / (nu * nvv) - s * (gv * ck).sum(1) / nvv ** 2
            cu = 2 * F[:, k]
            cv = cu + 1
            for base, fac, du, dv in ((0, st, -ddet_u, dnv_v - ddet_v),
                                      (nf, stt, dnu_u - ddet_u, -ddet_v),
                                      (2 * nf, ss, ds_u, ds_v)):
                filas += [base + ar, base + ar]
                cols += [cu, cv]
                vals += [fac * w * du, fac * w * dv]
        fila = 3 * nf
        filas.append(np.array([fila, fila + 1]))
        cols.append(np.array([2 * semilla_v, 2 * semilla_v + 1]))
        vals.append(np.array([W, W]))
        fila += 2
        for k in range(3):
            filas.append(np.array([fila]))
            cols.append(np.array([2 * F[fs, k] + 1]))
            vals.append(np.array([W * (c[fs, k] @ dloc)]))
        fila += 1
        if len(fijos):
            filas.append(fila + np.arange(2 * len(fijos)))
            cols.append(np.ravel(np.c_[2 * fijos, 2 * fijos + 1]))
            vals.append(np.full(2 * len(fijos), W))
            fila += 2 * len(fijos)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                             shape=(fila, 2 * nv))

    x = np.asarray(uv0, float).ravel().copy()
    r = residuos(x)
    coste0 = coste = 0.5 * float(r @ r)
    lam = 1e-4
    it = 0
    estado = "máximo de iteraciones"
    for it in range(1, max_iter + 1):
        J = jac(x)
        JtJ = (J.T @ J).tocsc()
        g = J.T @ r
        diag = JtJ.diagonal()
        mejora = False
        if not (np.isfinite(JtJ.data).all() and np.isfinite(g).all()):
            estado = "detenido (valores no finitos)"
            break
        for _ in range(12):
            A = (JtJ + sp.diags(lam * np.maximum(diag, 1e-9 * diag.max()))).tocsc()
            try:
                dx = spla.spsolve(A, -g)
            except RuntimeError:
                lam *= 10
                continue
            if not np.isfinite(dx).all():
                lam *= 10
                continue
            xn = x + dx
            rn = residuos(xn)
            cn = 0.5 * float(rn @ rn)
            if np.isfinite(cn) and cn < coste:
                mejora = True
                break
            lam *= 10
        if not mejora:
            estado = "convergido (sin mejora)"
            break
        rel = (coste - cn) / max(coste, 1e-300)
        x, r, coste = xn, rn, cn
        lam = max(lam / 5, 1e-10)
        if rel < tol:
            estado = "convergido"
            break
    uv = x.reshape(nv, 2)
    return uv, {"coste_inicial": coste0, "coste_final": coste, "iteraciones": it, "estado": estado}
