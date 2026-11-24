"""Teoría clásica de laminados (CLT) y criterio de Tsai-Wu."""
from __future__ import annotations

import numpy as np


def Q(m):
    nu21 = m.nu12 * m.E2 / m.E1
    d = 1 - m.nu12 * nu21
    return np.array([[m.E1 / d, m.nu12 * m.E2 / d, 0],
                     [m.nu12 * m.E2 / d, m.E2 / d, 0],
                     [0, 0, m.G12]])


def T(theta_deg):
    c = np.cos(np.radians(theta_deg))
    s = np.sin(np.radians(theta_deg))
    return np.array([[c * c, s * s, 2 * c * s],
                     [s * s, c * c, -2 * c * s],
                     [-c * s, c * s, c * c - s * s]])


def Qbar(m, theta):
    t = T(theta)
    R = np.diag([1, 1, 2])
    return np.linalg.inv(t) @ Q(m) @ R @ t @ np.linalg.inv(R)


def abd(capas):
    """capas: lista de (material, ángulo_grados, espesor). Devuelve (A, B, D, z)."""
    h = sum(c[2] for c in capas)
    z = -h / 2
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    zs = [z]
    for m, ang, t in capas:
        Qb = Qbar(m, ang)
        z1 = z + t
        A += Qb * (z1 - z)
        B += Qb * (z1 ** 2 - z ** 2) / 2
        D += Qb * (z1 ** 3 - z ** 3) / 3
        z = z1
        zs.append(z)
    return A, B, D, np.array(zs)


def constantes_ingenieria(capas):
    A, B, D, z = abd(capas)
    h = z[-1] - z[0]
    a = np.linalg.inv(A)
    return {"Ex": 1 / (a[0, 0] * h), "Ey": 1 / (a[1, 1] * h), "Gxy": 1 / (a[2, 2] * h),
            "nuxy": -a[0, 1] / a[0, 0], "espesor": h}


def tsai_wu(capas, N, M=(0, 0, 0)):
    """Índice de fallo de Tsai-Wu (máximo en cada capa) para esfuerzos N (N/mm) y M (N)."""
    A, B, D, z = abd(capas)
    ABD = np.block([[A, B], [B, D]])
    eps = np.linalg.solve(ABD, np.r_[N, M])
    e0, k = eps[:3], eps[3:]
    out = []
    for i, (m, ang, t) in enumerate(capas):
        peor = 0.0
        for zz in (z[i], z[i + 1]):
            e = e0 + zz * k
            s = Qbar(m, ang) @ e
            s12 = T(ang) @ s
            s1, s2, t12 = s12
            F1 = 1 / m.Xt - 1 / m.Xc
            F2 = 1 / m.Yt - 1 / m.Yc
            F11 = 1 / (m.Xt * m.Xc)
            F22 = 1 / (m.Yt * m.Yc)
            F66 = 1 / m.S ** 2
            F12 = -0.5 * np.sqrt(F11 * F22)
            a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2 * F12 * s1 * s2
            b = F1 * s1 + F2 * s2
            # factor de reserva R: a R² + b R = 1 → índice = 1/R
            if a > 0:
                R = (-b + np.sqrt(b * b + 4 * a)) / (2 * a)
            elif b > 0:
                R = 1 / b
            else:
                R = np.inf
            peor = max(peor, 1 / R if R > 0 else np.inf)
        out.append(peor)
    return np.array(out)
