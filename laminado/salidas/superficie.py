"""Superficie final del laminado (la otra cara de la pieza) en STL."""
from __future__ import annotations

import numpy as np


def escribir_stl(malla, ruta, nombre="laminado"):
    """STL binario."""
    V, F = malla.V, malla.F
    n = malla.face_normals.astype(np.float32)
    tri = V[F].astype(np.float32)
    reg = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    reg["n"] = n
    reg["v"] = tri
    with open(ruta, "wb") as f:
        f.write(nombre.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(np.uint32(len(F)).tobytes())
        f.write(reg.tobytes())
    return ruta


def exportar_superficie_laminado(proyecto, ruta):
    """Superficie del molde desplazada por el espesor total (con rampas y núcleos)."""
    sup, t = proyecto.superficie_laminado()
    escribir_stl(sup, ruta, f"Laminado {proyecto.nombre} - espesor max {t.max():.2f} mm")
    return ruta
