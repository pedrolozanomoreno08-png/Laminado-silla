"""Figuras (matplotlib) para el ply book y los informes: vista 3D de una capa, patrón plano,
mapa de espesores."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .patrones import contorno_xy  # noqa: E402

CMAP_PROD = LinearSegmentedColormap.from_list("prod", ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])


def color_producibilidad(g, aviso, limite):
    """Colores verde / amarillo / rojo según la cizalla."""
    g = np.abs(np.asarray(g))
    c = np.zeros((len(g), 4))
    c[:] = (0.10, 0.60, 0.31, 1)
    c[g > aviso] = (0.99, 0.80, 0.25, 1)
    c[g > limite] = (0.80, 0.10, 0.12, 1)
    return c


def _ejes_iguales(ax, V):
    c = (V.max(0) + V.min(0)) / 2
    r = (V.max(0) - V.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def figura_capa_3d(proyecto, nombre, vista=(28, -55), tam=(6, 4.5), modo="producibilidad"):
    rc = proyecto.resultados[nombre]
    capa = proyecto.capa(nombre)
    mat = proyecto.materiales[capa.material]
    molde = proyecto.malla
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111, projection="3d")
    paso = max(1, molde.nf // 6000)
    ax.add_collection3d(Poly3DCollection(molde.V[molde.F[::paso]], facecolors=(0.75, 0.75, 0.78, 0.35),
                                         edgecolor="none"))
    r = rc.drapeado
    g = np.abs(r.cizalla())
    if modo == "producibilidad":
        col = color_producibilidad(g, mat.cizalla_aviso, mat.cizalla_limite)
    else:
        col = CMAP_PROD(np.clip(g / max(mat.cizalla_limite * 1.2, 1), 0, 1))
    neto = rc.phi_neto[r.malla.F].mean(1) <= 1e-9
    col[~neto, 3] = 0.35
    ax.add_collection3d(Poly3DCollection(r.malla.V[r.malla.F], facecolors=col, edgecolor="none"))
    for b in rc.neto.boundary_loops():
        P = rc.neto.V[np.r_[b, b[:1]]]
        ax.plot(*P.T, color="k", lw=0.8)
    for d in r.darts:
        ax.plot(*np.asarray(d.puntos).T, color="m", lw=1.5)
    # flechas de urdimbre en una muestra de caras
    ea, eb = r.fibras()
    sel = np.linspace(0, r.malla.nf - 1, 30).astype(int)
    sel = sel[neto[sel]]
    L = 0.06 * np.ptp(molde.V, axis=0).max()
    C = r.malla.face_centroids[sel]
    ax.quiver(*C.T, *(ea[sel] * L).T, color="#1f3b73", lw=0.8, arrow_length_ratio=0.3)
    if capa.semilla is not None or True:
        s = r.malla.V[np.argmin(np.linalg.norm(r.uv, axis=1))]
        ax.scatter(*s, color="k", s=25, marker="o")
    _ejes_iguales(ax, molde.V)
    ax.view_init(*vista)
    fig.tight_layout()
    return fig


def figura_patron(patrones, tam=(6, 4.5), titulo=None, cizalla=None):
    fig, ax = plt.subplots(figsize=tam)
    for p in patrones:
        for k, anillo in enumerate(contorno_xy(p.corte)):
            ax.plot(anillo[:, 0], anillo[:, 1], color="k", lw=1.0)
        for n in p.neto:
            q = np.vstack([n, n[:1]])
            ax.plot(q[:, 0], q[:, 1], color="#1a9850", lw=0.8, ls="--")
        for d in p.darts:
            ax.plot(d[:, 0], d[:, 1], color="m", lw=1.2)
        if not p.corte.is_empty:
            c = p.corte.representative_point()
            L = 0.15 * np.sqrt(p.corte.area)
            ax.annotate("", xy=(c.x + L / 2, c.y), xytext=(c.x - L / 2, c.y),
                        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5))
            ax.text(c.x, c.y - 0.08 * np.sqrt(p.corte.area), f"{p.etiqueta}\n0° tejido", ha="center",
                    va="top", fontsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("urdimbre / largo del rollo (mm)")
    ax.set_ylabel("trama (mm)")
    if titulo:
        ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    return fig


def figura_espesores(proyecto, tam=(6, 4.5), vista=(28, -55)):
    esp, n = proyecto.espesores()
    m = proyecto.malla
    fig = plt.figure(figsize=tam)
    ax = fig.add_subplot(111, projection="3d")
    cm = plt.get_cmap("viridis")
    col = cm(esp / max(esp.max(), 1e-9))
    ax.add_collection3d(Poly3DCollection(m.V[m.F], facecolors=col, edgecolor="none"))
    _ejes_iguales(ax, m.V)
    ax.view_init(*vista)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, esp.max()))
    fig.colorbar(sm, ax=ax, shrink=0.6, label="espesor (mm)")
    fig.tight_layout()
    return fig


def figura_anidado(anidado, tam=(9, 4)):
    fig, ax = plt.subplots(figsize=tam)
    for h in anidado.hojas:
        ax.add_patch(plt.Rectangle((0, h.y0), h.largo, h.ancho, fill=False, ec="0.5"))
        for c in h.colocadas:
            q = c.patron.transformado(c.giro, c.x, c.y + h.y0)
            for anillo in contorno_xy(q.corte):
                ax.fill(anillo[:, 0], anillo[:, 1], alpha=0.35)
                ax.plot(anillo[:, 0], anillo[:, 1], color="k", lw=0.6)
            pt = q.corte.representative_point()
            ax.text(pt.x, pt.y, q.etiqueta, fontsize=6, ha="center")
        ax.text(0, h.y0 - 40, f"{h.material}: {h.largo:.0f} mm, {100 * h.aprovechamiento:.1f} %", fontsize=7)
    ax.set_aspect("equal")
    ax.autoscale()
    fig.tight_layout()
    return fig


def a_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf
