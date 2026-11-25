"""Vista 3D (pyvista dentro de Qt) con picado de puntos sobre el molde."""
from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore

from ..salidas.figuras import color_producibilidad


def a_pv(malla):
    F = np.c_[np.full(malla.nf, 3), malla.F].ravel()
    return pv.PolyData(malla.V, F)


class Vista3D(QtCore.QObject):
    """Envoltorio del QtInteractor: capas de dibujo con nombre y picado con botón derecho."""

    punto_picado = QtCore.Signal(object)       # np.ndarray (3,)

    def __init__(self, padre):
        super().__init__(padre)
        self.plotter = QtInteractor(padre)
        self.plotter.set_background("#f4f6fa", top="#d9e1ee")
        self.plotter.add_axes(interactive=False, line_width=2)
        self.widget = self.plotter.interactor
        self._picker = None
        self.modo_picado = None
        self.plotter.iren.add_observer("RightButtonPressEvent", self._clic_derecho)
        self._actores = {}

    # --------------------------------------------------------------- dibujo
    def quitar(self, nombre):
        a = self._actores.pop(nombre, None)
        if a is not None:
            self.plotter.remove_actor(a, render=False)

    def quitar_prefijo(self, prefijo):
        for k in [k for k in self._actores if k.startswith(prefijo)]:
            self.quitar(k)

    def poner(self, nombre, actor):
        self.quitar(nombre)
        self._actores[nombre] = actor

    def molde(self, malla, opacidad=1.0, color="#b9bec8"):
        self.poner("molde", self.plotter.add_mesh(a_pv(malla), color=color, opacity=opacidad,
                                                  smooth_shading=True, specular=0.2, name="molde",
                                                  pickable=True, render=False))

    def capa(self, res, modo, aviso, limite, phi_neto=None, ref=None, nombre="capa"):
        """Dibuja el resultado de drapeado coloreado según ``modo``."""
        m = res.malla
        pd = a_pv(m)
        # pequeño desplazamiento según la normal para que la capa se vea sobre el molde
        pd.points = m.V + m.vertex_normals * 0.3
        g = np.abs(res.cizalla())
        kw = dict(name=nombre, pickable=False, render=False, smooth_shading=False, show_edges=False)
        if modo == "producibilidad":
            col = (color_producibilidad(g, aviso, limite)[:, :3] * 255).astype(np.uint8)
            if phi_neto is not None:
                fuera = phi_neto[m.F].mean(1) > 1e-9
                col[fuera] = (col[fuera] * 0.55 + 255 * 0.45).astype(np.uint8)
            pd.cell_data["color"] = col
            actor = self.plotter.add_mesh(pd, scalars="color", rgb=True, **kw)
        elif modo == "cizalla":
            pd.cell_data["Cizalla (°)"] = g
            actor = self.plotter.add_mesh(pd, scalars="Cizalla (°)", cmap="RdYlGn_r",
                                          clim=(0, max(limite * 1.2, 1)), **kw,
                                          scalar_bar_args={"title": "Cizalla (°)", "vertical": True})
        elif modo == "desviacion" and ref is not None:
            d = res.angulo_respecto(ref)
            pd.cell_data["Ángulo (°)"] = d
            actor = self.plotter.add_mesh(pd, scalars="Ángulo (°)", cmap="twilight_shifted", **kw,
                                          scalar_bar_args={"title": "Ángulo de urdimbre (°)", "vertical": True})
        else:
            actor = self.plotter.add_mesh(pd, color="#4a78c2", **kw)
        self.poner(nombre, actor)

    def lineas(self, nombre, polilineas, color="k", ancho=3, cerrada=False):
        self.quitar(nombre)
        if not polilineas:
            return
        pts, lineas = [], []
        n = 0
        for P in polilineas:
            P = np.asarray(P, float)
            if cerrada:
                P = np.vstack([P, P[:1]])
            pts.append(P)
            lineas.append(np.r_[len(P), np.arange(n, n + len(P))])
            n += len(P)
        pd = pv.PolyData(np.vstack(pts), lines=np.concatenate(lineas))
        self.poner(nombre, self.plotter.add_mesh(pd, color=color, line_width=ancho, name=nombre,
                                                 pickable=False, render=False, render_lines_as_tubes=False))

    def flechas(self, nombre, origenes, direcciones, escala, color="#1f3b73"):
        self.quitar(nombre)
        if len(origenes) == 0:
            return
        pd = pv.PolyData(np.asarray(origenes, float))
        pd["v"] = np.asarray(direcciones, float)
        glifos = pd.glyph(orient="v", scale=False, factor=escala, geom=pv.Arrow(shaft_radius=0.03, tip_radius=0.08))
        self.poner(nombre, self.plotter.add_mesh(glifos, color=color, name=nombre, pickable=False, render=False))

    def puntos(self, nombre, P, color="k", tam=14):
        self.quitar(nombre)
        if P is None or len(P) == 0:
            return
        self.poner(nombre, self.plotter.add_points(np.atleast_2d(np.asarray(P, float)), color=color,
                                                   point_size=tam, render_points_as_spheres=True,
                                                   name=nombre, pickable=False, render=False))

    def malla_coloreada(self, nombre, malla, valores, titulo, cmap="viridis"):
        pd = a_pv(malla)
        pd.cell_data[titulo] = valores
        self.poner(nombre, self.plotter.add_mesh(pd, scalars=titulo, cmap=cmap, name=nombre, render=False,
                                                 scalar_bar_args={"title": titulo, "vertical": True}))

    def cobertura(self, nombre, malla, n):
        """Número de capas por cara: 0 en rojo (hueco sin tejido)."""
        pd = a_pv(malla)
        nmax = max(int(n.max()), 1)
        pd.cell_data["Nº de capas"] = n.astype(float)
        import matplotlib.colors as mcol
        import matplotlib.pyplot as plt
        base = plt.get_cmap("viridis", nmax)
        cols = ["#d62728"] + [mcol.to_hex(base(k)) for k in range(nmax)]
        self.poner(nombre, self.plotter.add_mesh(pd, scalars="Nº de capas", cmap=cols, clim=(-0.5, nmax + 0.5),
                                                 n_colors=nmax + 1, name=nombre, render=False,
                                                 scalar_bar_args={"title": "Nº de capas (rojo = hueco)",
                                                                  "vertical": True, "n_labels": min(nmax + 2, 8),
                                                                  "fmt": "%.0f"}))

    def red(self, nombre, red, color="#555555"):
        """Dibuja los hilos de la red cinemática."""
        self.quitar(nombre)
        if red is None:
            return
        G, ok = red.G, red.ok
        segs = []
        for eje in (0, 1):
            a = G[:-1] if eje == 0 else G[:, :-1]
            b = G[1:] if eje == 0 else G[:, 1:]
            oa = ok[:-1] & ok[1:] if eje == 0 else ok[:, :-1] & ok[:, 1:]
            segs.append(np.stack([a[oa], b[oa]], 1))
        S = np.concatenate(segs)
        if len(S) == 0:
            return
        n = len(S)
        pd = pv.PolyData(S.reshape(-1, 3), lines=np.c_[np.full(n, 2), 2 * np.arange(n), 2 * np.arange(n) + 1].ravel())
        self.poner(nombre, self.plotter.add_mesh(pd, color=color, line_width=1, opacity=0.6, name=nombre,
                                                 pickable=False, render=False))

    def render(self):
        self.plotter.render()

    def encuadrar(self):
        self.plotter.view_vector((1.0, -1.4, 1.1), viewup=(0, 0, 1))
        self.plotter.reset_camera()
        self.plotter.render()

    # --------------------------------------------------------------- picado
    def _clic_derecho(self, obj, ev):
        if self.modo_picado is None:
            return
        import vtk
        x, y = self.plotter.iren.interactor.GetEventPosition()
        if self._picker is None:
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.0005)
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        if self._picker.GetCellId() < 0:
            return
        P = np.array(self._picker.GetPickPosition())
        self.punto_picado.emit(P)
