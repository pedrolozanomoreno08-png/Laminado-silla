import numpy as np
import pytest

from laminado.geom.malla import TriMesh


def rejilla(f, nu, nv):
    u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    P = f(u.ravel(), v.ravel())
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    return TriMesh(P, np.vstack([np.c_[a, b, c], np.c_[a, c, d]]))


@pytest.fixture
def plano():
    return rejilla(lambda u, v: np.c_[200 * u - 100, 200 * v - 100, 0 * u], 41, 41)


@pytest.fixture
def medio_cilindro():
    m = rejilla(lambda u, v: np.c_[300 * u - 150, 100 * np.cos(np.pi * v), 100 * np.sin(np.pi * v)], 81, 81)
    return m if m.face_normals[:, 2].mean() > 0 else m.invertida()


@pytest.fixture
def casquete():
    """Casquete esférico R=100 hasta 60° de colatitud."""
    def f(u, v):
        th = 0.6 * np.pi / 2 * np.sqrt(u + 1e-12) * 1.0
        ph = 2 * np.pi * v
        return np.c_[100 * np.sin(th) * np.cos(ph), 100 * np.sin(th) * np.sin(ph), 100 * np.cos(th)]
    import pyvista as pv
    s = pv.Sphere(radius=100, theta_resolution=90, phi_resolution=90).triangulate().clean()
    m = TriMesh(np.asarray(s.points, float), s.faces.reshape(-1, 4)[:, 1:])
    m = m.submalla(m.face_centroids[:, 2] > 50)[0]
    del f
    return m if m.face_normals[:, 2].mean() > 0 else m.invertida()
