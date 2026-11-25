"""CLT y Tsai-Wu frente a resultados de referencia de libro."""
import numpy as np

from laminado.modelo import clt
from laminado.modelo.materiales import Material

UD = Material("T300/5208", "ud", 0.125, E1=181000, E2=10300, G12=7170, nu12=0.28,
              Xt=1500, Xc=1500, Yt=40, Yc=246, S=68)        # Tsai, "Composites Design"


def test_lamina_a_0_y_90():
    e0 = clt.constantes_ingenieria([(UD, 0, 1.0)])
    e90 = clt.constantes_ingenieria([(UD, 90, 1.0)])
    assert abs(e0["Ex"] - UD.E1) / UD.E1 < 1e-9 and abs(e0["Ey"] - UD.E2) / UD.E2 < 1e-9
    assert abs(e90["Ex"] - UD.E2) / UD.E2 < 1e-9 and abs(e0["nuxy"] - UD.nu12) < 1e-9


def test_lamina_a_45_formula_cerrada():
    # 1/Ex(θ) = c⁴/E1 + s⁴/E2 + (1/G12 − 2ν12/E1)·s²c²
    e = clt.constantes_ingenieria([(UD, 45, 1.0)])
    c = s = np.sqrt(0.5)
    inv = c**4 / UD.E1 + s**4 / UD.E2 + (1 / UD.G12 - 2 * UD.nu12 / UD.E1) * s**2 * c**2
    # en una lámina sola a 45° hay acoplamiento extensión-cortadura: comparar con la
    # flexibilidad de la lámina, no del laminado
    Sb = np.linalg.inv(clt.Qbar(UD, 45))
    assert abs(Sb[0, 0] - inv) / inv < 1e-9
    del e


def test_cuasi_isotropo_valor_de_libro():
    # [0/±45/90]s de T300/5208: Ex = Ey ≈ 69,7 GPa (Tsai, tabla de laminados cuasi-isótropos)
    lam = [(UD, a, 0.125) for a in (0, 45, -45, 90, 90, -45, 45, 0)]
    e = clt.constantes_ingenieria(lam)
    assert abs(e["Ex"] - e["Ey"]) < 1e-6 * e["Ex"]
    assert 68000 < e["Ex"] < 71500
    # cuasi-isótropo: Gxy = Ex / (2(1+ν))
    assert abs(e["Gxy"] - e["Ex"] / (2 * (1 + e["nuxy"]))) / e["Gxy"] < 1e-6


def test_tsai_wu_uniaxial_en_la_resistencia():
    # tracción uniaxial σ1 = Xt en una lámina a 0° → índice 1 (con F12 el término cruzado es nulo)
    t = 1.0
    N = [UD.Xt * t, 0, 0]
    fi = clt.tsai_wu([(UD, 0, t)], N)
    assert abs(fi[0] - 1.0) < 1e-6
    fi = clt.tsai_wu([(UD, 90, t)], [UD.Yt * t, 0, 0])
    assert abs(fi[0] - 1.0) < 1e-6
