"""Laminado con espesor: cada capa se drapea sobre las anteriores."""
import numpy as np

from laminado.ansys.lector import Molde
from laminado.modelo.proyecto import Capa, Contorno, Nucleo, Proyecto

MAT = "Carbono sarga 2x2 200 g/m² (prepreg epoxi)"


def _proyecto(m, n=4):
    p = Proyecto("cil")
    p.cargar_molde(Molde(m, {}, origen="prueba"))
    p.h = 5.0
    for k in range(n):
        c = Capa(f"P{k + 1}", MAT, 0.0, contorno=Contorno("molde"))
        c.empalme_auto = False
        p.capas.append(c)
    return p


def _ancho_trama(rc):
    """Desarrollo plano del arco a media longitud del medio cilindro (dirección de trama)."""
    r = rc.drapeado
    uv = r.uv
    centro = np.abs(r.malla.V[:, 0]) < 3
    return float(np.ptp(uv[centro, 1]))


def test_capas_exteriores_desarrollan_mas(medio_cilindro):
    """Medio cilindro R = 100: la capa k se drapea sobre un radio R + (k-1)·e y su patrón mide
    π·(R + (k-1)·e) (solución exacta)."""
    p = _proyecto(medio_cilindro)
    e = p.materiales[MAT].espesor
    for k, c in enumerate(p.capas):
        rc = p.simular(c)
        esperado = np.pi * (100.0 + k * e)
        assert abs(_ancho_trama(rc) - esperado) < 0.05 * max(k * e, 1) + 0.05, (k, _ancho_trama(rc), esperado)
        assert abs(rc.desplazamiento - k * e) < 1e-6
    # sin espesor todas iguales
    q = _proyecto(medio_cilindro)
    q.drapeado_con_espesor = False
    a = [_ancho_trama(q.simular(c)) for c in q.capas]
    assert np.ptp(a) < 0.02


def test_espesores_y_superficie_final(medio_cilindro):
    p = _proyecto(medio_cilindro, 3)
    p.nucleos.append(Nucleo("N", "Núcleo PVC 5 mm 80 kg/m³", Contorno("molde", desfase=-40), espesor=5.0,
                            chaflan=30.0, despues_de="P1"))
    e = p.materiales[MAT].espesor
    esp, _ = p.espesores()
    assert abs(esp.max() - (3 * e + 5.0)) < 1e-6
    sup, t = p.superficie_laminado()
    # en la cumbre del cilindro la superficie final está a R + 3e + 5
    cima = np.argmax(sup.V[:, 2])
    assert abs(sup.V[cima, 2] - (100 + 3 * e + 5.0)) < 0.2
    # la capa P2 se drapea sobre el núcleo: su superficie de drapeado sube 5 mm + e
    rc = p.simular("P2")
    assert abs(rc.desplazamiento - (e + 5.0)) < 1e-6
    # chaflán de 30°: el núcleo pasa de 0 a 5 mm en 5/tan(30°) = 8,66 mm
    assert t.max() > 0
