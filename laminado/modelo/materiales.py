"""Materiales: tejidos y unidireccionales preimpregnados, núcleos."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Material:
    nombre: str
    tipo: str = "tejido"                 # 'tejido' | 'ud' | 'nucleo'
    espesor: float = 0.22                # espesor curado por capa (mm)
    peso_area: float = 330.0             # g/m² del preimpregnado (fibra + resina)
    gramaje_fibra: float = 200.0         # g/m² de fibra seca
    contenido_resina: float = 0.40       # fracción en peso
    ancho_rollo: float = 1250.0          # mm
    cizalla_aviso: float = 25.0          # grados
    cizalla_limite: float = 35.0         # grados (ángulo de bloqueo práctico)
    E1: float = 60000.0                  # MPa
    E2: float = 60000.0
    G12: float = 4000.0
    nu12: float = 0.05
    Xt: float = 600.0                    # resistencias (MPa)
    Xc: float = 550.0
    Yt: float = 600.0
    Yc: float = 550.0
    S: float = 90.0
    simetrico: bool = True               # se puede girar 90° al anidar (tejido equilibrado)
    coste_m2: float = 0.0                # €/m²
    notas: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def unidireccional(self) -> bool:
        return self.tipo == "ud"

    def a_dict(self):
        return asdict(self)

    @staticmethod
    def de_dict(d):
        return Material(**d)


def biblioteca() -> dict[str, Material]:
    """Materiales de partida (valores típicos de catálogo; ajústalos con la ficha técnica)."""
    mats = [
        Material("Carbono sarga 2x2 200 g/m² (prepreg epoxi)", "tejido", 0.22, 330, 200, 0.40, 1250,
                 25, 35, 60000, 60000, 4000, 0.05, 600, 550, 600, 550, 90, True),
        Material("Carbono tafetán 3K 200 g/m² (prepreg epoxi)", "tejido", 0.22, 330, 200, 0.40, 1250,
                 20, 30, 58000, 58000, 4200, 0.05, 620, 560, 620, 560, 95, True),
        Material("Carbono sarga 2x2 400 g/m² (prepreg epoxi)", "tejido", 0.42, 650, 400, 0.38, 1250,
                 25, 35, 60000, 60000, 4000, 0.05, 600, 550, 600, 550, 90, True),
        Material("Vidrio 7781 satén 8H (prepreg epoxi)", "tejido", 0.24, 480, 300, 0.37, 1270,
                 30, 45, 24000, 23000, 4500, 0.12, 420, 380, 400, 360, 70, True),
        Material("Carbono UD 150 g/m² (prepreg epoxi)", "ud", 0.15, 250, 150, 0.36, 600,
                 3, 6, 135000, 9000, 4800, 0.30, 2200, 1300, 50, 200, 90, False),
        Material("Núcleo PVC 5 mm 80 kg/m³", "nucleo", 5.0, 400, 0, 0, 1000,
                 90, 90, 80, 80, 27, 0.3, 2, 1.4, 2, 1.4, 1.1, True),
    ]
    return {m.nombre: m for m in mats}
