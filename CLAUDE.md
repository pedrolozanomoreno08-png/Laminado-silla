# Laminado Studio — instrucciones para Claude Code

Programa de escritorio (Python + Qt) que reproduce el flujo de trabajo de un software de
diseño de laminados composite tipo Fibersim, con nombre e interfaz propios (no copiar marcas
ni interfaz literal). Caso de demostración: casco de silla de montar a caballo, solo tejido
preimpregnado. Idioma de la interfaz, el código y la documentación: **español**.

## Requisitos del usuario
- Programa de escritorio con interfaz gráfica (Windows es el destino; .exe con PyInstaller).
- Geometría desde **proyecto de Ansys (.wbpz)**; también .dat/.cdb/.rst, STEP/IGES (gmsh) y STL.
- Funciones imprescindibles: **producibilidad interactiva**, **estrategias de drapeado**,
  **ply book y salidas**, **darts y empalmes**. Objetivo: todas las funciones de Fibersim.
- Muy preciso.

## Pila
Python ≥3.11, numpy, scipy, PySide6, pyvista + pyvistaqt (vista 3D), matplotlib (vista 2D e
imágenes del ply book), ezdxf, shapely, reportlab, xlsxwriter, pyyaml, gmsh (opcional, STEP),
ansys-mapdl-reader (opcional, .rst), pytest, pyinstaller.

## Estado (v1.0.0, todo implementado y probado)
- `geom/` malla, punto más cercano, fast marching (exacto en bordes rectos), cortes por nivel,
  rendijas de darts, superficie extendida.
- `ansys/` lector .wbpz/.dat/.cdb/.rst/STEP/STL con selecciones y unidades.
- `drapeado/` red cinemática (geodésica, plano, proyección, curva de semilla), mínima energía
  (LM con jacobiano analítico en ∇u, ∇v), darts manuales/automáticos con relajación, empalmes.
- `modelo/` materiales, rosetas, contornos (desfase/exceso), capas, secuencias, núcleos, CLT,
  core sample, espesores, proyecto .lam.
- `salidas/` DXF, anidado (ráster + FFT), láser, ply book PDF, Excel, tablas ACP, `todo.py`.
- `gui/` ventana Qt + VTK con picado por clic derecho; `python -m laminado --demo`.
- 26 pruebas pytest (incluye archivos Ansys reales, CLT de libro, empalmes); CI en GitHub compila
  el .exe de Windows (artefacto LaminadoStudio-windows).
- `verificacion/`: esfera analítica, invariancias, archivos Ansys reales, class40, salidas,
  recorrido completo de la interfaz (`recorrido_gui.py`) y regresión de ejemplos
  (`regresion_ejemplos.py`).
- Empalmes: cada trozo conexo es una pieza; piezas redrapeadas desde su centro (geodésica si la
  capa usa curva); empalmes por cizalla aceptados sólo si reducen el área roja y no dejan retales.
- Laminado con espesor: cada capa se drapea sobre la superficie desplazada por el espesor de las
  capas y núcleos que tiene debajo (rampas de terminación y chaflanes); `sentido_laminado` ±1
  (molde macho/hembra); exacto en semicilindro (ancho plano π(R+k·e)); salida superficie_laminado.stl.
- `modelo/cobertura.py`: cobertura completa automática con parches producibles (regiones por tubo,
  relleno voraz, migas absorbidas, roseta por eje local); modo de color «Cobertura (nº de capas)».
  Ejemplo: cuadro de bicicleta completo (4 secuencias, 100 % cubierto, ningún parche rojo).
- Piezas grandes y planas: red limitada a ~200 000 nodos donde la curvatura lo permite (en class40
  la cizalla pico local baja de 27° a 20°; el área sobre aviso sigue en ~0 %).

## Pendiente / mejoras posibles
- Cobertura: menos parches en los nudos (ahora se fuerzan parches pequeños no producibles) y
  menos tiempo (~25 min por secuencia en la bici).
- Validar con el .wbpz real del usuario y su ficha de tejido; probar el .exe en Windows.
- Importación en ACP de las tablas de fibra (paso manual, sin probar).

## Notas de entorno
- Las pruebas de interfaz sin pantalla: `xvfb-run -a -s "-screen 0 1400x900x24" python ...`
  (no usar QT_QPA_PLATFORM=offscreen con VTK).
- Commits en español, pequeños y frecuentes.
