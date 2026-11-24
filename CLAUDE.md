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
- 12 pruebas pytest; CI en GitHub compila el .exe de Windows (artefacto LaminadoStudio-windows).

## Pendiente / mejoras posibles
- Drapear cada capa sobre la superficie desplazada por el espesor acumulado.
- Geometría de rampa de los núcleos (ahora sólo cuentan en espesores).
- Validar con el .wbpz real del usuario y su ficha de tejido; probar el .exe en Windows.
- Importación en ACP de las tablas de fibra (paso manual, sin probar).

## Notas de entorno
- Las pruebas de interfaz sin pantalla: `xvfb-run -a -s "-screen 0 1400x900x24" python ...`
  (no usar QT_QPA_PLATFORM=offscreen con VTK).
- Commits en español, pequeños y frecuentes.
