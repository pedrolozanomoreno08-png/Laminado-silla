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

## Estructura (paquete `laminado`)
- `geom/` **hecho (falta probar)**: `malla.py` (TriMesh: normales, aristas, bordes, laplaciano
  cotangente, submallas), `consulta.py` (punto más cercano vectorizado, normal Phong, rayos),
  `geodesia.py` (método del calor, Dijkstra, trazado de geodésicas), `cortes.py` (corte exacto
  por curva de nivel, rendijas para darts, superficie extendida).
- `ansys/lector.py` **hecho (falta probar)**: .wbpz (zip → ds.dat/.cdb/.rst/STEP/STL),
  NBLOCK/EBLOCK/CMBLOCK (componentes → selecciones), piel de sólidos, unidades → mm,
  `escribir_mapdl` para pruebas.
- `drapeado/` **pendiente**:
  - Red cinemática (fishnet): generadores desde el punto semilla y relleno por barridos
    vectorizados (nodo (i,j) a distancia h de (i±1,j) y (i,j±1), Newton en plano tangente +
    proyección). Rechazar celdas que crucen darts.
  - Estrategias: geodésica, plano natural (corte con plano que contiene semilla, normal y
    dirección), curva de semilla, dirección guiada (campo de la roseta proyectado),
    mínima energía (least_squares sobre coordenadas planas por vértice: |a|=|b|=1, a·b=0;
    tejido o UD).
  - Salida común: coordenadas de tejido (u,v) por vértice de la malla de la capa → direcciones
    de fibra, cizalla por cara, patrón plano = bucles de borde en (u,v).
  - Producibilidad: cizalla con umbrales de aviso/límite del material (verde/amarillo/rojo),
    zonas sin alcanzar.
  - Darts manuales (puntos picados → camino Dijkstra → `abrir_rendija`) y automáticos (zonas
    con cizalla > límite, desde el borde hacia el pico de cizalla).
  - Empalmes manuales (plano/curva, solape/tope/hueco) y automáticos por ancho de rollo.
- `modelo/` **pendiente**: materiales (espesor, gramaje, ángulo de bloqueo, ancho de rollo),
  rosetas, capas (orientación, contorno: todo/selección Ansys/polígono proyectado/curva 3D/
  planos; desfase geodésico, exceso de recorte), secuencias, zonas, núcleos, escalonado de
  caídas de capa, guardar/abrir proyecto (.lam JSON), CLT + Tsai-Wu, core sample.
- `salidas/` **pendiente**: DXF por capa (capas CORTE, NETO, FIBRA, TEXTO, DARTS), anidado en
  ancho de rollo (rotaciones permitidas según material), proyección láser (polilíneas 3D DXF +
  CSV), ply book PDF (reportlab + imágenes matplotlib), Excel, exportación a Ansys ACP (CSV de
  direcciones de fibra por elemento + script PyACP).
- `gui/` **pendiente**: ventana principal Qt: árbol del laminado, panel de propiedades, vista
  3D con picado (semilla, darts, core sample), producibilidad en vivo, patrón plano 2D, tabla
  de capas, mapa de espesores.
- `demo/silla.py` **pendiente**: casco de silla paramétrico (arco sobre el lomo, pomo y
  borrén) mallado a ~6 mm, escrito también como ds.dat para probar el lector.
- `tests/`, `README.md`, `requirements.txt`, `instalar.bat`, `ejecutar.bat`, spec de
  PyInstaller, GitHub Actions (pruebas en Ubuntu + .exe en windows-latest).

## Orden de trabajo
1. Pruebas de `geom/` y `ansys/` (con la silla de demostración).
2. `drapeado/` + pruebas (semiesfera: cizalla conocida; cilindro: cizalla nula).
3. `modelo/`, 4. `salidas/`, 5. `gui/` (verificar con capturas), 6. empaquetado y CI.

## Notas de entorno
- Las pruebas de interfaz sin pantalla: `xvfb-run -a -s "-screen 0 1400x900x24" python ...`
  (no usar QT_QPA_PLATFORM=offscreen con VTK).
- Commits en español, pequeños y frecuentes.
