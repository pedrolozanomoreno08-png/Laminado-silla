# -*- mode: python ; coding: utf-8 -*-
# PyInstaller:  pyinstaller laminado_studio.spec
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hidden = [], [], []
for pkg in ("pyvista", "pyvistaqt", "vtkmodules", "ezdxf", "reportlab", "shapely", "gmsh"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += h
    except Exception:
        pass
hidden += collect_submodules("laminado")
try:
    hidden += collect_submodules("ansys.mapdl.reader")
except Exception:
    pass

a = Analysis(["lanzar.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=["tkinter", "IPython", "jupyter", "trame", "pytest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LaminadoStudio", console=False,
          icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="LaminadoStudio")
