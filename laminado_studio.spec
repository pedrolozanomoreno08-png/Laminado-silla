# -*- mode: python ; coding: utf-8 -*-
# PyInstaller:  pyinstaller laminado_studio.spec
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules

EXCLUIR = ("mypy", "pytest", "trame", "IPython", "jupyter", "tkinter", "sphinx")


def filtro(nombre):
    return not any(x in nombre for x in ("mypy_plugin", "tests", "testing", "examples", "trame", "jupyter"))


datas, binaries, hidden = [], [], []
for pkg in ("pyvista", "pyvistaqt", "vtkmodules"):
    datas += collect_data_files(pkg)
    binaries += collect_dynamic_libs(pkg)
    hidden += collect_submodules(pkg, filter=filtro)
for pkg in ("ezdxf", "reportlab", "shapely", "gmsh", "h5py"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += [x for x in h if filtro(x)]
    except Exception:
        pass
hidden += collect_submodules("laminado")
try:
    hidden += collect_submodules("ansys.mapdl.reader", filter=filtro)
except Exception:
    pass

a = Analysis(["lanzar.py"], pathex=["."], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=list(EXCLUIR))
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LaminadoStudio", console=False, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="LaminadoStudio")
