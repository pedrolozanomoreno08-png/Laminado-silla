@echo off
chcp 65001 >nul
echo === Laminado Studio: instalación ===
where python >nul 2>nul
if errorlevel 1 (
  echo No se encuentra Python. Instala Python 3.11 o 3.12 desde https://www.python.org/downloads/
  echo y marca "Add python.exe to PATH" durante la instalación.
  pause
  exit /b 1
)
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Algún paquete opcional ha fallado; se instalan sólo los imprescindibles.
  python -m pip install numpy scipy PySide6 pyvista pyvistaqt vtk matplotlib shapely ezdxf reportlab xlsxwriter
)
echo.
echo Instalación terminada. Ejecuta ejecutar.bat para abrir el programa.
pause
