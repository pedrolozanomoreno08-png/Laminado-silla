"""Ply book (PDF), tabla Excel y exportación a Ansys (tablas de direcciones de fibra)."""
from __future__ import annotations

import datetime as _dt
import os

import numpy as np

from ..geom.consulta import Consultor
from . import figuras
from .patrones import patrones_de


# ------------------------------------------------------------------------------------ ply book
def ply_book(proyecto, ruta, anidado=None, autor=""):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    est = getSampleStyleSheet()
    _P = Paragraph
    _T = Table

    def _limpio(x):
        return x.replace("²", "2").replace("³", "3").replace("°", "°") if isinstance(x, str) else x

    def Paragraph(texto, estilo):  # noqa: N802
        return _P(_limpio(texto), estilo)

    def Table(datos, *a, **k):  # noqa: N802
        return _T([[_limpio(c) for c in fila] for fila in datos], *a, **k)
    doc = SimpleDocTemplate(ruta, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title=f"Ply book – {proyecto.nombre}", author=autor or "Laminado Studio")
    hoy = _dt.date.today().isoformat()
    els = []

    def tabla(datos, anchos=None, cab=True):
        t = Table(datos, colWidths=anchos, repeatRows=1 if cab else 0)
        st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8),
              ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        if cab:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                   ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8)]
        t.setStyle(TableStyle(st))
        return t

    def color_estado(e):
        return {"VERDE": colors.HexColor("#c8e6c9"), "AMARILLO": colors.HexColor("#fff59d"),
                "ROJO": colors.HexColor("#ef9a9a")}.get(e, colors.white)

    # ---------------------------------------------------------------- portada
    els.append(Paragraph(f"Ply book – {proyecto.nombre}", est["Title"]))
    els.append(Paragraph(f"Fecha: {hoy} &nbsp;&nbsp; Molde: {proyecto.molde.origen if proyecto.molde else '-'}"
                         f" &nbsp;&nbsp; Capas: {len(proyecto.capas)}", est["Normal"]))
    els.append(Spacer(1, 6 * mm))
    filas = [["Orden", "Capa", "Material", "Ori. (°)", "Estrategia", "Cizalla máx (°)", "Área corte (m²)",
              "Peso (g)", "Piezas", "Estado"]]
    estados = []
    peso_total = 0.0
    for k, c in enumerate(proyecto.capas, 1):
        rc = proyecto.resultados.get(c.nombre)
        s = rc.resumen if rc else {}
        peso_total += s.get("peso_g", 0)
        filas.append([k, c.nombre, c.material[:42], f"{c.orientacion:g}", c.estrategia,
                      f"{s.get('cizalla_max', float('nan')):.1f}", f"{s.get('area_corte_mm2', 0) / 1e6:.3f}",
                      f"{s.get('peso_g', 0):.0f}", s.get("piezas", "-"), s.get("estado", "sin simular")])
        estados.append(s.get("estado", ""))
    t = tabla(filas, [14 * mm, 22 * mm, 80 * mm, 16 * mm, 24 * mm, 26 * mm, 26 * mm, 18 * mm, 16 * mm, 24 * mm])
    for i, e in enumerate(estados, 1):
        t.setStyle(TableStyle([("BACKGROUND", (9, i), (9, i), color_estado(e))]))
    els.append(t)
    els.append(Spacer(1, 3 * mm))
    els.append(Paragraph(f"Peso total de preimpregnado cortado: <b>{peso_total / 1000:.3f} kg</b>", est["Normal"]))
    if proyecto.resultados:
        try:
            img = figuras.a_png(figuras.figura_espesores(proyecto, tam=(6, 3.6)))
            els.append(Image(img, width=150 * mm, height=90 * mm))
        except Exception:
            pass
    els.append(PageBreak())
    # --------------------------------------------------------- página por capa
    for k, c in enumerate(proyecto.capas, 1):
        rc = proyecto.resultados.get(c.nombre)
        if rc is None:
            continue
        mat = proyecto.materiales[c.material]
        s = rc.resumen
        els.append(Paragraph(f"Capa {k}: {c.nombre} &nbsp;&nbsp; {c.orientacion:g}° &nbsp;&nbsp; {c.material}",
                             est["Heading2"]))
        datos = [["Roseta", c.roseta, "Estrategia", c.estrategia, "Estado", s.get("estado", "")],
                 ["Cizalla máx", f"{s['cizalla_max']:.1f}°", "Aviso / límite",
                  f"{mat.cizalla_aviso:g}° / {mat.cizalla_limite:g}°", "Área > límite", f"{s.get('area_limite_%', 0):.2f} %"],
                 ["Área neta", f"{s['area_neta_mm2'] / 1e6:.4f} m²", "Área de corte",
                  f"{s['area_corte_mm2'] / 1e6:.4f} m²", "Peso", f"{s['peso_g']:.0f} g"],
                 ["Piezas", s["piezas"], "Empalmes", s["empalmes"], "Darts", s["n_darts"]],
                 ["Exceso recorte", f"{c.contorno.exceso:g} mm", "Desfase", f"{proyecto.desfase_total(c):g} mm",
                  "Secuencia", c.secuencia or "-"]]
        t = tabla(datos, [28 * mm, 40 * mm, 28 * mm, 40 * mm, 28 * mm, 40 * mm], cab=False)
        t.setStyle(TableStyle([("BACKGROUND", (5, 0), (5, 0), color_estado(s.get("estado")))]))
        els.append(t)
        els.append(Spacer(1, 3 * mm))
        i3d = figuras.a_png(figuras.figura_capa_3d(proyecto, c.nombre, tam=(5.2, 4.2)))
        pats = patrones_de(proyecto, c.nombre)
        i2d = figuras.a_png(figuras.figura_patron(pats, tam=(5.2, 4.2), titulo="Patrón plano (corte / neto - - -)"))
        els.append(Table([[Image(i3d, width=125 * mm, height=101 * mm), Image(i2d, width=125 * mm, height=101 * mm)]]))
        nota = ("Colocar desde la semilla (punto negro) alisando hacia los bordes a lo largo de la "
                "urdimbre (flechas). Zonas amarillas: alisar con cuidado; rojas: requieren dart o empalme.")
        if c.notas:
            nota = c.notas + " — " + nota
        els.append(Paragraph(nota, est["Normal"]))
        els.append(PageBreak())
    # --------------------------------------------------------------- anidado
    if anidado is not None:
        els.append(Paragraph("Anidado en rollo", est["Heading2"]))
        filas = [["Material", "Ancho (mm)", "Largo (mm)", "Piezas", "Aprovechamiento (%)"]]
        for r in anidado.resumen():
            filas.append([r["material"][:50], f"{r['ancho_mm']:.0f}", f"{r['largo_mm']:.0f}", r["piezas"],
                          f"{r['aprovechamiento_%']:.1f}"])
        els.append(tabla(filas))
        img = figuras.a_png(figuras.figura_anidado(anidado, tam=(10, 4.5)))
        els.append(Image(img, width=250 * mm, height=112 * mm))

    def pie(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(12 * mm, 7 * mm, f"{proyecto.nombre} – ply book – {hoy}")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"Página {d.page}")
        canvas.restoreState()
    doc.build(els, onFirstPage=pie, onLaterPages=pie)
    return ruta


# -------------------------------------------------------------------------------------- Excel
def excel(proyecto, ruta, anidado=None):
    import xlsxwriter
    wb = xlsxwriter.Workbook(ruta)
    neg = wb.add_format({"bold": True, "bg_color": "#1f3b73", "font_color": "white", "border": 1})
    num = wb.add_format({"num_format": "0.00", "border": 1})
    txt = wb.add_format({"border": 1})
    cols = {"VERDE": "#c8e6c9", "AMARILLO": "#fff59d", "ROJO": "#ef9a9a"}
    fmt_est = {k: wb.add_format({"bg_color": v, "border": 1}) for k, v in cols.items()}
    ws = wb.add_worksheet("Capas")
    cab = ["Orden", "Capa", "Material", "Orientación (°)", "Roseta", "Estrategia", "Secuencia",
           "Cizalla máx (°)", "Cizalla media (°)", "Área > aviso (%)", "Área > límite (%)", "Cubierto (%)",
           "Área neta (m²)", "Área corte (m²)", "Peso (g)", "Piezas", "Empalmes", "Darts", "Estado"]
    for j, c in enumerate(cab):
        ws.write(0, j, c, neg)
    for i, c in enumerate(proyecto.capas, 1):
        s = proyecto.resultados[c.nombre].resumen if c.nombre in proyecto.resultados else {}
        fila = [i, c.nombre, c.material, c.orientacion, c.roseta, c.estrategia, c.secuencia,
                s.get("cizalla_max"), s.get("cizalla_media"), s.get("area_aviso_%"), s.get("area_limite_%"),
                s.get("cubierto_%"), s.get("area_neta_mm2", 0) / 1e6, s.get("area_corte_mm2", 0) / 1e6,
                s.get("peso_g"), s.get("piezas"), s.get("empalmes"), s.get("n_darts"), s.get("estado", "")]
        for j, v in enumerate(fila):
            if j == len(fila) - 1:
                ws.write(i, j, v, fmt_est.get(v, txt))
            elif isinstance(v, float):
                ws.write_number(i, j, v, num)
            elif v is None:
                ws.write_blank(i, j, None, txt)
            else:
                ws.write(i, j, v, txt)
    ws.set_column(0, 0, 6)
    ws.set_column(1, 1, 10)
    ws.set_column(2, 2, 45)
    ws.set_column(3, 18, 13)
    ws.freeze_panes(1, 2)
    # consumo por material
    wm = wb.add_worksheet("Materiales")
    for j, c in enumerate(["Material", "Capas", "Área corte (m²)", "Peso (kg)", "Coste (€)"]):
        wm.write(0, j, c, neg)
    agg = {}
    for c in proyecto.capas:
        s = proyecto.resultados[c.nombre].resumen if c.nombre in proyecto.resultados else {}
        a = agg.setdefault(c.material, [0, 0.0, 0.0])
        a[0] += 1
        a[1] += s.get("area_corte_mm2", 0) / 1e6
        a[2] += s.get("peso_g", 0) / 1000
    for i, (mat, (n, area, peso)) in enumerate(agg.items(), 1):
        wm.write(i, 0, mat, txt)
        wm.write(i, 1, n, txt)
        wm.write_number(i, 2, area, num)
        wm.write_number(i, 3, peso, num)
        wm.write_number(i, 4, area * proyecto.materiales[mat].coste_m2, num)
    wm.set_column(0, 0, 50)
    wm.set_column(1, 4, 16)
    if anidado is not None:
        wa = wb.add_worksheet("Anidado")
        for j, c in enumerate(["Material", "Ancho (mm)", "Largo (mm)", "Piezas", "Aprovechamiento (%)"]):
            wa.write(0, j, c, neg)
        for i, r in enumerate(anidado.resumen(), 1):
            wa.write(i, 0, r["material"], txt)
            wa.write_number(i, 1, r["ancho_mm"], num)
            wa.write_number(i, 2, r["largo_mm"], num)
            wa.write(i, 3, r["piezas"], txt)
            wa.write_number(i, 4, r["aprovechamiento_%"], num)
        wa.set_column(0, 0, 50)
        wa.set_column(1, 4, 16)
    wb.close()
    return ruta


# --------------------------------------------------------------------------- Ansys / ACP
def exportar_ansys(proyecto, carpeta):
    """Por cada capa, tabla con la dirección real de fibra en cada elemento del molde cubierto.

    Archivos ``<capa>_fibras.csv`` (elemento, centroide, urdimbre, trama, ángulo real respecto a la
    roseta, desviación y cizalla) — formato de *Look-Up Table 3D* de ACP (coordenadas + vector) —
    y ``laminado.csv`` con la secuencia. En ACP: importar la tabla como Look-Up Table 3D y usarla
    para la corrección de ángulo / dirección de referencia de cada Modeling Ply."""
    os.makedirs(carpeta, exist_ok=True)
    m = proyecto.malla
    ids = proyecto.molde.id_elemento
    X = m.face_centroids
    rutas = []
    for c in proyecto.capas:
        rc = proyecto.resultados.get(c.nombre)
        if rc is None:
            continue
        r = rc.drapeado
        neto_d = Consultor(rc.neto_en_molde).mas_cercano(X)[3] < 0.5
        cr = Consultor(r.malla)
        _, fr, _, _ = cr.mas_cercano(X[neto_d])
        ea, eb = r.fibras()
        a = ea[fr]
        b = eb[fr]
        n = m.face_normals[neto_d]
        ros = proyecto.rosetas.get(c.roseta) or next(iter(proyecto.rosetas.values()))
        ref = ros.direccion(X[neto_d], n, 0.0)
        ang = np.degrees(np.arctan2((np.cross(ref, a) * n).sum(1), (ref * a).sum(1)))
        sg = np.sign((n * np.asarray(ros.normal)).sum(1))
        ang = ang * np.where(sg == 0, 1, sg)
        g = np.abs(r.cizalla()[fr])
        idx = np.nonzero(neto_d)[0]
        eid = ids[idx] if ids is not None else idx + 1
        ruta = os.path.join(carpeta, f"{c.nombre}_fibras.csv")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("# Laminado Studio – direcciones de fibra drapeadas\n")
            f.write(f"# capa={c.nombre}; material={c.material}; orientacion_nominal={c.orientacion}\n")
            f.write("elemento,x,y,z,urdimbre_x,urdimbre_y,urdimbre_z,trama_x,trama_y,trama_z,"
                    "angulo_real,desviacion,cizalla\n")
            for k in range(len(idx)):
                P = X[idx[k]]
                f.write(f"{int(eid[k])},{P[0]:.4f},{P[1]:.4f},{P[2]:.4f},"
                        f"{a[k, 0]:.6f},{a[k, 1]:.6f},{a[k, 2]:.6f},{b[k, 0]:.6f},{b[k, 1]:.6f},{b[k, 2]:.6f},"
                        f"{ang[k]:.3f},{ang[k] - c.orientacion:.3f},{g[k]:.3f}\n")
        rutas.append(ruta)
    ruta = os.path.join(carpeta, "laminado.csv")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("orden,capa,material,espesor_mm,orientacion,roseta,secuencia\n")
        for k, c in enumerate(proyecto.capas, 1):
            f.write(f"{k},{c.nombre},{c.material},{proyecto.materiales[c.material].espesor},"
                    f"{c.orientacion},{c.roseta},{c.secuencia}\n")
    rutas.append(ruta)
    return rutas
