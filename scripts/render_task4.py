"""Genera el informe de Task 4 con las mediciones del notebook ejecutado."""

import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = json.loads((ROOT / "artifacts/task4/resultados.json").read_text())
    pdfmetrics.registerFont(TTFont("DejaVu", font_manager.findfont("DejaVu Sans")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold"))))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold")
    body = ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=13, spaceAfter=8)
    heading = ParagraphStyle("heading", parent=body, fontName="DejaVuBold", fontSize=12,
                             leading=16, spaceBefore=8, spaceAfter=9, keepWithNext=True)
    title = ParagraphStyle("title", parent=heading, fontSize=17, leading=21)
    story = []

    def paragraph(text, style=body):
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))
        story.append(Paragraph(text, style))

    def table(rows, widths=None):
        item = Table(rows, colWidths=widths, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
            ("FONTNAME", (0, 0), (-1, 0), "DejaVuBold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf4")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2d0")),
        ]))
        story.extend([item, Spacer(1, 12)])

    paragraph("Laboratorio 9 — Task 4", title)
    paragraph("Curso: Deep Learning 10")
    paragraph("Autores: Diego Lopez, Ricardo Godinez, Vianka Castro y Felipe Aguilar")
    paragraph("Comparación de VAE y difusión; estimación de pass@k")
    paragraph("4.1(a) — Nitidez", heading)
    paragraph("Se mide la media de |x[i,j+1] − x[i,j]| sobre 500 imágenes de cada grupo, "
              "con todos los píxeles en [0,1]. Las reales son las primeras 500 de validación. "
              "Las de difusión son exactamente las 500 de la Task 3.4 con w=1, guardadas "
              "en uint8 y divididas entre 255. La cuantización puede cambiar la métrica "
              "como máximo en 1/255 respecto al tensor flotante original.")
    paragraph("El repositorio no incluía los pesos del VAE. Se repitió solo β=1 con la "
              "arquitectura de la Task 1 (784→400→2→400→784), semilla 3045, 20 épocas, "
              "Adam, lr=0.001 y lotes de 128. Se conservaron las primeras 55 000 imágenes "
              "para entrenamiento y las últimas 5 000 para validación. Las muestras "
              "proceden de z ~ N(0,I). Es una nueva ejecución, no los pesos originales.")
    table([["Grupo", "Imágenes", "Nitidez"]] +
          [[name, "500", f"{value:.6f}"] for name, value in results["nitidez"].items()],
          [250, 90, 150])
    story.append(Image(str(ROOT / "outputs/task4/comparacion.png"), width=490, height=490/3))
    paragraph("Figura 1. Muestras reales, VAE y difusión, con idéntica escala de visualización. "
              "La fila de difusión muestra una imagen de cada clase. No son pares de reconstrucción.")

    story.append(PageBreak())
    paragraph("4.1(b) — Velocidad de generación", heading)
    paragraph(f"Equipo: {results['sistema']}. PyTorch {results['torch']}; CPU, float32, "
              f"{results['hilos']} hilos. Ambos modelos se midieron en el mismo equipo, "
              "con lotes de 100 y sin gradientes. Se reporta la mediana de tres repeticiones "
              "después de calentar las redes. Se incluye el sorteo del ruido y el muestreo; "
              "se excluyen entrenamiento, carga, gráficos y escritura.")
    table([["Modelo", "Mediana, 100 imágenes", "Evaluaciones por lote"]] + [
        [name, f"{results['tiempos'][name]['mediana_s']:.6f} s",
         str(results["evaluaciones_por_lote"][name])] for name in ("VAE", "Difusión")], [90, 200, 200])
    for name in ("VAE", "Difusión"):
        paragraph(f"{name}, repeticiones (s): " + ", ".join(
            f"{t:.6f}" for t in results["tiempos"][name]["repeticiones_s"]))
    paragraph(f"Razón t(difusión)/t(VAE) = {results['razon_tiempos']:.1f}. "
              "Sin guía significa w=1: predicción condicional pura, sin una segunda "
              "evaluación incondicional. El decoder requiere una evaluación por imagen "
              "y la U-Net 1000; al procesar las 100 juntas son 1 y 1000 llamadas por lote.")
    paragraph("4.1(c) — Nitidez, velocidad y estabilidad", heading)
    for block in results["analisis"].split("\n\n"):
        paragraph(block)
    paragraph("Referencia para la propuesta de aceleración: Song, Meng y Ermon (2021), "
              "Denoising Diffusion Implicit Models, ICLR. No se implementó ni cronometró DDIM.")

    story.append(PageBreak())
    paragraph("4.2 — Muestreo de soluciones y pass@k", title)
    paragraph("(a) Estimador insesgado", heading)
    paragraph("Para n=20 y c éxitos: pass@k = 1 − C(n−c,k)/C(n,k). "
              "El cociente es la probabilidad de elegir solo fallos sin reemplazo. "
              "Se calculó en PyTorch float64 con el producto de (n−c−i)/(n−i).")
    rows = [["Problema", "c", "pass@1", "pass@5", "pass@10", "Ingenuo@5"]]
    for i, c in enumerate([0, 2, 5, 9, 14, 20]):
        rows.append([f"P{i+1}", str(c), *[f"{x:.6f}" for x in results["pass_at_k"][i]],
                     f"{results['ingenuo_5'][i]:.6f}"])
    rows.append(["Promedio", "—", *[f"{x:.6f}" for x in results["promedios"]],
                 f"{sum(results['ingenuo_5']) / 6:.6f}"])
    table(rows, [75, 25, 95, 95, 100, 100])
    paragraph("P2 con k=5: C(18,5)=8568 y C(20,5)=15504. "
              "Entonces pass@5 = 1 − 8568/15504 = 17/38 = 0.4473684211.")
    paragraph("(b) Menor k para P2", heading)
    paragraph("Al cancelar factoriales, C(18,k)/C(20,k) = "
              "18!(20−k)! / [20!(18−k)!] = (20−k)(19−k)/380, para 1≤k≤18. "
              "Así, pass@k = 1 − (20−k)(19−k)/380. Para k=19,20 también vale: "
              "no existen suficientes fallos y el producto es cero. Exigir pass@k &gt; 0.9 "
              "equivale a (20−k)(19−k) &lt; 38. Para k=13: 42 y pass@13=0.889474; "
              "para k=14: 30 y pass@14=0.921053. Por monotonía, el mínimo es 14."
              .replace("&gt;", ">").replace("&lt;", "<"))
    paragraph("(c) Por qué la versión ingenua no excede a la insesgada", heading)
    paragraph("Si n−c≥k, los factores son no negativos y "
              "(n−c−i)/(n−i) = 1−c/(n−i) ≤ 1−c/n. Al multiplicar y restar de 1, "
              "1−∏[(n−c−i)/(n−i)] ≥ 1−(1−c/n)^k. Si n−c&lt;k, el estimador "
              "insesgado vale 1 y la desigualdad se mantiene. P2: 0.447368 frente a "
              "0.409510; P3: 0.806308 frente a 0.762695. La versión ingenua supone "
              "intentos independientes con reemplazo y trata c/n como exacto."
              .replace("&lt;", "<"))
    paragraph("(d) Agente que reintenta hasta diez veces", heading)
    paragraph("En el conjunto finito, P1 no se resuelve (0); P2 tiene probabilidad 0.763158, "
              "P3 0.983746 y P4 0.999940. P5 y P6 tienen probabilidad 1 porque hay menos de "
              "diez fallos. No es correcto garantizar P2–P4. El verificador reconoce éxitos, "
              "pero no los genera; estas garantías no se trasladan a diez nuevas generaciones "
              "independientes ni a reintentos correlacionados con retroalimentación.")
    paragraph(results["conclusion"])

    output = ROOT / "report/RESPUESTAS_TASK4.pdf"
    output.parent.mkdir(exist_ok=True)

    def footer(canvas, doc):
        canvas.setFont("DejaVu", 8)
        canvas.drawString(50, 30, "Deep Learning 10 · Laboratorio 9 · Task 4")
        canvas.drawRightString(562, 30, str(doc.page))

    SimpleDocTemplate(str(output), pagesize=letter, leftMargin=50, rightMargin=50,
                      topMargin=40, bottomMargin=48, title="Laboratorio 9 — Task 4",
                      author="Diego Lopez, Ricardo Godinez, Vianka Castro y Felipe Aguilar"
                      ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe generado: {output}")


if __name__ == "__main__":
    main()
