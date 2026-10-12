"""Reúne las respuestas y salidas ejecutadas de las cuatro tasks en un PDF."""

from html import escape
from pathlib import Path
import re
import tempfile

import nbformat
from nbconvert.filters.markdown import markdown2html
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
AUTHORS = "Diego Lopez, Ricardo Godinez, Vianka Castro y Felipe Aguilar"
NOTEBOOKS = [
    "01_vae_fashion_mnist.ipynb",
    "02_difusion_forward.ipynb",
    "03_difusion_condicional.ipynb",
    "04_comparacion_passk.ipynb",
]

# Solo se excluyen preparación, registros de entrenamiento y gestión de artefactos.
# Las respuestas de análisis y todas las figuras permanecen en el informe.
SKIP = {1: {2, 3, 8, 9, 11}, 2: {1, 2}, 3: {1, 2, 8, 28, 29, 31},
        4: {1, 2, 3, 4, 5, 6}}

CSS = """
@page { size: Letter; margin: 18mm 17mm 19mm; }
body { font: 10pt/1.45 Arial, sans-serif; color: #172333; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 16pt; }
h2 { font-size: 14pt; line-height: 1.3; margin: 17pt 0 9pt; }
h3 { font-size: 11pt; margin: 13pt 0 7pt; }
h1,h2,h3 { break-after: avoid; }
p { margin: 7pt 0; orphans: 3; widows: 3; }
li { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 12pt 0; }
th,td { border-bottom: 1px solid #c6cfda; padding: 6pt; text-align: left; }
th { background: #e8edf4; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
pre { font: 7.6pt/1.4 monospace; white-space: pre-wrap; overflow-wrap: anywhere;
      background: #f3f5f8; padding: 8pt; break-inside: avoid; }
code { font-size: 9pt; }
figure { margin: 12pt 0; text-align: center; break-inside: avoid; }
img { max-width: 100%; max-height: 175mm; object-fit: contain; }
figcaption { font-size: 8pt; color: #4b5a6d; margin-top: 4pt; }
.task { break-before: page; }
.cover { padding-top: 28mm; }
.note { border-left: 3px solid #547496; padding: 8pt 12pt; background: #f3f5f8; }
mjx-container[display="true"] { break-inside: avoid; margin: 12pt 0 !important; }
"""


def main():
    parts = [f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
        <title>Laboratorio 9 — Informe completo</title><style>{CSS}</style></head><body>
        <section class="cover"><h1>Laboratorio 9<br>Modelos generativos</h1>
        <p><b>Curso:</b> Deep Learning 10</p><p><b>Autores:</b> {AUTHORS}</p>
        <h2>Informe final — Tasks 1–4</h2>
        <ol><li>VAE: reconstrucción, espacio latente y análisis cuantitativo.</li>
        <li>Difusión forward: verificación, calendarios y relación señal a ruido.</li>
        <li>Difusión condicional: U-Net, muestreo y guía sin clasificador.</li>
        <li>Comparación VAE–difusión y estimación de pass@k.</li></ol>
        <p>Semilla de los experimentos: <b>3045</b>. Este documento reúne las respuestas,
        tablas y figuras guardadas en los cuatro notebooks. El código comentado se
        entrega en esos notebooks y en el repositorio; no se repitieron los entrenamientos
        para compilar este informe.</p>
        <p>La Task 4 documenta una nueva ejecución del VAE β=1 y mide ambos tiempos en CPU.
        Sus valores no sustituyen los de la Task 1 ni los tiempos de GPU de la Task 3.</p>
        <div class="note"><b>Alcance de la evidencia:</b> la Task 2.5(c) existente calcula
        la varianza con 10 000 imágenes de entrenamiento, no con todo el conjunto solicitado.
        Se conserva ese resultado y se identifica su alcance, sin presentarlo como una
        medición completa. La interpretación inicial del SNR alto en la Task 2.5(b)
        se contrasta y corrige con los resultados de la Task 3.5(d).</div></section>"""]
    figures = 0
    for task, filename in enumerate(NOTEBOOKS, 1):
        notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
        parts.append(f'<section class="task" id="task{task}">')
        for index, cell in enumerate(notebook.cells):
            if index in SKIP[task]:
                continue
            if cell.cell_type == "markdown":
                source = cell.source
                # Se conserva la decisión del equipo de no incluir registros de IA ni texto oculto.
                source = re.split(r"(?m)^## .*Registro de uso de IA", source)[0]
                if task == 4 and index == 0:
                    source = re.sub(r"(?m)^\*\*(Curso|Autores):\*\*.*\n?", "", source)
                source = re.sub(r"<!--.*?-->", "", source, flags=re.S)
                parts.append(markdown2html(source))
                if task == 1 and index == 4:
                    parts.append(markdown2html(
                        "**Configuración:** batch_size=128, latent_dim=2, epochs=20, "
                        "lr=0.001, β∈{0.1, 1, 10}. Arquitectura compartida: "
                        "784→400→(μ, log σ²), con dos coordenadas latentes; decoder 2→400→784. "
                        "Se minimiza Lβ = SSE + β·KL, con SSE sumada sobre los 784 píxeles."))
                if task == 2 and index == 41:
                    parts.append('<p class="note"><b>Contraste posterior:</b> la afirmación '
                                 'anterior de que predecir ruido con SNR alto es trivial no coincide '
                                 'con la ejecución de la Task 3. El intervalo 1–100 es el más difícil '
                                 '(MSE=0.2166); la Task 3.5(d) desarrolla esta corrección.</p>')
                if task == 3 and index == 7:
                    parts.append('<p><b>Configuración:</b> 30 épocas, batch_size=128, '
                                 'lr=0.0002 y p_uncond=0.1; NVIDIA GeForce RTX 3050 6GB Laptop GPU.</p>')
                continue
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    raise ValueError(f"Salida con error en {filename}, celda {index}")
                data = output.get("data", {})
                if "image/png" in data:
                    figures += 1
                    parts.append(f'<figure><img src="data:image/png;base64,{data["image/png"]}" '
                                 f'alt="Figura {figures}, Task {task}"><figcaption>'
                                 f'Figura {figures}. Task {task} · salida de la celda {index} '
                                 f'de {escape(filename)}.</figcaption></figure>')
                elif "text/markdown" in data:
                    parts.append(markdown2html(data["text/markdown"]))
                elif output.output_type == "stream" and output.get("name") == "stdout":
                    parts.append(f'<pre>{escape(output.text.strip())}</pre>')
        parts.append("</section>")
    parts.append("</body></html>")
    output = ROOT / "report/RESPUESTAS_LAB9.pdf"
    output.parent.mkdir(exist_ok=True)
    # Los temporales se borran al terminar; no forman parte de la entrega.
    scratch = ROOT / ".amp/in"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        html = Path(directory) / "informe.html"
        html.write_text("\n".join(parts))
        raw_pdf = Path(directory) / "informe.pdf"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(html.as_uri())
            page.locator("a.anchor-link").evaluate_all("links => links.forEach(link => link.remove())")
            page.evaluate(r"""window.MathJax = {
                tex: {inlineMath: [['$', '$'], ['\\(', '\\)']]},
                svg: {fontCache: 'local'}, options: {enableMenu: false}
            }""")
            page.add_script_tag(url="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js")
            page.evaluate("MathJax.startup.promise")
            page.evaluate("document.fonts.ready")
            errors = page.locator('[data-mml-node="merror"]').count()
            if errors:
                raise ValueError(f"Hay {errors} fórmulas que no se pudieron renderizar")
            page.pdf(path=str(raw_pdf), prefer_css_page_size=True, print_background=True,
                     display_header_footer=True, header_template="<span></span>",
                     footer_template='<div style="font:8px Arial; width:100%; margin:0 17mm; '
                     'display:flex; justify-content:space-between"><span>Deep Learning 10 · '
                     'Laboratorio 9</span><span class="pageNumber"></span></div>')
            browser.close()
        writer = PdfWriter(clone_from=raw_pdf)
        writer.add_metadata({"/Title": "Laboratorio 9 — Informe completo (Tasks 1–4)",
                             "/Author": AUTHORS})
        writer.write(output)
    print(f"Informe generado: {output} ({len(PdfReader(output).pages)} páginas, {figures} figuras)")


if __name__ == "__main__":
    main()
