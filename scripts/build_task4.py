"""Construye y ejecuta la Task 4 sin volver a entrenar la difusión."""

from pathlib import Path
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main():
    # Copiamos las definiciones originales, no sus salidas ni el entrenamiento de los tres betas.
    task1 = nbf.read(ROOT / "notebooks/01_vae_fashion_mnist.ipynb", as_version=4)
    vae_source = next(c.source for c in task1.cells if c.source.startswith("class VAE("))
    vae_source = vae_source.split("\nmodel_check =")[0]
    train_source = next(c.source for c in task1.cells if c.source.startswith("def run_epoch("))
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.cells = [
        md(r"""
        # Laboratorio 9 — Task 4: comparación generativa y pass@k

        **Curso:** Deep Learning 10

        **Autor:** Felipe Aguilar

        Reutilizamos los pesos y las 500 muestras con $w=1$ de la Task 3.4.
        Como no se incluyeron pesos del VAE en el repositorio, repetimos únicamente
        el entrenamiento de $\beta=1$: misma arquitectura, semilla 3045, 20 épocas,
        Adam con lr=0.001 y lotes de 128. Es una nueva ejecución, no los pesos
        originales de la Task 1. Las definiciones del VAE y su entrenamiento se
        copian directamente del notebook 01 al construir este documento.
        """),
        code("""
        from pathlib import Path
        import json
        import math
        import platform
        import random
        import statistics
        import sys
        import time

        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
        from IPython.display import Markdown, display

        ROOT = Path.cwd()
        if ROOT.name == "notebooks":
            ROOT = ROOT.parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.difusion import UNet, linear_schedule, sample
        from src.evaluacion import horizontal_sharpness, pass_at_k

        SEED, LATENT_DIM, BATCH_SIZE, EPOCHS, LR = 3045, 2, 128, 20, 1e-3
        # CPU para el VAE, como en la Task 1; ambos tiempos se miden en este dispositivo.
        DEVICE = torch.device("cpu")
        torch.set_num_threads(4)
        DATA_DIR = ROOT / "data"
        CHECKPOINT_DIR = ROOT / "artifacts/task4"
        OUTPUT_DIR = ROOT / "outputs/task4"
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        def set_seed(seed=SEED):
            random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        set_seed()
        print(f"Semilla: {SEED} | Dispositivo: {DEVICE} | Hilos: {torch.get_num_threads()}")
        print(f"Sistema: {platform.platform()} | PyTorch: {torch.__version__}")
        """),
        md("## Preparación del VAE de β=1"),
        code("""
        full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True,
                                          transform=transforms.ToTensor())
        train_set = Subset(full_train, range(55_000))
        val_set = Subset(full_train, range(55_000, 60_000))

        def make_train_loader(seed=SEED):
            return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)

        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Entrenamiento: {len(train_set)} | Validación: {len(val_set)}")
        """),
        code(vae_source),
        code(train_source),
        code("""
        weights_path = CHECKPOINT_DIR / "vae_beta_1.0.pt"
        history_path = CHECKPOINT_DIR / "historial_vae.json"
        if weights_path.exists() and history_path.exists():
            vae = VAE().to(DEVICE)
            vae.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            history = json.loads(history_path.read_text())
        else:
            vae, history = train_one_beta(1.0)
            history_path.write_text(json.dumps(history, indent=2) + "\\n")
        vae.eval()
        print(f"Épocas: {len(history['val_loss'])} | Reconstrucción final: {history['val_recon'][-1]:.4f}"
              f" | KL final: {history['val_kl'][-1]:.4f}")
        """),
        md(r"""
        ## Task 4.1(a) — Nitidez de 500 imágenes por familia

        Medimos $\operatorname{media}|x_{i,j+1}-x_{i,j}|$ con todos los píxeles en
        $[0,1]$. Las muestras de difusión guardadas están cuantizadas a uint8:
        dividimos entre 255, **no** comparamos directamente su escala con la del VAE.
        La cuantización introduce como máximo $1/255$ de diferencia en esta métrica
        respecto a las muestras flotantes originales. Usamos las primeras 500
        imágenes de validación como referencia real y 500 muestras del prior del VAE.
        """),
        code("""
        saved = torch.load(ROOT / "artifacts/task3/muestras_cfg.pt", map_location="cpu", weights_only=True)
        diffusion_images = saved["w1"].float() / 255
        assert diffusion_images.shape == (500, 1, 28, 28)
        assert torch.equal(torch.bincount(saved["labels"]), torch.full((10,), 50))
        real_images = torch.stack([val_set[i][0] for i in range(500)])
        with torch.no_grad():
            z = torch.randn(500, LATENT_DIM, generator=torch.Generator().manual_seed(SEED))
            vae_images = vae.decode(z)
        groups = {"Reales": real_images, "VAE β=1": vae_images, "Difusión w=1": diffusion_images}
        sharpness = {name: horizontal_sharpness(imgs).item() for name, imgs in groups.items()}
        for name, value in sharpness.items():
            print(f"{name:15s} | n=500 | nitidez={value:.6f}")

        fig, axes = plt.subplots(3, 10, figsize=(12, 4))
        for row, (name, imgs) in enumerate(groups.items()):
            # Difusión está ordenada por clase: mostramos una imagen por clase.
            indices = range(0, 500, 50) if name == "Difusión w=1" else range(10)
            for col, idx in enumerate(indices):
                axes[row, col].imshow(imgs[idx, 0], cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if col == 0:
                    axes[row, col].set_ylabel(name)
        fig.suptitle("Muestras usadas en la comparación de nitidez")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "comparacion.png", dpi=150)
        plt.show()
        """),
        md(r"""
        ## Task 4.1(b) — Tiempo de generación de 100 imágenes

        «Sin guía» significa aquí $w=1$, la predicción condicional pura de la Task 3.3,
        sin una segunda evaluación incondicional. Ambos modelos usan CPU, float32,
        cuatro hilos y un lote de 100. Medimos la mediana de tres repeticiones después
        de un calentamiento. Incluimos el sorteo del ruido y el muestreo; excluimos carga
        de pesos, entrenamiento, gráficos y escritura. Son tiempos locales, no los
        tiempos de la GPU usados en la Task 3.

        El VAE evalúa su decoder una vez por lote (una vez por imagen). La difusión
        evalúa la U-Net 1000 veces por lote (1000 veces por imagen); no son 1000
        llamadas adicionales por cada imagen porque las 100 se procesan juntas.
        """),
        code("""
        unet = UNet().to(DEVICE).eval()
        unet.load_state_dict(torch.load(ROOT / "artifacts/task3/unet_cfg.pt",
                                       map_location=DEVICE, weights_only=True))
        betas, alphas, alpha_bars = linear_schedule()
        labels = torch.arange(10, device=DEVICE).repeat_interleave(10)

        @torch.no_grad()
        def generate_vae(generator):
            return vae.decode(torch.randn(100, LATENT_DIM, device=DEVICE, generator=generator))

        @torch.no_grad()
        def generate_diffusion(generator):
            return sample(unet, labels, 1, betas, alphas, alpha_bars, generator=generator)

        # Calentamiento con las mismas formas de lote; no forma parte de los tiempos.
        with torch.no_grad():
            generate_vae(torch.Generator(device=DEVICE).manual_seed(SEED))
            unet(torch.zeros(100, 1, 28, 28, device=DEVICE),
                 torch.ones(100, dtype=torch.long, device=DEVICE), labels)

        timings = {}
        for name, generate in [("VAE", generate_vae), ("Difusión", generate_diffusion)]:
            durations = []
            for repeat in range(3):
                generator = torch.Generator(device=DEVICE).manual_seed(SEED + repeat)
                start = time.perf_counter()
                generated = generate(generator)
                durations.append(time.perf_counter() - start)
                assert generated.shape == (100, 1, 28, 28) and torch.isfinite(generated).all()
                print(f"{name}: repetición {repeat+1}, {durations[-1]:.6f} s", flush=True)
            timings[name] = {"repeticiones_s": durations, "mediana_s": statistics.median(durations)}
        ratio = timings["Difusión"]["mediana_s"] / timings["VAE"]["mediana_s"]
        print(f"Razón difusión/VAE: {ratio:.1f} veces")
        """),
        md("## Task 4.1(c) — Compensaciones observadas"),
        code("""
        analysis = (
            f"El VAE obtuvo nitidez **{sharpness['VAE β=1']:.6f}**, frente a "
            f"**{sharpness['Difusión w=1']:.6f}** de difusión y **{sharpness['Reales']:.6f}** "
            f"de las imágenes reales. La difusión conserva más variación local; esta métrica "
            f"también puede aumentar por ruido y no demuestra por sí sola calidad semántica. "
            f"En la figura, el VAE suaviza contornos y texturas, mientras la difusión conserva "
            f"más detalle. El VAE tarda **{timings['VAE']['mediana_s']:.6f} s** por 100 imágenes "
            f"y la difusión **{timings['Difusión']['mediana_s']:.3f} s**: "
            f"**{ratio:.1f} veces** más tiempo, con 1 frente a 1000 evaluaciones por lote.\\n\\n"
            f"El VAE prioriza velocidad a costa de nitidez: comprime a dos dimensiones y "
            f"su reconstrucción cuadrática favorece promedios. Su pérdida de validación pasó "
            f"de **{history['val_loss'][0]:.4f}** a **{history['val_loss'][-1]:.4f}**. "
            f"En la Task 3 la pérdida de difusión pasó de **0.1250** a **0.0408** en 30 épocas. "
            f"Ambas corridas convergieron sin un entrenamiento adversarial, pero sus pérdidas "
            f"tienen escalas distintas y una sola semilla no prueba estabilidad general.\\n\\n"
            f"Para acelerar difusión implementaría un muestreador de pasos reducidos, por "
            f"ejemplo DDIM con 50 pasos, y volvería a medir nitidez y fidelidad. Eso reduciría "
            f"las evaluaciones teóricas de 1000 a 50; no basta con truncar los últimos pasos "
            f"del muestreador ancestral ni implica una aceleración medida de 20 veces."
        )
        display(Markdown(analysis))
        """),
        md(r"""
        ## Task 4.2(a) — Estimador insesgado

        El fracaso significa escoger $k$ soluciones entre las $n-c$ incorrectas, sin
        reemplazo. Por eso $\mathrm{pass@}k=1-\binom{n-c}{k}/\binom nk$.
        Usamos un producto de cocientes en float64, sin factoriales grandes.

        Para P2, $n=20$, $c=2$, $k=5$:

        $$\binom{18}{5}=8568,\qquad\binom{20}{5}=15504,$$
        $$\mathrm{pass@}5=1-\frac{8568}{15504}=\frac{17}{38}=0.4473684211.$$
        """),
        code("""
        counts = [0, 2, 5, 9, 14, 20]
        ks = [1, 5, 10]
        values = torch.tensor([[pass_at_k(20, c, k) for k in ks] for c in counts], dtype=torch.float64)
        naive = torch.tensor([1 - (1 - c / 20) ** 5 for c in counts], dtype=torch.float64)
        print("Problema | c | pass@1   | pass@5   | pass@10  | ingenuo@5 | diferencia@5")
        for i, c in enumerate(counts):
            print(f"P{i+1:1d}       | {c:2d}| {values[i,0]:.6f} | {values[i,1]:.6f} | "
                  f"{values[i,2]:.6f} | {naive[i]:.6f}  | {values[i,1]-naive[i]:.6f}")
        print("Promedios:", ", ".join(f"pass@{k}={v:.6f}" for k, v in zip(ks, values.mean(0))))
        min_k = next(k for k in range(1, 21) if pass_at_k(20, 2, k) > 0.9)
        print(f"P2: menor k con pass@k > 0.9: {min_k}")
        print(f"P2: pass@13={pass_at_k(20,2,13):.6f}, pass@14={pass_at_k(20,2,14):.6f}")
        assert torch.all(naive <= values[:, 1] + 1e-15)
        """),
        md(r"""
        ## Task 4.2(b) — Umbral para P2

        Al cancelar los factoriales, para $1\le k\le18$:

        $$\frac{\binom{18}{k}}{\binom{20}{k}}
        =\frac{18!\,(20-k)!}{20!\,(18-k)!}
        =\frac{(20-k)(19-k)}{380}.$$

        Para $k=19,20$ no pueden elegirse tantas soluciones incorrectas y pass@k vale 1;
        la misma expresión también da 1. Exigir pass@k > 0.9 equivale a
        $(20-k)(19-k)<38$. Con $k=13$ el producto es 42 y pass@13 = 0.889474;
        con $k=14$ es 30 y pass@14 = 0.921053. Como pass@k es monótono, **14 es el mínimo**.

        ## Task 4.2(c) — Comparación con la versión ingenua

        Si $n-c\ge k$, cada factor no negativo cumple

        $$\frac{n-c-i}{n-i}=1-\frac{c}{n-i}\le1-\frac cn.$$

        Al multiplicar para $i=0,\ldots,k-1$ y restar de 1:

        $$1-\prod_i\frac{n-c-i}{n-i}\ge1-(1-c/n)^k.$$

        Si $n-c<k$, el estimador insesgado vale 1 y la desigualdad también se cumple.
        En la tabla, P2 da 0.447368 frente a 0.409510; P3 da 0.806308 frente a
        0.762695. La versión ingenua nunca excede a la insesgada. Modela intentos
        independientes con reemplazo y trata c/n como una probabilidad exacta;
        no es el estimador insesgado a partir de las 20 soluciones observadas.

        ## Task 4.2(d) — Agente con hasta diez intentos

        No se puede afirmar que cada problema se resolverá a partir de una probabilidad.
        Bajo el muestreo sin reemplazo de las 20 soluciones, **P1 no se resuelve** (0),
        **P2** tiene probabilidad **0.763158**, **P3** **0.983746**, **P4** **0.999940**,
        y **P5 y P6** tienen probabilidad **1**: hay menos de diez fallos disponibles.
        Un verificador perfecto reconoce una solución correcta, pero no la genera.
        Estas garantías del conjunto finito no garantizan el éxito de diez nuevas
        generaciones independientes o de reintentos correlacionados con retroalimentación.
        """),
        code("""
        means = values.mean(0)
        conclusion = (f"El promedio aumenta de **{means[0]:.6f}** en pass@1 a **{means[2]:.6f}** "
                      f"en pass@10 (**{100*(means[2]-means[0]):.2f} puntos porcentuales**). "
                      "El modelo produce soluciones útiles que no siempre aparecen al primer intento; "
                      "muestrear y verificar ayuda, pero no corrige P1, donde no se observó ningún éxito.")
        display(Markdown(conclusion))
        results = {
            "semilla": SEED, "dispositivo": str(DEVICE), "sistema": platform.platform(),
            "torch": str(torch.__version__), "hilos": torch.get_num_threads(),
            "nitidez": sharpness, "tiempos": timings, "razon_tiempos": ratio,
            "evaluaciones_por_lote": {"VAE": 1, "Difusión": 1000},
            "analisis": analysis, "pass_at_k": values.tolist(), "ingenuo_5": naive.tolist(),
            "promedios": means.tolist(), "minimo_k_p2": min_k, "conclusion": conclusion,
        }
        (CHECKPOINT_DIR / "resultados.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\\n")
        """),
    ]
    target = ROOT / "notebooks/04_comparacion_passk.ipynb"
    nbf.write(notebook, target)
    NotebookClient(notebook, timeout=7200, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, target)
    print(f"Notebook ejecutado: {target}")


if __name__ == "__main__":
    main()
