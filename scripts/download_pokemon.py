"""Descarga y prepara los 898 sprites de Pokemon usados en la HDT2.

Los sprites se obtienen del repositorio publico de PokeAPI, se convierten a
RGB, se redimensionan a 64x64 con vecino mas cercano y se guardan con nombres
estables (001.png, ..., 898.png).

"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


URL_TEMPLATE = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
    "sprites/pokemon/{pokemon_id}.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y normaliza sprites de Pokemon para la HDT2."
    )
    parser.add_argument("--output", type=Path, default=Path("data/pokemon"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=898)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_image(
    image_bytes: bytes,
    size: int = 64,
    background: str = "black",
) -> Image.Image:
    """Convierte un PNG, incluyendo transparencia, a una imagen RGB cuadrada."""
    if size <= 0:
        raise ValueError("El tamano de salida debe ser positivo.")

    with Image.open(io.BytesIO(image_bytes)) as source:
        rgba = source.convert("RGBA")

    color = (0, 0, 0, 255) if background == "black" else (255, 255, 255, 255)
    canvas = Image.new("RGBA", rgba.size, color)
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB").resize((size, size), Image.Resampling.NEAREST)


def fetch_bytes(url: str, retries: int, timeout: float) -> bytes:
    """Descarga bytes con reintentos y un User-Agent identificable."""
    request = Request(url, headers={"User-Agent": "CC3092-HDT2-DCGAN/1.0"})
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(0.5 * attempt)

    raise RuntimeError(f"No se pudo descargar {url}: {last_error}")


def validate_saved_image(path: Path, size: int) -> None:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (size, size):
            raise ValueError(
                f"Imagen invalida {path}: mode={image.mode}, size={image.size}"
            )


def download_one(
    pokemon_id: int,
    output_dir: Path,
    size: int,
    background: str,
    retries: int,
    timeout: float,
    overwrite: bool,
) -> tuple[int, str]:
    destination = output_dir / f"{pokemon_id:03d}.png"

    if destination.exists() and not overwrite:
        try:
            validate_saved_image(destination, size)
            return pokemon_id, "skipped"
        except (OSError, ValueError):
            pass

    image_bytes = fetch_bytes(URL_TEMPLATE.format(pokemon_id=pokemon_id), retries, timeout)
    image = prepare_image(image_bytes, size=size, background=background)

    temporary = destination.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(destination)
    validate_saved_image(destination, size)
    return pokemon_id, "downloaded"


def main() -> int:
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise SystemExit("El rango solicitado no es valido.")
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("workers y retries deben ser mayores que cero.")

    args.output.mkdir(parents=True, exist_ok=True)
    pokemon_ids = list(range(args.start, args.end + 1))
    downloaded = 0
    skipped = 0
    failures: list[tuple[int, str]] = []

    print(
        f"Procesando {len(pokemon_ids)} sprites en {args.output.resolve()} "
        f"({args.size}x{args.size}, fondo {args.background})"
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                pokemon_id,
                args.output,
                args.size,
                args.background,
                args.retries,
                args.timeout,
                args.overwrite,
            ): pokemon_id
            for pokemon_id in pokemon_ids
        }

        for completed, future in enumerate(as_completed(futures), start=1):
            pokemon_id = futures[future]
            try:
                _, status = future.result()
                downloaded += status == "downloaded"
                skipped += status == "skipped"
            except Exception as error:  # muestra todos los fallos al final
                failures.append((pokemon_id, str(error)))

            if completed % 50 == 0 or completed == len(futures):
                print(f"Progreso: {completed}/{len(futures)}")

    print(f"Descargados: {downloaded} | existentes: {skipped} | fallidos: {len(failures)}")
    for pokemon_id, error in sorted(failures):
        print(f"  - Pokemon {pokemon_id}: {error}")

    expected = args.end - args.start + 1
    valid_files = list(args.output.glob("*.png"))
    print(f"PNG disponibles: {len(valid_files)} (esperados para este rango: {expected})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
