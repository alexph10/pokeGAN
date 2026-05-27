"""Download every Pokémon's official artwork as a clean, consistently sized
RGB JPEG, ready for GAN training.

Source:  PokeAPI sprite repository (the same images served by pokemondb /
         Bulbapedia "official artwork"). Each image is the high-resolution
         475 x 475 transparent PNG produced by Game Freak / The Pokémon Company.

         https://github.com/PokeAPI/sprites
         https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/<id>.png

Output:
    input/pokemon_dataset/img_<id>.jpg   (RGB, white background, square)

Usage:
    python scrape.py                                  # download all (default 1025)
    python scrape.py --max-id 1025 --size 256         # custom range + size
    python scrape.py --workers 16 --out input/pokemon_dataset
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

# Public sprite mirror — high quality 475 x 475 transparent PNGs.
ARTWORK_URL = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
    "sprites/pokemon/other/official-artwork/{id}.png"
)

# Fallback: PokeAPI HOME renders (slightly different style, also 512 x 512).
HOME_URL = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/"
    "sprites/pokemon/other/home/{id}.png"
)

# Total Pokémon through Generation IX (Scarlet/Violet + DLC).
DEFAULT_MAX_ID = 1025


def fetch(url: str, retries: int = 4, timeout: int = 20) -> bytes | None:
    """GET a URL with simple exponential backoff. Return content or None."""
    delay = 1.0
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(delay)
        delay *= 2
    return None


def to_square_rgb(png_bytes: bytes, size: int, bg=(255, 255, 255)) -> Image.Image:
    """Composite a transparent PNG onto a white background, pad to square,
    resize to (size, size) using high-quality LANCZOS resampling.
    """
    src = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # Crop tight to non-transparent pixels so all sprites have similar scale.
    bbox = src.getbbox()
    if bbox is not None:
        src = src.crop(bbox)

    # Pad to square.
    w, h = src.size
    side = max(w, h)
    square = Image.new("RGBA", (side, side), (bg[0], bg[1], bg[2], 0))
    square.paste(src, ((side - w) // 2, (side - h) // 2), src)

    # Composite onto white background → RGB.
    bg_img = Image.new("RGB", square.size, bg)
    bg_img.paste(square, mask=square.split()[3])

    # Resize to target.
    return bg_img.resize((size, size), Image.LANCZOS)


def download_one(pid: int, out_dir: str, size: int, quality: int,
                 overwrite: bool) -> tuple[int, str]:
    """Download a single Pokémon by national-dex id. Returns (id, status)."""
    out_path = os.path.join(out_dir, f"img_{pid:04d}.jpg")
    if os.path.exists(out_path) and not overwrite:
        return pid, "skip"

    data = fetch(ARTWORK_URL.format(id=pid))
    source = "artwork"
    if data is None:
        data = fetch(HOME_URL.format(id=pid))
        source = "home"
    if data is None:
        return pid, "missing"

    try:
        img = to_square_rgb(data, size=size)
        img.save(out_path, format="JPEG", quality=quality, optimize=True)
    except Exception as e:
        return pid, f"error: {e}"
    return pid, f"ok ({source})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all Pokémon as JPEGs.")
    parser.add_argument("--out", default="input/pokemon_dataset",
                        help="Output directory.")
    parser.add_argument("--max-id", type=int, default=DEFAULT_MAX_ID,
                        help="Highest national-dex id to download (default: 1025).")
    parser.add_argument("--min-id", type=int, default=1,
                        help="Lowest national-dex id to download (default: 1).")
    parser.add_argument("--size", type=int, default=256,
                        help="Output image size in pixels (square). Default 256.")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality (1-95). Default 95.")
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of concurrent download threads.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if the file already exists.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = list(range(args.min_id, args.max_id + 1))
    total = len(ids)
    print(f"Downloading {total} Pokemon -> {args.out} "
          f"({args.size}x{args.size} JPEG, {args.workers} workers)")

    ok = skipped = missing = errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, pid, args.out, args.size,
                        args.quality, args.overwrite): pid
            for pid in ids
        }
        for i, fut in enumerate(as_completed(futures), 1):
            pid, status = fut.result()
            if status.startswith("ok"):
                ok += 1
            elif status == "skip":
                skipped += 1
            elif status == "missing":
                missing += 1
                print(f"  #{pid:04d}: MISSING from both sources", flush=True)
            else:
                errors += 1
                print(f"  #{pid:04d}: {status}", flush=True)
            if i % 25 == 0 or i == total:
                elapsed = time.time() - start
                print(f"  [{i:>4}/{total}] ok={ok} skip={skipped} "
                      f"missing={missing} err={errors} "
                      f"({elapsed:.1f}s)", flush=True)

    print("\nDone.")
    print(f"  downloaded : {ok}")
    print(f"  skipped    : {skipped}")
    print(f"  missing    : {missing}")
    print(f"  errors     : {errors}")
    print(f"  output dir : {args.out}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
