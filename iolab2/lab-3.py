import numpy as np
import zlib
import struct


def generate_spectrum(colors, width=600, height=50):
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    num_colors = len(colors)

    for i in range(width):
        t = i / (width - 1)
        idx = int(t * (num_colors - 1))
        next_idx = min(idx + 1, num_colors - 1)
        local_t = (t * (num_colors - 1)) - idx
        color = tuple(int((1 - local_t) * colors[idx][j] + local_t * colors[next_idx][j]) for j in range(3))
        gradient[:, i] = color

    return gradient


def generate_spectrum_png(filename, colors, width, height):
    # Tworzenie danych obrazu
    num_colors = len(colors)
    raw_data = bytearray()
    for _ in range(height):
        raw_data.append(0)  # Filtr "zero"
        for i in range(width):
            t = i / (width - 1)
            idx = int(t * (num_colors - 1))
            next_idx = min(idx + 1, num_colors - 1)
            local_t = (t * (num_colors - 1)) - idx
            color = tuple(int((1 - local_t) * colors[idx][j] + local_t * colors[next_idx][j]) for j in range(3))
            raw_data.extend(color)

    # Kompresja danych obrazu
    compressed_data = zlib.compress(raw_data)

    # Tworzenie nagłówka PNG
    png_signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(
        "!IIBBBBB", width, height, 8, 2, 0, 0, 0
    )  # 8-bit, RGB, bez kompresji dodatkowej, bez filtrów, bez interlace
    ihdr_chunk = create_png_chunk(b"IHDR", ihdr_data)
    idat_chunk = create_png_chunk(b"IDAT", compressed_data)
    iend_chunk = create_png_chunk(b"IEND", b"")

    # Zapis pliku PNG
    with open(filename, 'wb') as f:
        f.write(png_signature)
        f.write(ihdr_chunk)
        f.write(idat_chunk)
        f.write(iend_chunk)


def create_png_chunk(chunk_type, data):
    chunk_length = struct.pack("!I", len(data))
    chunk_crc = struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return chunk_length + chunk_type + data + chunk_crc


# Parametry obrazu
width = 1800
height = 200
colors_1_8 = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255),(255,255,255), (0,0,0)]
rgb_spectrum = generate_spectrum(colors_1_8,width,height)

from pathlib import Path
downloads_path = str(Path.home() / "Downloads")

save_spectrum_png_path = f"{downloads_path}/rgb_spectrum.png"
print(f"Obraz zostal zapisany w {save_spectrum_png_path}")
generate_spectrum_png(save_spectrum_png_path,colors_1_8, width, height)
