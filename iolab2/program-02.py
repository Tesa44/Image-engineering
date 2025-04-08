import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import zlib
import struct
import cv2
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")

#Funkcje pomocnicze
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # Jeśli program jest uruchamiany jako EXE
        base_path = sys._MEIPASS
    else:
        # Tryb zwykły (skrypt)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def plot(data, title, rows=2, cols=2):
    plot.i += 1
    plt.subplot(rows,cols,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis("off")

def save_ppm_p3(filename, image):
    """Zapisuje obraz w formacie P3 (tekstowy PPM)."""
    im_data = np.array(image,dtype=np.uint8)
    height, width, _ = im_data.shape
    with open(filename, 'w') as f:
        f.write(f"P3\n{width} {height}\n255\n") # pierwszy wiersz P3. Drugi ilość kolumn i wierszy. Trzeci max wartość RGB
        for i in range(height):
            for j in range(width):
                f.write(f"{im_data[i][j][0]} {im_data[i][j][1]} {im_data[i][j][2]}  ")
            f.write("\n")


def save_ppm_p6(filename, image):
    """Zapisuje obraz w formacie P6 (binarny PPM)."""
    im_data = np.array(image,dtype=np.uint8)
    height, width, _ = im_data.shape
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        im_data.tofile(f)


def load_ppm_p3(filename):
    """Wczytuje obraz z formatu P3."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'P3':
        print("Niepoprawny format PPM")
        return None
    width, height = map(int, lines[1].split())
    max_val = int(lines[2])
    if max_val > 255:
        print("Format obsługuje tylko 8-bitowe kolory")
        return None
    pixels = list(map(int, " ".join(lines[3:]).split()))
    image_from_p3 = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
    return image_from_p3


def load_ppm_p6(filename):
    """Wczytuje obraz z formatu P6."""
    with open(filename, 'rb') as f:
        header = f.readline().strip()   # Pierwsza linia z formatem
        if header != b"P6":
            print("Niepoprawny format PPM")
            return None

        dimensions = f.readline().strip()   # Druga linia z rozmiarem
        width, height = map(int, dimensions.split())
        max_val = int(f.readline().strip())
        if max_val > 255:
            print("Format obsługuje tylko 8-bitowe kolory")
            return None
        image_from_p6 = np.fromfile(f, dtype=np.uint8).reshape((height, width, 3))  # Tworzy tablicę z pliku binarnego
    return image_from_p6

def generate_spectrum(colors, width=600, height=50):
    '''Funkcja do generowania tablicy kolorów, które płynnie przechodzą przez kolejne wierzchołki sześcianu barw RGB'''
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

def generate_spectrum_p6(filename, colors,width=600, height=50):
    '''Funkcja do generowania tablicy kolorów przejść, które zostają zapisywane do formatu ppm P6'''
    num_colors = len(colors)
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        for _ in range(height):
            for i in range(width):
                t = i / (width - 1)
                idx = int(t * (num_colors - 1))
                next_idx = min(idx + 1, num_colors - 1)
                local_t = (t * (num_colors - 1)) - idx
                color = tuple(int((1 - local_t) * colors[idx][j] + local_t * colors[next_idx][j]) for j in range(3))
                f.write(bytes(color))


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

def compress_data(data):
    return zlib.compress(data.tobytes())

def convert_color_space(image, conversion):
    return cv2.cvtColor(image, conversion)


def resample_channel(channel, factor, mode="down"):
    if factor == 1:
        return channel
    if mode == "down":
        return channel[::factor, ::factor]
    return cv2.resize(channel, (channel.shape[1] * factor, channel.shape[0] * factor), interpolation=cv2.INTER_NEAREST)

def jpeg_simulation(image_path, sampling_factor):
    image = convert_color_space(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ycbcr = convert_color_space(image, cv2.COLOR_RGB2YCrCb)

    Y, Cb, Cr = [ycbcr[:, :, i] for i in range(3)]
    Cb_ds, Cr_ds = [resample_channel(ch, sampling_factor, "down") for ch in (Cb, Cr)]
    Cb_us, Cr_us = [resample_channel(ch, sampling_factor, "up") for ch in (Cb_ds, Cr_ds)]

    reconstructed_ycbcr = np.stack((Y, Cb_us, Cr_us), axis=-1)
    compressed_image = convert_color_space(reconstructed_ycbcr, cv2.COLOR_YCrCb2RGB)

    compressed_sizes = sum(len(compress_data(ch)) for ch in (Y, Cb_ds, Cr_ds))
    print(f"Compressed size ({sampling_factor}x sampling): {compressed_sizes} bytes")

    return compressed_image, compressed_sizes


def print_menu():
    print("Wybierz jedna z dostepnych pozycji menu")
    print("1. Zadanie 1 - Generowanie obrazu w formacie PPM P3 i P6")
    print("2. Zadanie 2 - Generowanie spectrum rgb w formacie ppm p6")
    print("3. Zadanie 3 - Generowanie spectrum rgb w formacie PNG")
    print("4. Zadanie 4 - Generowanie spectrum rgb w formacie JPEG")
    print("0. - Exit")

print("Inzynieria Obrazow - Laboratorium 1")
print("Autor: Mateusz Tesarewicz 272909")
choice = -1
im = Image.open(resource_path('sunflower.png'))
im_spectrum_path = resource_path("rgb_spectrum.png")
while choice != 0:
    plot.i = 0
    print_menu()
    choice = int(input())
    if choice == 1:
        im_rgb = im.convert("RGB")

        im_data = np.array(im_rgb, dtype=np.uint8)
        downloads_path = str(Path.home() / "Downloads")
        save_p3_path = f"{downloads_path}/sunflower_p3.ppm"
        save_p6_path = f"{downloads_path}/sunflower_p6.ppm"
        save_ppm_p3(save_p3_path, im_rgb)
        save_ppm_p6(save_p6_path, im_rgb)

        # Odczyt i weryfikacja
        img_p3 = load_ppm_p3(save_p3_path)
        img_p6 = load_ppm_p6(save_p6_path)
        print(f"Obrazy zostaly zapisane w folderze {downloads_path}")

        # Porównanie rozmiarów plików
        size_p3 = os.path.getsize(save_p3_path)
        size_p6 = os.path.getsize(save_p6_path)
        print(f"Rozmiar P3: {size_p3} bajtów \n Rozmiar P6: {size_p6} bajtów")

        # Wyświetlenie obrazów
        plt.figure(figsize=(12, 10))
        plot(im_data, "Oryginalny")
        plot(img_p3, "P3")
        plot(img_p6, "P6")

        plt.show()

    elif choice == 2:
        width = 1800
        height = 200
        colors_1_8 = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255),
                      (255, 255, 255), (0, 0, 0)]
        downloads_path = str(Path.home() / "Downloads")
        save_p6_spectrum_path = f"{downloads_path}/rgb_spectrum.ppm"
        print(f"Obraz spectrum zostal zapisany w {save_p6_spectrum_path}")
        generate_spectrum_p6(save_p6_spectrum_path, colors_1_8, width, height)
        rgb_spectrum = generate_spectrum(colors_1_8, width, height)
        image_from_p6 = load_ppm_p6(save_p6_spectrum_path)
        # Wyświetlenie obrazu
        plt.figure(figsize=(12, 5))
        plot(rgb_spectrum, "RGB spectrum", 2, 1)
        plot(image_from_p6, "RGB spectrum z formatu PPM P6", 2, 1)
        plt.show()

    elif choice == 3:
        width = 1800
        height = 200
        colors_1_8 = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255),
                      (255, 255, 255), (0, 0, 0)]
        rgb_spectrum = generate_spectrum(colors_1_8, width, height)

        downloads_path = str(Path.home() / "Downloads")
        save_spectrum_png_path = f"{downloads_path}/rgb_spectrum.png"
        print(f"Obraz zostal zapisany w {save_spectrum_png_path}")
        generate_spectrum_png(save_spectrum_png_path, colors_1_8, width, height)

    elif choice == 4:
        sampling_factors = [1, 2, 4]
        results = [jpeg_simulation(im_spectrum_path, factor) for factor in sampling_factors]

        plt.figure(figsize=(12, 10))
        for idx, (img, size) in enumerate(results, 1):
            plt.subplot(len(sampling_factors), 1, idx)
            plt.axis("off")
            plt.imshow(img)
            plt.title(f"Sampling {sampling_factors[idx - 1]}x (Size: {size} bytes)")

        plt.tight_layout()
        plt.show()

    elif choice == 0:
        break
    else:
        print("Niepoprawny wybór")