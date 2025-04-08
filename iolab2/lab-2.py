import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def plot(data, title, rows=2, cols=2):
    plot.i += 1
    plt.subplot(rows,cols,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis("off")
plt.figure(figsize=(12,5))
plot.i = 0

def save_ppm_p6(filename, image):
    """Zapisuje obraz w formacie P6 (binarny PPM)."""
    height, width, _ = image.shape
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        image.tofile(f)

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


width = 1800
height = 200
colors_1_8 = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255),(255,255,255), (0,0,0)]
from pathlib import Path
downloads_path = str(Path.home() / "Downloads")
save_p6_spectrum_path = f"{downloads_path}/rgb_spectrum.ppm"
print(f"Obraz spectrum zostal zapisany w {save_p6_spectrum_path}")
generate_spectrum_p6(save_p6_spectrum_path,colors_1_8,width,height)
rgb_spectrum = generate_spectrum(colors_1_8,width,height)
image_from_p6 = load_ppm_p6(save_p6_spectrum_path)
# Wyświetlenie obrazu
plot(rgb_spectrum, "RGB spectrum",2,1)
plot(image_from_p6, "RGB spectrum z formatu PPM P6",2,1)
plt.show()

