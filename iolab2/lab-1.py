import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def plot(data, title, rows=2, cols=2):
    plot.i += 1
    plt.subplot(rows,cols,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis("off")
plt.figure(figsize=(12,10))
plot.i = 0


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


# Tworzenie testowego obrazu (szkic RGB) w OpenCV
# image = np.zeros((100, 200, 3), dtype=np.uint8)
# image[:, :100] = (255, 0, 0)  # Połowa obrazu na czerwono
# image[:, 100:] = (0, 255, 0)  # Druga połowa na zielono
image = Image.open("sunflower.png").convert('RGB')
im_data = np.array(image, dtype=np.uint8)

# Zapisywanie w PPM
from pathlib import Path
downloads_path = str(Path.home() / "Downloads")
save_p3_path = f"{downloads_path}/sunflower_p3.ppm"
save_p6_path = f"{downloads_path}/sunflower_p6.ppm"
save_ppm_p3(save_p3_path, image)
save_ppm_p6(save_p6_path, image)

# Odczyt i weryfikacja
img_p3 = load_ppm_p3(save_p3_path)
img_p6 = load_ppm_p6(save_p6_path)
print(f"Obrazy zostaly zapisane w folderze {downloads_path}")

# Porównanie rozmiarów plików
size_p3 = os.path.getsize(save_p3_path)
size_p6 = os.path.getsize(save_p6_path)
print(f"Rozmiar P3: {size_p3} bajtów \n Rozmiar P6: {size_p6} bajtów")

# Wyświetlenie obrazów
plot(im_data, "Oryginalny")
plot(img_p3, "P3")
plot(img_p6, "P6")

plt.show()
