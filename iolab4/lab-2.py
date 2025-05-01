from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def find_closest_palette_color(pixel,k=2):
    n = 255 // (k-1)
    return round(pixel / n) * n

#
# def find_closest_palette_color(value):
#     return 0 if value < 128 else 255

def floyd_steinberg_dithering_rgb(image_path, output_path,k):
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image, dtype=np.float32)

    height, width, _ = pixels.shape

    # Kopia do wygenerowania histogramu przed ditheringiem
    original_pixels = pixels.copy()

    for y in range(height):
        for x in range(width):
            for c in range(3):  # R, G, B
                old_pixel = pixels[y, x, c]
                new_pixel = find_closest_palette_color(old_pixel,k)
                quant_error = old_pixel - new_pixel
                pixels[y, x, c] = new_pixel

                if x + 1 < width:
                    pixels[y, x + 1, c] += quant_error * 7 / 16
                if y + 1 < height:
                    if x > 0:
                        pixels[y + 1, x - 1, c] += quant_error * 3 / 16
                    pixels[y + 1, x, c] += quant_error * 5 / 16
                    if x + 1 < width:
                        pixels[y + 1, x + 1, c] += quant_error * 1 / 16

    # Ograniczamy wartości do zakresu [0, 255]
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    # Zapisujemy obraz wynikowy
    dithered_image = Image.fromarray(pixels, mode="RGB")
    dithered_image.save(output_path)

    # Tworzymy histogramy przed i po ditheringu
    plot_color_histograms(original_pixels.astype(np.uint8), pixels)

def plot_color_histograms(before_pixels, after_pixels):
    colors = ['Red', 'Green', 'Blue']
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle("Histogramy kolorów: przed i po ditheringu", fontsize=16)

    for i, color in enumerate(colors):
        axes[i, 0].hist(before_pixels[..., i].flatten(), bins=256, color=color.lower(), alpha=0.7)
        axes[i, 0].set_title(f'{color} - Przed')
        axes[i, 0].set_xlim(0, 255)

        axes[i, 1].hist(after_pixels[..., i].flatten(), bins=256, color=color.lower(), alpha=0.7)
        axes[i, 1].set_title(f'{color} - Po')
        axes[i, 1].set_xlim(0, 255)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Przykład użycia
k = 9
floyd_steinberg_dithering_rgb("sunflower.png", "sunflower_dithered_rgb.png",k)
