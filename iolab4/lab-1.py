from PIL import Image
import numpy as np

def find_closest_palette_color(pixel,k=2):
    n = 255 // (k-1)
    return round(pixel / n) * n

# def find_closest_palette_color_rgb(pixel,k=2):
#     n = 255 // (k-1)
#     return round(pixel / n) * n



def to_black_white(image_path, output_path):
    image = Image.open(image_path).convert("L")
    pixels = np.array(image, dtype=np.float32)
    output_image = Image.fromarray(pixels.astype(np.uint8))
    output_image.save(output_path)

def floyd_steinberg_dithering(image_path, output_path):
    image = Image.open(image_path).convert("L")  # konwersja do skali szarości
    pixels = np.array(image, dtype=np.float32)

    height, width = pixels.shape

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = find_closest_palette_color(old_pixel,4)
            pixels[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < width:
                pixels[y, x + 1] += quant_error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    pixels[y + 1, x - 1] += quant_error * 3 / 16
                pixels[y + 1, x] += quant_error * 5 / 16
                if x + 1 < width:
                    pixels[y + 1, x + 1] += quant_error * 1 / 16

    # Upewniamy się, że wartości pikseli mieszczą się w przedziale [0, 255]
    pixels = np.clip(pixels, 0, 255)

    # Konwersja z powrotem do obrazu i zapis
    output_image = Image.fromarray(pixels.astype(np.uint8))
    output_image.save(output_path)

floyd_steinberg_dithering("sunflower.png", "sunflower_dithered.png")


