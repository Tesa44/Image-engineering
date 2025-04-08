from pickletools import uint8

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def matrix_multiply(matrix, vector):
    result = np.zeros(3)  # Wynikowy wektor [R_new, G_new, B_new]
    for i in range(3):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(3))
    return result

def plot(data, title):
    plot.i += 1
    plt.subplot(2,3,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis('off')
plot.i = 0

im = Image.open('sunflower_imresizer.png').convert("RGB")
#Konwersja obrazu na tablicę Numpy z wartościami [0-255]
im_data = np.array(im)
plot(im_data, "Oryginalny")
convert_matrix_ycbcr = np.array([
    [0.229, 0.587, 0.114],
    [0.500, -0.418, -0.082],
    [-0.168, -0.331, -0.500]
],dtype=np.float32)

#Tworzenie nowej macierzy pikseli dla pokolorowanego obrazu wypełnionej zerami
transformed_im = np.zeros_like(im_data, dtype=float)
#Przetwarzanie każdego piksela według wzoru
height, width, _ = im_data.shape
for i in range(height):
    for j in range(width):
        transformed_im[i, j] = matrix_multiply(convert_matrix_ycbcr, im_data[i, j])
transformed_im = np.clip(transformed_im + np.array([0, 128, 128]), 0, 255).astype(int)
plot(transformed_im, "Konwersja YCrCb")

# Rozdzielanie składowych Y, Cb, Cr
Y = transformed_im[:, :, 0]  # Pierwszy kanał (luminancja)
Cb = transformed_im[:, :, 1]  # Drugi kanał (chroma blue)
Cr = transformed_im[:, :, 2]  # Trzeci kanał (chroma red)

plot(Y, "Składowa Y")
plot(Cb, "Składowa Cb")
plot(Cr, "Składowa Cr")

plt.show()

