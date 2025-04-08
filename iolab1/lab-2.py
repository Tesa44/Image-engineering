import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def matrix_multiply(matrix, vector):
    result = np.zeros(3)  # Wynikowy wektor [R_new, G_new, B_new]
    for i in range(3):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(3))
    return np.clip(result, 0, 1)  # Ograniczenie wartości do 0.0 - 1.0

def plot(data, title, rows=2, cols=2):
    plot.i += 1
    plt.subplot(rows,cols,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis("off")
plot.i = 0

im = Image.open('sunflower_imresizer.png').convert("RGB")
#Konwersja obrazu na tablicę Numpy z wartościami [0-255]
im_data = np.array(im)
plot(im_data, "Oryginalny", 1,2)
#konwersja na format zmiennoprzecinkowy [0.0-1.0]
float_im = np.array(im_data, dtype=np.float32) / 255.0
print("Przykładowy piksel przed:", im_data[0, 0])
print("Przykładowy piksel po:", float_im[0, 0])

color_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.689, 0.168],
    [0.272, 0.534, 0.131]
], dtype=np.float32)

#Tworzenie nowej macierzy pikseli dla pokolorowanego obrazu wypełnionej zerami
colored_im = np.zeros_like(float_im)
#Przetwarzanie każdego piksela według wzoru
height, width, _ = float_im.shape
for i in range(height):
    for j in range(width):
        colored_im[i, j] = matrix_multiply(color_matrix, float_im[i, j])

plot(colored_im, "Sepia",1,2)
plt.show()

