import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

def convolve(image, kernel):
    """Własna implementacja konwolucji 3x3 dla filtrów górnoprzepustowych."""
    height, width = image.shape
    output = np.zeros((height, width))

    # Przechodzimy przez piksele, omijając krawędzie
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]  # Pobieramy sąsiedztwo 3x3
            output[i, j] = np.sum(region * kernel)  # Splot macierzy

    return output

def plot(data, title, rows=2, cols=2):
    plot.i += 1
    plt.subplot(rows,cols,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis("off")
plot.i = 0

# Wczytywanie obrazu
im = Image.open('sunflower_imresizer.png')

#################### Zadanie 1 - Filtr górnoprzepustowy ####################
im_grayscale = im.convert('L')  #Konwertowanie na skalę szarości
im_data = np.array(im_grayscale, dtype=float)
plot(im_data, 'Oryginalny',1,2)

# Filtr górnoprzepustowy
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
# highpass_3x3 = ndimage.convolve(im_data, kernel)
highpass_3x3 = convolve(im_data, kernel)
plot(highpass_3x3, 'Filtr górnoprzepustowy 3x3',1,2)
plt.show()
