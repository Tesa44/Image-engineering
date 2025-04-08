import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot(data, title):
    plot.i += 1
    plt.subplot(2,2,plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)
    plt.axis('off')
plot.i = 0

im = Image.open('sunflower_imresizer.png').convert("RGB")
#Konwersja obrazu na tablicę Numpy z wartościami [0-255]
data = np.array(im)
plot(data, "Oryginalny")
#Macierz do konwersji z RGB do YCbCr
convert_matrix_ycbcr = np.array([
    [0.229, 0.587, 0.144],
    [0.500, -0.418, -0.082],
    [-0.168, -0.331, -0.500]
])
#wzór YCrCb
transformed_im = np.tensordot(data, convert_matrix_ycbcr, axes=([2], [0])).astype(int) + np.array([0, 128, 128])

# Rozdzielanie składowych Y, Cb, Cr
Y = transformed_im[:, :, 0]  # Pierwszy kanał (luminancja)
Cb = transformed_im[:, :, 1]  # Drugi kanał (chroma blue)
Cr = transformed_im[:, :, 2]  # Trzeci kanał (chroma red)
#Downsampling - bierzemy wartości na co drugim wierszu i kolumnie
Cb_downsampled = Cb[::2, ::2]
Cr_downsampled = Cr[::2, ::2]

#Upsampling
h, w = Cb_downsampled.shape
Cb_upsampled = np.zeros((h*2, w*2), dtype=Cb_downsampled.dtype)
#Wypełnianie kwadratów 2x2 wartością z downsamplingu
Cb_upsampled[::2, ::2] =  Cb_downsampled  # Oryginalne próbki
Cb_upsampled[::2, 1::2] = Cb_downsampled  # Powielenie w poziomie
Cb_upsampled[1::2, ::2] = Cb_downsampled  # Powielenie w pionie
Cb_upsampled[1::2, 1::2] = Cb_downsampled  # Powielenie w poziomie i pionie

h, w = Cr_downsampled.shape
Cr_upsampled = np.zeros((h*2, w*2), dtype=Cr_downsampled.dtype)
#Wypełnianie kwadratów 2x2 wartością z downsamplingu
Cr_upsampled[::2, ::2] =  Cr_downsampled  # Oryginalne próbki
Cr_upsampled[::2, 1::2] = Cr_downsampled  # Powielenie w poziomie
Cr_upsampled[1::2, ::2] = Cr_downsampled  # Powielenie w pionie
Cr_upsampled[1::2, 1::2] = Cr_downsampled  # Powielenie w poziomie i pionie

plot(Cb_upsampled, 'Cb po upsamplingu')
plot(Cr_upsampled, 'Cr po upsamplingu')
#Składanie obrazu YCbCr
rebuilt_im_ycbcr = np.dstack((Y, Cb_upsampled, Cr_upsampled))

#Konwersja obrazu YCbCr na RGB za pomocą równania odwrotnego
rebuilt_im_ycbcr -= np.array([0, 128, 128])
rebuilt_im_rgb = np.tensordot(rebuilt_im_ycbcr, np.linalg.inv(convert_matrix_ycbcr), axes=([2], [0])).clip( 0, 255).astype(np.uint8)
plot(rebuilt_im_rgb, 'Po transmisji')

# Inicjalizacja sumy błędów kwadratowych
data = data.astype(np.int32)
rebuilt_im_rgb = rebuilt_im_rgb.astype(np.int32)

error_sum = 0.0
num_pixels = np.prod(data.shape)  # Liczba pikseli w obrazie
height = data.shape[0]
width = data.shape[1]
channels = data.shape[2]
# Obliczanie sumy kwadratów różnic dla każdego piksela
for i in range(height):  # Iteracja po wierszach
    for j in range(width):  # Iteracja po kolumnach
        for k in range(channels):  # Iteracja po kanałach (RGB)
            diff = data[i, j, k] - rebuilt_im_rgb[i, j, k]
            error_sum += diff ** 2
# Obliczanie średniego błędu kwadratowego
mse = error_sum / num_pixels / channels
max_mse = height * width * channels * (255**2)
images_simmilarity = 100 - (mse/max_mse*100)
print("MSE wynosi: ",mse)
print("Obrazy sa podobne w", images_simmilarity, "%")

plt.show()