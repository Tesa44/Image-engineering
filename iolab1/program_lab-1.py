import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image

#Funkcje pomocnicze
def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # Jeśli program jest uruchamiany jako EXE
        base_path = sys._MEIPASS
    else:
        # Tryb zwykły (skrypt)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

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

def print_menu():
    print("Wybierz jedna z dostepnych pozycji menu")
    print("1. Zadanie 1 - Filtr Górnoprzepustowy")
    print("2. Zadanie 2 - Konwersja RGB [0-255] na RGB [0.0-1.0")
    print("3. Zadanie 3 - Konwersja RGB do modelu YCbCr")
    print("4. Zadanie 4 - Symulacja transmisji obrazu w systemie DVB")
    print("5. Zadanie 5 - Błąd średniokwadratowy dla zadania 4")
    print("0. - Exit")


# Wczytywanie obrazu
im = Image.open(resource_path('sunflower_imresizer.png'))
choice = -1
print("Inzynieria Obrazow - Laboratorium 1")
print("Autor: Mateusz Tesarewicz 272909")
while choice != 0:
    plot.i = 0
    print_menu()
    choice = int(input())
    if choice == 1:
        #################### Zadanie 1 - Filtr górnoprzepustowy ####################
        im_grayscale = im.convert('L')  # Konwertowanie na skalę szarości
        im_data = np.array(im_grayscale, dtype=float)
        plot(im_data, 'Oryginalny', 1, 2)

        # Prosty filtr górnoprzepustowy
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        highpass_3x3 = ndimage.convolve(im_data, kernel)
        plot(highpass_3x3, 'Filtr górnoprzepustowy 3x3', 1, 2)
        plt.show()
    elif choice == 2:
        #################### Zadanie 2 - Konwersja RGB [0-255] na RGB [0.0-1.0] ####################
        im_rgb = im.convert('RGB')
        im_data = np.array(im_rgb)
        plot(im_data, "Oryginalny", 1, 2)
        # konwersja na format zmiennoprzecinkowy [0.0-1.0]
        float_im = np.array(im_data, dtype=np.float32) / 255.0
        print("Oryginalny zakres wartości RGB:", im_data.dtype)
        print("Nowy zakres wartości RGB:", float_im.dtype)
        print("Przykładowy piksel przed:", im_data[0, 0])
        print("Przykładowy piksel po:", float_im[0, 0])
        print()

        color_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.689, 0.168],
            [0.272, 0.534, 0.131]
        ], dtype=np.float32)

        # Tworzenie nowej macierzy pikseli dla pokolorowanego obrazu wypełnionej zerami
        colored_im = np.zeros_like(float_im)
        # Przetwarzanie każdego piksela według wzoru
        height, width, _ = float_im.shape
        for i in range(height):
            for j in range(width):
                colored_im[i, j] = matrix_multiply(color_matrix, float_im[i, j])

        plot(colored_im, "Sepia", 1, 2)
        plt.show()
    elif choice == 3:
        #################### Zadanie 3 - Konwersja RGB do modelu YCbCr ####################
        im_rgb = im.convert('RGB')
        # Konwersja obrazu na tablicę Numpy z wartościami [0-255]
        im_data = np.array(im_rgb)
        plot(im_data, "Oryginalny",2,3)

        pattern = np.array([
            [0.229, 0.587, 0.144],
            [0.500, -0.418, -0.082],
            [-0.168, -0.331, -0.500]
        ])
        # wzór na konwertowanie RGB do YCrCb
        transformed_im = np.tensordot(im_data, pattern, axes=([2], [0])).astype(int) + np.array([0, 128, 128])

        plot(transformed_im, "Transformacja YCbCr",2,3)

        # Rozdzielanie składowych Y, Cb, Cr
        Y = transformed_im[:, :, 0]  # Pierwszy kanał (luminancja)
        Cb = transformed_im[:, :, 1]  # Drugi kanał (chroma blue)
        Cr = transformed_im[:, :, 2]  # Trzeci kanał (chroma red)

        plot(Y, "Składowa Y",2,3)
        plot(Cb, "Składowa Cb",2,3)
        plot(Cr, "Składowa Cr",2,3)

        plt.show()

    elif choice == 4 or choice == 5:
        #################### Zadanie 4 - Symulacja transmisji obrazu w systemie DVB ####################
        im_rgb = im.convert("RGB")
        # Konwersja obrazu na tablicę Numpy z wartościami [0-255]
        im_data = np.array(im_rgb)
        plot(im_data, "Oryginalny",2,3)

        convert_matrix_ycbcr = np.array([
            [0.229, 0.587, 0.144],
            [0.500, -0.418, -0.082],
            [-0.168, -0.331, -0.500]
        ])
        # wzór YCrCb
        transformed_im = np.tensordot(im_data, convert_matrix_ycbcr, axes=([2], [0])).astype(int) + np.array([0, 128, 128])
        plot(transformed_im, "Transformacja YCbCr",2,3)
        # Rozdzielanie składowych Y, Cb, Cr
        Y = transformed_im[:, :, 0]  # Pierwszy kanał (luminancja)
        Cb = transformed_im[:, :, 1]  # Drugi kanał (chroma blue)
        Cr = transformed_im[:, :, 2]  # Trzeci kanał (chroma red)
        # Downsampling - bierzemy wartości na co drugim wierszu i kolumnie
        Cb_downsampled = Cb[::2, ::2]
        Cr_downsampled = Cr[::2, ::2]

        # Upsampling
        h, w = Cb_downsampled.shape
        Cb_upsampled = np.zeros((h * 2, w * 2), dtype=Cb_downsampled.dtype)
        # Wypełnianie kwadratów 2x2 wartością z downsamplingu
        Cb_upsampled[::2, ::2] = Cb_downsampled  # Oryginalne próbki
        Cb_upsampled[::2, 1::2] = Cb_downsampled  # Powielenie w poziomie
        Cb_upsampled[1::2, ::2] = Cb_downsampled  # Powielenie w pionie
        Cb_upsampled[1::2, 1::2] = Cb_downsampled  # Powielenie w poziomie i pionie

        h, w = Cr_downsampled.shape
        Cr_upsampled = np.zeros((h * 2, w * 2), dtype=Cr_downsampled.dtype)
        # Wypełnianie kwadratów 2x2 wartością z downsamplingu
        Cr_upsampled[::2, ::2] = Cr_downsampled  # Oryginalne próbki
        Cr_upsampled[::2, 1::2] = Cr_downsampled  # Powielenie w poziomie
        Cr_upsampled[1::2, ::2] = Cr_downsampled  # Powielenie w pionie
        Cr_upsampled[1::2, 1::2] = Cr_downsampled  # Powielenie w poziomie i pionie

        plot(Cb_upsampled, 'Cb po upsamplingu',2,3)
        plot(Cr_upsampled, 'Cr po upsamplingu',2,3)
        # Składanie obrazu YCbCr
        rebuilt_im_ycbcr = np.dstack((Y, Cb_upsampled, Cr_upsampled))
        plot(rebuilt_im_ycbcr, 'Odtworzony obraz YCbCr',2,3)

        # Konwersja obrazu YCbCr na RGB za pomocą równania odwrotnego
        rebuilt_im_ycbcr -= np.array([0, 128, 128])
        rebuilt_im_rgb = np.tensordot(rebuilt_im_ycbcr, np.linalg.inv(convert_matrix_ycbcr), axes=([2], [0])).clip(0,255).astype(np.uint8)
        plot(rebuilt_im_rgb, 'Odtworzony obraz RGB',2,3)

        plt.show()

        if choice == 5:
            #################### Zadanie 5 - Błąd średniokwadratowy dla zadania 4 ####################
            #Konwersja z uint8 na int32 w celu uniknięcia błędu podczas liczenia różnicy wartości
            im_data_int32 = im_data.astype(np.int32)
            rebuilt_im_rgb_int32 = rebuilt_im_rgb.astype(np.int32)

            error_sum = 0.0
            num_pixels = np.prod(im_data_int32.shape)  # Liczba wszystkich pikseli w obrazie razy 3 kanały (RGB)
            height = im_data_int32.shape[0]
            width = im_data_int32.shape[1]
            channels = im_data_int32.shape[2]
            # Obliczanie sumy kwadratów różnic dla każdego piksela
            for i in range(height):  # Iteracja po wierszach
                for j in range(width):  # Iteracja po kolumnach
                    for k in range(channels):  # Iteracja po kanałach (RGB)
                        diff = im_data_int32[i, j, k] - rebuilt_im_rgb_int32[i, j, k]
                        error_sum += diff ** 2
            # Obliczanie średniego błędu kwadratowego
            mse = error_sum / num_pixels
            max_mse = height * width * channels * (255 ** 2)
            images_simmilarity = 100 - (mse / max_mse * 100)
            print("MSE wynosi: ", mse)
            print("Obrazy są podobne w", images_simmilarity, "%")
            print()

    elif choice == 0:
        break
    else:
        print("Niepoprawny wybór")