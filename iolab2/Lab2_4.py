import cv2
import numpy as np
import zlib # importowanie biblioteki zlib do kompresji danych
import matplotlib.pyplot as plt


def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb) # konwertuje obraz z RGB na YCbCr

def ycbcr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB) # konwertuje obraz z YCbCr na RGB


# Funkcja do downsamplingu
def downsample(channel, factor):
    if factor == 1:  # Bez próbkowania
        return channel
    elif factor == 2:  # Co drugi element
        return channel[::2, ::2] # zwraca kanał próbkowany co drugi piksel w obu wymiarach
    elif factor == 4:  # Co czwarty element
        return channel[::4, ::4] # zwraca kanał próbkowany co czwarty piksel w obu wymiarach
    else:
        raise ValueError("Nieprawidłowy współczynnik próbkowania.")


# Funkcja do upsamplingu
def upsample(channel, original_shape):
    return cv2.resize(channel, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST) # przywraca kanał do pierwotnego rozmiaru za pomocą interpolacji najbliższego sąsiada


# Funkcja do podziału na bloki 8x8
def split_into_blocks(channel):
    h, w = channel.shape  # pobiera wysokość i szerokość kanału
    blocks = []  # inicjalizuje listę bloków

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blocks.append(channel[i:i + 8, j:j + 8])  # dodaje blok 8x8 do listy
    return blocks


# Funkcja do kodowania współczynników
def zigzag(block):
    rows, cols = block.shape  # pobiera rozmiar bloku
    result = []  # inicjalizuje listę wynikową

    for sum_idx in range(rows + cols - 1):  # iteruje po przekątnych bloku
        if sum_idx % 2 == 0:
            # dla parzystych przekątnych iteruje od góry do dołu i od lewej do prawej
            for i in range(rows):
                for j in range(cols):
                    if i + j == sum_idx:
                        result.append(block[i][j])
        else:
            # dla nieparzystych przekątnych iteruje od dołu do góry i od prawej do lewej
            for i in range(rows - 1, -1, -1):
                for j in range(cols - 1, -1, -1):
                    if i + j == sum_idx:
                        result.append(block[i][j])

    return np.array(result)  # zwraca zakodowaną tablicę


# Funkcja odwrotna
def inverse_zigzag(array, rows=8, cols=8):
    block = np.zeros((rows, cols))  # inicjalizuje pusty blok
    idx = 0  # indeks dla elementów tablicy wejściowej

    for sum_idx in range(rows + cols - 1):
        if sum_idx % 2 == 0:
            for i in range(rows):
                for j in range(cols):
                    if i + j == sum_idx:
                        block[i][j] = array[idx]
                        idx += 1
        else:
            for i in range(rows - 1, -1, -1):
                for j in range(cols - 1, -1, -1):
                    if i + j == sum_idx:
                        block[i][j] = array[idx]
                        idx += 1
    return block


# Funkcja kompresji danych
def compress_data(data):
    return zlib.compress(data.tobytes()) # kompresuje dane wejściowe za pomocą zlib


def jpeg_algorithm(image_path, sampling_factor):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ycbcr_image = rgb_to_ycbcr(image)

    # rozdzielenie obraz YCbCr na kanały: Y, Cb, Cr
    Y_channel = ycbcr_image[:, :, 0]
    Cb_channel = ycbcr_image[:, :, 1]
    Cr_channel = ycbcr_image[:, :, 2]

    # wykonanie próbkowanie kanałów Cb i Cr zgodnie z podanym współczynnikiem
    Cb_downsampled = downsample(Cb_channel, sampling_factor)
    Cr_downsampled = downsample(Cr_channel, sampling_factor)

    # przywrócenie kanałów Cb i Cr do pierwotnego rozmiaru obrazu
    Cb_upsampled = upsample(Cb_downsampled, Y_channel.shape)
    Cr_upsampled = upsample(Cr_downsampled, Y_channel.shape)

    # tworzenie nowego obrazu YCbCr po próbkowaniu przez połączenie kanałów
    ycbcr_downsampled_image = np.stack((Y_channel, Cb_upsampled, Cr_upsampled), axis=-1)

    # konwertowanie obrazu YCbCr z powrotem na RGB - do wizualizacji
    rgb_downsampled_image = ycbcr_to_rgb(ycbcr_downsampled_image)

    # kompresowanie danych dla każdego kanału za pomocą algorytmu zlib
    compressed_Y = compress_data(Y_channel)
    compressed_Cb = compress_data(Cb_downsampled)
    compressed_Cr = compress_data(Cr_downsampled)

    # obliczanie całkowitego rozmiaru skompresowanego obrazu jako sumę rozmiarów skompresowanych kanałów
    total_size = len(compressed_Y) + len(compressed_Cb) + len(compressed_Cr)

    print(f"Rozmiar skompresowanego obrazu dla próbkowania {sampling_factor}: {total_size} bajtów")

    return rgb_downsampled_image, total_size


# image_path = "../../../../../Downloads/lab2/lab2/rainbow.png"
image_path = "rgb_spectrum.png"

# generowanie obrazów
image_no_sampling, size_no_sampling = jpeg_algorithm(image_path, sampling_factor=1)
image_half_sampling, size_half_sampling = jpeg_algorithm(image_path, sampling_factor=2)
image_quarter_sampling, size_quarter_sampling = jpeg_algorithm(image_path, sampling_factor=4)


plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.imshow(image_no_sampling)
plt.title(f"No sampling (Rozmiar: {size_no_sampling} bajtów)")

plt.subplot(3, 1, 2)
plt.imshow(image_half_sampling)
plt.title(f"Sampling: every 2nd (Rozmiar: {size_half_sampling} bajtów)")

plt.subplot(3, 1, 3)
plt.imshow(image_quarter_sampling)
plt.title(f"Sampling: every 4th (Rozmiar: {size_quarter_sampling} bajtów)")


plt.tight_layout()
plt.show()

# Notatka:
# Obraz bez próbkowania ma pełny rozmiar wynikający z wymiarów WYSOKOŚĆ x SZEROKOŚĆ x KANAŁY x GŁĘBIA (8 bitów).
# Próbkowanie kanałów Cr i Cb co drugi element zmniejsza rozmiar obrazu o około jedną trzecią, a co czwarty element — prawie o połowę.
# Przy głębszym wpatrzeniu różnice w jakości są widoczne przy próbkowaniu co drugi element, a przy próbkowaniu co czwarty element deformacje są jeszcze bardziej zauważalne.
# W przypadku większych obrazów różnice mogą być mniej dostrzegalne, ale przy powiększaniu obrazu utrata jakości staje się znacząca.
