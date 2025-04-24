import cv2
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
from lorem.text import TextLorem
import os
import sys

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # Jeśli program jest uruchamiany jako EXE
        base_path = sys._MEIPASS
    else:
        # Tryb zwykły (skrypt)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def convert_color_space(image, conversion):
    return cv2.cvtColor(image, conversion)

def resample_channel(channel, factor, mode="down"):
    if factor == 1:
        return channel
    if mode == "down":
        return channel[::factor, ::factor]
    return cv2.resize(channel, (channel.shape[1] * factor, channel.shape[0] * factor), interpolation=cv2.INTER_NEAREST)

def quantize(block, q_matrix):
    return np.round(block / q_matrix).astype(np.int32)

def dequantize(block, q_matrix):
    return (block * q_matrix).astype(np.float32)


def split_into_blocks(channel):
    h, w = channel.shape
    return [channel[i:i+8, j:j+8] for i in range(0, h, 8) for j in range(0, w, 8)]

#Dysretna transformacja cosinusowa 2D
def dct2(array):
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(array):
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

#Macierz kwantyzacji
_QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

_QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 48, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

def _scale(QF):
    if QF < 50 and QF >= 1:
        scale = np.floor(5000 / QF)
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        raise ValueError('Quality Factor must be in the range [1..99]')
    scale = scale / 100.0
    return scale

def QY(QF=85):
    return _QY * _scale(QF)
def QC(QF=85):
    return _QC * _scale(QF)

def estimate_compressed_size(blocks):
    size = 0
    for block in blocks:
        size += np.count_nonzero(block)
    return size

def simulate_jpeg(image_rgb, QF=85):
    image_ycbcr = convert_color_space(image_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(image_ycbcr)

    # 4:2:0
    Cb_down = resample_channel(Cb, 2)
    Cr_down = resample_channel(Cr, 2)

    QY_matrix = QY(QF)
    QC_matrix = QC(QF)

    # Po kompresji:
    Y_blocks = split_into_blocks(Y.astype(np.float32))
    Cb_blocks = split_into_blocks(Cb_down.astype(np.float32))
    Cr_blocks = split_into_blocks(Cr_down.astype(np.float32))

    Y_dct = [dct2(b - 128) for b in Y_blocks]
    Cb_dct = [dct2(b - 128) for b in Cb_blocks]
    Cr_dct = [dct2(b - 128) for b in Cr_blocks]

    Y_quant = [quantize(b, QY_matrix) for b in Y_dct]
    Cb_quant = [quantize(b, QC_matrix) for b in Cb_dct]
    Cr_quant = [quantize(b, QC_matrix) for b in Cr_dct]

    compressed_size = (
            estimate_compressed_size(Y_quant) +
            estimate_compressed_size(Cb_quant) +
            estimate_compressed_size(Cr_quant)
    )

    def encode_decode(blocks, q_matrix):
        processed_blocks = []
        for block in blocks:
            block = block.astype(np.float32)
            dct_block = dct2(block - 128)
            quant = quantize(dct_block, q_matrix)
            dequant = dequantize(quant, q_matrix)
            idct_block = idct2(dequant) + 128
            idct_block = np.clip(idct_block, 0, 255)
            processed_blocks.append(idct_block)
        return processed_blocks

    # Przetwarzamy kanały
    Y_processed = encode_decode(Y_blocks, QY_matrix)
    Cb_processed = encode_decode(Cb_blocks, QC_matrix)
    Cr_processed = encode_decode(Cr_blocks, QC_matrix)

    def merge_blocks(blocks, h, w):
        out = np.zeros((h, w))
        i = 0
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                out[y:y+8, x:x+8] = blocks[i]
                i += 1
        return out

    h, w = Y.shape
    Y_rec = merge_blocks(Y_processed, h, w)
    Cb_rec = merge_blocks(Cb_processed, h//2, w//2)
    Cr_rec = merge_blocks(Cr_processed, h//2, w//2)

    # Upsampling do pełnego rozmiaru
    Cb_up = cv2.resize(Cb_rec, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_rec, (w, h), interpolation=cv2.INTER_LINEAR)
    ycbcr_rec = cv2.merge([Y_rec, Cr_up, Cb_up])
    rgb_rec = convert_color_space(ycbcr_rec.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    return rgb_rec, compressed_size


def encode_as_binary_array(msg):
    """Encode a message as a binary string."""
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = [ "{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)
def decode_from_binary_array(array):
    """Decode a binary string to utf8."""
    array = [array[i:i+8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))
    array = [ "{:02x}".format(int(el, 2)) for el in array]
    array = "".join(array)
    result = binascii.unhexlify(array)
    return result.decode("utf-8", errors="replace")
def load_image(path, pad=False):
    """Load an image.
    If pad is set then pad an image to multiple of 8 pixels.
    """
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(
        image, ((0, y_pad), (0, x_pad) ,(0, 0)), mode='constant')
    return image
def save_image(path, image):
    """Save an image."""
    plt.imsave(path, image)
def clamp(n, minn, maxn):
    """Clamp the n value to be in range (minn, maxn)."""
    return max(min(maxn, n), minn)
def count_mse(original_image, image):
    error_sum = 0.0
    num_pixels = np.prod(original_image.shape)  # Liczba pikseli w obrazie
    height = original_image.shape[0]
    width = original_image.shape[1]
    channels = original_image.shape[2]
    # Obliczanie sumy kwadratów różnic dla każdego piksela
    for i in range(height):  # Iteracja po wierszach
        for j in range(width):  # Iteracja po kolumnach
            for k in range(channels):  # Iteracja po kanałach (RGB)
                diff = original_image[i, j, k] - image[i, j, k]
                error_sum += diff ** 2
    # Obliczanie średniego błędu kwadratowego
    mse = error_sum / num_pixels / channels
    return mse

def hide_message(image, message, nbits=1, spos=0):
    """Hide a message in an image (LSB).
    nbits: number of least significant bits
    spos: start position to hide message
    """
    nbits = clamp(nbits, 1, 8) # nbits musi być z przedziału 1-8
    shape = image.shape
    image = np.copy(image).flatten() # Tworzenie tablicy jednowymiarowej
    pixels = shape[0] * shape[1] * shape[2]
    if len(message) > len(image) * nbits:
        raise ValueError("Message is to long :(")
    # Podział wiadomości na bloki o długości nbits
    chunks = [message[i:i + nbits] for i in range(0, len(message),
    nbits)]
    for i, chunk in enumerate(chunks):
        # Zamiana każdej wartości RGB na 8 bitów
        byte = "{:08b}".format(image[(i + spos) % pixels])
        # Sklejanie nowego bajta z części niezmienionej i bloku wiadomości
        new_byte = byte[:-nbits] + chunk
        # Zamiana spowrotem na liczbę całkowitą
        image[(i + spos) % pixels] = int(new_byte, 2)
    return image.reshape(shape)

def reveal_message(image, nbits=1, length=0, spos=0):
    """Reveal the hidden message.
    nbits: number of least significant bits
    length: length of the message in bits.
    spos: start position to hide message
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    length_in_pixels = math.ceil(length/nbits)
    pixels = shape[0] * shape[1] * shape[2]
    if len(image) < length_in_pixels or length_in_pixels <= 0:
        length_in_pixels = len(image)
    message = ""
    i = 0
    while i < length_in_pixels:
        byte = "{:08b}".format(image[(i + spos) % pixels])
        message += byte[-nbits:]
        i += 1
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    return message

def hide_image(image, secret_image_path, nbits=1):
    """Hide the image.
            secret_image_path: path to the secret image
            nbits: number of least significant bits
            """
    with open(secret_image_path, "rb") as file:
        secret_img = file.read()
    # Zamiana na ciąg szesnastkowy
    secret_img = secret_img.hex()
    # Podzielenie ciągu na dwójki (po 8 bitów)
    secret_img = [secret_img[i:i + 2] for i in range(0, len(secret_img), 2)]
    # Zamiana hex na bin
    secret_img = ["{:08b}".format(int(el, base=16)) for el in secret_img]
    # Ciąg binarny
    secret_img = "".join(secret_img)
    return hide_message(image, secret_img, nbits), len(secret_img)

def reveal_image(image, length, nbits=1):
    """Reveal the hidden image.
        image: image with hidden image
        nbits: number of least significant bits
        length: length of the image in bits.
        """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    length_in_pixels = math.ceil(length/nbits)
    if len(image) < length_in_pixels or length_in_pixels <= 0:
        length_in_pixels = len(image)
    message = ""
    i = 0
    while i < length_in_pixels:
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
        i += 1
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    # Podział ciągu binarnego na 8 bitów
    secret_img = [message[i:i + 8] for i in range(0, len(message), 8)]
    # Zamiana 8 bitów na hex
    secret_img = ["{:02x}".format(int(el, base=2)) for el in secret_img]
    # Połączenie w jeden ciąg
    secret_img = "".join(secret_img)
    # Przekonwertowanie stringa na dane heksadecymalne
    data = binascii.a2b_hex(secret_img)
    # Utworzenie obrazu jpg
    with open('reveal_image.jpg', 'wb') as file:
        file.write(data)

    return load_image("reveal_image.jpg")

def reveal_image_no_length(image, nbits=1):
    """Reveal the hidden image without giving length of the image.
        image: image with hidden image
        nbits: number of least significant bits
        """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    max_message_length = shape[0] * shape[1] * nbits

    message = ""
    i = 0
    while i < max_message_length:
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
        i += 1
    message = [message[i:i + 8] for i in range(0, len(message), 8)]  #Podział ciągu binarnego na 8 bitów
    message = ["{:02x}".format(int(el, base=2)) for el in message]    #Zamiana 8 bitów na hex

    secret_img = []
    i = 0
    # Przepisywanie danych dopóki nie trafimy na pierwsze wystąpienie numeru stopki (0xFFD9)
    while message[i] != "ff" or message[i+1] != "d9":
        secret_img.append(message[i])
        i+=1
    # Dodajemy ręcznie stopkę na koniec
    secret_img.append("ff")
    secret_img.append("d9")
    secret_img = "".join(secret_img)    # Połączenie w jeden ciąg
    data = binascii.a2b_hex(secret_img) # Przekonwertowanie stringa na dane heksadecymalne
    with open('reveal_image-no_length.jpg', 'wb') as file:
        file.write(data)

    return load_image("reveal_image-no_length.jpg")


def print_menu():
    print("Wybierz jedna z dostepnych pozycji menu")
    print("1. Zadanie 1(5) - Wpływ czynnika QF")
    print("2. Zadanie 1 - Ukrywanie wiadomości w obrazku")
    print("3. Zadanie 2 - Porównanie obrazków dla ilości kodowanych bitów")
    print("4. Zadanie 3 - Zapisywanie i odczytywanie wiadomości od zadanej pozycji")
    print("5. Zadanie 4 - Odzyskiwanie obrazka zakodowanego w obrazku")
    print("6. Zadanie 5 - Odzyskiwanie obrazka bez podawania długości")
    print("0. - Exit")

print("Inzynieria Obrazow - Laboratorium 1")
print("Autor: Mateusz Tesarewicz 272909")
choice = -1
im_spectrum_path = resource_path("spectrum.png")
im_sunflower_path = resource_path("sunflower.png")
im_rembrandt_path = resource_path("rembrandt.png")
im_spanish_path = resource_path("spanish.jpg")
im_hidden = resource_path("hidden-image.jpg")
while choice != 0:
    print_menu()
    choice = int(input())
    if choice == 1:
        image = cv2.imread(im_spectrum_path)
        compressed_50, size_50 = simulate_jpeg(image, QF=50)
        # compare_images(image, compressed_50)
        compressed_90, size_90 = simulate_jpeg(image, QF=90)
        # compare_images(image, compressed_90)
        QFs = [50, 90]
        results = [simulate_jpeg(image, QF=qf) for qf in QFs]

        plt.figure(figsize=(12, 10))
        plt.subplot(len(QFs) + 1, 1, 1)
        plt.axis('off')
        plt.imshow(image)
        plt.title(f"Original Image (Size: {image.nbytes} Bytes")

        for idx, (img, size) in enumerate(results, 2):
            plt.subplot(len(QFs) + 1, 1, idx)
            plt.axis("off")
            plt.imshow(img)
            plt.title(f"Quantized Image QF = {QFs[idx - 2]} (Size: {size} Bytes)")

        plt.show()

    elif choice == 2:
        original_image = load_image(im_sunflower_path)  # Wczytanie obrazka
        message = "Nad ludy i nad krole podniesiony; Na trzech stoi koronach, a sam bez korony; A zycie jego - trud trudow, A tytul jego - lud ludow; Z matki obcej, krew jego dawne bohatery, A imie jego czterdziesci i cztery."
        n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości
        message = encode_as_binary_array(message)  # Zakodowanie wiadomości jako ciąg 0 i 1
        image_with_message = hide_message(original_image, message, n)  # Ukrycie wiadomości w obrazku
        save_image("image_with_message.png", image_with_message)  # Zapisanie obrazka w formacie PNG
        save_image("image_with_message.jpg", image_with_message)  # Zapisanie obrazka w formacie JPG
        # Wczytanie obrazka PNG
        image_with_message_png = load_image("image_with_message.png")
        # Wczytanie obrazka JPG
        image_with_message_jpg = load_image("image_with_message.jpg")

        # Odczytanie ukrytej wiadomości z PNG
        secret_message_png = decode_from_binary_array(
            reveal_message(image_with_message_png, nbits=n,
                           length=len(message)))
        # Odczytanie ukrytej wiadomości z JPG
        secret_message_jpg = decode_from_binary_array(
            reveal_message(image_with_message_jpg, nbits=n,
                           length=len(message)))

        print(secret_message_png)
        print(secret_message_jpg)
        # Wyświetlenie obrazków
        f, ar = plt.subplots(2, 2)
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.imshow(original_image)
        plt.title("Oryginalny obraz")
        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.imshow(image_with_message)
        plt.title("Obraz z ukrytym obrazkiem")
        plt.subplot(2, 2, 3)
        plt.axis("off")
        plt.imshow(image_with_message_png)
        plt.title("Obraz PNG")
        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.imshow(image_with_message_jpg)
        plt.title("Obraz JPG")
        plt.show()

    elif choice == 3:
        image = load_image(im_sunflower_path)  # Wczytanie obrazka
        lorem = TextLorem(srange=(10000, 10001))  # funkcja do generowania tekstu
        message = lorem.sentence()
        n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości

        def simulate(original_image, message, nbits):
            new_message = message * nbits  # Zwiększamy rozmiar wiadomości, gdy możemy kodować na n bitach
            encoded_message = encode_as_binary_array(new_message)  # Zakodowanie wiadomości jako ciąg 0 i 1
            image_with_message = hide_message(original_image, encoded_message, nbits)  # Ukrycie wiadomości w obrazku
            mse = count_mse(original_image, image_with_message)  # Liczymy MSE
            return image_with_message, mse

        results = [simulate(image, message, n) for n in range(1, 9)]
        # Wyświetlenie obrazków
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 3, 1)
        plt.axis('off')
        plt.imshow(image)
        plt.title(f"Original Image")

        for idx, (img, mse) in enumerate(results, 2):
            plt.subplot(3, 3, idx)
            plt.axis("off")
            plt.imshow(img)
            plt.title(f"Nbits = {idx - 1} MSE = {round(mse, 5)}")
        plt.show()
        # Wyświetlanie wykresu
        x = [int(i) for i in range(1, 9)]
        y = [result[1] for result in results]
        plt.plot(x, y)
        plt.xlabel("NBits")
        plt.ylabel("MSE")
        plt.title("Podobieństwo obrazów w zależności od ilości kodowanych bitów")
        plt.show()

    elif choice == 4:
        original_image = load_image(im_sunflower_path)  # Wczytanie obrazka

        def simulate(original_image, message, nbits, spos):
            new_message = message
            encoded_message = encode_as_binary_array(new_message)  # Zakodowanie wiadomości jako ciąg 0 i 1
            image_with_message = hide_message(original_image, encoded_message, nbits, spos)  # Ukrycie wiadomości w obrazku
            return image_with_message

        lorem = TextLorem(srange=(50000, 50001))
        message = lorem.sentence()
        results = [simulate(original_image, message, n, 200000) for n in range(6, 9)]
        for idx, img in enumerate(results, 1):
            plt.subplot(1, 3, idx)
            plt.axis("off")
            plt.imshow(img)
            plt.title(f"Nbits = {idx + 5}")
        plt.show()

    elif choice == 5:
        image = load_image(im_rembrandt_path)
        image_with_secret, length_of_secret = hide_image(image, im_spanish_path, 1)
        secret_image = reveal_image(image_with_secret, length_of_secret)
        # Wyświetlanie
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Oryginalny obraz")
        plt.subplot(2, 2, 2)
        plt.imshow(image_with_secret)
        plt.title("Obraz z ukrytym obrazkiem")
        plt.imshow(image_with_secret)
        plt.subplot(2, 2, 3)
        plt.imshow(secret_image)
        plt.title("Ukryty obraz")
        plt.show()

    elif choice == 6:
        image = load_image(im_rembrandt_path)
        image_with_secret, length_of_secret = hide_image(image, im_hidden, 1)
        secret_image = reveal_image_no_length(image_with_secret)
        # Wyświetlanie
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Oryginalny obraz")
        plt.subplot(2, 2, 2)
        plt.imshow(image_with_secret)
        plt.title("Obraz z ukrytym obrazkiem")
        plt.imshow(image_with_secret)
        plt.subplot(2, 2, 3)

        plt.imshow(secret_image)
        plt.title("Ukryty obraz")
        plt.show()

    elif choice == 0:
        break
    else:
        print("Niepoprawny wybór")