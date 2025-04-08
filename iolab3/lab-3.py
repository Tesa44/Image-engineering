from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
# import lorem
from lorem.text import TextLorem


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
def hide_message(image, message, nbits=1):
    """Hide a message in an image (LSB).
    nbits: number of least significant bits
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    if len(message) > len(image) * nbits:
        raise ValueError("Message is to long :(")
    chunks = [message[i:i + nbits] for i in range(0, len(message),
    nbits)]
    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[i])
        new_byte = byte[:-nbits] + chunk
        image[i] = int(new_byte, 2)
    return image.reshape(shape)
def reveal_message(image, nbits=1, length=0):
    """Reveal the hidden message.
    nbits: number of least significant bits
    length: length of the message in bits.
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
    return message


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



image = load_image("sunflower.png") # Wczytanie obrazka
lorem = TextLorem(srange=(10000,10001))
message = lorem.sentence()
n = 1 # liczba najmłodszych bitów używanych do ukrycia wiadomości

def simulate(original_image, message, nbits):
    new_message = message * nbits
    encoded_message = encode_as_binary_array(new_message) # Zakodowanie wiadomości jako ciąg 0 i 1
    image_with_message = hide_message(original_image, encoded_message, nbits) # Ukrycie wiadomości w obrazku
    # print(image_with_message.shape)
    # input()
    mse = count_mse(original_image, image_with_message)
    return image_with_message, mse

results = [ simulate(image, message, n) for n in range(1,9)]
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
    plt.title(f"Nbits = {idx-1} MSE = {round(mse,5)}")
plt.show()

#Wyświetlanie wykresu
x = [int(i) for i in range(1, 9)]
y = [ result[1] for result in results]
plt.plot(x, y)
plt.xlabel("NBits")
plt.ylabel("MSE")
plt.title("Podobieństwo obrazów w zależności od ilości kodowanych bitów")
plt.show()
