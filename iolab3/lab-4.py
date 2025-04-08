"""Function definitions that are used in LSB steganography."""
from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math

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
def hide_message(image, message, nbits=1, spos=0):
    """Hide a message in an image (LSB).
    nbits: number of least significant bits
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    if len(message) > len(image) * nbits + spos:
        raise ValueError("Message is to long :(")
    chunks = [message[i:i + nbits] for i in range(0, len(message),
    nbits)]
    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[i + spos])
        new_byte = byte[:-nbits] + chunk
        image[i + spos] = int(new_byte, 2)
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

# message = "Moja tajna wiadomość"
# binary = encode_as_binary_array(message)
# print("Binary:", binary)
# message = decode_from_binary_array(binary)
# print("Retrieved message:", message)


original_image = load_image("sunflower.png") # Wczytanie obrazka

def simulate(original_image, message, nbits):
    new_message = message
    encoded_message = encode_as_binary_array(new_message) # Zakodowanie wiadomości jako ciąg 0 i 1
    image_with_message = hide_message(original_image, encoded_message, nbits) # Ukrycie wiadomości w obrazku

    return image_with_message

plt.figure(figsize=(12,10))


plt.show()
