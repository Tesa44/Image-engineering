from os import error

import numpy as np
import cv2
import struct
import zlib
from scipy.fftpack import dct, idct
from PIL import Image
from collections import Counter
import heapq


# *** ETAP 1: Konwersja RGB -> YCbCr ***
def rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def ycbcbr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)


# *** ETAP 2: Próbkowanie (Downsampling) ***
def downsampling(channel, factor=1):

    # subsampled = np.zeros((h, w, c), dtype=np.uint8)
    # subsampled[:, :, 0] = image[:, :, 0]  # Luma (Y) bez zmian
    # subsampled[:, :, 1] = image[:, ::factor, 1]  # Chrominancja Cb
    # subsampled[:, :, 2] = image[:, ::factor, 2]  # Chrominancja Cr
    return channel[::factor, ::factor]
    # Y = transformed_im[:, :, 0]  # Pierwszy kanał (luminancja)
    # Cb = transformed_im[:, :, 1]  # Drugi kanał (chroma blue)
    # Cr = transformed_im[:, :, 2]  # Trzeci kanał (chroma red)
    # # Downsampling - bierzemy wartości na co drugim wierszu i kolumnie
    # Cb_downsampled = Cb[::2, ::2]
    # Cr_downsampled = Cr[::2, ::2]
    # return subsampled

# def upsampling(channel, factor=1):
#     h, w = channel.shape
#     if factor == 1:
#         return channel
#     elif factor == 2:
#         scale_w = 1
#         scale_h = 2
#     elif factor == 4:
#         scale_w = 2
#         scale_h = 2
#     else:
#         return error("unknow factor")
#
#     upsampled = np.zeros((h * 2, w * 2), dtype=Cr_downsampled.dtype)
#     # Wypełnianie kwadratów 2x2 wartością z downsamplingu
#     Cr_upsampled[::2, ::2] = Cr_downsampled  # Oryginalne próbki
#     Cr_upsampled[::scale_h, 1::scale_w] = Cr_downsampled  # Powielenie w poziomie
#     Cr_upsampled[1::scale_h, ::scale_w] = Cr_downsampled  # Powielenie w pionie
#     Cr_upsampled[1::scale_h, 1::scale_w] = Cr_downsampled  # Powielenie w poziomie i pionie

def upsampling(channel, original_shape):
    return cv2.resize(channel, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST) # przywraca kanał do pierwotnego rozmiaru za pomocą interpolacji najbliższego sąsiada


# *** ETAP 3: Podział na bloki 8x8 ***
def split_into_blocks(channel):
    h, w = channel.shape
    blocks = []
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            blocks.append(channel[y:y + 8, x:x + 8])
    return blocks


# *** ETAP 4: Transformata DCT ***
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


# *** ETAP 5: Kwantyzacja ***
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


def quantize(block, Q=Q_MATRIX):
    return np.round(block / Q).astype(np.int16)


# *** ETAP 6: Kodowanie RLE (Run-Length Encoding) ***
def run_length_encode(block):
    flat = block.flatten()
    encoded = []
    count = 0
    prev = flat[0]

    for val in flat:
        if val == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = val
            count = 1
    encoded.append((prev, count))
    return encoded


# *** ETAP 7: Kodowanie Huffmana ***

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    frequency = Counter(data)
    heap = [HuffmanNode(k, v) for k, v in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def build_huffman_codes(tree, prefix="", code_dict={}):
    if tree is None:
        return

    if tree.value is not None:
        code_dict[tree.value] = prefix

    build_huffman_codes(tree.left, prefix + "0", code_dict)
    build_huffman_codes(tree.right, prefix + "1", code_dict)

    return code_dict


def huffman_encode(data):
    tree = build_huffman_tree(data)
    huffman_codes = build_huffman_codes(tree)
    encoded_data = "".join([huffman_codes[val] for val in data])
    return encoded_data, tree


# *** ETAP 8: Zapis do pliku JPEG ***
def save_jpeg(filename, image):
    ycbcr = rgb_to_ycbcr(image)
    # subsampled = chroma_subsampling(ycbcr)
    Y = ycbcr[:, :, 0]  # Pierwszy kanał (luminancja)
    Cb = ycbcr[:, :, 1]  # Drugi kanał (chroma blue)
    Cr = ycbcr[:, :, 2]  # Trzeci kanał (chroma red)
    # Downsampling - bierzemy wartości na co drugim wierszu i kolumnie
    Cb_downsampled = Cb[::2, ::2]
    Cr_downsampled = Cr[::2, ::2]
    blocks = split_into_blocks(Y)  # Tylko Y
    dct_blocks = [apply_dct(block) for block in blocks]
    quantized_blocks = [quantize(block) for block in dct_blocks]

    rle_encoded = [run_length_encode(block) for block in quantized_blocks]
    flattened_data = [item for sublist in rle_encoded for item in sublist]

    huffman_encoded, _ = huffman_encode(flattened_data)

    with open(filename, "wb") as f:
        f.write(b"\xFF\xD8")  # Start JPEG
        f.write(b"\xFF\xDB")  # Define Quantization Table
        f.write(struct.pack(">H", len(Q_MATRIX.flatten()) + 2))
        f.write(Q_MATRIX.flatten().astype(np.uint8).tobytes())

        f.write(b"\xFF\xC0")  # Start of Frame
        f.write(struct.pack(">BHHBB", 8, image.shape[0], image.shape[1], 3, 1))

        f.write(b"\xFF\xDA")  # Start of Scan
        compressed_data = zlib.compress(huffman_encoded.encode(), 9)
        f.write(struct.pack(">H", len(compressed_data) + 2))
        f.write(compressed_data)

        f.write(b"\xFF\xD9")  # End of Image


# *** GŁÓWNY PROGRAM ***
if __name__ == "__main__":
    width, height = 128, 128
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        img[:, x] = [x % 256, (x * 2) % 256, (x * 3) % 256]  # Gradient RGB

    save_jpeg("generated.jpeg", img)
    print("Plik JPEG został wygenerowany!")
