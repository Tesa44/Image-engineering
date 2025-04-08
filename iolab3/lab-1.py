import cv2
from scipy.fftpack import dct, idct
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pyplot as plt
from PIL import Image

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

def process_blocks(blocks, q_matrix):
    result = []
    for block in blocks:
        dct_block = dct2(block - 128)  # przesunięcie zakresu
        quant_block = quantize(dct_block, q_matrix)
        zigzagged = zigzag(quant_block)
        result.append(zigzagged)
    return result

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

def divide_matrices(A, B):
    return np.dot(A, np.linalg.inv(B))

def round_to_int(matrix):
    return np.round(matrix)

def zigzag(block):
    rows, cols = block.shape
    result = []

    for sum_idx in range(rows + cols - 1):
        indices = [(i, sum_idx - i) for i in range(rows) if 0 <= sum_idx - i < cols]
        if sum_idx % 2 == 0:
            indices.reverse()

        for i, j in indices:
            result.append(block[i, j])

    return result

import huffman
from collections import Counter

def flatten(blocks):
    return [val for block in blocks for val in block]

def huffman_encode(data):
    freq = Counter(data)
    codebook = huffman.codebook(freq.items())
    encoded_bits = ''.join(codebook[symbol] for symbol in data)
    return encoded_bits, codebook

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

    # Podział na bloki
    Y_blocks = split_into_blocks(Y)
    Cb_blocks = split_into_blocks(Cb_down)
    Cr_blocks = split_into_blocks(Cr_down)

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

# def compare_images(original, compressed):
#     psnr_val = psnr(original, compressed)
#     print(f"PSNR: {psnr_val:.2f} dB")
#
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1,2,1)
#     plt.imshow(original)
#     plt.title("Oryginał")
#     plt.axis('off')
#
#     plt.subplot(1,2,2)
#     plt.imshow(compressed)
#     plt.title("Po kompresji")
#     plt.axis('off')
#     plt.show()

image = cv2.imread("zad_4.png")
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
    plt.title(f"Quantized Image QF = {QFs[idx-2]} (Size: {size} Bytes)")

# plt.tight_layout()
plt.show()