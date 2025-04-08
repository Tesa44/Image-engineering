import cv2
import numpy as np
import zlib
import matplotlib.pyplot as plt
from PIL import Image


def convert_color_space(image, conversion):
    return cv2.cvtColor(image, conversion)


def resample_channel(channel, factor, mode="down"):
    if factor == 1:
        return channel
    if mode == "down":
        return channel[::factor, ::factor]
    return cv2.resize(channel, (channel.shape[1] * factor, channel.shape[0] * factor), interpolation=cv2.INTER_NEAREST)


def process_blocks(channel, block_size=8, encode=True):
    h, w = channel.shape
    processed = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i + block_size, j:j + block_size]
            processed.append(zigzag(block) if encode else inverse_zigzag(block))

    return processed

def split_into_blocks(channel):
    h, w = channel.shape
    return [channel[i:i+8, j:j+8] for i in range(0, h, 8) for j in range(0, w, 8)]


def zigzagNew(block):
    indices = sorted(((i, j) for i in range(8) for j in range(8)), key=lambda x: (x[0] + x[1], (x[0] + x[1]) % 2))
    return np.array([block[i, j] for i, j in indices])

def inverse_zigzagNew(array):
    block = np.zeros((8, 8))
    indices = sorted(((i, j) for i in range(8) for j in range(8)), key=lambda x: (x[0] + x[1], (x[0] + x[1]) % 2))
    for idx, (i, j) in enumerate(indices):
        block[i, j] = array[idx]
    return block

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

def inverse_zigzag(array, rows=8, cols=8):
    block = np.zeros((rows, cols))
    idx = 0

    for sum_idx in range(rows + cols - 1):
        indices = [(i, sum_idx - i) for i in range(rows) if 0 <= sum_idx - i < cols]
        if sum_idx % 2 == 0:
            indices.reverse()

        for i, j in indices:
            block[i, j] = array[idx]
            idx += 1

    return block


def compress_data(data):
    return zlib.compress(data.tobytes())


def jpeg_simulation(image_path, sampling_factor):
    image = convert_color_space(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    ycbcr = convert_color_space(image, cv2.COLOR_RGB2YCrCb)

    Y, Cb, Cr = [ycbcr[:, :, i] for i in range(3)]
    Cb_ds, Cr_ds = [resample_channel(ch, sampling_factor, "down") for ch in (Cb, Cr)]
    Cb_us, Cr_us = [resample_channel(ch, sampling_factor, "up") for ch in (Cb_ds, Cr_ds)]

    reconstructed_ycbcr = np.stack((Y, Cb_us, Cr_us), axis=-1)
    compressed_image = convert_color_space(reconstructed_ycbcr, cv2.COLOR_YCrCb2RGB)

    compressed_sizes = sum(len(compress_data(ch)) for ch in (Y, Cb_ds, Cr_ds))
    print(f"Compressed size ({sampling_factor}x sampling): {compressed_sizes} bytes")

    return compressed_image, compressed_sizes


image_path = "rgb_spectrum.png"

sampling_factors = [1, 2, 4]
results = [jpeg_simulation(image_path, factor) for factor in sampling_factors]

plt.figure(figsize=(12, 10))
for idx, (img, size) in enumerate(results, 1):
    plt.subplot(len(sampling_factors), 1, idx)
    plt.axis("off")
    plt.imshow(img)
    plt.title(f"Sampling {sampling_factors[idx - 1]}x (Size: {size} bytes)")

plt.tight_layout()
plt.show()
